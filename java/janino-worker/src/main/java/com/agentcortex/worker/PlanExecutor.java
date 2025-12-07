package com.agentcortex.worker;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;

/** Loads previously compiled classes and executes the Java plan. */
final class PlanExecutor {
    private static final Gson GSON = new Gson();

    private final WorkerIO io;

    PlanExecutor(WorkerIO io) {
        this.io = io;
    }

    PlanExecutionResult execute(RunPlanRequest request) throws Exception {
        Path classesDir = request.classesDir;
        PlanTrace trace = new PlanTrace(request.captureTrace);
        JsonObject startPayload = jsonBuilder();
        startPayload.addProperty("planClass", request.planClass);
        trace.record("execution_start", startPayload);
        PlanRuntimeInvoker runtimeInvoker = null;
        try (URLClassLoader loader = new URLClassLoader(new URL[]{classesDir.toUri().toURL()}, WorkerMain.class.getClassLoader())) {
            Class<?> planClass = loader.loadClass(request.planClass);
            Object instance = planClass.getDeclaredConstructor().newInstance();
            Method mainMethod = resolveMainMethod(planClass);
            Class<?> toolErrorClass = loader.loadClass("ToolError");

            ToolInvocationGateway gateway = new ToolInvocationGateway(io, request.requestId, trace, toolErrorClass);
            runtimeInvoker = new PlanRuntimeInvoker(loader);
            runtimeInvoker.install(gateway);
            Object returnValue = null;
            JsonArray failureErrors = null;
            boolean success = false;
            ByteArrayOutputStream stdoutBuffer = new ByteArrayOutputStream();
            ByteArrayOutputStream stderrBuffer = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            PrintStream originalErr = System.err;
            PrintStream stdoutCapture = new PrintStream(stdoutBuffer, true, StandardCharsets.UTF_8);
            PrintStream stderrCapture = new PrintStream(stderrBuffer, true, StandardCharsets.UTF_8);
            System.setOut(stdoutCapture);
            System.setErr(stderrCapture);
            try {
                Object[] invokeArgs = buildMainArgs(mainMethod.getParameterCount());
                returnValue = mainMethod.invoke(instance, invokeArgs);
                success = true;
                JsonObject endPayload = jsonBuilder();
                endPayload.addProperty("status", "success");
                trace.record("execution_end", endPayload);
            } catch (InvocationTargetException ex) {
                Throwable rootCause = ex.getCause() != null ? ex.getCause() : ex;
                failureErrors = formatErrors(rootCause);
                JsonObject endPayload = jsonBuilder();
                endPayload.addProperty("status", "failure");
                trace.record("execution_end", endPayload);
            } catch (Exception ex) {
                failureErrors = formatErrors(ex);
                JsonObject endPayload = jsonBuilder();
                endPayload.addProperty("status", "failure");
                trace.record("execution_end", endPayload);
            } finally {
                stdoutCapture.flush();
                stderrCapture.flush();
                System.setOut(originalOut);
                System.setErr(originalErr);
                if (runtimeInvoker != null) {
                    runtimeInvoker.clear();
                }
            }
            recordCapturedOutput(trace, stdoutBuffer, "stdout");
            recordCapturedOutput(trace, stderrBuffer, "stderr");
            if (success) {
                return PlanExecutionResult.success(request.requestId, returnValue, trace.snapshot());
            }
            JsonArray errors = failureErrors != null ? failureErrors : new JsonArray();
            return PlanExecutionResult.failure(request.requestId, errors, trace.snapshot());
        } catch (Exception ex) {
            JsonArray errors = formatErrors(ex);
            JsonObject endPayload = jsonBuilder();
            endPayload.addProperty("status", "failure");
            trace.record("execution_end", endPayload);
            return PlanExecutionResult.failure(request.requestId, errors, trace.snapshot());
        } finally {
            if (runtimeInvoker != null) {
                try {
                    runtimeInvoker.clear();
                } catch (ReflectiveOperationException ignored) {
                    // best-effort cleanup
                }
            }
        }
    }

    private Method resolveMainMethod(Class<?> planClass) throws NoSuchMethodException {
        try {
            return planClass.getMethod("main");
        } catch (NoSuchMethodException ex) {
            return planClass.getMethod("main", String[].class);
        }
    }

    private Object[] buildMainArgs(int parameterCount) {
        if (parameterCount == 0) {
            return new Object[0];
        }
        if (parameterCount == 1) {
            return new Object[]{new String[0]};
        }
        throw new IllegalArgumentException("Planner main method must accept zero arguments or a single String[] parameter");
    }

    private static JsonObject jsonBuilder() {
        return new JsonObject();
    }

    private void recordCapturedOutput(PlanTrace trace, ByteArrayOutputStream buffer, String stream) {
        if (buffer.size() == 0) {
            return;
        }
        String data = buffer.toString(StandardCharsets.UTF_8);
        if (data.isEmpty()) {
            return;
        }
        JsonObject payload = jsonBuilder();
        payload.addProperty("stream", stream);
        payload.addProperty("data", data);
        trace.record("plan_output", payload);
    }

    private JsonArray formatErrors(Throwable error) {
        JsonArray errors = new JsonArray();
        errors.add(formatError(error));
        return errors;
    }

    private JsonObject formatError(Throwable error) {
        JsonObject payload = new JsonObject();
        payload.addProperty("type", classify(error));
        payload.addProperty("message", error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage());
        payload.addProperty("exception", error.getClass().getName());
        JsonArray stack = new JsonArray();
        Arrays.stream(error.getStackTrace()).forEach(frame -> {
            JsonObject frameJson = new JsonObject();
            frameJson.addProperty("class", frame.getClassName());
            frameJson.addProperty("method", frame.getMethodName());
            frameJson.addProperty("line", frame.getLineNumber());
            frameJson.addProperty("file", frame.getFileName());
            stack.add(frameJson);
        });
        payload.add("stack", stack);
        Throwable cause = error.getCause();
        if (cause != null && cause != error) {
            payload.add("cause", formatError(cause));
        }
        return payload;
    }

    private String classify(Throwable error) {
        String name = error.getClass().getSimpleName();
        if ("ToolError".equals(name)) {
            return "tool_error";
        }
        if (error instanceof IOException) {
            return "io_error";
        }
        return "runtime_error";
    }

    /** Handles syscall requests over the Python bridge. */
    private static final class ToolInvocationGateway implements BiFunction<String, Object[], Object> {
        private final WorkerIO io;
        private final String requestId;
        private final PlanTrace trace;
        private final Class<?> toolErrorClass;
        private final AtomicLong callCounter = new AtomicLong();

        ToolInvocationGateway(WorkerIO io, String requestId, PlanTrace trace, Class<?> toolErrorClass) {
            this.io = io;
            this.requestId = requestId;
            this.trace = trace;
            this.toolErrorClass = toolErrorClass;
        }

        @Override
        public Object apply(String toolName, Object[] args) {
            long callId = callCounter.incrementAndGet();
            JsonObject payload = new JsonObject();
            payload.addProperty("type", "syscall_request");
            payload.addProperty("requestId", requestId);
            payload.addProperty("callId", callId);
            payload.addProperty("tool", toolName);
            payload.add("args", GSON.toJsonTree(args == null ? new Object[0] : args));
            JsonObject startPayload = jsonBuilder();
            startPayload.addProperty("name", toolName);
            trace.record("syscall_start", startPayload);
            try {
                io.write(payload);
                JsonObject response = awaitResponse(callId);
                boolean success = response.get("success").getAsBoolean();
                if (success) {
                    JsonObject endPayload = jsonBuilder();
                    endPayload.addProperty("name", toolName);
                    trace.record("syscall_end", endPayload);
                    return decodeResult(response);
                }
                JsonObject errorPayload = jsonBuilder();
                errorPayload.addProperty("name", toolName);
                trace.record("syscall_error", errorPayload);
                JsonObject error = response.getAsJsonObject("error");
                String message = error.has("message") ? error.get("message").getAsString() : "Tool invocation failed";
                throwToolError(message);
            } catch (RuntimeException ex) {
                throw ex;
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
            throw new IllegalStateException("Syscall gateway returned without a response");
        }

        private JsonObject awaitResponse(long callId) throws IOException {
            while (true) {
                JsonObject message = io.read();
                if (!message.has("type")) {
                    continue;
                }
                if (!"syscall_response".equals(message.get("type").getAsString())) {
                    continue;
                }
                if (message.get("callId").getAsLong() != callId) {
                    continue;
                }
                return message;
            }
        }

        private void throwToolError(String message) throws ReflectiveOperationException {
            try {
                throw (RuntimeException) toolErrorClass.getConstructor(String.class).newInstance(message);
            } catch (NoSuchMethodException e) {
                RuntimeException runtime = (RuntimeException) toolErrorClass.getConstructor().newInstance();
                throw runtime;
            }
        }

        private Object decodeResult(JsonObject response) {
            if (!response.has("result") || response.get("result").isJsonNull()) {
                return null;
            }
            return GSON.fromJson(response.get("result"), Object.class);
        }
    }
}
