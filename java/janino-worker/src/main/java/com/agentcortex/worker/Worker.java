package com.agentcortex.worker;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Coordinates inbound messages for plan execution. */
final class Worker {
    private final WorkerIO io;
    private final Gson gson = new Gson();

    Worker(WorkerIO io) {
        this.io = io;
    }

    void run() throws Exception {
        JsonObject ready = new JsonObject();
        ready.addProperty("type", "ready");
        io.write(ready);
        while (true) {
            JsonObject message = io.read();
            String type = message.get("type").getAsString();
            switch (type) {
                case "run_plan" -> handleRunPlan(message);
                case "shutdown" -> {
                    JsonObject response = new JsonObject();
                    response.addProperty("type", "shutdown_ack");
                    io.write(response);
                    return;
                }
                default -> {
                    JsonObject error = new JsonObject();
                    error.addProperty("type", "error");
                    error.addProperty("message", "Unknown message type: " + type);
                    io.write(error);
                }
            }
        }
    }

    private void handleRunPlan(JsonObject payload) throws Exception {
        RunPlanRequest request = parseRunPlanRequest(payload);
        PlanExecutor executor = new PlanExecutor(io);
        PlanExecutionResult result = executor.execute(request);
        io.write(result.toJson());
    }

    private RunPlanRequest parseRunPlanRequest(JsonObject payload) throws IOException {
        String requestId = payload.get("requestId").getAsString();
        String planClass = payload.get("planClass").getAsString();
        String classesDirRaw = payload.get("classesDir").getAsString();
        Path classesDir = Path.of(classesDirRaw);
        if (!Files.isDirectory(classesDir)) {
            throw new IOException("classesDir does not exist: " + classesDir);
        }
        boolean captureTrace = payload.has("captureTrace") && payload.get("captureTrace").getAsBoolean();
        String planSource = payload.has("planSource") && !payload.get("planSource").isJsonNull()
            ? payload.get("planSource").getAsString()
            : null;
        String toolStubSource = payload.has("toolStubSource") && !payload.get("toolStubSource").isJsonNull()
            ? payload.get("toolStubSource").getAsString()
            : null;
        return new RunPlanRequest(requestId, planClass, classesDir, captureTrace, planSource, toolStubSource);
    }
}
