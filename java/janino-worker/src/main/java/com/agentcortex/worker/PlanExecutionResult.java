package com.agentcortex.worker;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;

/** Serialized payload returned to the Python coordinator. */
final class PlanExecutionResult {
    private static final Gson GSON = new Gson();

    private final String requestId;
    private final boolean success;
    private final JsonElement returnValue;
    private final JsonArray errors;
    private final JsonArray trace;

    private PlanExecutionResult(String requestId, boolean success, JsonElement returnValue,
                                JsonArray errors, JsonArray trace) {
        this.requestId = requestId;
        this.success = success;
        this.returnValue = returnValue;
        this.errors = errors;
        this.trace = trace;
    }

    static PlanExecutionResult success(String requestId, Object returnValue, JsonArray trace) {
        JsonElement serialized = returnValue == null ? JsonNull.INSTANCE : GSON.toJsonTree(returnValue);
        return new PlanExecutionResult(requestId, true, serialized, new JsonArray(), trace);
    }

    static PlanExecutionResult failure(String requestId, JsonArray errors, JsonArray trace) {
        return new PlanExecutionResult(requestId, false, JsonNull.INSTANCE, errors, trace);
    }

    JsonObject toJson() {
        JsonObject root = new JsonObject();
        root.addProperty("type", "result");
        root.addProperty("requestId", requestId);
        root.addProperty("success", success);
        root.add("returnValue", returnValue);
        root.add("errors", errors);
        root.add("trace", trace);
        return root;
    }
}
