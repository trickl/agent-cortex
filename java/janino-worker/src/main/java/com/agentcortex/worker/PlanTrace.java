package com.agentcortex.worker;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/** Collects execution trace events to return with the final payload. */
final class PlanTrace {
    private final boolean enabled;
    private final JsonArray events = new JsonArray();

    PlanTrace(boolean enabled) {
        this.enabled = enabled;
    }

    void record(String type, JsonObject payload) {
        if (!enabled) {
            return;
        }
        JsonObject event = new JsonObject();
        event.addProperty("type", type);
        if (payload != null) {
            for (String key : payload.keySet()) {
                JsonElement value = payload.get(key);
                event.add(key, value);
            }
        }
        events.add(event);
    }

    JsonArray snapshot() {
        return events.deepCopy();
    }
}
