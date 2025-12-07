package com.agentcortex.worker;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

/** Utility for newline-delimited JSON communication. */
final class WorkerIO {
    private final BufferedReader reader;
    private final BufferedWriter writer;
    private final Gson gson = new Gson();
    private final AtomicLong messageCounter = new AtomicLong();

    WorkerIO(BufferedReader reader, BufferedWriter writer) {
        this.reader = reader;
        this.writer = writer;
    }

    synchronized void write(JsonObject payload) throws IOException {
        payload.addProperty("seq", messageCounter.incrementAndGet());
        writer.write(gson.toJson(payload));
        writer.write('\n');
        writer.flush();
    }

    synchronized JsonObject read() throws IOException {
        String line = reader.readLine();
        if (line == null) {
            throw new IOException("Worker stdin closed");
        }
        return JsonParser.parseString(line).getAsJsonObject();
    }
}
