package com.agentcortex.worker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

/** Entry point for the long-lived Janino worker process. */
public final class WorkerMain {
    private WorkerMain() {
    }

    public static void main(String[] args) throws Exception {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))) {
            Worker worker = new Worker(new WorkerIO(reader, writer));
            worker.run();
        }
    }
}
