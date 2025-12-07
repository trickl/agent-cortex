package com.agentcortex.worker;

import java.nio.file.Path;

final class RunPlanRequest {
    final String requestId;
    final String planClass;
    final Path classesDir;
    final boolean captureTrace;
    final String planSource;
    final String toolStubSource;

    RunPlanRequest(String requestId, String planClass, Path classesDir, boolean captureTrace,
                   String planSource, String toolStubSource) {
        this.requestId = requestId;
        this.planClass = planClass;
        this.classesDir = classesDir;
        this.captureTrace = captureTrace;
        this.planSource = planSource;
        this.toolStubSource = toolStubSource;
    }
}
