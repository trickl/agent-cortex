package com.agentcortex.worker;

import java.lang.reflect.Method;
import java.util.function.BiFunction;

/** Binds the generated PlanningToolRuntime helper to a Java-side invoker. */
final class PlanRuntimeInvoker {
    private final Method setInvoker;
    private final Method clearInvoker;

    PlanRuntimeInvoker(ClassLoader loader) throws ReflectiveOperationException {
        Class<?> runtimeClass = loader.loadClass("PlanningToolRuntime");
        this.setInvoker = runtimeClass.getMethod("setInvoker", BiFunction.class);
        this.clearInvoker = runtimeClass.getMethod("clearInvoker");
    }

    void install(BiFunction<String, Object[], Object> invoker) throws ReflectiveOperationException {
        setInvoker.invoke(null, invoker);
    }

    void clear() throws ReflectiveOperationException {
        clearInvoker.invoke(null);
    }
}
