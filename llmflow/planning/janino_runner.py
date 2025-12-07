"""Bridge that executes Java plans inside a long-lived Janino JVM worker."""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry

from .execution_artifacts import PlanExecutionArtifacts
from .java_plan_analysis import analyze_java_plan
from .plan_runner import detect_stub_method_errors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JANINO_JAR = PROJECT_ROOT / "third_party" / "janino" / "janino-3.1.11.jar"
GSON_JAR = PROJECT_ROOT / "third_party" / "gson" / "gson-2.11.0.jar"
WORKER_SRC = PROJECT_ROOT / "java" / "janino-worker" / "src" / "main" / "java"
WORKER_BUILD = PROJECT_ROOT / "build" / "janino_worker" / "classes"


class JaninoWorkerError(RuntimeError):
    """Raised when the JVM worker cannot be started or communicated with."""


def _ensure_dependency(path: Path, description: str) -> None:
    if not path.exists():
        raise JaninoWorkerError(f"Missing {description} at {path}. Run the Janino setup instructions.")


def ensure_worker_compiled() -> Path:
    """Compile the Java worker sources on-demand."""

    _ensure_dependency(JANINO_JAR, "Janino JAR")
    _ensure_dependency(GSON_JAR, "Gson JAR")
    if not WORKER_SRC.exists():
        raise JaninoWorkerError(f"Worker sources not found at {WORKER_SRC}")
    marker = WORKER_BUILD / "com" / "agentcortex" / "worker" / "WorkerMain.class"
    source_paths = sorted(WORKER_SRC.rglob("*.java"))
    if not source_paths:
        raise JaninoWorkerError("No Java worker sources were found to compile.")
    if marker.exists():
        try:
            latest_source_time = max(path.stat().st_mtime for path in source_paths)
        except OSError as exc:
            raise JaninoWorkerError(f"Unable to stat worker sources: {exc}") from exc
        marker_time = marker.stat().st_mtime
        if marker_time >= latest_source_time:
            return WORKER_BUILD
    WORKER_BUILD.mkdir(parents=True, exist_ok=True)
    source_files = [str(path) for path in source_paths]
    javac = shutil.which("javac")
    if not javac:
        raise JaninoWorkerError("javac is not available on PATH. Install a JDK to run the Janino worker.")
    classpath = os.pathsep.join(str(path) for path in (JANINO_JAR, GSON_JAR))
    command = [
        javac,
        "-cp",
        classpath,
        "-d",
        str(WORKER_BUILD),
        *source_files,
    ]
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise JaninoWorkerError(
            "javac failed to compile the Janino worker:\n" f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )
    return WORKER_BUILD


@dataclass
class WorkerPlanRequest:
    request_id: str
    plan_class: str
    classes_dir: Path
    capture_trace: bool
    plan_source: str
    tool_stub_source: Optional[str]


class JaninoWorkerProcess:
    """Wraps a single long-lived JVM worker process."""

    def __init__(self, syscall_registry: SyscallRegistry) -> None:
        self._syscall_registry = syscall_registry
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._closed = False
        ensure_worker_compiled()
        classpath = os.pathsep.join(str(path) for path in (WORKER_BUILD, JANINO_JAR, GSON_JAR))
        java = shutil.which("java")
        if not java:
            raise JaninoWorkerError("java runtime not available on PATH.")
        self._process = subprocess.Popen(
            [java, "-cp", classpath, "com.agentcortex.worker.WorkerMain"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if not self._process.stdin or not self._process.stdout:
            raise JaninoWorkerError("Failed to initialize worker process pipes.")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read_message()
        if ready.get("type") != "ready":
            raise JaninoWorkerError(f"Worker failed to start: {ready}")

    def _drain_stderr(self) -> None:
        if not self._process.stderr:
            return
        for line in self._process.stderr:
            line = line.rstrip()
            if not line:
                continue
            # stderr is best-effort diagnostics; route to console for now
            print(f"[janino-worker] {line}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send_message({"type": "shutdown"})
            self._read_message()
        except Exception:
            pass
        finally:
            self._process.kill()

    def run_plan(self, request: WorkerPlanRequest) -> Dict[str, Any]:
        payload = {
            "type": "run_plan",
            "requestId": request.request_id,
            "planClass": request.plan_class,
            "classesDir": str(request.classes_dir),
            "captureTrace": request.capture_trace,
            "planSource": request.plan_source,
            "toolStubSource": request.tool_stub_source,
        }
        self._send_message(payload)
        while True:
            message = self._read_message()
            message_type = message.get("type")
            if message_type == "syscall_request":
                response = self._handle_syscall(message)
                self._send_message(response)
                continue
            if message_type == "result":
                return message
            raise JaninoWorkerError(f"Unexpected worker message: {message}")

    def _handle_syscall(self, message: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = message.get("tool")
        args = message.get("args") or []
        call_id = message.get("callId")
        request_id = message.get("requestId")
        try:
            fn = self._syscall_registry.get(str(tool_name))
        except KeyError:
            return self._syscall_error(request_id, call_id, "unknown_tool", f"Unknown tool '{tool_name}'")
        try:
            result = fn(*args)
        except ToolError as exc:
            return self._syscall_error(request_id, call_id, "tool_error", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return self._syscall_error(request_id, call_id, "execution_error", str(exc))
        return {
            "type": "syscall_response",
            "requestId": request_id,
            "callId": call_id,
            "success": True,
            "result": result,
        }

    @staticmethod
    def _syscall_error(request_id: str, call_id: int, error_type: str, message: str) -> Dict[str, Any]:
        return {
            "type": "syscall_response",
            "requestId": request_id,
            "callId": call_id,
            "success": False,
            "error": {"type": error_type, "message": message},
        }

    def _send_message(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, default=_json_default)
        with self._write_lock:
            self._stdin.write(data + "\n")
            self._stdin.flush()

    def _read_message(self) -> Dict[str, Any]:
        with self._read_lock:
            line = self._stdout.readline()
        if not line:
            raise JaninoWorkerError("Worker process terminated unexpectedly.")
        return json.loads(line)


class JaninoWorkerPool:
    """Thread-safe pool for reusing a limited set of JVM workers."""

    def __init__(self, syscall_registry: SyscallRegistry, max_workers: int = 4) -> None:
        self._syscall_registry = syscall_registry
        self._max_workers = max(1, max_workers)
        self._available: "queue.Queue[JaninoWorkerProcess]" = queue.Queue()
        self._all_workers: List[JaninoWorkerProcess] = []
        self._lock = threading.Lock()
        self._closed = False

    def acquire(self) -> JaninoWorkerProcess:
        with self._lock:
            if self._available.qsize() > 0:
                try:
                    return self._available.get_nowait()
                except queue.Empty:
                    pass
            if len(self._all_workers) < self._max_workers:
                worker = JaninoWorkerProcess(self._syscall_registry)
                self._all_workers.append(worker)
                return worker
        return self._available.get()

    def release(self, worker: JaninoWorkerProcess) -> None:
        if self._closed:
            worker.close()
            return
        self._available.put(worker)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        while not self._available.empty():
            worker = self._available.get_nowait()
            worker.close()
        for worker in self._all_workers:
            worker.close()


class JaninoPlanRunner:
    """Plan runner that executes compiled artifacts via the Janino worker pool."""

    def __init__(self, worker_pool: JaninoWorkerPool) -> None:
        self._worker_pool = worker_pool
        self._artifacts: Optional[PlanExecutionArtifacts] = None

    def bind_plan_artifacts(self, artifacts: PlanExecutionArtifacts) -> None:
        self._artifacts = artifacts

    def execute(
        self,
        plan_source: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        goal_summary: Optional[str] = None,
        deferred_metadata: Optional[Dict[str, Any]] = None,
        deferred_constraints: Optional[Iterable[str]] = None,
        tool_stub_class_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del goal_summary, deferred_metadata, deferred_constraints
        if not self._artifacts:
            raise JaninoWorkerError("Plan artifacts were not provided to the Janino runtime.")
        graph = analyze_java_plan(plan_source, tool_stub_class_name=tool_stub_class_name)
        metadata_payload = dict(metadata or {})
        stub_errors = detect_stub_method_errors(graph)
        if stub_errors:
            metadata_payload.setdefault("functions", len(graph.functions))
            metadata_payload.setdefault("function_names", [fn.name for fn in graph.functions])
            metadata_payload.setdefault(
                "tool_call_count",
                sum(len(fn.tool_calls) for fn in graph.functions),
            )
            return {
                "success": False,
                "errors": stub_errors,
                "graph": graph.to_dict(),
                "metadata": metadata_payload,
                "trace": [],
            }
        metadata_payload.setdefault("functions", len(graph.functions))
        metadata_payload.setdefault("function_names", [fn.name for fn in graph.functions])
        metadata_payload.setdefault(
            "tool_call_count",
            sum(len(fn.tool_calls) for fn in graph.functions),
        )
        request = WorkerPlanRequest(
            request_id=uuid.uuid4().hex,
            plan_class=self._artifacts.plan_class_name or "Plan",
            classes_dir=self._artifacts.classes_dir,
            capture_trace=capture_trace,
            plan_source=plan_source,
            tool_stub_source=self._read_stub_source(),
        )
        worker = self._worker_pool.acquire()
        try:
            result = worker.run_plan(request)
        finally:
            self._worker_pool.release(worker)
        execution = {
            "success": bool(result.get("success")),
            "errors": result.get("errors") or [],
            "graph": graph.to_dict(),
            "metadata": metadata_payload,
            "trace": result.get("trace") or [],
            "return_value": result.get("returnValue"),
        }
        return execution

    def _read_stub_source(self) -> Optional[str]:
        if not self._artifacts or not self._artifacts.stub_source_path:
            return None
        try:
            return self._artifacts.stub_source_path.read_text(encoding="utf-8")
        except OSError:
            return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


__all__ = [
    "JaninoPlanRunner",
    "JaninoWorkerPool",
    "JaninoWorkerError",
    "ensure_worker_compiled",
]
