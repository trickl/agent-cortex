"""Instrumentation and logging helpers for the Agent."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmflow.logging_utils import (
    LLM_LOGGER_NAME,
    PLAN_LOGGER_NAME,
    TOOLS_LOGGER_NAME,
    RunArtifactManager,
    RunLogContext,
    setup_run_logging,
    summarize_messages,
    summarize_response,
)
from llmflow.telemetry.mermaid_recorder import MermaidSequenceRecorder


_RUN_DIR_ENV = "LLMFLOW_ACTIVE_RUN_DIR"
_RUN_ID_ENV = "LLMFLOW_ACTIVE_RUN_ID"


class AgentInstrumentationMixin:
    """Provides logging and instrumentation helpers for the Agent."""

    def _start_run_instrumentation(self) -> None:
        self._run_failed = False
        self._last_prompt_summary = None
        if not self.enable_run_logging:
            self._run_log_context = None
            self._run_artifact_manager = None
            self._mermaid_recorder = None
            return

        parent_run_dir = os.getenv(_RUN_DIR_ENV)
        parent_run_id = os.getenv(_RUN_ID_ENV)
        if parent_run_dir and parent_run_id:
            context = setup_run_logging(
                run_id=parent_run_id,
                logs_root=Path(parent_run_dir).resolve().parent,
                run_directory=Path(parent_run_dir),
            )
            self._owns_run_directory = False
        else:
            context = setup_run_logging()
            os.environ[_RUN_DIR_ENV] = str(context.run_dir)
            os.environ[_RUN_ID_ENV] = context.run_id
            self._owns_run_directory = True
        self._run_log_context = context
        self._run_artifact_manager = RunArtifactManager(context)
        mermaid_dir = context.run_dir / "mermaid"
        self._mermaid_recorder = MermaidSequenceRecorder(context.run_id, output_dir=mermaid_dir)

    def _finalize_run_instrumentation(self) -> None:
        if not self._run_log_context or not self._run_artifact_manager:
            self._reset_instrumentation_handles()
            return

        if self._mermaid_recorder:
            mermaid_path = self._mermaid_recorder.write()
            self._run_artifact_manager.register_mermaid(mermaid_path)

        self._run_artifact_manager.write_manifest(success=not self._run_failed)
        if self._owns_run_directory:
            os.environ.pop(_RUN_DIR_ENV, None)
            os.environ.pop(_RUN_ID_ENV, None)
            self._owns_run_directory = False
        self._reset_instrumentation_handles()

    def _reset_instrumentation_handles(self) -> None:
        self._run_log_context = None
        self._run_artifact_manager = None
        self._mermaid_recorder = None
        self._last_prompt_summary = None
        self._owns_run_directory = False

    def _mark_run_failure(self) -> None:
        self._run_failed = True

    @staticmethod
    def _serialize_payload(payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(payload)

    def _log_llm_prompt(self, messages: List[Dict[str, Any]]) -> None:
        if not self._run_log_context:
            return
        summary = summarize_messages(messages)
        self._last_prompt_summary = summary
        logger = logging.getLogger(LLM_LOGGER_NAME)
        logger.info(
            "prompt iteration=%s message_count=%s summary=%s",
            self.current_iteration,
            len(messages),
            summary,
        )
        serialized = self._serialize_payload(messages)
        logger.info("prompt_detail iteration=%s\n%s", self.current_iteration, serialized)

    def _log_llm_response(self, response: Dict[str, Any]) -> None:
        if not self._run_log_context:
            return
        summary = summarize_response(response)
        tool_calls = response.get("tool_calls") or []
        logger = logging.getLogger(LLM_LOGGER_NAME)
        logger.info(
            "response iteration=%s tool_calls=%s summary=%s",
            self.current_iteration,
            len(tool_calls),
            summary,
        )
        serialized = self._serialize_payload(response)
        logger.info("response_detail iteration=%s\n%s", self.current_iteration, serialized)
        if self._mermaid_recorder and self._last_prompt_summary is not None:
            self._mermaid_recorder.record_llm_exchange(self._last_prompt_summary, summary)

    def _log_tool_result(
        self,
        tool_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
        content: Optional[str],
    ) -> None:
        if not self._run_log_context:
            return
        payload = metadata or {}
        status = "success" if payload.get("success", True) else "failed"
        retryable = payload.get("retryable", False)
        fatal = payload.get("fatal", False)
        error_text = payload.get("error") or ""
        preview = (content or "").replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        logging.getLogger(TOOLS_LOGGER_NAME).info(
            "tool=%s status=%s retryable=%s fatal=%s detail=%s preview=%s",
            tool_name or "unknown_tool",
            status,
            retryable,
            fatal,
            error_text,
            preview,
        )
        if self._mermaid_recorder:
            status_hint = "success" if payload.get("success", True) else (error_text or "failed")
            self._mermaid_recorder.record_tool_call(tool_name or "unknown_tool", status_hint)

    def _record_context_snapshot(
        self,
        stage: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        snapshot = messages if messages is not None else self.memory.get_history()
        msg_count = len(snapshot)
        approx_chars = sum(
            len(str(msg.get("content", "")))
            for msg in snapshot
            if isinstance(msg, dict)
        )
        tool_messages = sum(
            1 for msg in snapshot if isinstance(msg, dict) and msg.get("role") == "tool"
        )
        entry = {
            "stage": stage,
            "message_count": msg_count,
            "approx_chars": approx_chars,
            "tool_messages": tool_messages,
            "timestamp": time.time(),
        }
        self.context_trace.append(entry)
        if len(self.context_trace) > self._MAX_CONTEXT_TRACE:
            self.context_trace.pop(0)

        if self._run_log_context:
            logging.getLogger(PLAN_LOGGER_NAME).info(
                "stage=%s iteration=%s pending_goals=%s messages=%s approx_chars=%s tool_messages=%s",
                stage,
                self.current_iteration,
                self._pending_goal_count(),
                msg_count,
                approx_chars,
                tool_messages,
            )
            if self._mermaid_recorder and stage == "pre_llm_prompt":
                self._mermaid_recorder.record_plan_attempt(
                    self.current_iteration,
                    "snapshot",
                    f"pending_goals={self._pending_goal_count()}",
                )
        if self.verbose:
            print(
                f"[context] {stage}: {msg_count} msgs, ~{approx_chars} chars, {tool_messages} tool msgs."
            )
