"""Java-plan Agent orchestrator.

This module provides a single entry point for building agents that rely on the
Cortex Java planning workflow. Instead of iteratively calling tools directly
from LLM responses, the agent now:

1. Synthesizes a Java plan with :class:`~llmflow.planning.JavaPlanner`.
2. Executes the plan via :class:`~llmflow.planning.plan_runner.PlanRunner`.
3. Surfaces the orchestrator summary (or plan return value) back to the user.

Tool exposure is controlled through PlanningToolStubs generated from the
registered planning tools, optionally filtered via tool tags. Conversation
memory is retained for traceability, but the execution flow is entirely
plan-driven.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from llmflow.llm_client import LLMClient
from llmflow.logging_utils import RunArtifactManager, RunLogContext
from llmflow.planning import (
    JavaPlanner,
    JavaPlanFixer,
    JavaPlanRequest,
    PlanOrchestrator,
    ToolStubGenerationError,
    generate_tool_stub_class,
)
from llmflow.planning.plan_runner import PlanRunner
from llmflow.telemetry.mermaid_recorder import MermaidSequenceRecorder
from llmflow.tools import get_module_for_tool_name, load_tool_module
from llmflow.tools.tool_registry import (
    get_tool_schema,
    get_tool_tags,
    get_tools_by_tags,
)

from .agent_instrumentation import AgentInstrumentationMixin
from .memory import Memory


_JAVA_PLANNING_GUIDANCE = (
    "Create formatted Java code to accomplish the user's task and emit only Java source."
    "Include a single main method and aim to call a series of two to seven methods to decompose the work."
    "Use the provided PlanningToolStubs class to invoke the registered planning tools by calling "
    "PlanningToolStubs.<toolName>(...) exactly as shown in the stub source; never call raw tool "
    "functions directly."
    "Define helper methods as needed to break down complex or long logic."
    "Do not implement detailed logic directly in the main method; instead, delegate to helper methods."
    "Each helper method should either call one or more PlanningToolStubs methods or contain a descriptive comment "
    "explaining the intended functionality."
    "Comment out helper bodies when a step cannot yet be implemented, but prefer concrete tool "
    "calls whenever a registered tool can perform the work.\n\n"
    "Example Planner:\n"
    "```java\n"
    "public class Planner {\n"
    "    public static void main(String[] args) throws Exception {\n"
    "        Map<String, Object> returnValue = PlanningToolStubs.getItemOfInterest();\n"
    "        if (returnValue == null) {\n"
    "            System.out.println(\"No item of interest.\");\n"
    "            return;\n"
    "        }\n"
    "        PlanningToolStubs.atomicToolAction(\"example\" + returnValue.get(\"slug\"));\n"
    "        processValue(returnValue);\n"
    "    }\n\n"
    "    private static void processValue(Map<String, Object> returnValue) throws Exception {\n"
    "        String path = (String) returnValue.get(\"path\");\n"
    "        PlanningToolStubs.doAction(path);\n"
    "        // TODO: add concrete fixes using other PlanningToolStubs helpers.\n"
    "    }\n"
    "}\n"
    "```"
)
_SYSTEM_PROMPT_TEMPLATE = (
    "{{ base_prompt }}\n\n"
    "{{ planning_guidance }}"
)
_SYSTEM_PROMPT_PREVIEW = (
    "{{ base_prompt }}\n\n"
    "{{ planning_guidance }}"
)

_TOOL_STUB_CLASS_NAME = "PlanningToolStubs"


class Agent(AgentInstrumentationMixin):
    """Goal-aware Java plan orchestrator that scopes tool usage via stubs."""

    _MAX_CONTEXT_TRACE = 100

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = (
            "You are a helpful AI assistant. You coordinate structured Java plans to solve user"
            " requests using the available planning tools."
        ),
        initial_goals: Optional[List[Dict[str, Any]]] = None,
        available_tool_tags: Optional[List[str]] = None,
        match_all_tags: bool = True,
        allowed_tools: Optional[Sequence[str]] = None,
        runner_factory: Optional[Callable[[], PlanRunner]] = None,
        planner: Optional[JavaPlanner] = None,
        plan_fixer: Optional[JavaPlanFixer] = None,
        plan_max_retries: int = 0,
        capture_trace: bool = False,
        verbose: bool = True,
        enable_run_logging: bool = True,
    ):
        self.llm_client = llm_client
        self.system_prompt, self._system_prompt_template_preview = self._render_system_prompt(
            system_prompt
        )
        self.memory = Memory(system_prompt=self.system_prompt)
        self.available_tool_tags = available_tool_tags
        self.match_all_tags_for_tools = match_all_tags
        self.verbose = verbose
        self.active_tools_schemas: List[Dict[str, Any]] = []
        self.context_trace: List[Dict[str, Any]] = []
        self.current_iteration = 0
        self._capture_trace = capture_trace
        self.plan_max_retries = max(plan_max_retries, 0)
        self._planner = planner or JavaPlanner(llm_client)
        self._plan_fixer = plan_fixer or JavaPlanFixer(llm_client)
        if runner_factory is not None:
            self._runner_factory = runner_factory
        else:
            self._runner_factory = self._build_default_runner_factory
        self._orchestrator = PlanOrchestrator(
            self._planner,
            self._runner_factory,
            max_retries=self.plan_max_retries,
            plan_fixer=self._plan_fixer,
        )

        available_tool_names = self._discover_tool_names(
            available_tool_tags,
            match_all_tags,
        )
        self.allowed_tools = self._resolve_allowed_tools(
            available_tool_names,
            allowed_tools,
        )
        if not self.allowed_tools:
            raise ValueError("Agent must be configured with at least one planning tool.")
        self.active_tools_schemas = self._build_tool_schemas(self.allowed_tools)
        self._tool_stub_class_name = _TOOL_STUB_CLASS_NAME
        self._tool_stub_source = self._build_tool_stub_source()

        self.enable_run_logging = enable_run_logging
        self._run_log_context: Optional[RunLogContext] = None
        self._run_artifact_manager: Optional[RunArtifactManager] = None
        self._mermaid_recorder: Optional[MermaidSequenceRecorder] = None
        self._run_failed: bool = False
        self._last_prompt_summary: Optional[str] = None
        self._owns_run_directory = False
        self._last_run_summary: Optional[str] = None

        if self.verbose:
            print("Agent initialized for Java planning.")
            print("System Prompt Template (spec omitted in logs):")
            print(self._system_prompt_template_preview)
            print(
                f"Allowed tools ({len(self.allowed_tools)}): "
                + ", ".join(self.allowed_tools)
            )
            print(f"Plan retries: {self.plan_max_retries}")

    # ------------------------------------------------------------------
    # Public API

    def add_user_message_and_run(self, user_input: str) -> Optional[str]:
        """Record ``user_input`` and execute a Java plan once."""

        if not user_input or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        self._start_run_instrumentation()
        self.current_iteration = 1
        try:
            self.memory.add_user_message(user_input)
            self._record_context_snapshot("pre_plan_request")
            plan_request = self._build_plan_request(user_input)
            if self.verbose:
                print("Submitting Java plan request...")
            result = self._orchestrator.execute_with_retries(
                plan_request,
                capture_trace=self._capture_trace,
                metadata={"allowed_tools": list(self.allowed_tools)},
            )
            final_message = self._finalize_plan_result(result)
            if final_message:
                self.memory.add_assistant_message(final_message)
            self._append_context_trace(result)
            self._last_run_summary = result.get("summary")
            if not result.get("success", False):
                self._mark_run_failure()
            return final_message
        finally:
            self._finalize_run_instrumentation()

    def get_context_trace(self) -> List[Dict[str, Any]]:
        """Expose the captured execution context."""

        return list(self.context_trace)

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_default_runner_factory(self) -> PlanRunner:
        return PlanRunner()

    def _discover_tool_names(
        self,
        tags: Optional[Sequence[str]],
        match_all: bool,
    ) -> List[str]:
        requested_tags = [tag for tag in (tags or []) if tag and tag.strip()]
        registry_entries = get_tools_by_tags(requested_tags, match_all=match_all)
        names = sorted(registry_entries.keys())
        if requested_tags and not names:
            raise ValueError(
                f"No planning tools matched tags {requested_tags} (match_all={match_all})."
            )
        if not names:
            raise RuntimeError("No planning tools are registered. Import tool modules first.")
        return names

    def _resolve_allowed_tools(
        self,
        discovered: Sequence[str],
        explicit: Optional[Sequence[str]],
    ) -> List[str]:
        if explicit:
            normalized = [name.strip() for name in explicit if name and name.strip()]
            unique = sorted(set(normalized))
            missing: List[str] = []
            for name in unique:
                schema = get_tool_schema(name)
                if schema is None:
                    module_name = get_module_for_tool_name(name)
                    if module_name:
                        load_tool_module(module_name, warn=self.verbose)
                        schema = get_tool_schema(name)
                if schema is None:
                    missing.append(name)
            if missing:
                raise ValueError(f"Unknown planning tools requested: {', '.join(missing)}")
            return unique
        return list(discovered)

    def _build_tool_schemas(self, tool_names: Sequence[str]) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool_name in tool_names:
            schema = get_tool_schema(tool_name)
            if schema is None:
                module_name = get_module_for_tool_name(tool_name)
                if module_name:
                    load_tool_module(module_name, warn=self.verbose)
                    schema = get_tool_schema(tool_name)
            if schema:
                schemas.append(schema)
        return schemas

    def _build_tool_stub_source(self) -> Optional[str]:
        tool_names: List[str] = []
        for schema in self.active_tools_schemas:
            if not isinstance(schema, dict):
                continue
            function_meta = schema.get("function")
            if not isinstance(function_meta, dict):
                continue
            name = function_meta.get("name")
            if name:
                tool_names.append(str(name))
        unique_names = sorted(set(tool_names))
        if not unique_names:
            return None
        try:
            return generate_tool_stub_class(self._tool_stub_class_name, unique_names)
        except ToolStubGenerationError as exc:
            if self.verbose:
                print(f"Skipping tool stub generation: {exc}")
            return None

    def _build_plan_request(self, user_input: str) -> JavaPlanRequest:
        task = self._format_planner_task(user_input)
        context = self._build_planner_context()
        return JavaPlanRequest(
            task=task,
            context=context,
            tool_names=self.allowed_tools,
            tool_schemas=self.active_tools_schemas,
            tool_stub_source=self._tool_stub_source,
            tool_stub_class_name=self._tool_stub_class_name if self._tool_stub_source else None,
            metadata={
                "source": "llmflow.core.agent",
            },
        )

    def _format_planner_task(self, user_input: str) -> str:
        header = (
            "Create a Java class named Planner that carries out the user's request."
            " Keep the structure minimal, lean on helper methods for decomposition, and"
            " invoke planning tools via the provided stub class when a step can be executed directly."
            " Break down complex tasks into smaller sub-tasks. Each sub-task should either be a stubbed method or a helper method."
            " Stubbed methods should just contain a descriptive comment of their intended functionality."
            " The problem should be decomposed into manageable pieces using helper methods."
        )
        normalized = user_input.strip()
        if not normalized:
            raise ValueError("user_input must be a non-empty string")
        return f"{header}\n\nUser request:\n{normalized}"

    def _build_planner_context(self) -> Optional[str]:
        sections: List[str] = []
        if self._last_run_summary:
            sections.append(f"Previous plan summary:\n{self._last_run_summary.strip()}")
        if not sections:
            return None
        return "\n\n".join(sections).strip()

    def _finalize_plan_result(self, result: Dict[str, Any]) -> str:
        summary = result.get("summary")
        execution = result.get("final_execution") or {}
        if execution.get("success"):
            return self._format_success_message(execution, summary)
        errors = execution.get("errors") or []
        if errors:
            return self._format_failure_message(execution, errors, summary)
        if summary:
            return summary
        return "Plan run finished without additional details."

    def _format_success_message(
        self,
        execution: Dict[str, Any],
        summary: Optional[str],
    ) -> str:
        return_value = execution.get("return_value")
        if isinstance(return_value, str) and return_value.strip():
            return return_value.strip()
        if return_value not in (None, ""):
            try:
                return json.dumps(return_value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(return_value)
        base_message = summary or "✅ Java plan run completed successfully."
        capability_note = self._summarize_execution_capabilities(execution)
        if capability_note:
            return f"{base_message}\n{capability_note}"
        return base_message

    def _summarize_execution_capabilities(self, execution: Dict[str, Any]) -> Optional[str]:
        metadata = execution.get("metadata") or {}
        allowed_tools = metadata.get("allowed_tools")
        if not allowed_tools:
            return None
        if isinstance(allowed_tools, (str, bytes)):
            normalized_tools = {str(allowed_tools).strip()}
        else:
            normalized_tools = {str(tool).strip() for tool in allowed_tools if str(tool).strip()}
        if not normalized_tools:
            return None
        lowered = {tool.lower() for tool in normalized_tools}
        notes: List[str] = []
        if any("pull_request" in name or "pullrequest" in name for name in lowered):
            notes.append("ready to create a pull request")
        if any("qlty" in name and "issue" in name for name in lowered):
            notes.append("will report when no lint issues remain")
        for tag_note in self._capabilities_from_tags():
            if tag_note not in notes:
                notes.append(tag_note)
        if not notes:
            return None
        detail = self._join_human_list(notes)
        return f"Capabilities: {detail}."

    @staticmethod
    def _join_human_list(items: Sequence[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        head = ", ".join(items[:-1])
        return f"{head}, and {items[-1]}"

    def _capabilities_from_tags(self) -> List[str]:
        tags = {
            str(tag).strip().lower()
            for tag in (self.available_tool_tags or [])
            if tag and str(tag).strip()
        }
        notes: List[str] = []
        if "git" in tags:
            notes.append("ready to create a pull request")
        if "qlty" in tags:
            notes.append("will report when no lint issues remain")
        return notes

    def _format_failure_message(
        self,
        execution: Dict[str, Any],
        errors: Sequence[Dict[str, Any]],
        summary: Optional[str],
    ) -> str:
        first = errors[0] if errors else {}
        err_type = first.get("type") or "execution_error"
        message = first.get("message") or "Plan execution failed."
        location = []
        if first.get("function"):
            location.append(f"function {first['function']}")
        if first.get("line") is not None:
            loc = f"line {first['line']}"
            if first.get("column") is not None:
                loc += f", column {first['column']}"
            location.append(loc)
        suffix = f" ({'; '.join(location)})" if location else ""
        base = f"❌ {err_type}: {message}{suffix}"
        lines = [base]
        if summary:
            lines.append(summary)
        capability_note = self._summarize_execution_capabilities(execution)
        if capability_note:
            lines.append(capability_note)
        return "\n".join(lines)

    def _append_context_trace(self, result: Dict[str, Any]) -> None:
        entry = {
            "stage": "plan_run",
            "success": result.get("success", False),
            "attempts": len(result.get("attempts") or []),
            "summary": result.get("summary"),
        }
        self.context_trace.append(entry)
        if len(self.context_trace) > self._MAX_CONTEXT_TRACE:
            self.context_trace.pop(0)
        self._record_context_snapshot("post_plan_execution")

    def _render_system_prompt(self, base_prompt: str) -> tuple[str, str]:
        prompt = (base_prompt or "").strip()
        if not prompt:
            prompt = "You are a helpful AI assistant that coordinates Java plans."
        replacements = {
            "{{ base_prompt }}": prompt,
            "{{ planning_guidance }}": _JAVA_PLANNING_GUIDANCE,
        }
        rendered = _SYSTEM_PROMPT_TEMPLATE
        preview = _SYSTEM_PROMPT_PREVIEW
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
            preview = preview.replace(placeholder, value)
        return rendered.strip(), preview.strip()

