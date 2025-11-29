"""Helpers for progressively building Mermaid sequence diagrams."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


class MermaidSequenceRecorder:
    """Accumulates sequence-diagram lines for a single agent run."""

    def __init__(self, run_id: str, output_dir: Optional[Path] = None) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir or Path("artifacts") / "mermaid").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"sequence-{run_id}.mmd"
        self._participants: Dict[str, str] = {}
        self._events: List[str] = []
        self._ensure_participant("Agent0", "Agent Cortex")
        self._ensure_participant("LLM", "LLM")
        self._ensure_participant("Tools", "Tools")

    def _ensure_participant(self, name: str, label: str) -> None:
        if name not in self._participants:
            self._participants[name] = label

    def _actor_for_depth(self, depth: int) -> str:
        actor = f"Agent{depth}"
        label = "Agent Cortex" if depth == 0 else f"Agent Depth {depth}"
        self._ensure_participant(actor, label)
        return actor

    @staticmethod
    def _sanitize(text: str, limit: int = 80) -> str:
        trimmed = (text or "").replace("\n", " ").strip()
        if len(trimmed) > limit:
            trimmed = trimmed[:limit] + "…"
        return trimmed or "(empty)"

    def record_llm_exchange(self, prompt_summary: str, response_summary: str) -> None:
        prompt = self._sanitize(prompt_summary)
        response = self._sanitize(response_summary)
        self._events.append(f"    Agent0->>LLM: {prompt}")
        self._events.append(f"    LLM-->>Agent0: {response}")

    def record_tool_call(self, tool_name: str, status: str) -> None:
        safe_tool = self._sanitize(tool_name, limit=40)
        safe_status = self._sanitize(status, limit=40)
        self._events.append(f"    Agent0->>Tools: {safe_tool}")
        self._events.append(f"    Tools-->>Agent0: {safe_status}")

    def record_plan_attempt(self, attempt: int, status: str, hint: Optional[str] = None) -> None:
        hint_text = f" | {self._sanitize(hint)}" if hint else ""
        self._events.append(f"    Agent0->>Agent0: Plan attempt {attempt} -> {status}{hint_text}")

    def record_function_event(self, depth: int, function_name: str) -> None:
        actor = self._actor_for_depth(depth)
        safe_fn = self._sanitize(function_name, limit=50)
        self._events.append(f"    Agent0->>{actor}: enter {safe_fn}")

    def write(self) -> Path:
        if not self._events:
            self._events.append("    Agent0->>Agent0: No recorded events")
        participant_lines = [f"    participant {name} as {label}" for name, label in self._participants.items()]
        content = ["sequenceDiagram", *participant_lines, *self._events]
        self.output_path.write_text("\n".join(content), encoding="utf-8")
        return self.output_path
