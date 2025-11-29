"""Compatibility shims for the retired CPL execution stack."""
from __future__ import annotations


class PlanExecutor:  # pragma: no cover - compatibility shim
    """Placeholder that explains the removal of the CPL runtime.

    The new Java workflow graph architecture no longer executes arbitrary Java code
    in-process. Importers that previously relied on :class:`PlanExecutor` should
    switch to :class:`llmflow.planning.plan_runner.PlanRunner` and the
    ``analyze_java_plan`` utilities instead.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - short shim
        raise RuntimeError(
            "PlanExecutor has been removed. Use PlanRunner/analyze_java_plan for"
            " static workflow analysis."
        )


__all__ = ["PlanExecutor"]
