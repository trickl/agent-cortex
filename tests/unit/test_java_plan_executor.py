"""Tests for interpreting Java plans into actionable sub-goals."""
from __future__ import annotations

from textwrap import dedent

from llmflow.planning.java_plan_executor import JavaPlanNavigator


_SAMPLE_PLAN = dedent(
    """
    public class Planner {
        public static void main(String[] args) {
            repairLintIssue();
        }

        public static void repairLintIssue() {
            // Step 1: Prepare the workspace and create a feature branch
            createBranch();

            // Step 2: Clone or checkout the repository
            cloneRepo();

            // Step 3: Retrieve the first open lint issue from Qlty tools
            applyTextRewrite();
            qltyGetFirstIssue();
            String lintIssueId = readTextFile("queries/qlty_llvm-lint_results.md");

            // Step 4: Understand the issue details for the parse error on line 7
            runUnderstandIssueSubgoal(lintIssueId);

            // Step 5: Apply changes to resolve the parse error
            applyTextRewrite();

            // Step 6: Verify the fix with tests and commit evidence
            getUncommittedChanges();
            commitChanges();
            pushBranch();

            // Step 7: Finalize by opening a pull request summarizing the delivery
            createPullRequest();
        }
    }
    """
)


def test_navigator_extracts_first_subgoal_comment() -> None:
    navigator = JavaPlanNavigator.from_source(_SAMPLE_PLAN)
    intent = navigator.next_subgoal()

    assert intent is not None
    assert intent.action_kind == "helper"
    assert intent.action_name == "createBranch"
    assert intent.parent_function == "repairLintIssue"
    assert intent.goal.lower().startswith("step 1")
    assert "create a feature branch" in intent.goal.lower()