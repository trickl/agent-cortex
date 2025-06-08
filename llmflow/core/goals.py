"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Goals Module - Core component managing agent objectives and priorities.
This module implements the Goals component of the GAME methodology:
- Goal Class: Represents individual agent objectives with priorities and instructions
- GoalManager Class: Orchestrates multiple goals with sophisticated management:
  * Dynamic goal addition and removal
  * Priority-based goal ordering
  * Goal completion tracking
  * Instruction management for complex goals
  * Formatted goal reporting for LLM prompts

The Goals module ensures agents maintain clear objectives and can track
progress towards completion while adapting to changing priorities and requirements.
Default goals ensure basic agent functionality even without specific objectives.
"""

from typing import List, Optional, Dict, Any

class Goal:
    """Represents a single goal for the agent."""
    def __init__(self, description: str, priority: int = 0, instructions: Optional[List[str]] = None, completed: bool = False):
        self.description = description
        self.priority = priority
        self.instructions = instructions if instructions else []
        self.completed = completed

    def __repr__(self):
        return f"Goal(description='{self.description}', priority={self.priority}, completed={self.completed})"

class GoalManager:
    """Manages the agent's goals."""
    def __init__(self, initial_goals: Optional[List[Dict[str, Any]]] = None):
        self.goals: List[Goal] = []
        if initial_goals:
            for g_data in initial_goals:
                self.add_goal(
                    description=g_data.get("description", "(Unnamed goal from initial_goals)"),
                    priority=g_data.get("priority", 0),
                    instructions=g_data.get("instructions")
                )
        
        if not self.goals:
            self.add_goal(description="Understand and respond to user queries accurately and helpfully.", priority=1)
            self.add_goal(description="Utilize available tools effectively when requested or appropriate to fulfill user needs.", priority=0)

    def add_goal(self, description: str, priority: int = 0, instructions: Optional[List[str]] = None):
        """Adds a new goal."""
        if not description:
            raise ValueError("Goal description cannot be empty.")
        goal = Goal(description, priority, instructions)
        self.goals.append(goal)
        self.sort_goals()

    def remove_goal(self, description: str):
        """Removes a goal by its description."""
        self.goals = [g for g in self.goals if g.description != description]

    def complete_goal(self, description: str):
        """Marks a goal as completed."""
        for goal in self.goals:
            if goal.description == description:
                goal.completed = True
                break
        else:
            print(f"Warning: Goal '{description}' not found to mark as completed.")

    def get_current_goal(self) -> Optional[Goal]:
        """Returns the highest priority uncompleted goal."""
        uncompleted_goals = [g for g in self.goals if not g.completed]
        if not uncompleted_goals:
            return None
        return uncompleted_goals[0] # Assumes goals are sorted by priority

    def all_goals_completed(self) -> bool:
        """Checks if all goals are completed."""
        return all(g.completed for g in self.goals)

    def sort_goals(self):
        """Sorts goals by priority (descending)."""
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def get_goals_for_prompt(self) -> str:
        """Formats the goals and instructions for inclusion in a prompt."""
        if not self.goals:
            return "No current goals."

        output_lines = ["Current Goals (sorted by priority):"]
        for i, goal in enumerate(self.goals):
            status = "[COMPLETED]" if goal.completed else "[PENDING]"
            output_lines.append(f"{i+1}. {status} {goal.description} (Priority: {goal.priority})")
            if goal.instructions:
                output_lines.append("  Instructions:")
                for instr in goal.instructions:
                    output_lines.append(f"    - {instr}")
        return "\n".join(output_lines)

# Example Usage:
if __name__ == '__main__':
    goal_manager = GoalManager()
    goal_manager.add_goal("Analyze project files", priority=1, instructions=["Scan all .py files", "Identify main modules"])
    goal_manager.add_goal("Generate documentation", priority=0, instructions=["Use comments to generate docs"])
    goal_manager.add_goal("Write unit tests for critical functions", priority=2)

    print("Initial Goals:")
    print(goal_manager.get_goals_for_prompt())

    current_goal = goal_manager.get_current_goal()
    if current_goal:
        print("\n" + f"Current highest priority goal: {current_goal.description}")
        goal_manager.complete_goal(current_goal.description)
        print(f"Completed '{current_goal.description}'.")


    goal_manager.add_goal("Refactor database module", priority=1)
    print("\n" + "Goals after completing one and adding another:")
    print(goal_manager.get_goals_for_prompt())

    print("\n" + f"All goals completed: {goal_manager.all_goals_completed()}")

    # Completing all goals
    for goal in goal_manager.goals:
        if not goal.completed:
            goal_manager.complete_goal(goal.description)

    print("\n" + f"All goals completed: {goal_manager.all_goals_completed()}")
    print("\n" + "Final Goals Status:")
    print(goal_manager.get_goals_for_prompt()) 