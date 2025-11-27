# TODO

- [ ] **Add sub-goal orchestration to the agent loop**  
  Ensure the quality agent can spin up scoped child goals (e.g., branch prep, lint fixes) so each runs with minimal context, isolated tool memory, and a clearly defined exit criterion.
- [ ] **Halt workflows when no lint issues are returned**  
  Short-circuit after `qlty_get_first_issue` reports `issue_found: false`, preventing unnecessary git operations and highlighting that lint is already clean.
- [ ] **Document the sub-goal lifecycle**  
  Produce developer docs covering how sub-goals are created, what state they inherit, and how their results roll back into the parent goal for review.
