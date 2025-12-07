# Agent Configuration

This directory contains declarative presets that describe how an agent should be
instantiated (tools, prompts, workflow constraints, and environment bindings).

## quality_issue_agent_ollama.yaml

This is the canonical configuration for the "quality issue autofixer" agent.
It ships with the repository and powers both the Ollama-backed integration test
and any other runner that wants a ready-made preset. Highlights:

- **Base context** – opinionated system prompt plus user prompt template that
  instruct the agent to fetch Qlty issues, edit the repository, and ship fixes
  via pull requests.
- **Tooling** – enables git, file discovery, editing, and Qlty API helpers.
  Add/remove entries under `tools.tags.include` or explicitly list individual
  tool names under `tools.explicit` for tighter control.
- **Environment** – identifiers (repository root, owner/project keys, tokens)
  resolve via environment variables so deployments can override them without
  editing the preset.
- **Workflow** – iteration budget, reflection cadence, and branch naming rules
  are parameterised and can be overridden by downstream orchestration code.

To target a different LLM provider (e.g., OpenAI), load the YAML with
`yaml.safe_load`, mutate the `agent.llm` block in-memory, and then pass the
resulting structure into the `Agent` initialiser. Keeping a single source of
truth avoids the divergence issues we previously hit with multiple files.
