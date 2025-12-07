# **Agent-Cortex**  
*A Decomposition-Driven, Self-Refining Orchestration Engine for Large Language Models*

Agent-Cortex is an experiment in building *reliable intelligence atop unreliable substrates*.  
At its heart lies a simple question:

> **Can we turn the inherently fuzzy output of LLMs into a system that behaves with the precision of classical software?**

The project explores this question by constructing a hierarchical, self-correcting planning system in which LLMs write *typed, executable plans*; a refinement engine ensures those plans compile; and a Python-based runtime executes them step-by-step, deferring to tools or generating new sub-plans as required.

This is not merely another agent framework. Agent-Cortex is an attempt at a **principled architecture** — one inspired as much by *compiler theory* and *soft robotics* as by contemporary AI.

---

## **Why Agent-Cortex Exists**

Contemporary LLM agents often suffer from the same ailments:

- Tool invocation is inconsistent and brittle.  
- Plans are ephemeral, untyped, and difficult to validate.  
- Context rapidly degrades.  
- Minor deviations (indentation, hallucinated fields, inconsistent schema) cause catastrophic failure.  
- The agent behaves as though improvising rather than reasoning.

Agent-Cortex attempts to remedy these structural issues by imposing discipline and *form*.  
Its design philosophy rests on four key insights:

### **1. Decomposition is the only scalable form of reasoning**
LLMs perform best when asked to break a problem into steps.  
Agent-Cortex leverages this by allowing the model to **invent arbitrary functions** (sub-goals) inside a pure Java plan.  
Each function must reduce complexity relative to its parent.  
Eventually, the leaves of the tree must resolve into concrete tool invocations.
This approach purposely avoids the traditional agent loop employed by chat agents that add each new question
and response to the context window of the LLM. The chat agent loop suffers over time from increased cost, lower speed
and hallucinations as the context window grows. It is forced eventually to summarise prior conversational history
as a compromise, in the process losing essential information.
The hierarchical approach instead works in the same way as traditional programming - each sub-problem only
as a context-window to solve it's immediate concern. This is faster, cheaper and removes the context
pollution issue.

### **2. Typed languages are a blessing, not a burden**
Rather than invent a bespoke DSL in planning actions (and wrestle with schema complexity), we let the model write **real Java**.  
Java’s type system offers a natural scaffolding in which the plan is:

- structured  
- hierarchical  
- machine-verifiable  

LLMs are trained on a wealth of Java already, being one of the most popular typed languages in the world, so we don't have
to fill our system context with instructions on how to write it.

If the model produces poor Java — no matter: a refinement loop will fix it.

### **3. Refinement is not optional — it *is* the method**
The first plan an LLM produces will usually be wrong.  
The second will be better.  
By the tenth, you often have something correct.

Agent-Cortex embraces this reality by implementing a **tight refinement loop**:

1. The model produces Java during it's planning phase to decompose the task at hand.  
2. We compile it.  
3. On error, the model receives the original code, the compiler logs, and is instructed to make *minimal, localised patches*.  
4. We repeat until the code compiles (or we abandon the attempt).

This is the same loop later used for tool-level code repair.

### **4. Tools are terminal nodes of reasoning**
We remove the notion of “normal” vs “deferred”.  
All invented functions are *implicitly deferred* unless they call a tool.  
A tool call is a leaf node, and therefore a base case of reasoning.  

If a function can be solved by a tool, it *must* be solved by a tool.

---

## **System Architecture**

Agent-Cortex consists of three interacting layers:

---

### **1. The Planner (LLM-Driven, Java-Based)**

- Accepts a natural language request.
- Produces a **hierarchical Java file** in which:
  - Functions represent sub-goals.
  - Leaf functions call concrete tools.
- Adheres to four structural rules:
  1. The model may invent functions freely.  
  2. Sub-functions must reduce complexity.  
  3. Tool-solvable steps must call tools directly.  
  4. Excessive depth (e.g., >7 levels) invalidates the plan.  

The result is a static but deeply structured representation of intent.

---

### **2. The Refinement Engine (Java Compiler + Patch Loop)**

- Compiles the generated Java.  
- On failure:
  - Sends the model the code + compiler output.  
  - Requests a **surgical patch**, not a complete rewrite.  
- Repeats until:
  - The code compiles,  
  - or a retry threshold is met.

This mechanism dramatically improves reliability over naïve code generation.

---

### **3. The Executor (Python Runtime + Tooling Layer)**

Although the plan itself is expressed in Java, the execution engine is Python-based — pragmatic, lightweight, and easily containerised.

Execution proceeds as follows:

1. The plan’s call graph is traversed.  
2. When reaching a stub method, the system:
   - Constructs a new planning request for that sub-goal.  
   - Generates and refines a sub-plan.  
   - Returns the output upwards.  
3. A call to `PlanningTools.someTool()` maps to **real registered tools** in Python.  
4. The executor integrates results, folding them back through the hierarchy.

This Python/Java duality proves surprisingly elegant:  
Java provides structure; Python provides operational flexibility.

---

## **Key Design Ambitions**

Agent-Cortex aspires to become a **general-purpose orchestration layer** offering:

- **Repeatability** — Plans are deterministic artefacts, cacheable and testable.  
- **Traceability** — Every decision is stored as code.  
- **Composability** — Tools and sub-plans form a natural architecture.  
- **Robustness** — Refinement and decomposition counteract LLM noisiness.  
- **Extensibility** — Any callable tool (weather APIs, email handlers, CUDA debuggers) can join the ecosystem.

---

## **A Note on the Philosophy**

Agent-Cortex is, in spirit, closer to *compilers* and *hierarchical planners* than to typical LLM “agents”.

We do not chase emergent magic.  
We build robustness through structure.

The system assumes:

- LLMs will hallucinate.  
- They will be inconsistent.  
- They will occasionally produce things of unalloyed absurdity.

And yet: with the right scaffolding, they can behave like extraordinarily capable junior engineers — messy, but improvable.

Agent-Cortex is that scaffolding.

---

## **Roadmap**

- Working examples of building useful agents using just natural language
- Better observability and logging
- Better robustness / reliability
- Separation of tooling into a separate repository 
- Dynamic tool search and lading

---

## **Contributing**

PRs are warmly welcomed.  
This project grew out of curiosity, and will grow further through conversation, critique, and playful experimentation.

If Agent-Cortex intrigues you — build with it, break it, refine it, extend it.

---

## **Credits**

This project began as a fork of **LLMFlow** by **kamikaze2020**, whose work provided the core inspiration for a lightweight, structured orchestration layer around LLMs.
