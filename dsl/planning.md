📘 Cortex Planning DSL (CPL) Specification

This document defines the Cortex Planning Language (CPL) — a structured, typed DSL used by the Agent Cortex to produce hierarchical plans.
The LLM must strictly follow this specification when generating plans.

🧠 1. Overview

CPL is a Java-like, minimal programming language used to express multi-step agent plans.

A CPL program:

Consists of functions.

Must define a top-level function called main().

May define additional sub-goal functions.

Uses syscalls to interact with external tools.

Is fully typed.

Uses clean hierarchical structure to decompose problems.

Your output must be a single valid CPL program, and nothing else.

🏗️ 2. CPL Structure

A CPL plan has the following structure:

plan {

    function main() : Void {
        # statements...
    }

    function functionName(arg1: Type, arg2: Type) : ReturnType {
        # statements...
    }

    # additional functions...

}


The outer plan { ... } block is required.

main() is the entry point.

Functions must be pure except for calls to syscalls.

🧩 3. Types

CPL supports the following types:

Void
String
Int
Bool
ToolResult
List<T>
Map<String, T>


Where T may be any valid type.

No other types are allowed.

📝 4. Statements

A function body consists of statements.
Each statement ends with a semicolon — except control blocks.

Variable declaration
let x: Type = expression;

Assignment
x = expression;

Return
return expression;
return;              # if ReturnType is Void

Conditional
if (condition) {
    # statements
} else {
    # statements
}

Loop
for (item in listVariable) {
    # statements
}

Try/catch
try {
    # statements
} catch (ToolError e) {
    # statements
}

🧮 5. Expressions

Expressions may include:

String, Int, Bool literals
variable names
function calls
syscall invocations
list literals
map literals
string concatenation using +


Example valid expressions:

"message"
42
true
x + " world"
doSomething(a, b)
syscall.fetchEmail("inbox")

⚙️ 6. Functions

Functions follow this format:

function name(arg1: Type, arg2: Type) : ReturnType {
    # statements
}


Rules:

Function names must be camelCase.

Argument names must be camelCase.

The number of statements inside a single function should be no more than 7, unless necessary.

Only functions and syscalls may be invoked.

Functions are pure except for syscalls.

### Deferred functions

Annotate a function with `@Deferred` when its body should be synthesized at execution
time instead of being planned up front. A deferred function may omit its body entirely
by ending the declaration with a semicolon:

```
@Deferred
function fixIssue(repo: ToolResult) : Void;
```

During execution, when a deferred function is invoked, the Agent Cortex runtime will
pause and request a function body from the planner using the current runtime context.
The planner must respond with only the function body block (`{ ... }`). The newly
generated body executes immediately, and it may be cached for subsequent invocations
depending on interpreter configuration.

Deferred functions may also include a placeholder body in the initial plan, but the
runtime is free to overwrite it when synthesis occurs. Non-deferred functions MUST
always provide a body in the plan.

🔧 7. Syscalls (Tool Calls)

Syscalls are the only way to interact with external systems.

They are invoked through the reserved identifier syscall:

syscall.toolName(arg1, arg2, ...)


All syscalls return either:

ToolResult

String, Int, or Bool depending on tool definition

Examples (your installation may define more):

syscall.fetchEmail(query: String) : ToolResult
syscall.cloneRepo(branch: String) : ToolResult
syscall.analyseRepo(repo: ToolResult) : ToolResult
syscall.applyPatch(repo: ToolResult, patch: ToolResult, branch: String) : ToolResult
syscall.createPullRequest(branch: String) : ToolResult
syscall.queryWeather(city: String) : ToolResult
syscall.extractCity(geo: ToolResult) : String
syscall.log(text: String) : Void


Do not invent new syscalls unless instructed. Use only those defined in the system prompt.

🧱 8. Error Handling

Only syscalls may throw ToolError.

Use try/catch:

try {
    let result: ToolResult = syscall.analyseRepo(repo);
    return result;
} catch (ToolError e) {
    syscall.log("Retrying...");
    return syscall.analyseRepo(repo);
}

🔒 9. Constraints the LLM Must Follow

When generating CPL:

Output only valid CPL code. No explanations.

All code must be inside a single plan { ... } block.

Must include a main() function.

Each function should implement one coherent sub-goal.

Functions should call no more than 7 other functions.

Avoid long functions — use sub-goals.

Use only allowed syntax and types.

Syscalls must be used exactly as defined.

📘 10. Example CPL Program
plan {

    function main() : Void {
        let repo: ToolResult = syscall.cloneRepo("origin/main");
        let analysis: ToolResult = analyse(repo);

        if (syscall.hasIssues(analysis)) {
            let branch: String = createBranch(repo);
            applyFixes(repo, analysis, branch);
            syscall.createPullRequest(branch);
        } else {
            syscall.log("No issues found.");
        }

        return;
    }

    function analyse(repo: ToolResult) : ToolResult {
        try {
            return syscall.analyseRepo(repo);
        } catch (ToolError e) {
            syscall.log("Retry after error.");
            return syscall.analyseRepo(repo);
        }
    }

    function createBranch(repo: ToolResult) : String {
        return syscall.createBranch(repo, "quality-fixes");
    }

    function applyFixes(repo: ToolResult, analysis: ToolResult, branch: String) : Void {
        let patches: List<ToolResult> = syscall.extractPatches(analysis);

        for (p in patches) {
            syscall.applyPatch(repo, p, branch);
        }

        return;
    }
}

🎯 11. Output Requirements

When you are asked to produce a plan:

Output only a CPL program.

The program must be valid according to this specification.

No prose, no commentary, no explanation — only the CPL code.