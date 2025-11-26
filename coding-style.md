# Python Coding Style Guide

This document distills the public-domain guidance from [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/). Follow these practices for every commit, pull request, and AI-assisted suggestion in this repository.

---

## 1. Philosophy & Consistency
- Favor readability and local consistency over rigid rule-following.
- Remain backward compatible unless there is a compelling reason to change.
- Follow surrounding conventions when editing legacy sections, but clean them up when practical.
- Treat public APIs differently from internal helpers: document what is public and prefix internals with underscores.

## 2. File & Layout Structure
- **Indentation:** Use 4 spaces per level; continuation lines align vertically or use a hanging indent with a clear visual offset.
- **Tabs:** Avoid unless required to match an existing tab-indented file; never mix tabs and spaces in the same block.
- **Line length:** Limit code to 79 characters; wrap comments and docstrings at 72 characters. Prefer implicit parentheses over backslashes when breaking lines.
- **Blank lines:** Two blank lines around top-level classes/functions; one blank line between class methods. Use additional blank lines sparingly to separate logical sections.
- **Module size:** Keep source files under 300 lines. When a file nears this limit, refactor helpers into dedicated modules before adding new logic.
- **Imports:** Place at the top of the file after the module docstring. Group as standard library, third-party, and local; separate groups with blank lines. Avoid wildcard imports.
- **Module dunders:** Keep `__all__`, `__author__`, etc., immediately after the module docstring and any `from __future__` imports.

## 3. Whitespace & Expressions
- No extra spaces inside parentheses, brackets, or braces, nor before commas/colons/semicolons.
- Keep spacing symmetric around binary operators, but skip spaces around `=` in keyword arguments or default values. Add spaces around `=` when combined with annotations and defaults.
- Break long boolean or arithmetic expressions before the operator for clarity, especially when vertically aligning operands.
- Avoid trailing whitespace and gratuitous alignment via multi-space padding.

## 4. Naming Conventions
- **Modules/Packages:** Short, lowercase names; underscores allowed for modules but discouraged for packages.
- **Classes & Exceptions:** `CapWords`; add the `Error` suffix for true error types.
- **Functions, Methods, Variables:** `lower_case_with_underscores`.
- **Constants:** `UPPER_CASE_WITH_UNDERSCORES` at the module level.
- **Method arguments:** Use `self` for instance methods and `cls` for class methods. Append a trailing underscore when a name would otherwise shadow a keyword.
- **Visibility:** Single leading underscore for non-public members, double leading underscores only when name-mangling is necessary to avoid subclass collisions.

## 5. Functions, Complexity & Flow
- Keep functions focused; split large routines into logical helpers.
- Ensure every code path returns consistently. If any branch returns a value, make all branches explicit (use `return None` when needed).
- Prefer `def` over assigning lambdas to names—this aids tracebacks and readability.
- Avoid multiple statements on the same line. Keep `try` blocks minimal so exception handlers do not hide unrelated errors.
- Use context managers (`with`) for resource handling and keep cleanup localized.
- Prefer fail-fast behavior over fallbacks: if a dependency, capability, or environment requirement is missing, raise an explicit exception rather than silently substituting alternate logic.

## 6. Comments & Docstrings
- Keep comments truthful, concise, and in full sentences; update them with code changes.
- Use block comments aligned with the code they describe. Inline comments require two spaces before the `#` and should add non-obvious context.
- Provide docstrings for all public modules, classes, and functions following [PEP 257](https://peps.python.org/pep-0257/). Use triple double quotes, putting the closing quotes on their own line for multi-line docs.

## 7. Programming Practices
- Compare to singletons with `is`/`is not`; prefer `is not None` over `not ... is None` or truthiness when `None` is a legal sentinel.
- Implement the full set of rich comparisons or use `functools.total_ordering()`.
- Chain exceptions with `raise ... from ...`; catch the most specific exception possible. Reserve bare `except:` for logging or re-raising after cleanup.
- Use `with` statements for resources, `''.join()` for multi-part string assembly, `startswith`/`endswith` for prefix checks, and `isinstance` for type checks.
- Treat empty sequences as falsey (`if not seq:`) instead of comparing lengths explicitly.

## 8. Type Hints & Annotations
- Follow [PEP 484](https://peps.python.org/pep-0484/) and [PEP 526](https://peps.python.org/pep-0526/) formatting: one space after the colon, none before it; surround `=` with spaces when combining annotations and default values.
- Place `# type: ignore` comments near the top only when intentionally opting out of type checking for the entire file.
- Keep annotations lightweight in the stdlib-style sections, but feel free to adopt full typing in new modules when it improves clarity.

## 9. Using This Guide with AI Assistants
- Reference this document in GitHub Copilot “Custom Instructions” or similar tooling so generated suggestions comply with our standards.
- When prompting Copilot (or any AI assistant), mention: “Follow the project’s `coding-style.md`.”
- Review AI-generated diffs for conformance before accepting suggestions.

## 10. References
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)

_By adhering to these conventions, we keep the codebase approachable for humans and AI collaborators alike._