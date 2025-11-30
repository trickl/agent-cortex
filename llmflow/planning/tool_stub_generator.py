"""Utilities for generating Java tool stub classes for planning."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import pathlib
import re
from textwrap import wrap
from typing import (
	Any,
	List,
	Mapping,
	MutableMapping,
	MutableSequence,
	Sequence,
	Set,
	Tuple,
	Union,
	get_args,
	get_origin,
)

from llmflow.tools import load_all_tools
from llmflow.tools.tool_decorator import get_registered_tools

try:  # Python 3.13 keeps Annotated in typing
	from typing import Annotated  # type: ignore
except ImportError:  # pragma: no cover - fallback for older runtimes
	Annotated = None  # type: ignore


_CLASS_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_METHOD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LIST_ORIGINS = {list, List, tuple, Sequence, MutableSequence}
_MAP_ORIGINS = {dict, Mapping, MutableMapping}


class ToolStubGenerationError(RuntimeError):
	"""Raised when tool stubs cannot be generated."""


@dataclass
class JavaParameter:
	name: str
	java_type: str
	optional: bool


@dataclass
class JavaMethodSpec:
	name: str
	return_type: str
	summary_lines: List[str]
	parameters: List[JavaParameter]
	optional_parameters: List[str]
	return_nullable: bool
	imports: Set[str]


def generate_tool_stub_class(
	class_name: str,
	tool_names: Sequence[str],
	*,
	package: str | None = None,
) -> str:
	"""Render a Java class containing static stubs for the requested tools."""

	if not tool_names:
		raise ToolStubGenerationError("At least one tool name must be provided")
	if not _CLASS_NAME_RE.match(class_name):
		raise ToolStubGenerationError(
			f"Class name '{class_name}' is not a valid Java identifier; "
			"start with an uppercase letter and use alphanumerics/underscores only."
		)
	if package and not _is_valid_package(package):
		raise ToolStubGenerationError(
			f"Package '{package}' contains invalid segments; use dot-separated identifiers"
		)

	load_all_tools()
	registry = get_registered_tools()
	missing = [name for name in tool_names if name not in registry]
	if missing:
		raise ToolStubGenerationError(
			"Unregistered tool names: " + ", ".join(sorted(missing))
		)

	method_specs: List[JavaMethodSpec] = []
	for tool_name in tool_names:
		entry = registry[tool_name]
		method_specs.append(_build_method_spec(tool_name, entry))

	imports: Set[str] = set()
	for spec in method_specs:
		imports.update(spec.imports)

	lines: List[str] = []
	if package:
		lines.append(f"package {package};")
		lines.append("")
	for imp in sorted(imports):
		lines.append(f"import {imp};")
	if imports:
		lines.append("")
	lines.append("@SuppressWarnings(\"unused\")")
	lines.append(f"public final class {class_name} {{")
	lines.append(f"    private {class_name}() {{")
	lines.append('        throw new AssertionError("Utility class");')
	lines.append("    }")
	lines.append("")

	for idx, spec in enumerate(method_specs):
		lines.extend(_render_method(spec))
		if idx != len(method_specs) - 1:
			lines.append("")

	lines.append("}")
	return "\n".join(lines) + "\n"


def _build_method_spec(tool_name: str, registry_entry: dict[str, Any]) -> JavaMethodSpec:
	func = registry_entry["function"]
	signature = inspect.signature(func)
	type_hints = inspect.get_annotations(func, eval_str=True)
	summary = registry_entry["schema"]["function"].get("description", tool_name)
	summary_lines = _wrap_comment(summary.strip() or tool_name)

	parameters: List[JavaParameter] = []
	imports: Set[str] = set()
	optional_parameters: List[str] = []

	for name, param in signature.parameters.items():
		if name in {"self", "cls"}:
			continue
		annotation = type_hints.get(name, param.annotation)
		annotation = _strip_annotated(annotation)
		annotation, optional_from_annotation = _unwrap_optional(annotation)
		is_optional = optional_from_annotation or param.default is not inspect._empty
		java_type, java_imports = _python_type_to_java(annotation, boxed=is_optional)
		imports.update(java_imports)
		parameters.append(
			JavaParameter(name=name, java_type=java_type, optional=is_optional)
		)
		if is_optional:
			optional_parameters.append(name)

	return_annotation = type_hints.get("return", signature.return_annotation)
	return_annotation = _strip_annotated(return_annotation)
	return_annotation, return_optional = _unwrap_optional(return_annotation)

	if return_annotation in {inspect._empty, type(None)}:
		return_type = "void"
	else:
		return_type, return_imports = _python_type_to_java(
			return_annotation,
			boxed=return_optional,
		)
		imports.update(return_imports)

	return JavaMethodSpec(
		name=_sanitize_method_name(tool_name),
		return_type=return_type,
		summary_lines=summary_lines,
		parameters=parameters,
		optional_parameters=optional_parameters,
		return_nullable=return_optional,
		imports=imports,
	)


def _python_type_to_java(py_type: Any, *, boxed: bool) -> Tuple[str, Set[str]]:
	py_type = _resolve_literal(py_type)

	if py_type is str:
		return "String", set()
	if py_type in {int}:
		return ("Integer" if boxed else "int"), set()
	if py_type in {float}:
		return ("Double" if boxed else "double"), set()
	if py_type in {bool}:
		return ("Boolean" if boxed else "boolean"), set()
	if py_type in {bytes, bytearray}:
		return "byte[]", set()
	if isinstance(py_type, type) and issubclass(py_type, pathlib.Path):
		return "String", set()

	origin = get_origin(py_type)
	args = get_args(py_type)

	if origin in _LIST_ORIGINS:
		element_type = args[0] if args else Any
		element_java, element_imports = _python_type_to_java(element_type, boxed=True)
		imports = set(element_imports)
		imports.add("java.util.List")
		return f"List<{element_java}>", imports

	if origin in _MAP_ORIGINS:
		value_type = args[1] if len(args) >= 2 else Any
		value_java, value_imports = _python_type_to_java(value_type, boxed=True)
		imports = set(value_imports)
		imports.add("java.util.Map")
		return f"Map<String, {value_java}>", imports

	if origin is set:
		element_type = args[0] if args else Any
		element_java, element_imports = _python_type_to_java(element_type, boxed=True)
		imports = set(element_imports)
		imports.add("java.util.Set")
		return f"Set<{element_java}>", imports

	return "Object", set()


def _unwrap_optional(py_type: Any) -> Tuple[Any, bool]:
	origin = get_origin(py_type)
	if origin is None:
		return py_type, False
	args = [arg for arg in get_args(py_type) if arg is not type(None)]
	if origin is Union and len(args) == 1:
		return args[0], True
	return py_type, False


def _resolve_literal(py_type: Any) -> Any:
	origin = get_origin(py_type)
	if origin is None:
		return py_type
	if getattr(origin, "__name__", None) == "Literal":  # type: ignore[attr-defined]
		args = get_args(py_type)
		if args:
			return type(args[0])
	return py_type


def _strip_annotated(py_type: Any) -> Any:
	if Annotated is None:
		return py_type
	origin = get_origin(py_type)
	if origin is Annotated:
		return get_args(py_type)[0]
	return py_type


def _sanitize_method_name(name: str) -> str:
	if not _METHOD_NAME_RE.match(name):
		raise ToolStubGenerationError(
			f"Tool name '{name}' is not a valid Java method name"
		)
	return name


def _render_method(spec: JavaMethodSpec) -> List[str]:
	lines: List[str] = []
	lines.append("    /**")
	for summary_line in spec.summary_lines:
		lines.append(f"     * {summary_line}")
	if spec.optional_parameters:
		optional_list = ", ".join(spec.optional_parameters)
		lines.append("     *")
		lines.append(f"     * Optional parameters: {optional_list}.")
	if spec.return_nullable:
		lines.append("     *")
		lines.append("     * May return null when the tool does not provide a value.")
	lines.append("     */")
	params = ", ".join(f"{param.java_type} {param.name}" for param in spec.parameters)
	lines.append(
		f"    public static {spec.return_type} {spec.name}({params}) {{"
	)
	lines.append(
		'        throw new UnsupportedOperationException("Stub for planning only.");'
	)
	lines.append("    }")
	return lines


def _wrap_comment(text: str) -> List[str]:
	if not text:
		return [""]
	wrapped: List[str] = []
	for paragraph in text.splitlines():
		if not paragraph.strip():
			wrapped.append("")
			continue
		for line in wrap(paragraph, width=94):
			wrapped.append(line)
	return wrapped


def _is_valid_package(package: str) -> bool:
	segments = package.split('.')
	return all(_METHOD_NAME_RE.match(segment) for segment in segments)


__all__ = ["generate_tool_stub_class", "ToolStubGenerationError"]
