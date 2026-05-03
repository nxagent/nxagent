"""
Tool decorator and schema extraction.

Usage
-----
    @tool
    def web_search(query: str) -> str:
        \"\"\"Search the web for current information.\"\"\"
        ...

The decorator:
  - preserves the original function (callable as normal Python)
  - attaches a `.schema` attribute (ToolSchema) used by agents
  - registers the function in a global registry for introspection
"""

from __future__ import annotations

import inspect
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_type_hints

_TOOL_REGISTRY: Dict[str, "ToolSchema"] = {}


@dataclass
class ParameterSchema:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = inspect.Parameter.empty

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "description": self.description}
        if not self.required:
            d["default"] = None if self.default is inspect.Parameter.empty else self.default
        return d


@dataclass
class ToolSchema:
    """JSON-schema-like description of a tool, auto-built from type hints + docstring."""

    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    fn: Optional[Callable] = field(default=None, repr=False)

    # ── helpers ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return an OpenAI-style function-call schema dict."""
        props: dict = {}
        required: list = []
        for p in self.parameters:
            props[p.name] = p.to_dict()
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def __call__(self, *args, **kwargs):
        """Invoke the underlying function."""
        if self.fn is None:
            raise RuntimeError(f"Tool '{self.name}' has no attached function.")
        return self.fn(*args, **kwargs)


# ── type-to-jsonschema mapping ────────────────────────────────────────────────

_PY_TO_JSON: Dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _py_type_to_json(annotation) -> str:
    name = getattr(annotation, "__name__", str(annotation))
    return _PY_TO_JSON.get(name, "string")


def _parse_docstring_params(doc: str) -> Dict[str, str]:
    """Extract param descriptions from a numpy/google/plain docstring."""
    descriptions: Dict[str, str] = {}
    if not doc:
        return descriptions
    in_params = False
    for line in doc.splitlines():
        stripped = line.strip()
        # Google-style: "Args:" section
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params = True
            continue
        if in_params:
            if stripped == "" or (stripped.endswith(":") and not stripped.startswith(" ")):
                in_params = False
                continue
            if ":" in stripped:
                param, _, desc = stripped.partition(":")
                descriptions[param.strip().split()[0]] = desc.strip()
    return descriptions


def _build_schema(fn: Callable) -> ToolSchema:
    doc = inspect.getdoc(fn) or ""
    # First line of docstring → description
    description = doc.splitlines()[0] if doc else fn.__name__

    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    param_docs = _parse_docstring_params(doc)

    parameters: List[ParameterSchema] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = hints.get(pname, str)
        json_type = _py_type_to_json(ann)
        desc = param_docs.get(pname, f"The {pname} parameter.")
        required = param.default is inspect.Parameter.empty
        parameters.append(
            ParameterSchema(
                name=pname,
                type=json_type,
                description=desc,
                required=required,
                default=param.default,
            )
        )

    return ToolSchema(
        name=fn.__name__,
        description=description,
        parameters=parameters,
        fn=fn,
    )


# ── public decorator ──────────────────────────────────────────────────────────

def tool(fn: Callable) -> Callable:
    """
    Decorate a plain Python function to make it an NxAgent tool.

    The decorated function remains fully callable. A `.schema` (ToolSchema)
    attribute is attached and the tool is added to the global registry.

    Example
    -------
        @tool
        def web_search(query: str) -> str:
            \"\"\"Search the web for current information.\"\"\"
            return requests.get(f"https://api.search.example?q={query}").text
    """
    schema = _build_schema(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.schema = schema  # type: ignore[attr-defined]
    wrapper.is_nx_tool = True  # type: ignore[attr-defined]

    # Register globally so Router / Workflow can discover tools
    _TOOL_REGISTRY[schema.name] = schema

    return wrapper


def get_tool_registry() -> Dict[str, ToolSchema]:
    """Return a copy of the global tool registry."""
    return dict(_TOOL_REGISTRY)


def is_tool(fn: Any) -> bool:
    """Return True if *fn* was decorated with @tool."""
    return callable(fn) and getattr(fn, "is_nx_tool", False)
