"""Tests for the @tool decorator and ToolSchema."""

from nx_agent import tool
from nx_agent.tool import is_tool, get_tool_registry, ToolSchema


@tool
def add(x: int, y: int) -> int:
    """Add two integers together.

    Args:
        x: First integer.
        y: Second integer.
    """
    return x + y


@tool
def greet(name: str, greeting: str = "Hello") -> str:
    """Return a greeting string."""
    return f"{greeting}, {name}!"


class TestToolDecorator:
    def test_callable(self):
        assert add(2, 3) == 5

    def test_is_tool(self):
        assert is_tool(add)

    def test_not_tool(self):
        assert not is_tool(lambda x: x)

    def test_schema_attached(self):
        assert hasattr(add, "schema")
        assert isinstance(add.schema, ToolSchema)

    def test_schema_name(self):
        assert add.schema.name == "add"

    def test_schema_description(self):
        assert "Add two integers" in add.schema.description

    def test_schema_parameters(self):
        params = {p.name: p for p in add.schema.parameters}
        assert "x" in params
        assert params["x"].type == "integer"
        assert params["x"].required is True

    def test_optional_parameter(self):
        params = {p.name: p for p in greet.schema.parameters}
        assert params["greeting"].required is False

    def test_to_dict(self):
        d = add.schema.to_dict()
        assert d["name"] == "add"
        assert "parameters" in d
        assert "x" in d["parameters"]["properties"]
        assert "x" in d["parameters"]["required"]

    def test_registry(self):
        registry = get_tool_registry()
        assert "add" in registry
        assert "greet" in registry
