"""Tests for optional backend adapters without live provider dependencies."""

import sys
import types

import pytest

from nx_agent.backends import (
    anthropic_backend,
    grok_backend,
    huggingface_backend,
    ollama_backend,
    openai_backend,
)
from nx_agent.exceptions import ProviderRateLimitError


TOOL_SCHEMA = {
    "name": "lookup",
    "description": "Look up a value.",
    "parameters": {"type": "object", "properties": {}},
}


class FakeChatClient:
    def __init__(self, message, error=None):
        self.message = message
        self.error = error
        self.requests = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self.message)]
        )


def install_fake_openai(monkeypatch, message, error=None):
    client = FakeChatClient(message, error)
    constructed = {}

    def factory(**kwargs):
        constructed.update(kwargs)
        return client

    module = types.SimpleNamespace(OpenAI=factory)
    monkeypatch.setitem(sys.modules, "openai", module)
    return client, constructed


class TestOpenAICompatibleBackends:
    def test_openai_text_response_and_configuration(self, monkeypatch):
        message = types.SimpleNamespace(content="hello", tool_calls=None)
        client, constructed = install_fake_openai(monkeypatch, message)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

        backend = openai_backend(model="test-model")
        output = backend("be helpful", "hello", [])

        assert output == "hello"
        assert constructed == {"api_key": "openai-secret"}
        assert client.requests[0]["model"] == "test-model"
        assert "tools" not in client.requests[0]

    def test_grok_configures_xai_endpoint_and_maps_tool_call(self, monkeypatch):
        function = types.SimpleNamespace(name="lookup", arguments='{"id": "7"}')
        message = types.SimpleNamespace(
            content=None,
            tool_calls=[types.SimpleNamespace(function=function)],
        )
        client, constructed = install_fake_openai(monkeypatch, message)
        monkeypatch.setenv("XAI_API_KEY", "xai-secret")

        backend = grok_backend()
        output = backend("system", "find it", [TOOL_SCHEMA])

        assert output == 'TOOL_CALL:lookup:{"id": "7"}'
        assert constructed == {
            "api_key": "xai-secret",
            "base_url": "https://api.x.ai/v1",
        }
        request = client.requests[0]
        assert request["model"] == "grok-4.3"
        assert request["tools"] == [{"type": "function", "function": TOOL_SCHEMA}]
        assert request["tool_choice"] == "auto"

    def test_openai_maps_sdk_rate_limit_by_class_name(self, monkeypatch):
        RateLimitError = type("RateLimitError", (Exception,), {})
        client, _ = install_fake_openai(
            monkeypatch,
            types.SimpleNamespace(content="", tool_calls=None),
            error=RateLimitError("quota exceeded"),
        )

        backend = openai_backend(model="test-model", api_key="secret")

        with pytest.raises(ProviderRateLimitError) as exc_info:
            backend("system", "hello", [])

        assert client.requests[0]["model"] == "test-model"
        assert exc_info.value.provider == "openai"
        assert "quota exceeded" in str(exc_info.value)

    def test_grok_maps_compatible_http_429(self, monkeypatch):
        error = RuntimeError("too many requests")
        error.status_code = 429
        install_fake_openai(
            monkeypatch,
            types.SimpleNamespace(content="", tool_calls=None),
            error=error,
        )

        backend = grok_backend(api_key="secret")

        with pytest.raises(ProviderRateLimitError) as exc_info:
            backend("system", "hello", [])

        assert exc_info.value.provider == "grok"


class TestHuggingFaceBackend:
    def test_chat_completion_maps_tools_and_output(self, monkeypatch):
        function = types.SimpleNamespace(name="lookup", arguments={"id": "7"})
        message = types.SimpleNamespace(
            content=None,
            tool_calls=[types.SimpleNamespace(function=function)],
        )
        constructed = {}

        class FakeInferenceClient:
            def __init__(self, **kwargs):
                constructed.update(kwargs)
                self.requests = []

            def chat_completion(self, **kwargs):
                self.requests.append(kwargs)
                constructed["request"] = kwargs
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        module = types.SimpleNamespace(InferenceClient=FakeInferenceClient)
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        monkeypatch.setenv("HF_TOKEN", "hf-secret")

        backend = huggingface_backend("provider/model")
        output = backend("system", "find it", [TOOL_SCHEMA])

        assert output == 'TOOL_CALL:lookup:{"id": "7"}'
        assert constructed["model"] == "provider/model"
        assert constructed["token"] == "hf-secret"
        assert constructed["request"]["tools"] == [
            {"type": "function", "function": TOOL_SCHEMA}
        ]

    def test_maps_http_response_rate_limit(self, monkeypatch):
        constructed = {}

        class FakeInferenceClient:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

            def chat_completion(self, **kwargs):
                error = RuntimeError("provider throttled")
                error.response = types.SimpleNamespace(status_code=429)
                raise error

        module = types.SimpleNamespace(InferenceClient=FakeInferenceClient)
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

        backend = huggingface_backend("provider/model")

        with pytest.raises(ProviderRateLimitError) as exc_info:
            backend("system", "hello", [])

        assert exc_info.value.provider == "huggingface"

    def test_preserves_max_new_tokens_alias(self, monkeypatch):
        class FakeInferenceClient:
            def __init__(self, **kwargs):
                pass

            def chat_completion(self, **kwargs):
                assert kwargs["max_tokens"] == 42
                message = types.SimpleNamespace(content="done", tool_calls=None)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        module = types.SimpleNamespace(InferenceClient=FakeInferenceClient)
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

        backend = huggingface_backend("provider/model", max_new_tokens=42)

        assert backend("system", "hello", []) == "done"


class TestAnthropicBackend:
    def test_maps_rate_limit_by_class_name(self, monkeypatch):
        RateLimitError = type("RateLimitError", (Exception,), {})

        class FakeAnthropicClient:
            def __init__(self, **kwargs):
                self.messages = types.SimpleNamespace(create=self.create)

            def create(self, **kwargs):
                raise RateLimitError("requests exhausted")

        module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
        monkeypatch.setitem(sys.modules, "anthropic", module)

        backend = anthropic_backend(api_key="secret")

        with pytest.raises(ProviderRateLimitError) as exc_info:
            backend("system", "hello", [])

        assert exc_info.value.provider == "anthropic"


class TestOllamaBackend:
    def test_local_chat_maps_tools_and_response(self, monkeypatch):
        constructed = {}

        class FakeOllamaClient:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

            def chat(self, **kwargs):
                constructed["request"] = kwargs
                function = types.SimpleNamespace(
                    name="lookup", arguments={"id": "local"}
                )
                message = types.SimpleNamespace(
                    content="",
                    tool_calls=[types.SimpleNamespace(function=function)],
                )
                return types.SimpleNamespace(message=message)

        module = types.SimpleNamespace(Client=FakeOllamaClient)
        monkeypatch.setitem(sys.modules, "ollama", module)

        backend = ollama_backend(model="qwen3", host="http://localhost:11434")
        output = backend("system", "find locally", [TOOL_SCHEMA])

        assert output == 'TOOL_CALL:lookup:{"id": "local"}'
        assert constructed["host"] == "http://localhost:11434"
        assert constructed["request"]["model"] == "qwen3"
        assert constructed["request"]["tools"] == [
            {"type": "function", "function": TOOL_SCHEMA}
        ]

    def test_uses_ollama_default_host_configuration(self, monkeypatch):
        constructed = {}

        class FakeOllamaClient:
            def __init__(self, **kwargs):
                constructed["constructor"] = kwargs

            def chat(self, **kwargs):
                message = types.SimpleNamespace(content="local answer", tool_calls=None)
                return types.SimpleNamespace(message=message)

        module = types.SimpleNamespace(Client=FakeOllamaClient)
        monkeypatch.setitem(sys.modules, "ollama", module)

        output = ollama_backend()("system", "hello", [])

        assert output == "local answer"
        assert constructed["constructor"] == {}

    def test_maps_ollama_status_code_rate_limit(self, monkeypatch):
        class FakeOllamaClient:
            def chat(self, **kwargs):
                error = RuntimeError("slow down")
                error.status_code = 429
                raise error

        module = types.SimpleNamespace(Client=lambda **kwargs: FakeOllamaClient())
        monkeypatch.setitem(sys.modules, "ollama", module)

        backend = ollama_backend()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            backend("system", "hello", [])

        assert exc_info.value.provider == "ollama"
