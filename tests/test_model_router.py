"""Tests for model_router._call_local and _call_groq implementations.

All tests are mock-based — no GPU, no API key, no network needed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dermarbiter.core.config import AgentConfig, DermArbiterConfig
from dermarbiter.core.model_router import ModelBackend, ModelRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config() -> DermArbiterConfig:
    """Config with no API keys — only LOCAL_HF is available."""
    return DermArbiterConfig(
        google_api_key="",
        groq_api_key="",
        agents={
            "generalist": AgentConfig(
                role="generalist",
                model_backend="local_hf",
                model_name="test-model",
                device="cuda",
                quantization="4bit",
                temperature=0.3,
            ),
            "skeptic": AgentConfig(
                role="skeptic",
                model_backend="local_hf",
                model_name="Qwen/Qwen3-8B-Instruct",
                device="cuda",
                quantization="4bit",
                temperature=0.5,
            ),
        },
    )


@pytest.fixture
def groq_config() -> DermArbiterConfig:
    """Config with a Groq API key."""
    # Inject a fake groq module so _init_backends can import it
    fake_groq = MagicMock()
    sys.modules.setdefault("groq", fake_groq)

    return DermArbiterConfig(
        google_api_key="",
        groq_api_key="gsk_test_fake_key_12345",
        agents={
            "test_agent": AgentConfig(
                role="test_agent",
                model_backend="groq_api",
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
            ),
        },
    )


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]


# ---------------------------------------------------------------------------
# _call_local Tests
# ---------------------------------------------------------------------------

class TestCallLocal:
    """Tests for the local HuggingFace backend."""

    @patch("dermarbiter.core.model_router.ModelRouter._call_local")
    def test_dispatch_routes_to_local(self, mock_local, minimal_config):
        """Verify _dispatch routes LOCAL_HF to _call_local."""
        mock_local.return_value = "mocked response"
        router = ModelRouter(minimal_config)

        result = router.call("generalist", SAMPLE_MESSAGES)
        mock_local.assert_called_once()
        assert result == "mocked response"

    def test_call_local_loads_and_generates(self, minimal_config):
        """End-to-end _call_local with mocked transformers."""
        router = ModelRouter(minimal_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.decode.return_value = "Generated diagnosis text"

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("torch.inference_mode"):

            # Mock the tokenizer __call__ to return a dict with proper shape
            input_ids_mock = MagicMock()
            input_ids_mock.shape = [1, 10]
            input_ids_mock.to.return_value = input_ids_mock
            mock_tokenizer.return_value = {"input_ids": input_ids_mock}

            # Mock generate to return tensor-like output
            gen_output = MagicMock()
            gen_output.__getitem__ = lambda s, i: list(range(20))
            mock_model.generate.return_value = gen_output

            result = router._call_local(
                SAMPLE_MESSAGES,
                model="test-model",
                device="cpu",
                quantization=None,
            )

        assert result == "Generated diagnosis text"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_call_local_caches_model(self, minimal_config):
        """Model should be loaded once and cached on subsequent calls."""
        router = ModelRouter(minimal_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        input_ids.to.return_value = input_ids
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "response"

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        gen_out = MagicMock()
        gen_out.__getitem__ = lambda s, i: list(range(10))
        mock_model.generate.return_value = gen_out

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as tok_load, \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as model_load, \
             patch("torch.inference_mode"):

            # Call twice
            router._call_local(SAMPLE_MESSAGES, "test-model", "cpu", None)

            # Reset mock iterator for second call
            mock_model.parameters.return_value = iter([mock_param])
            router._call_local(SAMPLE_MESSAGES, "test-model", "cpu", None)

        # Model should only be loaded once (cached)
        assert tok_load.call_count == 1
        assert model_load.call_count == 1

    def test_call_local_4bit_quantization_kwargs(self, minimal_config):
        """4bit quantization should use BitsAndBytesConfig."""
        router = ModelRouter(minimal_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        input_ids.to.return_value = input_ids
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "response"

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        gen_out = MagicMock()
        gen_out.__getitem__ = lambda s, i: list(range(10))
        mock_model.generate.return_value = gen_out

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as model_load, \
             patch("transformers.BitsAndBytesConfig") as bnb_config, \
             patch("torch.inference_mode"):

            router._call_local(SAMPLE_MESSAGES, "test-model", "cuda", "4bit")

        # BitsAndBytesConfig should have been instantiated
        bnb_config.assert_called_once()
        # Model load should include quantization_config and device_map
        call_kwargs = model_load.call_args[1]
        assert "quantization_config" in call_kwargs
        assert call_kwargs["device_map"] == "auto"


# ---------------------------------------------------------------------------
# _call_groq Tests
# ---------------------------------------------------------------------------

class TestCallGroq:
    """Tests for the Groq Cloud API backend."""

    def test_call_groq_creates_client_and_calls(self, groq_config):
        """Groq client should be created and chat.completions.create called."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Groq says hello"))
        ]

        fake_groq = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        fake_groq.Groq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": fake_groq}):
            router = ModelRouter(groq_config)
            result = router._call_groq(
                SAMPLE_MESSAGES,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=4096,
            )

        assert result == "Groq says hello"
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            messages=SAMPLE_MESSAGES,
            temperature=0.3,
            max_tokens=4096,
        )

    def test_call_groq_caches_client(self, groq_config):
        """Groq client should be created once and reused."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="response"))
        ]

        fake_groq = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        fake_groq.Groq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": fake_groq}):
            router = ModelRouter(groq_config)
            router._call_groq(SAMPLE_MESSAGES, "test-model")
            router._call_groq(SAMPLE_MESSAGES, "test-model")

        # Groq() instantiated only once (cached via hasattr check)
        assert fake_groq.Groq.call_count == 1

    @patch("dermarbiter.core.model_router.ModelRouter._call_groq")
    def test_dispatch_routes_to_groq(self, mock_groq, groq_config):
        """Verify _dispatch routes GROQ_API to _call_groq."""
        mock_groq.return_value = "groq response"
        router = ModelRouter(groq_config)

        result = router.call("test_agent", SAMPLE_MESSAGES)
        mock_groq.assert_called_once()
        assert result == "groq response"


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests that agent config properly forwards device/quantization."""

    def test_agent_config_has_device_and_quantization(self):
        """AgentConfig should accept device and quantization fields."""
        cfg = AgentConfig(
            role="test",
            model_backend="local_hf",
            model_name="test-model",
            device="cuda",
            quantization="4bit",
        )
        assert cfg.device == "cuda"
        assert cfg.quantization == "4bit"

    def test_agent_config_defaults(self):
        """AgentConfig should default to cpu and no quantization."""
        cfg = AgentConfig(role="test")
        assert cfg.device == "cpu"
        assert cfg.quantization is None

    def test_agent_config_ignores_extra_fields(self):
        """AgentConfig should not crash on extra YAML fields."""
        cfg = AgentConfig(
            role="test",
            name="Test Agent",
            description="A test agent",
            top_p=0.9,
        )
        assert cfg.role == "test"

    def test_call_forwards_device_from_config(self, minimal_config):
        """call() should forward device and quantization from AgentConfig."""
        router = ModelRouter(minimal_config)

        with patch.object(router, "_call_local", return_value="ok") as mock_local:
            router.call("generalist", SAMPLE_MESSAGES)

        call_kwargs = mock_local.call_args
        # device and quantization should come from agent config
        assert call_kwargs[0][2] == "cuda"  # device positional arg
        assert call_kwargs[0][3] == "4bit"  # quantization positional arg
