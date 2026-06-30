import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"

    def test_missing_azure_ad_token_provider_scope_raises(self):
        with pytest.raises(ValueError, match="scope"):
            LitellmModelConfig(model_name="test", azure_ad_token_provider={"credentials": ["azure_cli"]})

    def test_unknown_azure_ad_token_provider_credential_raises(self):
        with pytest.raises(ValueError, match="azure_cli"):
            LitellmModelConfig(
                model_name="test", azure_ad_token_provider={"scope": "api://trapi/.default", "credentials": ["bogus"]}
            )


def _install_fake_azure_identity(monkeypatch):
    calls = []

    class AzureCliCredential:
        pass

    class ManagedIdentityCredential:
        pass

    class DefaultAzureCredential:
        pass

    class ChainedTokenCredential:
        def __init__(self, *credentials):
            calls.append(("chain", [type(credential).__name__ for credential in credentials]))

    def get_bearer_token_provider(credential, scope):
        calls.append(("provider", type(credential).__name__, scope))
        return lambda: "token"

    azure = types.ModuleType("azure")
    identity = types.ModuleType("azure.identity")
    identity.AzureCliCredential = AzureCliCredential
    identity.ManagedIdentityCredential = ManagedIdentityCredential
    identity.DefaultAzureCredential = DefaultAzureCredential
    identity.ChainedTokenCredential = ChainedTokenCredential
    identity.get_bearer_token_provider = get_bearer_token_provider
    azure.identity = identity
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.identity", identity)
    return calls


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    def test_azure_ad_token_provider_configures_callable(self, monkeypatch):
        calls = _install_fake_azure_identity(monkeypatch)

        model = LitellmModel(
            model_name="azure/gpt-4o_2024-11-20",
            azure_ad_token_provider={
                "scope": "api://trapi/.default",
                "credentials": ["azure_cli", "managed_identity"],
            },
            model_kwargs={
                "api_base": "https://trapi.research.microsoft.com/redmond/interactive",
                "api_version": "v1",
                "drop_params": True,
            },
            cost_tracking="ignore_errors",
        )

        assert callable(model.config.model_kwargs["azure_ad_token_provider"])
        assert model.config.model_kwargs["api_base"] == "https://trapi.research.microsoft.com/redmond/interactive"
        assert calls == [
            ("chain", ["AzureCliCredential", "ManagedIdentityCredential"]),
            ("provider", "ChainedTokenCredential", "api://trapi/.default"),
        ]

    def test_azure_ad_token_provider_rejects_conflicting_model_kwargs(self):
        for conflict in ["api_key", "azure_ad_token_provider"]:
            with pytest.raises(ValueError, match=conflict):
                LitellmModel(
                    model_name="azure/gpt-4o_2024-11-20",
                    azure_ad_token_provider={"scope": "api://trapi/.default"},
                    model_kwargs={conflict: "existing"},
                )

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []
