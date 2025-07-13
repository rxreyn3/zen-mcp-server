"""Tests for Azure OpenAI provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.azure_openai import AzureOpenAIModelProvider
from providers.base import ProviderType


class TestAzureOpenAIProvider:
    """Test Azure OpenAI provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "gpt-4-deployment", "gpt-35-turbo": "gpt-35-turbo-deployment"}',
        },
    )
    def test_initialization(self):
        """Test provider initialization with Azure configuration."""
        provider = AzureOpenAIModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.AZURE_OPENAI
        assert provider.azure_endpoint == "https://test-resource.openai.azure.com"
        assert provider.api_version == "2025-04-01-preview"  # Default version
        assert provider.deployment_mappings == {"gpt-4": "gpt-4-deployment", "gpt-35-turbo": "gpt-35-turbo-deployment"}

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com/",  # With trailing slash
            "AZURE_OPENAI_API_VERSION": "2024-06-01",
        },
    )
    def test_initialization_custom_version_and_trailing_slash(self):
        """Test provider initialization with custom API version and trailing slash handling."""
        provider = AzureOpenAIModelProvider("test-key")
        assert provider.azure_endpoint == "https://test-resource.openai.azure.com"  # Should strip trailing slash
        assert provider.api_version == "2024-06-01"

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"})
    def test_initialization_missing_endpoint_raises_error(self):
        """Test that missing endpoint raises an error."""
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT environment variable is required"):
            AzureOpenAIModelProvider("test-key")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": "invalid-json",
        },
    )
    def test_invalid_deployment_mappings(self):
        """Test handling of invalid JSON in deployment mappings."""
        provider = AzureOpenAIModelProvider("test-key")
        # Should fall back to empty mappings on invalid JSON
        assert provider.deployment_mappings == {}

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "my-gpt4-deployment"}',
        },
    )
    def test_model_validation(self):
        """Test model name validation."""
        provider = AzureOpenAIModelProvider("test-key")

        # Test valid models in our supported list
        assert provider.validate_model_name("gpt-4") is True
        assert provider.validate_model_name("gpt-35-turbo") is True
        assert provider.validate_model_name("gpt-4o") is True
        assert provider.validate_model_name("gpt-4o-mini") is True

        # Test valid aliases
        assert provider.validate_model_name("gpt4") is True
        assert provider.validate_model_name("gpt4o") is True
        assert provider.validate_model_name("gpt35-turbo") is True

        # Test models not in our list but with deployment mapping should still validate
        # (Azure allows custom models through deployments)
        # In this case gpt-4 has a deployment mapping so it validates

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "my-gpt4-deployment", "custom-model": "custom-deployment"}',
        },
    )
    def test_deployment_name_mapping(self):
        """Test deployment name mapping functionality."""
        provider = AzureOpenAIModelProvider("test-key")

        # Test mapped models
        assert provider._get_deployment_name("gpt-4") == "my-gpt4-deployment"
        assert provider._get_deployment_name("custom-model") == "custom-deployment"

        # Test case-insensitive mapping
        assert provider._get_deployment_name("GPT-4") == "my-gpt4-deployment"
        assert provider._get_deployment_name("Custom-Model") == "custom-deployment"

        # Test unmapped model (should fall back to model name with warning)
        assert provider._get_deployment_name("unmapped-model") == "unmapped-model"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "my-gpt4-deployment"}',
        },
    )
    def test_resolve_model_name_with_aliases(self):
        """Test model name resolution with aliases."""
        provider = AzureOpenAIModelProvider("test-key")

        # Test alias resolution
        assert provider._resolve_model_name("gpt4") == "gpt-4"
        assert provider._resolve_model_name("gpt4o") == "gpt-4o"
        assert provider._resolve_model_name("gpt35-turbo") == "gpt-35-turbo"

        # Test deployment name resolution with aliases
        assert provider._get_deployment_name("gpt4") == "my-gpt4-deployment"  # gpt4 -> gpt-4 -> deployment

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    def test_get_capabilities_known_model(self):
        """Test getting model capabilities for known Azure OpenAI models."""
        provider = AzureOpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-4")
        assert capabilities.model_name == "gpt-4"
        assert capabilities.friendly_name == "Azure OpenAI (GPT-4)"
        assert capabilities.context_window == 128_000
        assert capabilities.provider == ProviderType.AZURE_OPENAI
        assert capabilities.supports_extended_thinking is False
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_images is True

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    def test_get_capabilities_unknown_model(self):
        """Test getting model capabilities for unknown Azure OpenAI models."""
        provider = AzureOpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("unknown-model")
        assert capabilities.model_name == "unknown-model"
        assert capabilities.friendly_name == "Azure OpenAI (unknown-model)"
        assert capabilities.context_window == 128_000  # Default
        assert capabilities.provider == ProviderType.AZURE_OPENAI
        assert capabilities.supports_images is False  # Conservative default

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "my-gpt4-deployment"}',
        },
    )
    @patch("providers.openai_compatible.AzureOpenAI")
    def test_generate_content_uses_deployment_name(self, mock_azure_openai_class):
        """Test that generate_content uses deployment name in API calls."""
        # Set up mock Azure OpenAI client
        mock_client = MagicMock()
        mock_azure_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"  # API returns the deployment name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = AzureOpenAIModelProvider("test-key")

        # Call generate_content with model name
        result = provider.generate_content(
            prompt="Test prompt",
            model_name="gpt-4",
            temperature=0.7,
        )

        # Verify the API was called with the DEPLOYMENT name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive "my-gpt4-deployment", not "gpt-4"
        assert (
            call_kwargs["model"] == "my-gpt4-deployment"
        ), f"Expected 'my-gpt4-deployment' but API received '{call_kwargs['model']}'"

        # Verify response contains original model name for user
        assert result.model_name == "gpt-4"
        assert result.friendly_name == "Azure OpenAI"
        assert result.content == "Test response"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENTS": '{"gpt-4": "my-gpt4-deployment"}',
        },
    )
    @patch("providers.openai_compatible.AzureOpenAI")
    def test_generate_content_with_alias_resolution(self, mock_azure_openai_class):
        """Test that generate_content resolves aliases and uses correct deployment."""
        # Set up mock Azure OpenAI client
        mock_client = MagicMock()
        mock_azure_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = AzureOpenAIModelProvider("test-key")

        # Call generate_content with alias 'gpt4' (should resolve to 'gpt-4' then to deployment)
        result = provider.generate_content(
            prompt="Test prompt",
            model_name="gpt4",  # Alias for gpt-4
            temperature=0.7,
        )

        # Verify the API was called with the deployment name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # The API should receive the deployment name for gpt-4
        assert call_kwargs["model"] == "my-gpt4-deployment"

        # Response should show the original alias provided by user
        assert result.model_name == "gpt4"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    def test_supports_thinking_mode(self):
        """Test that Azure OpenAI models don't support thinking mode."""
        provider = AzureOpenAIModelProvider("test-key")

        # Azure OpenAI models currently don't support extended thinking
        assert provider.supports_thinking_mode("gpt-4") is False
        assert provider.supports_thinking_mode("gpt-35-turbo") is False
        assert provider.supports_thinking_mode("gpt-4o") is False

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    @patch("tiktoken.encoding_for_model")
    def test_count_tokens_with_tiktoken(self, mock_encoding_for_model):
        """Test token counting with tiktoken."""
        provider = AzureOpenAIModelProvider("test-key")

        # Mock tiktoken encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding_for_model.return_value = mock_encoding

        tokens = provider.count_tokens("test text", "gpt-4")
        assert tokens == 5
        mock_encoding_for_model.assert_called_once_with("gpt-4")
        mock_encoding.encode.assert_called_once_with("test text")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    def test_count_tokens_fallback_without_tiktoken(self):
        """Test token counting fallback when tiktoken is not available."""
        provider = AzureOpenAIModelProvider("test-key")

        # Test with a string that's 12 characters (should be ~3 tokens)
        tokens = provider.count_tokens("test string!", "gpt-4")
        assert tokens == 3  # 12 characters / 4 = 3 tokens

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    )
    def test_azure_openai_client_initialization(self):
        """Test that Azure OpenAI client is initialized with correct parameters."""
        provider = AzureOpenAIModelProvider("test-key")

        # Check that the client property creates an AzureOpenAI client (not OpenAI client)
        with patch("providers.azure_openai.AzureOpenAI") as mock_azure_openai:
            mock_client = MagicMock()
            mock_azure_openai.return_value = mock_client

            # Access client property to trigger initialization
            client = provider.client

            # Verify AzureOpenAI was called with correct parameters
            mock_azure_openai.assert_called_once()
            call_kwargs = mock_azure_openai.call_args[1]

            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["azure_endpoint"] == "https://test-resource.openai.azure.com"
            assert call_kwargs["api_version"] == "2025-04-01-preview"
            assert "http_client" in call_kwargs

            assert client == mock_client
