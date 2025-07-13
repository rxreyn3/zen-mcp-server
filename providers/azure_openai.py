"""Azure OpenAI model provider implementation."""

import json
import logging
import os
from typing import Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    create_temperature_constraint,
)
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class AzureOpenAIModelProvider(OpenAICompatibleProvider):
    """Azure OpenAI API provider with deployment-based routing."""

    FRIENDLY_NAME = "Azure OpenAI"

    # Default model configurations - users can override via AZURE_OPENAI_DEPLOYMENTS
    # These are common Azure OpenAI models based on standard deployments
    SUPPORTED_MODELS = {
        "gpt-4": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="gpt-4",
            friendly_name="Azure OpenAI (GPT-4)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="GPT-4 via Azure OpenAI (128K context) - Advanced reasoning and analysis",
            aliases=["gpt4"],
        ),
        "gpt-4-32k": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="gpt-4-32k",
            friendly_name="Azure OpenAI (GPT-4-32k)",
            context_window=32_000,  # 32K tokens
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,  # Original GPT-4 doesn't support vision
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="GPT-4-32k via Azure OpenAI (32K context) - Legacy high-context model",
            aliases=["gpt4-32k"],
        ),
        "gpt-35-turbo": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="gpt-35-turbo",
            friendly_name="Azure OpenAI (GPT-3.5-Turbo)",
            context_window=16_384,  # 16K tokens
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="GPT-3.5-Turbo via Azure OpenAI (16K context) - Fast and cost-effective",
            aliases=["gpt-3.5-turbo", "gpt35-turbo"],
        ),
        "gpt-4o": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="gpt-4o",
            friendly_name="Azure OpenAI (GPT-4o)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="GPT-4o via Azure OpenAI (128K context) - Optimized multimodal model",
            aliases=["gpt4o"],
        ),
        "gpt-4o-mini": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="gpt-4o-mini",
            friendly_name="Azure OpenAI (GPT-4o-mini)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=16_384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="GPT-4o-mini via Azure OpenAI (128K context) - Balanced performance and cost",
            aliases=["gpt4o-mini"],
        ),
        "o3-mini": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="o3-mini",
            friendly_name="Azure OpenAI (O3-mini)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # 100K tokens
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,  # O3 models don't support temperature
            temperature_constraint=None,
            description="O3-mini via Azure - Fast reasoning model with 200K context and extended thinking",
            aliases=[],
        ),
        "o3": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="o3",
            friendly_name="Azure OpenAI (O3)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # 100K tokens
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,  # O3 models don't support temperature
            temperature_constraint=None,
            description="O3 via Azure - Powerful reasoning model for coding, math, science (200K context)",
            aliases=[],
        ),
        "o3-pro": ModelCapabilities(
            provider=ProviderType.AZURE_OPENAI,
            model_name="o3-pro",
            friendly_name="Azure OpenAI (O3-pro)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # 100K tokens
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,  # O3 models don't support temperature
            temperature_constraint=None,
            description="O3-pro via Azure - Professional grade reasoning model (EXTREMELY EXPENSIVE)",
            aliases=[],
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Azure OpenAI provider with required Azure configuration.

        Args:
            api_key: Azure OpenAI API key
            **kwargs: Additional configuration options
        """
        # Parse Azure-specific configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable is required for Azure OpenAI provider. "
                "Set it to your Azure OpenAI resource endpoint (e.g., https://myresource.openai.azure.com)"
            )

        # Ensure endpoint doesn't have trailing slash
        self.azure_endpoint = self.azure_endpoint.rstrip("/")

        # Get API version
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        # Parse deployment mappings
        self.deployment_mappings = self._parse_deployment_mappings()

        # Create base URL for Azure OpenAI
        # Azure OpenAI uses a different URL structure than standard OpenAI
        base_url = f"{self.azure_endpoint}/openai"

        # Initialize parent with Azure-specific base URL
        super().__init__(api_key, base_url=base_url, **kwargs)

        logger.info(f"Azure OpenAI provider initialized with endpoint: {self.azure_endpoint}")
        logger.info(f"API version: {self.api_version}")
        logger.info(f"Deployment mappings: {self.deployment_mappings}")

    def _parse_deployment_mappings(self) -> dict[str, str]:
        """Parse deployment mappings from environment variable.

        Returns:
            Dict mapping model names to deployment names
        """
        deployments_str = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "{}")

        try:
            mappings = json.loads(deployments_str)
            if not isinstance(mappings, dict):
                raise ValueError("AZURE_OPENAI_DEPLOYMENTS must be a JSON object")

            # Convert to lowercase keys for case-insensitive matching
            normalized_mappings = {}
            for model_name, deployment_name in mappings.items():
                normalized_mappings[model_name.lower()] = deployment_name

            return normalized_mappings
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in AZURE_OPENAI_DEPLOYMENTS: {e}")
            logger.error(
                "Using empty deployment mappings. Configure AZURE_OPENAI_DEPLOYMENTS with JSON like: "
                '{"gpt-4": "gpt-4-deployment", "gpt-35-turbo": "gpt-35-turbo-deployment"}'
            )
            return {}

    def _get_deployment_name(self, model_name: str) -> str:
        """Get deployment name for a model.

        Args:
            model_name: User-provided model name

        Returns:
            Azure deployment name

        Raises:
            ValueError: If no deployment mapping found
        """
        # Resolve any aliases first
        resolved_model = self._resolve_model_name(model_name)

        # Check deployment mappings (case-insensitive)
        deployment = self.deployment_mappings.get(resolved_model.lower())
        if deployment:
            return deployment

        # Try original model name as well
        deployment = self.deployment_mappings.get(model_name.lower())
        if deployment:
            return deployment

        # Fall back to using model name as deployment name
        logger.warning(
            f"No deployment mapping found for model '{model_name}'. "
            f"Using model name as deployment name. Configure AZURE_OPENAI_DEPLOYMENTS "
            f"to map model names to deployment names."
        )
        return resolved_model

    @property
    def client(self):
        """Create Azure OpenAI client with proper authentication and endpoints."""
        if self._client is None:
            import httpx
            from openai import AzureOpenAI

            # Configure timeouts
            timeout_config = (
                self.timeout_config if hasattr(self, "timeout_config") and self.timeout_config else httpx.Timeout(30.0)
            )

            # Create httpx client
            http_client = httpx.Client(
                timeout=timeout_config,
                follow_redirects=True,
            )

            # Create Azure OpenAI client
            client_kwargs = {
                "api_key": self.api_key,
                "azure_endpoint": self.azure_endpoint,
                "api_version": self.api_version,
                "http_client": http_client,
            }

            # Add default headers if any
            if self.DEFAULT_HEADERS:
                client_kwargs["default_headers"] = self.DEFAULT_HEADERS.copy()

            logger.debug(f"Creating Azure OpenAI client with endpoint: {self.azure_endpoint}")
            self._client = AzureOpenAI(**client_kwargs)

        return self._client

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific Azure OpenAI model."""
        # Resolve shorthand
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.SUPPORTED_MODELS:
            # For Azure, we might have models not in our default list
            # Create a generic capability for unknown models
            logger.warning(f"Unknown Azure OpenAI model: {model_name}. Using generic capabilities.")
            return ModelCapabilities(
                provider=ProviderType.AZURE_OPENAI,
                model_name=resolved_name,
                friendly_name=f"Azure OpenAI ({resolved_name})",
                context_window=128_000,  # Default to reasonable context window
                max_output_tokens=4_096,
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=True,
                supports_json_mode=True,
                supports_images=False,  # Conservative default
                max_image_size_mb=0.0,
                supports_temperature=True,
                temperature_constraint=create_temperature_constraint("range"),
                description=f"Azure OpenAI model: {resolved_name}",
                aliases=[],
            )

        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.AZURE_OPENAI, resolved_name, model_name):
            raise ValueError(f"Azure OpenAI model '{model_name}' is not allowed by restriction policy.")

        return self.SUPPORTED_MODELS[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.AZURE_OPENAI

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed."""
        resolved_name = self._resolve_model_name(model_name)

        # For Azure, we're more permissive since deployments can have custom models
        # Just check if we have a deployment mapping or if it's in our supported models
        if resolved_name not in self.SUPPORTED_MODELS:
            # Check if we have a deployment mapping for this model
            deployment = self.deployment_mappings.get(resolved_name.lower())
            if not deployment:
                deployment = self.deployment_mappings.get(model_name.lower())

            if not deployment:
                logger.warning(f"No deployment mapping found for Azure OpenAI model: {model_name}")
                # Still allow it in case the user has configured the deployment correctly
                # The actual validation will happen when we try to make the API call

        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.AZURE_OPENAI, resolved_name, model_name):
            logger.debug(f"Azure OpenAI model '{model_name}' -> '{resolved_name}' blocked by restrictions")
            return False

        return True

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using Azure OpenAI API with deployment name mapping."""
        # Get deployment name for the model
        deployment_name = self._get_deployment_name(model_name)

        logger.debug(f"Mapping model '{model_name}' to deployment '{deployment_name}'")

        # Call parent implementation with deployment name instead of model name
        # Azure OpenAI uses deployment names in the API calls
        response = super().generate_content(
            prompt=prompt,
            model_name=deployment_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        # Update response to show the original model name requested by user
        response.model_name = model_name
        response.friendly_name = self.FRIENDLY_NAME

        return response

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        # Currently Azure OpenAI models don't support extended thinking
        # This may change with future model deployments
        return False

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for the given text using the specified model's tokenizer."""
        # For Azure OpenAI, we use the same tokenization as OpenAI
        # since the underlying models are the same
        try:
            import tiktoken

            # Map Azure model names to OpenAI equivalents for tokenization
            resolved_name = self._resolve_model_name(model_name)

            # Map common Azure model names to tiktoken model names
            tokenizer_mapping = {
                "gpt-4": "gpt-4",
                "gpt-4-32k": "gpt-4-32k",
                "gpt-35-turbo": "gpt-3.5-turbo",
                "gpt-4o": "gpt-4o",
                "gpt-4o-mini": "gpt-4o-mini",
            }

            tiktoken_model = tokenizer_mapping.get(resolved_name, "gpt-4")  # Default to gpt-4

            encoding = tiktoken.encoding_for_model(tiktoken_model)
            return len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, using approximate token count")
            # Fallback: approximate 4 characters per token
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Error counting tokens for {model_name}: {e}")
            # Fallback: approximate 4 characters per token
            return len(text) // 4
