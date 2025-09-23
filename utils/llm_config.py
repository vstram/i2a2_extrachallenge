"""
LLM Configuration Manager

This module provides a unified interface for configuring and managing different
LLM providers (OpenAI API and Ollama local) with streaming support, error handling,
and fallback mechanisms for the AI analytics chatbot.
"""

import os
import sys
from typing import Dict, Any, Optional, Iterator, List
from enum import Enum
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Error handling
import requests
from requests.exceptions import ConnectionError, Timeout
import openai
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMConfiguration:
    """Configuration settings for LLM providers."""
    provider: LLMProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, callback_function=None):
        """
        Initialize the streaming callback handler.

        Args:
            callback_function: Function to call with each streamed token
        """
        self.callback_function = callback_function
        self.tokens = []

    def on_llm_new_token(self, token: str, **_kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        if self.callback_function:
            self.callback_function(token)

    def on_llm_end(self, _response, **_kwargs) -> None:
        """Handle end of LLM response."""
        if self.callback_function:
            self.callback_function(None, is_end=True)

    def get_full_response(self) -> str:
        """Get the complete response."""
        return "".join(self.tokens)


class LLMConfigurationError(Exception):
    """Custom exception for LLM configuration errors."""
    pass


class LLMConnectionError(Exception):
    """Custom exception for LLM connection errors."""
    pass


class LLMManager:
    """
    Manages LLM configurations and provides unified interface for different providers.

    Supports OpenAI API and Ollama with streaming, error handling, and fallback mechanisms.
    """

    def __init__(self):
        """Initialize the LLM manager."""
        self.configurations: Dict[LLMProvider, LLMConfiguration] = {}
        self.active_provider: Optional[LLMProvider] = None
        self.llm_instances: Dict[LLMProvider, BaseChatModel] = {}
        self._setup_default_configurations()

    def _setup_default_configurations(self):
        """Setup default configurations for supported providers."""
        # OpenAI configuration
        openai_config = LLMConfiguration(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30,
            max_retries=3
        )
        self.configurations[LLMProvider.OPENAI] = openai_config

        # Ollama configuration
        ollama_config = LLMConfiguration(
            provider=LLMProvider.OLLAMA,
            model_name="llama2",
            temperature=0.7,
            max_tokens=2000,
            streaming=True,
            base_url="http://localhost:11434",
            timeout=60,
            max_retries=2
        )
        self.configurations[LLMProvider.OLLAMA] = ollama_config

    def configure_provider(self, provider: LLMProvider, **kwargs) -> None:
        """
        Configure a specific LLM provider.

        Args:
            provider: The LLM provider to configure
            **kwargs: Configuration parameters
        """
        if provider not in self.configurations:
            raise LLMConfigurationError(f"Unsupported provider: {provider}")

        config = self.configurations[provider]

        # Update configuration with provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Clear cached instance to force recreation
        if provider in self.llm_instances:
            del self.llm_instances[provider]

        logger.info(f"Updated configuration for {provider.value}")

    def validate_openai_configuration(self, config: LLMConfiguration) -> bool:
        """
        Validate OpenAI configuration.

        Args:
            config: OpenAI configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        if not config.api_key:
            logger.error("OpenAI API key is required")
            return False

        try:
            # Test API key validity
            client = OpenAI(api_key=config.api_key)
            client.models.list()
            logger.info("OpenAI API key validated successfully")
            return True
        except openai.AuthenticationError:
            logger.error("Invalid OpenAI API key")
            return False
        except Exception as e:
            logger.error(f"Error validating OpenAI configuration: {e}")
            return False

    def validate_ollama_configuration(self, config: LLMConfiguration) -> bool:
        """
        Validate Ollama configuration.

        Args:
            config: Ollama configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Test Ollama server connectivity
            response = requests.get(f"{config.base_url}/api/tags", timeout=config.timeout)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                # Get both full names (with tags) and base names
                full_model_names = [model.get("name", "") for model in available_models]
                base_model_names = [model.get("name", "").split(":")[0] for model in available_models]

                # Check if the model name matches either full name or base name
                if config.model_name in full_model_names or config.model_name in base_model_names:
                    logger.info(f"Ollama model '{config.model_name}' is available")
                    return True
                else:
                    logger.warning(f"Ollama model '{config.model_name}' not found. Available models: {full_model_names}")
                    return False
            else:
                logger.error(f"Ollama server returned status code: {response.status_code}")
                return False
        except ConnectionError:
            logger.error("Cannot connect to Ollama server. Make sure Ollama is running.")
            return False
        except Timeout:
            logger.error("Timeout connecting to Ollama server")
            return False
        except Exception as e:
            logger.error(f"Error validating Ollama configuration: {e}")
            return False

    def _create_openai_instance(self, config: LLMConfiguration) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        if not self.validate_openai_configuration(config):
            raise LLMConfigurationError("Invalid OpenAI configuration")

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=config.streaming,
            api_key=config.api_key,
            request_timeout=config.timeout,
            max_retries=config.max_retries
        )

    def _create_ollama_instance(self, config: LLMConfiguration) -> ChatOllama:
        """Create Ollama LLM instance."""
        if not self.validate_ollama_configuration(config):
            raise LLMConfigurationError("Invalid Ollama configuration")

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=config.base_url,
            timeout=config.timeout
        )

    def get_llm_instance(self, provider: LLMProvider) -> BaseChatModel:
        """
        Get or create LLM instance for the specified provider.

        Args:
            provider: The LLM provider

        Returns:
            LLM instance

        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        if provider not in self.configurations:
            raise LLMConfigurationError(f"Provider {provider} not configured")

        # Return cached instance if available
        if provider in self.llm_instances:
            return self.llm_instances[provider]

        config = self.configurations[provider]

        try:
            if provider == LLMProvider.OPENAI:
                instance = self._create_openai_instance(config)
            elif provider == LLMProvider.OLLAMA:
                instance = self._create_ollama_instance(config)
            else:
                raise LLMConfigurationError(f"Unsupported provider: {provider}")

            # Cache the instance
            self.llm_instances[provider] = instance
            logger.info(f"Created LLM instance for {provider.value}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create {provider.value} instance: {e}")
            raise LLMConfigurationError(f"Failed to create {provider.value} instance: {e}")

    def set_active_provider(self, provider: LLMProvider) -> None:
        """
        Set the active LLM provider.

        Args:
            provider: Provider to set as active

        Raises:
            LLMConfigurationError: If provider is not available
        """
        try:
            # Test if provider works by getting instance
            self.get_llm_instance(provider)
            self.active_provider = provider
            logger.info(f"Set active provider to {provider.value}")
        except Exception as e:
            raise LLMConfigurationError(f"Cannot set {provider.value} as active provider: {e}")

    def get_active_llm(self) -> BaseChatModel:
        """
        Get the active LLM instance.

        Returns:
            Active LLM instance

        Raises:
            LLMConfigurationError: If no active provider is set
        """
        if not self.active_provider:
            raise LLMConfigurationError("No active provider set")

        return self.get_llm_instance(self.active_provider)

    def invoke_llm(self, messages: List[BaseMessage], provider: Optional[LLMProvider] = None) -> str:
        """
        Invoke LLM with messages.

        Args:
            messages: List of messages to send to LLM
            provider: Optional specific provider to use

        Returns:
            LLM response text

        Raises:
            LLMConnectionError: If LLM invocation fails
        """
        target_provider = provider or self.active_provider
        if not target_provider:
            raise LLMConfigurationError("No provider specified and no active provider set")

        try:
            llm = self.get_llm_instance(target_provider)
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error invoking {target_provider.value}: {e}")
            raise LLMConnectionError(f"Failed to invoke {target_provider.value}: {e}")

    def stream_llm(self, messages: List[BaseMessage],
                   callback_function=None,
                   provider: Optional[LLMProvider] = None) -> Iterator[str]:
        """
        Stream LLM response.

        Args:
            messages: List of messages to send to LLM
            callback_function: Optional callback for each token
            provider: Optional specific provider to use

        Yields:
            Streamed tokens

        Raises:
            LLMConnectionError: If streaming fails
        """
        target_provider = provider or self.active_provider
        if not target_provider:
            raise LLMConfigurationError("No provider specified and no active provider set")

        try:
            llm = self.get_llm_instance(target_provider)

            # Create streaming callback handler
            callback_handler = StreamingCallbackHandler(callback_function)

            # Stream the response
            for chunk in llm.stream(messages, config={"callbacks": [callback_handler]}):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Error streaming from {target_provider.value}: {e}")
            raise LLMConnectionError(f"Failed to stream from {target_provider.value}: {e}")

    def try_providers_with_fallback(self, messages: List[BaseMessage],
                                   preferred_order: Optional[List[LLMProvider]] = None) -> str:
        """
        Try multiple providers with fallback mechanism.

        Args:
            messages: Messages to send to LLM
            preferred_order: Optional order of providers to try

        Returns:
            LLM response from first successful provider

        Raises:
            LLMConnectionError: If all providers fail
        """
        order = preferred_order or [LLMProvider.OPENAI, LLMProvider.OLLAMA]

        last_error = None
        for provider in order:
            try:
                logger.info(f"Trying provider: {provider.value}")
                response = self.invoke_llm(messages, provider)
                logger.info(f"Success with provider: {provider.value}")
                return response
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise LLMConnectionError(f"All providers failed. Last error: {last_error}")

    def get_available_providers(self) -> List[LLMProvider]:
        """
        Get list of available (working) providers.

        Returns:
            List of available providers
        """
        available = []
        for provider in LLMProvider:
            try:
                # Test if provider configuration is valid
                config = self.configurations[provider]
                if provider == LLMProvider.OPENAI:
                    if self.validate_openai_configuration(config):
                        available.append(provider)
                elif provider == LLMProvider.OLLAMA:
                    if self.validate_ollama_configuration(config):
                        available.append(provider)
            except Exception as e:
                logger.debug(f"Provider {provider.value} not available: {e}")
                continue

        return available

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about all providers.

        Returns:
            Dictionary with provider information
        """
        info = {
            "available_providers": [p.value for p in self.get_available_providers()],
            "active_provider": self.active_provider.value if self.active_provider else None,
            "configurations": {}
        }

        for provider, config in self.configurations.items():
            info["configurations"][provider.value] = {
                "model_name": config.model_name,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "streaming": config.streaming,
                "timeout": config.timeout,
                "has_api_key": bool(config.api_key) if config.api_key else False,
                "base_url": config.base_url
            }

        return info

    def reset_configurations(self):
        """Reset all configurations to defaults."""
        self.configurations.clear()
        self.llm_instances.clear()
        self.active_provider = None
        self._setup_default_configurations()
        logger.info("Reset all configurations to defaults")


# Convenience functions
def create_llm_manager(openai_api_key: Optional[str] = None,
                      ollama_base_url: Optional[str] = None,
                      auto_detect_available: bool = True) -> LLMManager:
    """
    Create and configure LLM manager with common settings.

    Args:
        openai_api_key: OpenAI API key
        ollama_base_url: Ollama server URL
        auto_detect_available: Whether to auto-detect and set active provider

    Returns:
        Configured LLM manager
    """
    manager = LLMManager()

    # Configure OpenAI if API key provided
    if openai_api_key:
        manager.configure_provider(LLMProvider.OPENAI, api_key=openai_api_key)

    # Configure Ollama if URL provided
    if ollama_base_url:
        manager.configure_provider(LLMProvider.OLLAMA, base_url=ollama_base_url)

    # Auto-detect and set active provider
    if auto_detect_available:
        available = manager.get_available_providers()
        if available:
            # Prefer OpenAI if available, fallback to Ollama
            preferred = LLMProvider.OPENAI if LLMProvider.OPENAI in available else available[0]
            try:
                manager.set_active_provider(preferred)
                logger.info(f"Auto-detected and set active provider: {preferred.value}")
            except Exception as e:
                logger.warning(f"Could not set auto-detected provider: {e}")

    return manager


def quick_llm_response(prompt: str,
                      system_message: Optional[str] = None,
                      openai_api_key: Optional[str] = None) -> str:
    """
    Quick utility function for getting LLM response.

    Args:
        prompt: User prompt
        system_message: Optional system message
        openai_api_key: Optional OpenAI API key

    Returns:
        LLM response

    Raises:
        LLMConnectionError: If no providers are available
    """
    manager = create_llm_manager(openai_api_key=openai_api_key)

    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))

    return manager.try_providers_with_fallback(messages)


if __name__ == "__main__":
    # Example usage and testing
    import sys

    def test_llm_manager():
        """Test the LLM manager functionality."""
        print("=== LLM Manager Test ===")

        # Create manager
        manager = create_llm_manager()

        # Show provider info
        info = manager.get_provider_info()
        print(f"Available providers: {info['available_providers']}")
        print(f"Active provider: {info['active_provider']}")

        # Test simple response if any provider is available
        if info['available_providers']:
            try:
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Say hello and describe yourself in one sentence.")
                ]

                response = manager.try_providers_with_fallback(messages)
                print(f"LLM Response: {response}")

            except Exception as e:
                print(f"Error testing LLM: {e}")
        else:
            print("No providers available for testing")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_llm_manager()
    else:
        print("LLM Configuration Manager")
        print("Usage: python llm_config.py test")
        print("Set OPENAI_API_KEY environment variable for OpenAI support")
        print("Start Ollama server for local LLM support")