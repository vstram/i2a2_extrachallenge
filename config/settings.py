"""
Configuration Management System for Streamlit AI Analytics Chatbot

This module provides comprehensive configuration management including environment
variables, default parameters, logging configuration, and development/production
settings for the fraud detection analysis application.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 4000
    openai_timeout: int = 60

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    ollama_temperature: float = 0.1
    ollama_timeout: int = 60

    # General LLM Settings
    default_provider: str = "openai"  # or "ollama"
    streaming_enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class DataProcessingConfig:
    """Data processing configuration."""
    # CSV Processing
    chunk_size: int = 10000
    max_file_size_mb: int = 150
    max_rows_sample: int = 50000
    encoding: str = "utf-8"
    delimiter: str = ","

    # Memory Management
    memory_limit_mb: int = 1024
    enable_memory_optimization: bool = True
    garbage_collection_threshold: int = 100

    # Performance
    enable_multiprocessing: bool = False
    max_workers: int = 4
    progress_update_interval: float = 1.0


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    max_size_mb: int = 500
    max_entries: int = 100
    default_ttl_hours: int = 24
    cache_dir: str = "cache"
    compression_enabled: bool = False
    cleanup_interval_hours: int = 6


@dataclass
class UIConfig:
    """User interface configuration."""
    # Streamlit Settings
    page_title: str = "AI Analytics Chatbot"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"

    # Theme
    theme_primary_color: str = "#FF6B6B"
    theme_background_color: str = "#FFFFFF"
    theme_secondary_background_color: str = "#F0F2F6"
    theme_text_color: str = "#262730"

    # Features
    enable_file_upload: bool = True
    enable_charts: bool = True
    enable_statistics_display: bool = True
    max_chat_history: int = 50


@dataclass
class SecurityConfig:
    """Security configuration."""
    # API Security
    validate_api_keys: bool = True
    mask_sensitive_data: bool = True
    log_sensitive_data: bool = False

    # File Security
    allowed_file_types: List[str] = field(default_factory=lambda: ["csv"])
    scan_uploaded_files: bool = True
    max_upload_size_mb: int = 150

    # Session Security
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Log Levels
    app_log_level: LogLevel = LogLevel.INFO
    root_log_level: LogLevel = LogLevel.WARNING
    streamlit_log_level: LogLevel = LogLevel.WARNING
    langchain_log_level: LogLevel = LogLevel.WARNING

    # Log Files
    log_dir: str = "logs"
    app_log_file: str = "app.log"
    error_log_file: str = "error.log"
    performance_log_file: str = "performance.log"

    # Log Format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Log Rotation
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_rotation: bool = True

    # Console Logging
    enable_console_logging: bool = True
    console_log_level: LogLevel = LogLevel.INFO


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: float = 10.0
    export_metrics: bool = True
    metrics_export_interval: int = 300  # seconds

    # Error Monitoring
    enable_error_tracking: bool = True
    max_error_history: int = 1000
    error_notification_threshold: int = 10

    # Health Checks
    enable_health_checks: bool = True
    health_check_interval: int = 60
    memory_threshold_mb: int = 2048
    cpu_threshold_percent: float = 80.0


class Settings:
    """
    Comprehensive application settings with environment variable support.
    """

    def __init__(self, environment: Environment = None, config_file: str = None):
        """
        Initialize settings.

        Args:
            environment: Application environment (auto-detected if None)
            config_file: Optional configuration file path
        """
        self.environment = environment or self._detect_environment()
        self.config_file = config_file

        # Load configuration
        self._load_base_config()
        self._load_environment_variables()
        if config_file:
            self._load_config_file(config_file)
        self._apply_environment_overrides()
        self._validate_configuration()

    def _detect_environment(self) -> Environment:
        """Auto-detect environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()

        try:
            return Environment(env_name)
        except ValueError:
            warnings.warn(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT

    def _load_base_config(self):
        """Load base configuration with defaults."""
        # Core configuration sections
        self.llm = LLMConfig()
        self.data_processing = DataProcessingConfig()
        self.cache = CacheConfig()
        self.ui = UIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()

        # Apply environment-specific defaults
        self._apply_environment_defaults()

    def _apply_environment_defaults(self):
        """Apply environment-specific default configurations."""
        if self.environment == Environment.DEVELOPMENT:
            # Development settings
            self.logging.app_log_level = LogLevel.DEBUG
            self.logging.enable_console_logging = True
            self.monitoring.enable_performance_monitoring = True
            self.cache.enabled = True
            self.security.validate_api_keys = False  # More relaxed for dev

        elif self.environment == Environment.TESTING:
            # Testing settings
            self.logging.app_log_level = LogLevel.WARNING
            self.cache.enabled = False  # Fresh state for tests
            self.data_processing.chunk_size = 1000  # Smaller for faster tests
            self.monitoring.enable_performance_monitoring = False

        elif self.environment == Environment.STAGING:
            # Staging settings (production-like)
            self.logging.app_log_level = LogLevel.INFO
            self.monitoring.enable_performance_monitoring = True
            self.security.validate_api_keys = True
            self.cache.max_size_mb = 200  # Smaller cache

        elif self.environment == Environment.PRODUCTION:
            # Production settings
            self.logging.app_log_level = LogLevel.INFO
            self.logging.enable_console_logging = False
            self.monitoring.enable_performance_monitoring = True
            self.monitoring.export_metrics = True
            self.security.validate_api_keys = True
            self.security.scan_uploaded_files = True
            self.cache.cleanup_interval_hours = 2  # More frequent cleanup

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # LLM Configuration
        self.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm.openai_model = os.getenv("OPENAI_MODEL", self.llm.openai_model)
        self.llm.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", self.llm.openai_temperature))

        self.llm.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.llm.ollama_base_url)
        self.llm.ollama_model = os.getenv("OLLAMA_MODEL", self.llm.ollama_model)
        self.llm.default_provider = os.getenv("LLM_PROVIDER", self.llm.default_provider)

        # Data Processing
        self.data_processing.chunk_size = int(os.getenv("CHUNK_SIZE", self.data_processing.chunk_size))
        self.data_processing.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", self.data_processing.max_file_size_mb))
        self.data_processing.max_rows_sample = int(os.getenv("MAX_ROWS_SAMPLE", self.data_processing.max_rows_sample))

        # Cache Configuration
        self.cache.enabled = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        self.cache.max_size_mb = int(os.getenv("CACHE_MAX_SIZE_MB", self.cache.max_size_mb))
        self.cache.cache_dir = os.getenv("CACHE_DIR", self.cache.cache_dir)

        # Logging
        log_level = os.getenv("LOG_LEVEL", self.logging.app_log_level.value)
        try:
            self.logging.app_log_level = LogLevel(log_level.upper())
        except ValueError:
            warnings.warn(f"Invalid log level '{log_level}', using default")

        self.logging.log_dir = os.getenv("LOG_DIR", self.logging.log_dir)

        # UI Configuration
        self.ui.page_title = os.getenv("PAGE_TITLE", self.ui.page_title)
        self.ui.enable_charts = os.getenv("ENABLE_CHARTS", "true").lower() in ("true", "1", "yes")

        # Security
        self.security.max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", self.security.max_upload_size_mb))

        # Monitoring
        self.monitoring.enable_performance_monitoring = os.getenv("ENABLE_MONITORING", "true").lower() in ("true", "1", "yes")

    def _load_config_file(self, config_file: str):
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            warnings.warn(f"Configuration file not found: {config_file}")
            return

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Update configuration sections
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                        else:
                            warnings.warn(f"Unknown configuration key: {section_name}.{key}")
                else:
                    warnings.warn(f"Unknown configuration section: {section_name}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file {config_file}: {e}")

    def _apply_environment_overrides(self):
        """Apply environment-specific overrides."""
        # Additional environment-specific logic can be added here
        pass

    def _validate_configuration(self):
        """Validate configuration values."""
        errors = []

        # Validate LLM configuration
        if self.llm.default_provider not in ["openai", "ollama"]:
            errors.append("LLM default_provider must be 'openai' or 'ollama'")

        if self.llm.default_provider == "openai" and not self.llm.openai_api_key and self.environment == Environment.PRODUCTION:
            errors.append("OpenAI API key is required in production when using OpenAI")

        # Validate data processing
        if self.data_processing.chunk_size <= 0:
            errors.append("Chunk size must be positive")

        if self.data_processing.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")

        # Validate cache configuration
        if self.cache.max_size_mb <= 0:
            errors.append("Cache max size must be positive")

        # Validate security configuration
        if self.security.max_upload_size_mb > self.data_processing.max_file_size_mb:
            errors.append("Max upload size cannot exceed max file size")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))

    def setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        log_dir = Path(self.logging.log_dir)
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.logging.root_log_level.value))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            self.logging.log_format,
            datefmt=self.logging.date_format
        )

        # Console handler
        if self.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.logging.console_log_level.value))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handlers
        if self.logging.enable_rotation:
            from logging.handlers import RotatingFileHandler

            # App log file
            app_log_path = log_dir / self.logging.app_log_file
            app_handler = RotatingFileHandler(
                app_log_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            app_handler.setLevel(getattr(logging, self.logging.app_log_level.value))
            app_handler.setFormatter(formatter)
            root_logger.addHandler(app_handler)

            # Error log file (ERROR and above only)
            error_log_path = log_dir / self.logging.error_log_file
            error_handler = RotatingFileHandler(
                error_log_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

        # Configure specific loggers
        self._configure_specific_loggers()

        # Log startup message
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured for {self.environment.value} environment")
        logger.info(f"Log level: {self.logging.app_log_level.value}")
        logger.info(f"Log directory: {log_dir.absolute()}")

    def _configure_specific_loggers(self):
        """Configure specific library loggers."""
        # Streamlit logger
        streamlit_logger = logging.getLogger("streamlit")
        streamlit_logger.setLevel(getattr(logging, self.logging.streamlit_log_level.value))

        # LangChain logger
        langchain_logger = logging.getLogger("langchain")
        langchain_logger.setLevel(getattr(logging, self.logging.langchain_log_level.value))

        # Suppress overly verbose loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration as dictionary."""
        return {
            'enabled': self.cache.enabled,
            'max_size_mb': self.cache.max_size_mb,
            'max_entries': self.cache.max_entries,
            'default_ttl_hours': self.cache.default_ttl_hours,
            'cache_dir': self.cache.cache_dir,
            'compression_enabled': self.cache.compression_enabled
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary."""
        return {
            'openai_api_key': self.llm.openai_api_key,
            'openai_model': self.llm.openai_model,
            'openai_temperature': self.llm.openai_temperature,
            'openai_max_tokens': self.llm.openai_max_tokens,
            'ollama_base_url': self.llm.ollama_base_url,
            'ollama_model': self.llm.ollama_model,
            'default_provider': self.llm.default_provider,
            'streaming_enabled': self.llm.streaming_enabled,
            'retry_attempts': self.llm.retry_attempts
        }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration as dictionary."""
        return {
            'chunk_size': self.data_processing.chunk_size,
            'max_file_size_mb': self.data_processing.max_file_size_mb,
            'max_rows_sample': self.data_processing.max_rows_sample,
            'encoding': self.data_processing.encoding,
            'memory_limit_mb': self.data_processing.memory_limit_mb,
            'enable_memory_optimization': self.data_processing.enable_memory_optimization
        }

    def export_config(self, file_path: str = None) -> str:
        """Export current configuration to JSON file."""
        if file_path is None:
            file_path = f"config/settings_{self.environment.value}.json"

        config_data = {
            'environment': self.environment.value,
            'llm': self.get_llm_config(),
            'data_processing': self.get_processing_config(),
            'cache': self.get_cache_config(),
            'ui': {
                'page_title': self.ui.page_title,
                'enable_charts': self.ui.enable_charts,
                'enable_file_upload': self.ui.enable_file_upload,
                'max_chat_history': self.ui.max_chat_history
            },
            'security': {
                'allowed_file_types': self.security.allowed_file_types,
                'max_upload_size_mb': self.security.max_upload_size_mb,
                'validate_api_keys': self.security.validate_api_keys
            },
            'logging': {
                'app_log_level': self.logging.app_log_level.value,
                'log_dir': self.logging.log_dir,
                'enable_console_logging': self.logging.enable_console_logging
            },
            'monitoring': {
                'enable_performance_monitoring': self.monitoring.enable_performance_monitoring,
                'enable_error_tracking': self.monitoring.enable_error_tracking,
                'metrics_collection_interval': self.monitoring.metrics_collection_interval
            }
        }

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Export configuration
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        return file_path

    def __str__(self) -> str:
        """String representation of settings."""
        return f"Settings(environment={self.environment.value})"

    def __repr__(self) -> str:
        """Detailed representation of settings."""
        return (f"Settings(environment={self.environment.value}, "
                f"llm_provider={self.llm.default_provider}, "
                f"cache_enabled={self.cache.enabled}, "
                f"log_level={self.logging.app_log_level.value})")


# Global settings instance
_global_settings: Optional[Settings] = None

def get_settings(environment: Environment = None, config_file: str = None,
                reload: bool = False) -> Settings:
    """
    Get global settings instance.

    Args:
        environment: Environment to use (auto-detected if None)
        config_file: Optional configuration file
        reload: Force reload of settings

    Returns:
        Settings instance
    """
    global _global_settings

    if _global_settings is None or reload:
        _global_settings = Settings(environment=environment, config_file=config_file)
        _global_settings.setup_logging()

    return _global_settings

def init_settings(environment: Environment = None, config_file: str = None) -> Settings:
    """
    Initialize application settings.

    Args:
        environment: Environment to use
        config_file: Optional configuration file

    Returns:
        Initialized settings
    """
    settings = get_settings(environment=environment, config_file=config_file, reload=True)

    # Create necessary directories
    Path(settings.cache.cache_dir).mkdir(exist_ok=True)
    Path(settings.logging.log_dir).mkdir(exist_ok=True)

    return settings


if __name__ == "__main__":
    # Example usage and testing
    def test_settings():
        """Test the settings system."""
        print("=== Settings System Test ===")

        # Test 1: Development environment
        dev_settings = Settings(Environment.DEVELOPMENT)
        print(f"Development settings: {dev_settings}")
        print(f"Log level: {dev_settings.logging.app_log_level.value}")
        print(f"Cache enabled: {dev_settings.cache.enabled}")

        # Test 2: Production environment (with API key)
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        prod_settings = Settings(Environment.PRODUCTION)
        print(f"Production settings: {prod_settings}")
        print(f"Console logging: {prod_settings.logging.enable_console_logging}")
        del os.environ["OPENAI_API_KEY"]  # Clean up

        # Test 3: Environment variables
        os.environ["CHUNK_SIZE"] = "5000"
        os.environ["LOG_LEVEL"] = "DEBUG"
        env_settings = Settings(Environment.DEVELOPMENT)
        print(f"With env vars - Chunk size: {env_settings.data_processing.chunk_size}")
        print(f"With env vars - Log level: {env_settings.logging.app_log_level.value}")

        # Test 4: Configuration export
        export_path = env_settings.export_config("test_config.json")
        print(f"Configuration exported to: {export_path}")

        # Test 5: Global settings
        global_settings = get_settings()
        print(f"Global settings: {global_settings}")

        print("\n✅ Settings system test completed!")

    test_settings()