"""
Comprehensive Error Handling for Streamlit AI Analytics Chatbot

This module provides centralized error handling for all components of the application,
including large file processing, LLM API failures, CSV format issues, network
connectivity problems, and user-friendly error message generation.
"""

import logging
import traceback
import time
import os
import json
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import requests
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"          # Minor issues that don't prevent operation
    MEDIUM = "medium"    # Significant issues that may affect functionality
    HIGH = "high"        # Major issues that prevent core functionality
    CRITICAL = "critical"  # System-breaking issues


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILE_PROCESSING = "file_processing"
    LLM_API = "llm_api"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"
    WORKFLOW = "workflow"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    user_input: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ErrorRecord:
    """Detailed error record for logging and analysis."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    technical_details: str
    context: ErrorContext
    exception: Optional[Exception] = None
    recovery_suggestions: List[str] = None
    retry_possible: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class ApplicationError(Exception):
    """Base application error with enhanced context."""

    def __init__(self, message: str, category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: ErrorContext = None,
                 recovery_suggestions: List[str] = None,
                 retry_possible: bool = False):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context
        self.recovery_suggestions = recovery_suggestions or []
        self.retry_possible = retry_possible


class FileProcessingError(ApplicationError):
    """Errors related to file processing operations."""

    def __init__(self, message: str, file_path: str = None,
                 file_size: int = None, **kwargs):
        super().__init__(message, ErrorCategory.FILE_PROCESSING, **kwargs)
        self.file_path = file_path
        self.file_size = file_size


class LLMAPIError(ApplicationError):
    """Errors related to LLM API operations."""

    def __init__(self, message: str, provider: str = None,
                 api_response: Dict = None, rate_limited: bool = False, **kwargs):
        super().__init__(message, ErrorCategory.LLM_API, **kwargs)
        self.provider = provider
        self.api_response = api_response
        self.rate_limited = rate_limited


class NetworkError(ApplicationError):
    """Errors related to network connectivity."""

    def __init__(self, message: str, url: str = None,
                 status_code: int = None, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, **kwargs)
        self.url = url
        self.status_code = status_code


class ValidationError(ApplicationError):
    """Errors related to data validation."""

    def __init__(self, message: str, field: str = None,
                 value: Any = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field = field
        self.value = value


class ErrorHandler:
    """
    Centralized error handling system for the application.

    Provides comprehensive error handling including:
    - Error classification and severity assessment
    - User-friendly message generation
    - Recovery suggestion systems
    - Retry mechanism management
    - Error logging and analytics
    """

    def __init__(self, enable_analytics: bool = True,
                 log_file: str = "logs/error_log.json"):
        """
        Initialize the error handler.

        Args:
            enable_analytics: Whether to collect error analytics
            log_file: Path to error log file
        """
        self.enable_analytics = enable_analytics
        self.log_file = Path(log_file)
        self.error_history: List[ErrorRecord] = []

        # Create log directory if it doesn't exist
        self.log_file.parent.mkdir(exist_ok=True)

        # Retry configuration
        self.retry_config = {
            ErrorCategory.NETWORK: {"max_attempts": 3, "backoff": 2.0},
            ErrorCategory.LLM_API: {"max_attempts": 2, "backoff": 5.0},
            ErrorCategory.FILE_PROCESSING: {"max_attempts": 2, "backoff": 1.0},
        }

    def handle_error(self, error: Exception, context: ErrorContext = None) -> ErrorRecord:
        """
        Handle an error with comprehensive processing.

        Args:
            error: The exception that occurred
            context: Additional context about the error

        Returns:
            ErrorRecord with processed error information
        """
        # Generate unique error ID
        error_id = f"ERR_{int(time.time())}{hash(str(error)) % 10000:04d}"

        # Classify the error
        category, severity = self._classify_error(error)

        # Generate user-friendly message
        user_message = self._generate_user_message(error, category, severity)

        # Generate technical details
        technical_details = self._generate_technical_details(error)

        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(error, category)

        # Determine if retry is possible
        retry_possible = self._is_retry_possible(error, category)

        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            user_message=user_message,
            technical_details=technical_details,
            context=context or ErrorContext("unknown", "unknown"),
            exception=error,
            recovery_suggestions=recovery_suggestions,
            retry_possible=retry_possible
        )

        # Log the error
        self._log_error(error_record)

        # Store in history
        self.error_history.append(error_record)

        # Log to application logger
        log_level = self._get_log_level(severity)
        logger.log(log_level, f"[{error_id}] {user_message}")

        return error_record

    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity."""

        # Import error classes to avoid circular imports
        try:
            from utils.llm_config import LLMConfigurationError, LLMConnectionError
            from agents.analyser import AnalyserAgentError
            from agents.reporter import ReporterAgentError
            from workflows.analysis_workflow import WorkflowError
            from components.file_uploader import FileValidationError
        except ImportError:
            # Define fallback classes if imports fail
            class LLMConfigurationError(Exception): pass
            class LLMConnectionError(Exception): pass
            class AnalyserAgentError(Exception): pass
            class ReporterAgentError(Exception): pass
            class WorkflowError(Exception): pass
            class FileValidationError(Exception): pass

        # Classify based on error type and message
        error_type = type(error).__name__
        error_message = str(error).lower()

        # File processing errors
        if (isinstance(error, (FileNotFoundError, PermissionError, FileValidationError)) or
            error_type in ['FileNotFoundError', 'PermissionError', 'FileValidationError'] or
            any(term in error_message for term in ['file', 'csv', 'upload', 'size', 'format'])):
            severity = ErrorSeverity.HIGH if isinstance(error, FileNotFoundError) else ErrorSeverity.MEDIUM
            return ErrorCategory.FILE_PROCESSING, severity

        # LLM API errors
        if (isinstance(error, (LLMConfigurationError, LLMConnectionError, AnalyserAgentError, ReporterAgentError)) or
            any(term in error_message for term in ['api', 'llm', 'openai', 'ollama', 'rate limit', 'quota'])):
            severity = ErrorSeverity.HIGH if 'configuration' in error_message else ErrorSeverity.MEDIUM
            return ErrorCategory.LLM_API, severity

        # Network errors
        if (isinstance(error, (requests.RequestException, ConnectionError)) or
            any(term in error_message for term in ['network', 'connection', 'timeout', 'unreachable'])):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM

        # Validation errors
        if (isinstance(error, (ValueError, TypeError)) or
            any(term in error_message for term in ['invalid', 'validation', 'format', 'schema'])):
            return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM

        # Workflow errors
        if isinstance(error, WorkflowError) or 'workflow' in error_message:
            return ErrorCategory.WORKFLOW, ErrorSeverity.HIGH

        # Memory and system errors
        if isinstance(error, MemoryError) or 'memory' in error_message:
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL

        # Default classification
        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM

    def _generate_user_message(self, error: Exception, category: ErrorCategory,
                              severity: ErrorSeverity) -> str:
        """Generate user-friendly error message."""

        error_message = str(error).lower()

        # File processing messages
        if category == ErrorCategory.FILE_PROCESSING:
            if 'not found' in error_message or 'does not exist' in error_message:
                return "The selected file could not be found. Please check the file path and try again."
            elif 'permission' in error_message:
                return "Permission denied accessing the file. Please check file permissions."
            elif 'size' in error_message or 'large' in error_message:
                return "The file is too large to process. Please try with a smaller file or check available memory."
            elif 'format' in error_message or 'csv' in error_message:
                return "The file format is not valid. Please ensure you're uploading a properly formatted CSV file."
            else:
                return "An error occurred while processing the file. Please check the file and try again."

        # LLM API messages
        elif category == ErrorCategory.LLM_API:
            if 'api key' in error_message or 'authentication' in error_message:
                return "LLM service authentication failed. Please check your API key configuration."
            elif 'rate limit' in error_message or 'quota' in error_message:
                return "API rate limit exceeded. Please wait a moment and try again."
            elif 'connection' in error_message or 'network' in error_message:
                return "Cannot connect to the LLM service. Please check your internet connection."
            elif 'model' in error_message:
                return "The requested LLM model is not available. Please check your configuration."
            else:
                return "An error occurred with the AI service. Please try again or check your configuration."

        # Network messages
        elif category == ErrorCategory.NETWORK:
            if 'timeout' in error_message:
                return "Connection timed out. Please check your internet connection and try again."
            elif 'unreachable' in error_message:
                return "Service is unreachable. Please check your network connection."
            else:
                return "Network connectivity issue. Please check your internet connection."

        # Validation messages
        elif category == ErrorCategory.VALIDATION:
            return "Invalid data format detected. Please check your input and try again."

        # Workflow messages
        elif category == ErrorCategory.WORKFLOW:
            return "An error occurred during data processing. The operation has been stopped."

        # System messages
        elif category == ErrorCategory.SYSTEM:
            if severity == ErrorSeverity.CRITICAL:
                return "A critical system error occurred. Please restart the application."
            else:
                return "A system error occurred. Please try again."

        # Default message
        return "An unexpected error occurred. Please try again or contact support if the problem persists."

    def _generate_technical_details(self, error: Exception) -> str:
        """Generate technical error details for debugging."""
        return f"""
Error Type: {type(error).__name__}
Error Message: {str(error)}
Stack Trace:
{traceback.format_exc()}
"""

    def _generate_recovery_suggestions(self, error: Exception,
                                     category: ErrorCategory) -> List[str]:
        """Generate recovery suggestions based on error type."""

        suggestions = []
        error_message = str(error).lower()

        if category == ErrorCategory.FILE_PROCESSING:
            suggestions.extend([
                "Verify the file exists and is accessible",
                "Check file permissions and ownership",
                "Ensure the file is a valid CSV format",
                "Try with a smaller file if memory issues occur"
            ])

            if 'size' in error_message:
                suggestions.extend([
                    "Reduce the file size by sampling the data",
                    "Increase available memory/RAM",
                    "Use chunked processing if available"
                ])

        elif category == ErrorCategory.LLM_API:
            suggestions.extend([
                "Check your API key configuration",
                "Verify your internet connection",
                "Try switching to a different LLM provider",
                "Check API service status"
            ])

            if 'rate limit' in error_message:
                suggestions.extend([
                    "Wait before retrying the request",
                    "Reduce request frequency",
                    "Check your API quota limits"
                ])

        elif category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check your internet connection",
                "Try again in a few moments",
                "Verify firewall settings",
                "Check if the service is running"
            ])

        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Verify input data format",
                "Check for missing required fields",
                "Ensure data types are correct"
            ])

        elif category == ErrorCategory.WORKFLOW:
            suggestions.extend([
                "Retry the operation",
                "Check input data quality",
                "Verify system resources are available"
            ])

        elif category == ErrorCategory.SYSTEM:
            suggestions.extend([
                "Restart the application",
                "Check system resources (memory, disk space)",
                "Update application dependencies"
            ])

        return suggestions

    def _is_retry_possible(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if the error is retryable."""

        error_message = str(error).lower()

        # Non-retryable conditions
        non_retryable = [
            'not found', 'permission denied', 'invalid format',
            'authentication failed', 'api key', 'quota exceeded'
        ]

        if any(term in error_message for term in non_retryable):
            return False

        # Retryable categories
        retryable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.LLM_API,
            ErrorCategory.SYSTEM
        ]

        return category in retryable_categories

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get logging level based on error severity."""
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map.get(severity, logging.ERROR)

    def _log_error(self, error_record: ErrorRecord):
        """Log error to file for analytics."""
        if not self.enable_analytics:
            return

        try:
            log_entry = {
                "error_id": error_record.error_id,
                "timestamp": error_record.timestamp.isoformat(),
                "category": error_record.category.value,
                "severity": error_record.severity.value,
                "message": error_record.message,
                "user_message": error_record.user_message,
                "context": {
                    "operation": error_record.context.operation,
                    "component": error_record.context.component,
                    "request_id": error_record.context.request_id
                } if error_record.context else None,
                "retry_possible": error_record.retry_possible
            }

            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.warning(f"Failed to log error: {e}")

    def retry_with_backoff(self, func: Callable, category: ErrorCategory,
                          context: ErrorContext = None, *args, **kwargs):
        """
        Execute function with retry logic and exponential backoff.

        Args:
            func: Function to execute
            category: Error category for retry configuration
            context: Error context
            *args, **kwargs: Arguments for the function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        config = self.retry_config.get(category, {"max_attempts": 1, "backoff": 1.0})
        max_attempts = config["max_attempts"]
        backoff_factor = config["backoff"]

        last_error = None

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_record = self.handle_error(e, context)

                if not error_record.retry_possible or attempt == max_attempts - 1:
                    break

                wait_time = backoff_factor ** attempt
                logger.info(f"Retrying {func.__name__} in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait_time)

        raise last_error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}

        # Calculate statistics
        total_errors = len(self.error_history)

        # Group by category
        by_category = {}
        for record in self.error_history:
            category = record.category.value
            by_category[category] = by_category.get(category, 0) + 1

        # Group by severity
        by_severity = {}
        for record in self.error_history:
            severity = record.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Recent errors (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_errors = sum(1 for r in self.error_history if r.timestamp > recent_cutoff)

        return {
            "total_errors": total_errors,
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors_1h": recent_errors,
            "retry_capable_errors": sum(1 for r in self.error_history if r.retry_possible)
        }

    def clear_error_history(self):
        """Clear error history (useful for testing)."""
        self.error_history.clear()

    def handle_file_processing_error(self, error: Exception, file_path: str,
                                   file_size: int = None) -> ErrorRecord:
        """Handle file processing specific errors."""
        context = ErrorContext(
            operation="file_processing",
            component="csv_processor",
            user_input={"file_path": file_path, "file_size": file_size}
        )

        # Enhance error with file-specific information
        if isinstance(error, FileNotFoundError):
            enhanced_error = FileProcessingError(
                f"File not found: {file_path}",
                file_path=file_path,
                severity=ErrorSeverity.HIGH,
                context=context,
                recovery_suggestions=[
                    "Check if the file path is correct",
                    "Verify the file exists",
                    "Ensure proper file permissions"
                ]
            )
        elif isinstance(error, MemoryError):
            enhanced_error = FileProcessingError(
                f"Insufficient memory to process file: {file_path}",
                file_path=file_path,
                file_size=file_size,
                severity=ErrorSeverity.CRITICAL,
                context=context,
                recovery_suggestions=[
                    "Try with a smaller file",
                    "Increase available system memory",
                    "Use chunked processing",
                    "Close other applications to free memory"
                ]
            )
        else:
            enhanced_error = error

        return self.handle_error(enhanced_error, context)

    def handle_llm_api_error(self, error: Exception, provider: str,
                           operation: str) -> ErrorRecord:
        """Handle LLM API specific errors."""
        context = ErrorContext(
            operation=operation,
            component="llm_api",
            user_input={"provider": provider}
        )

        return self.handle_error(error, context)

    def handle_network_error(self, error: Exception, url: str) -> ErrorRecord:
        """Handle network connectivity errors."""
        context = ErrorContext(
            operation="network_request",
            component="network",
            user_input={"url": url}
        )

        return self.handle_error(error, context)


# Global error handler instance
_global_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def handle_error(error: Exception, context: ErrorContext = None) -> ErrorRecord:
    """Convenience function to handle an error using the global handler."""
    return get_error_handler().handle_error(error, context)

# Decorator for automatic error handling
def handle_errors(category: ErrorCategory = ErrorCategory.SYSTEM,
                 context_operation: str = None):
    """
    Decorator for automatic error handling.

    Args:
        category: Error category for classification
        context_operation: Operation name for context
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=context_operation or func.__name__,
                    component=func.__module__
                )
                error_record = handle_error(e, context)

                # Re-raise the error with enhanced information
                raise ApplicationError(
                    error_record.user_message,
                    category=error_record.category,
                    severity=error_record.severity,
                    context=context,
                    recovery_suggestions=error_record.recovery_suggestions,
                    retry_possible=error_record.retry_possible
                ) from e

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    def test_error_handler():
        """Test the error handler with various error types."""
        print("=== Error Handler Test ===")

        handler = ErrorHandler(enable_analytics=False)

        # Test file processing error
        try:
            raise FileNotFoundError("test.csv not found")
        except Exception as e:
            record = handler.handle_file_processing_error(e, "test.csv")
            print(f"File Error: {record.user_message}")
            print(f"Category: {record.category.value}, Severity: {record.severity.value}")
            print(f"Retry possible: {record.retry_possible}")
            print()

        # Test LLM API error
        try:
            raise ConnectionError("Failed to connect to OpenAI API")
        except Exception as e:
            record = handler.handle_llm_api_error(e, "openai", "text_generation")
            print(f"LLM Error: {record.user_message}")
            print(f"Recovery suggestions: {record.recovery_suggestions[:2]}")
            print()

        # Test network error
        try:
            raise requests.RequestException("Connection timeout")
        except Exception as e:
            record = handler.handle_network_error(e, "https://api.openai.com")
            print(f"Network Error: {record.user_message}")
            print()

        # Test retry mechanism
        attempt_count = 0
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "Success!"

        try:
            result = handler.retry_with_backoff(
                failing_function,
                ErrorCategory.NETWORK,
                ErrorContext("test_retry", "test")
            )
            print(f"Retry test result: {result} (attempts: {attempt_count})")
        except Exception as e:
            print(f"Retry test failed: {e}")

        # Show statistics
        stats = handler.get_error_statistics()
        print(f"\nError Statistics: {stats}")

        print("\n✅ Error Handler test completed!")

    test_error_handler()