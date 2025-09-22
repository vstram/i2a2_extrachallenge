"""
File Upload Interface for AI Analytics Chatbot

This module implements a comprehensive CSV file upload interface with validation,
progress tracking, automatic statistics processing, and caching functionality.
"""

import streamlit as st
import pandas as pd
import hashlib
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.csv_processor import CSVProcessor
from utils.pattern_analyzer import PatternAnalyzer
from utils.chart_generator import ChartGenerator
from utils.error_handler import (
    get_error_handler, ErrorContext, ErrorCategory,
    FileProcessingError, ValidationError, handle_errors
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class FileUploader:
    """
    Comprehensive file upload interface for CSV files with validation,
    processing, and caching capabilities.
    """

    def __init__(self, max_file_size_mb: int = 150, cache_enabled: bool = True):
        """
        Initialize the FileUploader.

        Args:
            max_file_size_mb: Maximum allowed file size in MB
            cache_enabled: Whether to enable result caching
        """
        self.max_file_size_mb = max_file_size_mb
        self.cache_enabled = cache_enabled

        # Initialize error handler
        self.error_handler = get_error_handler()

        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables for file upload."""
        defaults = {
            # File upload state
            'uploaded_file': None,
            'file_hash': None,
            'file_validated': False,
            'file_info': {},

            # Processing state
            'processing_started': False,
            'processing_complete': False,
            'processing_progress': 0,
            'processing_status': '',
            'processing_error': None,

            # Processed data
            'statistics_data': None,
            'pattern_data': None,
            'chart_data': None,
            'processing_timestamp': None,

            # Cache
            'file_cache': {},
            'cache_hits': 0,
            'cache_misses': 0
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def render(self) -> Optional[Dict[str, Any]]:
        """
        Render the file upload interface.

        Returns:
            Dictionary containing processed data if available, None otherwise
        """
        st.subheader("📁 Data Upload")

        # File upload widget
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help=f"Upload CSV files up to {self.max_file_size_mb}MB. Supports fraud detection datasets with PCA features.",
            key="csv_file_uploader"
        )

        # Handle file upload
        if uploaded_file is not None:
            try:
                # Validate and process file
                if self._handle_file_upload(uploaded_file):
                    # Show file information
                    self._display_file_info()

                    # Show processing status
                    if st.session_state.processing_started and not st.session_state.processing_complete:
                        self._display_processing_status()

                    # Show results if processing is complete
                    elif st.session_state.processing_complete:
                        self._display_processing_results()
                        return self._get_processed_data()

            except FileValidationError as e:
                st.error(f"❌ File validation failed: {e}")
                logger.error(f"File validation error: {e}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in file upload: {e}")

        else:
            # Show upload instructions when no file is uploaded
            self._display_upload_instructions()

        # Show cache statistics if enabled
        if self.cache_enabled and st.session_state.get('show_debug', False):
            self._display_cache_statistics()

        return None

    def _handle_file_upload(self, uploaded_file) -> bool:
        """
        Handle the uploaded file validation and processing.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            True if file is successfully handled, False otherwise
        """
        # Calculate file hash for caching
        file_hash = self._calculate_file_hash(uploaded_file)

        # Check if file has changed
        if (st.session_state.uploaded_file != uploaded_file.name or
            st.session_state.file_hash != file_hash):

            # Reset processing state for new file
            self._reset_processing_state()

            # Update file information
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.file_hash = file_hash

            # Validate file
            self._validate_file(uploaded_file)
            st.session_state.file_validated = True

            # Extract file information
            st.session_state.file_info = self._extract_file_info(uploaded_file)

            # Check cache first
            if self.cache_enabled and self._check_cache(file_hash):
                st.success("✅ File loaded from cache!")
                st.session_state.processing_complete = True
                return True

            # Start processing
            self._start_processing(uploaded_file)

        return st.session_state.processing_complete

    def _validate_file(self, uploaded_file) -> None:
        """
        Validate the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Raises:
            FileValidationError: If file validation fails
        """
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise FileValidationError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)"
            )

        # Check file extension
        if not uploaded_file.name.lower().endswith('.csv'):
            raise FileValidationError("Only CSV files are supported")

        # Try to read a sample of the CSV to validate format
        try:
            # Read first few lines to validate CSV format
            uploaded_file.seek(0)
            sample_data = uploaded_file.read(1024).decode('utf-8')
            uploaded_file.seek(0)

            # Check for CSV-like structure
            if ',' not in sample_data and ';' not in sample_data and '\t' not in sample_data:
                raise FileValidationError("File does not appear to be a valid CSV format")

            # Try to read with pandas
            df_sample = pd.read_csv(uploaded_file, nrows=5)

            if df_sample.empty:
                raise FileValidationError("CSV file appears to be empty")

            # Reset file pointer
            uploaded_file.seek(0)

        except pd.errors.EmptyDataError:
            raise FileValidationError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise FileValidationError(f"CSV parsing error: {e}")
        except UnicodeDecodeError:
            raise FileValidationError("File encoding not supported. Please use UTF-8 encoding.")
        except Exception as e:
            raise FileValidationError(f"Unable to read CSV file: {e}")

    def _calculate_file_hash(self, uploaded_file) -> str:
        """
        Calculate hash of uploaded file for caching.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            SHA256 hash of the file content
        """
        uploaded_file.seek(0)
        content = uploaded_file.read()
        uploaded_file.seek(0)

        file_hash = hashlib.sha256(content).hexdigest()
        return file_hash

    def _extract_file_info(self, uploaded_file) -> Dict[str, Any]:
        """
        Extract information about the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary containing file information
        """
        file_size_mb = uploaded_file.size / (1024 * 1024)

        # Try to get basic CSV info
        try:
            uploaded_file.seek(0)
            df_info = pd.read_csv(uploaded_file, nrows=0)  # Just headers
            uploaded_file.seek(0)

            # Get row count estimate (for large files, sample)
            if file_size_mb > 50:  # For large files, estimate
                sample_df = pd.read_csv(uploaded_file, nrows=1000)
                uploaded_file.seek(0)
                estimated_rows = int((uploaded_file.size / len(uploaded_file.read(10000))) * 1000)
                uploaded_file.seek(0)
            else:
                df_full = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                estimated_rows = len(df_full)

            return {
                'filename': uploaded_file.name,
                'size_mb': file_size_mb,
                'size_bytes': uploaded_file.size,
                'estimated_rows': estimated_rows,
                'columns': list(df_info.columns),
                'num_columns': len(df_info.columns),
                'upload_timestamp': datetime.now().isoformat(),
                'has_fraud_columns': self._check_fraud_dataset_format(df_info.columns)
            }

        except Exception as e:
            logger.warning(f"Could not extract detailed file info: {e}")
            return {
                'filename': uploaded_file.name,
                'size_mb': file_size_mb,
                'size_bytes': uploaded_file.size,
                'upload_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _check_fraud_dataset_format(self, columns: List[str]) -> bool:
        """
        Check if the dataset appears to be in fraud detection format.

        Args:
            columns: List of column names

        Returns:
            True if dataset appears to be fraud detection format
        """
        columns_lower = [col.lower() for col in columns]

        # Check for typical fraud detection columns
        has_time = any('time' in col for col in columns_lower)
        has_amount = any('amount' in col for col in columns_lower)
        has_class = any('class' in col for col in columns_lower)
        has_pca_features = any(col.startswith('v') and col[1:].isdigit() for col in columns_lower)

        return has_time and has_amount and has_class and has_pca_features

    def _check_cache(self, file_hash: str) -> bool:
        """
        Check if processed results exist in cache.

        Args:
            file_hash: Hash of the file

        Returns:
            True if cached results found and loaded
        """
        cache = st.session_state.file_cache

        if file_hash in cache:
            cached_data = cache[file_hash]

            # Load cached results
            st.session_state.statistics_data = cached_data['statistics_data']
            st.session_state.pattern_data = cached_data['pattern_data']
            st.session_state.chart_data = cached_data['chart_data']
            st.session_state.processing_timestamp = cached_data['timestamp']

            st.session_state.cache_hits += 1
            logger.info(f"Cache hit for file hash: {file_hash[:8]}...")
            return True

        st.session_state.cache_misses += 1
        return False

    def _start_processing(self, uploaded_file) -> None:
        """
        Start processing the uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object
        """
        st.session_state.processing_started = True
        st.session_state.processing_complete = False
        st.session_state.processing_error = None
        st.session_state.processing_progress = 0

        try:
            # Process in steps with progress updates
            with st.status("Processing CSV file...", expanded=True) as status:

                # Step 1: Load CSV data
                status.write("📊 Loading CSV data...")
                st.session_state.processing_progress = 10

                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

                # Step 2: Generate statistics
                status.write("📈 Generating statistics...")
                st.session_state.processing_progress = 30

                csv_processor = CSVProcessor()
                statistics_data = csv_processor.process_dataframe(df)
                st.session_state.statistics_data = statistics_data

                # Step 3: Analyze patterns
                status.write("🔍 Analyzing patterns...")
                st.session_state.processing_progress = 60

                pattern_analyzer = PatternAnalyzer()
                pattern_data = pattern_analyzer.analyze_all_patterns(df)
                st.session_state.pattern_data = pattern_data

                # Step 4: Generate charts
                status.write("📊 Generating visualizations...")
                st.session_state.processing_progress = 80

                chart_generator = ChartGenerator()
                chart_data = chart_generator.generate_all_charts(df)
                st.session_state.chart_data = chart_data

                # Step 5: Cache results
                status.write("💾 Caching results...")
                st.session_state.processing_progress = 95

                if self.cache_enabled:
                    self._cache_results()

                # Complete
                st.session_state.processing_progress = 100
                st.session_state.processing_complete = True
                st.session_state.processing_timestamp = datetime.now().isoformat()

                status.update(
                    label="✅ Processing complete!",
                    state="complete"
                )

        except Exception as e:
            st.session_state.processing_error = str(e)
            st.session_state.processing_complete = False
            logger.error(f"Processing error: {e}")
            st.error(f"❌ Processing failed: {e}")

    def _cache_results(self) -> None:
        """Cache the processing results."""
        if st.session_state.file_hash:
            cache_data = {
                'statistics_data': st.session_state.statistics_data,
                'pattern_data': st.session_state.pattern_data,
                'chart_data': st.session_state.chart_data,
                'timestamp': datetime.now().isoformat(),
                'file_info': st.session_state.file_info
            }

            st.session_state.file_cache[st.session_state.file_hash] = cache_data
            logger.info(f"Cached results for file hash: {st.session_state.file_hash[:8]}...")

    def _reset_processing_state(self) -> None:
        """Reset processing state for new file."""
        st.session_state.processing_started = False
        st.session_state.processing_complete = False
        st.session_state.processing_progress = 0
        st.session_state.processing_status = ''
        st.session_state.processing_error = None
        st.session_state.statistics_data = None
        st.session_state.pattern_data = None
        st.session_state.chart_data = None

    def _display_upload_instructions(self) -> None:
        """Display upload instructions when no file is uploaded."""
        st.info("👆 **Upload a CSV file to get started**")

        with st.expander("📋 File Requirements", expanded=False):
            st.markdown(f"""
            ### Supported Files
            - **Format**: CSV files (.csv)
            - **Size**: Up to {self.max_file_size_mb}MB
            - **Encoding**: UTF-8 (recommended)

            ### Fraud Detection Format (Recommended)
            - **Time**: Transaction time feature
            - **Amount**: Transaction amount
            - **V1-V28**: PCA-transformed features
            - **Class**: Fraud indicator (0=normal, 1=fraud)

            ### General CSV Requirements
            - First row should contain column headers
            - Data should be comma-separated
            - No special characters in column names
            """)

    def _display_file_info(self) -> None:
        """Display information about the uploaded file."""
        if not st.session_state.file_info:
            return

        info = st.session_state.file_info

        # File info in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📁 File Size", f"{info.get('size_mb', 0):.1f} MB")

        with col2:
            st.metric("📊 Rows", f"{info.get('estimated_rows', 'Unknown'):,}")

        with col3:
            st.metric("📋 Columns", info.get('num_columns', 'Unknown'))

        with col4:
            fraud_format = info.get('has_fraud_columns', False)
            st.metric("🔍 Format", "Fraud Detection" if fraud_format else "General CSV")

        # Additional file details in expander
        with st.expander("📄 File Details", expanded=False):
            st.json({
                'filename': info.get('filename'),
                'upload_time': info.get('upload_timestamp'),
                'size_bytes': info.get('size_bytes'),
                'columns': info.get('columns', [])[:10],  # Show first 10 columns
                'fraud_detection_format': info.get('has_fraud_columns', False)
            })

    def _display_processing_status(self) -> None:
        """Display processing progress and status."""
        if st.session_state.processing_error:
            st.error(f"❌ Processing failed: {st.session_state.processing_error}")
            return

        # Progress bar
        progress = st.session_state.processing_progress / 100
        st.progress(progress, text=f"Processing: {st.session_state.processing_progress}%")

        # Processing steps
        steps = [
            (10, "📊 Loading CSV data"),
            (30, "📈 Generating statistics"),
            (60, "🔍 Analyzing patterns"),
            (80, "📊 Creating visualizations"),
            (95, "💾 Caching results"),
            (100, "✅ Complete")
        ]

        current_step = None
        for threshold, step_name in steps:
            if st.session_state.processing_progress >= threshold:
                current_step = step_name
            else:
                break

        if current_step:
            st.info(f"⚙️ {current_step}")

    def _display_processing_results(self) -> None:
        """Display processing results summary."""
        if not st.session_state.processing_complete:
            return

        st.success("✅ **File processed successfully!**")

        # Processing summary
        col1, col2, col3 = st.columns(3)

        with col1:
            stats_available = st.session_state.statistics_data is not None
            st.metric("📊 Statistics", "✅ Ready" if stats_available else "❌ Failed")

        with col2:
            patterns_available = st.session_state.pattern_data is not None
            st.metric("🔍 Patterns", "✅ Ready" if patterns_available else "❌ Failed")

        with col3:
            charts_available = st.session_state.chart_data is not None
            chart_count = len(st.session_state.chart_data) if charts_available else 0
            st.metric("📈 Charts", f"✅ {chart_count}" if charts_available else "❌ Failed")

        # Processing timestamp
        if st.session_state.processing_timestamp:
            processing_time = datetime.fromisoformat(st.session_state.processing_timestamp)
            st.caption(f"⏰ Processed at {processing_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _display_cache_statistics(self) -> None:
        """Display cache statistics for debugging."""
        if not self.cache_enabled:
            return

        st.subheader("🗄️ Cache Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cache Hits", st.session_state.cache_hits)

        with col2:
            st.metric("Cache Misses", st.session_state.cache_misses)

        with col3:
            cached_files = len(st.session_state.file_cache)
            st.metric("Cached Files", cached_files)

        # Cache hit rate
        total_requests = st.session_state.cache_hits + st.session_state.cache_misses
        if total_requests > 0:
            hit_rate = (st.session_state.cache_hits / total_requests) * 100
            st.info(f"📊 Cache hit rate: {hit_rate:.1f}%")

    def _get_processed_data(self) -> Optional[Dict[str, Any]]:
        """
        Get processed data if available.

        Returns:
            Dictionary containing all processed data
        """
        if not st.session_state.processing_complete:
            return None

        return {
            'file_info': st.session_state.file_info,
            'statistics_data': st.session_state.statistics_data,
            'pattern_data': st.session_state.pattern_data,
            'chart_data': st.session_state.chart_data,
            'processing_timestamp': st.session_state.processing_timestamp
        }

    def clear_cache(self) -> None:
        """Clear the file processing cache."""
        st.session_state.file_cache = {}
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        logger.info("File cache cleared")

    def get_cache_size(self) -> int:
        """
        Get the number of cached files.

        Returns:
            Number of files in cache
        """
        return len(st.session_state.file_cache)

    def is_file_processed(self) -> bool:
        """
        Check if a file has been processed.

        Returns:
            True if file is processed and data is available
        """
        return (st.session_state.processing_complete and
                st.session_state.statistics_data is not None)


# Convenience function for easy integration
def render_file_uploader(max_file_size_mb: int = 150,
                        cache_enabled: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convenience function to render file uploader.

    Args:
        max_file_size_mb: Maximum file size in MB
        cache_enabled: Whether to enable caching

    Returns:
        Processed data if available, None otherwise
    """
    uploader = FileUploader(max_file_size_mb=max_file_size_mb, cache_enabled=cache_enabled)
    return uploader.render()


if __name__ == "__main__":
    # Example usage for testing
    st.set_page_config(
        page_title="File Uploader Test",
        page_icon="📁",
        layout="wide"
    )

    st.title("📁 File Uploader Component Test")

    # Test the file uploader
    processed_data = render_file_uploader()

    if processed_data:
        st.subheader("🎉 Processing Complete!")
        st.json(processed_data)
    else:
        st.info("Upload a CSV file to see the processing results.")