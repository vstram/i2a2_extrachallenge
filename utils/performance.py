"""
Performance Optimization Module for Streamlit AI Analytics Chatbot

This module provides comprehensive performance optimization features including
intelligent caching, progress tracking, memory optimization, chunked processing,
and performance monitoring for handling large datasets efficiently.
"""

import os
import gc
import time
import json
import hashlib
import logging

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import threading
import queue
import weakref
from contextlib import contextmanager
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    cache_hit: bool = False
    data_size_mb: Optional[float] = None
    rows_processed: Optional[int] = None
    throughput_rows_per_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self, end_time: datetime = None, **kwargs):
        """Finalize metrics calculation."""
        self.end_time = end_time or datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

        # Calculate throughput if rows processed
        if self.rows_processed and self.duration_seconds > 0:
            self.throughput_rows_per_sec = self.rows_processed / self.duration_seconds

        # Update metadata
        self.metadata.update(kwargs)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)

    def access(self):
        """Mark entry as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class AdvancedCache:
    """
    Advanced caching system with features:
    - LRU eviction
    - Size-based limits
    - TTL support
    - Tag-based invalidation
    - Compression
    - Persistence
    """

    def __init__(self, max_size_mb: int = 500, max_entries: int = 100,
                 default_ttl_hours: int = 24, cache_dir: str = "cache"):
        """
        Initialize advanced cache.

        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            default_ttl_hours: Default TTL in hours
            cache_dir: Cache directory for persistence
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_evictions': 0,
            'ttl_evictions': 0
        }

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            if isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str))
            elif isinstance(data, str):
                return len(data.encode())
            elif isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, np.ndarray):
                return data.nbytes
            else:
                return len(str(data))
        except Exception:
            return 1024  # Default estimate

    def _evict_lru(self, required_space: int = 0):
        """Evict least recently used entries."""
        with self._lock:
            # Sort by last accessed time
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )

            for key, entry in sorted_entries:
                if (self._current_size_bytes + required_space <= self.max_size_bytes and
                    len(self._cache) < self.max_entries):
                    break

                # Remove entry
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                self.stats['evictions'] += 1
                self.stats['size_evictions'] += 1

                logger.debug(f"Evicted cache entry: {key}")

    def _evict_expired(self):
        """Evict expired entries."""
        now = datetime.now()
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if now - entry.created_at > self.default_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                self.stats['evictions'] += 1
                self.stats['ttl_evictions'] += 1

                logger.debug(f"Evicted expired cache entry: {key}")

    def get(self, key: str, default=None) -> Any:
        """Get item from cache."""
        with self._lock:
            # Clean expired entries
            self._evict_expired()

            if key in self._cache:
                entry = self._cache[key]
                entry.access()
                self.stats['hits'] += 1
                return entry.data
            else:
                self.stats['misses'] += 1
                return default

    def put(self, key: str, data: Any, tags: List[str] = None, ttl_hours: int = None):
        """Put item in cache."""
        tags = tags or []
        size_bytes = self._estimate_size(data)

        with self._lock:
            # Check if we need to evict
            if (self._current_size_bytes + size_bytes > self.max_size_bytes or
                len(self._cache) >= self.max_entries):
                self._evict_lru(size_bytes)

            # Create entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                tags=tags
            )

            # Store entry
            if key in self._cache:
                # Update existing
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes

            self._cache[key] = entry
            self._current_size_bytes += size_bytes

            logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")

    def invalidate_by_tag(self, tag: str):
        """Invalidate all entries with specific tag."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                entry = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes

            logger.info(f"Invalidated {len(keys_to_remove)} entries with tag: {tag}")

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)

            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self._current_size_bytes / self.max_size_bytes,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'size_evictions': self.stats['size_evictions'],
                'ttl_evictions': self.stats['ttl_evictions']
            }


class ProgressTracker:
    """
    Advanced progress tracking with multiple progress bars and callbacks.
    """

    def __init__(self, callback: Callable[[str, float, Dict], None] = None):
        """
        Initialize progress tracker.

        Args:
            callback: Progress callback function (operation, progress, metadata)
        """
        self.callback = callback
        self._progress_stack: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    @contextmanager
    def track_operation(self, operation: str, total_steps: int = 100, **metadata):
        """Context manager for tracking operation progress."""
        progress_info = {
            'operation': operation,
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': datetime.now(),
            'metadata': metadata
        }

        with self._lock:
            self._progress_stack.append(progress_info)

        try:
            yield self
        finally:
            with self._lock:
                if self._progress_stack and self._progress_stack[-1]['operation'] == operation:
                    self._progress_stack.pop()

    def update_progress(self, steps: int = 1, message: str = None, **metadata):
        """Update progress for current operation."""
        with self._lock:
            if not self._progress_stack:
                return

            current = self._progress_stack[-1]
            current['current_step'] = min(current['current_step'] + steps, current['total_steps'])

            if message:
                current['metadata']['message'] = message
            current['metadata'].update(metadata)

            progress_percent = (current['current_step'] / current['total_steps']) * 100

            if self.callback:
                self.callback(current['operation'], progress_percent, current['metadata'])

    def set_progress(self, step: int, message: str = None, **metadata):
        """Set absolute progress for current operation."""
        with self._lock:
            if not self._progress_stack:
                return

            current = self._progress_stack[-1]
            current['current_step'] = min(step, current['total_steps'])

            if message:
                current['metadata']['message'] = message
            current['metadata'].update(metadata)

            progress_percent = (current['current_step'] / current['total_steps']) * 100

            if self.callback:
                self.callback(current['operation'], progress_percent, current['metadata'])


class MemoryOptimizer:
    """
    Memory optimization utilities for large file processing.
    """

    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        else:
            # Fallback for systems without psutil
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux

    @staticmethod
    def get_available_memory() -> float:
        """Get available memory in MB."""
        if HAS_PSUTIL:
            return psutil.virtual_memory().available / 1024 / 1024
        else:
            # Conservative fallback - assume 4GB available
            return 4096.0

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype('category')
            except (ValueError, TypeError):
                pass

        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        savings = original_memory - optimized_memory

        logger.info(f"Memory optimization: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                   f"(saved {savings:.1f}MB, {savings/original_memory*100:.1f}%)")

        return df

    @staticmethod
    @contextmanager
    def memory_limit_check(operation: str, max_memory_mb: float = None):
        """Context manager to check memory limits during operations."""
        if max_memory_mb is None:
            max_memory_mb = MemoryOptimizer.get_available_memory() * 0.8  # Use 80% of available

        start_memory = MemoryOptimizer.get_memory_usage()

        try:
            yield
        finally:
            end_memory = MemoryOptimizer.get_memory_usage()
            used_memory = end_memory - start_memory

            if used_memory > max_memory_mb:
                logger.warning(f"Operation '{operation}' used {used_memory:.1f}MB "
                             f"(limit: {max_memory_mb:.1f}MB)")

            # Force garbage collection
            gc.collect()

    @staticmethod
    def suggest_chunk_size(file_size_mb: float, available_memory_mb: float = None) -> int:
        """Suggest optimal chunk size based on file size and available memory."""
        if available_memory_mb is None:
            available_memory_mb = MemoryOptimizer.get_available_memory()

        # Use conservative memory allocation (25% of available)
        usable_memory_mb = available_memory_mb * 0.25

        # Estimate rows based on average row size (assume ~1KB per row)
        estimated_rows = int(file_size_mb * 1024)

        # Calculate chunk size to fit in memory
        max_chunk_rows = int(usable_memory_mb * 1024)

        # Choose reasonable chunk size
        if estimated_rows <= max_chunk_rows:
            return estimated_rows  # Process entire file
        else:
            # Use chunks that fit in memory, minimum 1000 rows
            chunk_size = max(1000, max_chunk_rows // 4)
            logger.info(f"Suggested chunk size: {chunk_size} rows for {file_size_mb:.1f}MB file")
            return chunk_size


class ChunkedProcessor:
    """
    Chunked processing utilities for large datasets.
    """

    def __init__(self, progress_tracker: ProgressTracker = None):
        """Initialize chunked processor."""
        self.progress_tracker = progress_tracker

    def process_csv_chunks(self, file_path: str, chunk_size: int,
                          processor_func: Callable[[pd.DataFrame], Any],
                          aggregator_func: Callable[[List[Any]], Any] = None) -> Any:
        """
        Process CSV file in chunks.

        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            processor_func: Function to process each chunk
            aggregator_func: Function to aggregate chunk results

        Returns:
            Aggregated results
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / 1024 / 1024

        # Estimate total chunks
        try:
            total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            total_chunks = (total_rows + chunk_size - 1) // chunk_size
        except Exception:
            total_chunks = max(1, int(file_size_mb / 10))  # Rough estimate

        logger.info(f"Processing {file_path.name} in {total_chunks} chunks of {chunk_size} rows")

        chunk_results = []

        if self.progress_tracker:
            context = self.progress_tracker.track_operation(
                f"Processing {file_path.name}",
                total_chunks,
                file_size_mb=file_size_mb
            )
        else:
            context = None

        try:
            with (context if context else nullcontext()):
                for chunk_idx, chunk_df in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):

                    # Optimize chunk memory
                    chunk_df = MemoryOptimizer.optimize_dataframe_memory(chunk_df)

                    # Process chunk
                    with MemoryOptimizer.memory_limit_check(f"chunk_{chunk_idx}"):
                        chunk_result = processor_func(chunk_df)
                        chunk_results.append(chunk_result)

                    # Update progress
                    if self.progress_tracker:
                        self.progress_tracker.update_progress(
                            1, f"Processed chunk {chunk_idx + 1}/{total_chunks}"
                        )

                    # Clean up chunk
                    del chunk_df
                    gc.collect()

        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            raise

        # Aggregate results
        if aggregator_func and chunk_results:
            logger.info(f"Aggregating {len(chunk_results)} chunk results")
            return aggregator_func(chunk_results)
        else:
            return chunk_results


class PerformanceMonitor:
    """
    Performance monitoring and profiling system.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

    @contextmanager
    def monitor_operation(self, operation_name: str, **metadata):
        """Context manager for monitoring operation performance."""
        # Get initial metrics
        start_time = datetime.now()
        memory_before = MemoryOptimizer.get_memory_usage()

        # CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1) if HAS_PSUTIL else 0.0

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            memory_before_mb=memory_before,
            cpu_percent=cpu_percent,
            metadata=metadata
        )

        try:
            yield metrics
        finally:
            # Finalize metrics
            end_time = datetime.now()
            memory_after = MemoryOptimizer.get_memory_usage()
            memory_delta = memory_after - memory_before

            metrics.finalize(
                end_time=end_time,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_delta
            )

            # Store metrics
            with self._lock:
                self.metrics_history.append(metrics)
                self.operation_stats[operation_name].append(metrics.duration_seconds)

            logger.info(f"Operation '{operation_name}' completed in {metrics.duration_seconds:.2f}s "
                       f"(memory: {memory_delta:+.1f}MB)")

    def get_operation_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        with self._lock:
            if operation_name:
                durations = self.operation_stats.get(operation_name, [])
                if not durations:
                    return {}

                return {
                    'operation': operation_name,
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
            else:
                # All operations
                stats = {}
                for op_name, durations in self.operation_stats.items():
                    stats[op_name] = {
                        'count': len(durations),
                        'avg_duration': np.mean(durations),
                        'total_duration': sum(durations)
                    }
                return stats

    def get_recent_metrics(self, limit: int = 10) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        with self._lock:
            return list(self.metrics_history)[-limit:]

    def export_metrics(self, file_path: str):
        """Export metrics to file."""
        with self._lock:
            metrics_data = []
            for metric in self.metrics_history:
                metrics_data.append({
                    'operation_name': metric.operation_name,
                    'start_time': metric.start_time.isoformat(),
                    'duration_seconds': metric.duration_seconds,
                    'memory_delta_mb': metric.memory_delta_mb,
                    'cache_hit': metric.cache_hit,
                    'rows_processed': metric.rows_processed,
                    'throughput_rows_per_sec': metric.throughput_rows_per_sec,
                    'metadata': metric.metadata
                })

            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            logger.info(f"Exported {len(metrics_data)} metrics to {file_path}")


# Global instances
_global_cache = None
_global_progress_tracker = None
_global_performance_monitor = None

def get_cache() -> AdvancedCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdvancedCache()
    return _global_cache

def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance."""
    global _global_progress_tracker
    if _global_progress_tracker is None:
        _global_progress_tracker = ProgressTracker()
    return _global_progress_tracker

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


# Decorators
def cached(tags: List[str] = None, ttl_hours: int = None):
    """Decorator for automatic caching."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            key = cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(key, result, tags=tags, ttl_hours=ttl_hours)

            return result
        return wrapper
    return decorator

def monitored(operation_name: str = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            monitor = get_performance_monitor()

            with monitor.monitor_operation(op_name) as metrics:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


# Utility functions
def nullcontext():
    """Null context manager for Python < 3.7 compatibility."""
    class NullContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NullContext()

def optimize_for_large_files(file_path: str) -> Dict[str, Any]:
    """Analyze file and suggest optimizations."""
    file_path = Path(file_path)
    file_size_mb = file_path.stat().st_size / 1024 / 1024
    available_memory_mb = MemoryOptimizer.get_available_memory()

    suggestions = {
        'file_size_mb': file_size_mb,
        'available_memory_mb': available_memory_mb,
        'use_chunked_processing': file_size_mb > 50,
        'suggested_chunk_size': MemoryOptimizer.suggest_chunk_size(file_size_mb, available_memory_mb),
        'enable_memory_optimization': True,
        'enable_progress_tracking': file_size_mb > 10,
        'estimated_processing_time_minutes': max(1, file_size_mb / 50),  # Rough estimate
        'recommended_cache_size_mb': min(200, file_size_mb * 2)
    }

    return suggestions


if __name__ == "__main__":
    # Example usage and testing
    def test_performance_system():
        """Test the performance optimization system."""
        print("=== Performance Optimization System Test ===")

        # Test 1: Cache system
        print("\n1. Testing Cache System...")
        cache = get_cache()

        # Store some data
        cache.put("test_key", {"data": "test_value"}, tags=["test"])

        # Retrieve data
        result = cache.get("test_key")
        print(f"Cache retrieval: {result}")

        # Show stats
        stats = cache.get_stats()
        print(f"Cache stats: Hit rate: {stats['hit_rate']:.2f}, Size: {stats['size_mb']:.2f}MB")

        # Test 2: Progress tracking
        print("\n2. Testing Progress Tracking...")

        def progress_callback(operation, progress, metadata):
            print(f"Progress: {operation} - {progress:.1f}% - {metadata.get('message', '')}")

        tracker = ProgressTracker(progress_callback)

        with tracker.track_operation("Test Operation", 100):
            for i in range(0, 101, 20):
                tracker.set_progress(i, f"Step {i}")
                time.sleep(0.1)

        # Test 3: Memory optimization
        print("\n3. Testing Memory Optimization...")

        # Create test dataframe
        import pandas as pd
        df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': [float(i) for i in range(1000)],
            'category_col': (['A', 'B', 'C'] * 334)[:1000]  # Ensure exact length
        })

        print(f"Original memory: {df.memory_usage(deep=True).sum() / 1024:.1f}KB")
        df_optimized = MemoryOptimizer.optimize_dataframe_memory(df)
        print(f"Optimized memory: {df_optimized.memory_usage(deep=True).sum() / 1024:.1f}KB")

        # Test 4: Performance monitoring
        print("\n4. Testing Performance Monitoring...")

        monitor = get_performance_monitor()

        @monitored("test_operation")
        def test_operation():
            time.sleep(0.1)
            return "completed"

        result = test_operation()

        # Show operation stats
        stats = monitor.get_operation_stats("test_operation")
        print(f"Operation stats: {stats}")

        # Test 5: Chunked processing simulation
        print("\n5. Testing Chunked Processing...")

        processor = ChunkedProcessor(tracker)

        # Simulate processing function
        def process_chunk(chunk_df):
            return len(chunk_df)

        def aggregate_results(results):
            return sum(results)

        # Create temporary CSV for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            for i in range(1000):
                f.write(f"{i},{i*2},{i*3}\n")
            temp_file = f.name

        try:
            total_rows = processor.process_csv_chunks(
                temp_file, 100, process_chunk, aggregate_results
            )
            print(f"Processed {total_rows} total rows in chunks")
        finally:
            os.unlink(temp_file)

        print("\n✅ Performance optimization system test completed!")

    test_performance_system()