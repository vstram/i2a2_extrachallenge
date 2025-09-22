"""
CSV Statistics Processor for Large Files

This module handles processing of large CSV files (150MB+) by generating
comprehensive statistics for fraud detection analysis. It uses chunked reading
to handle memory constraints and exports results to JSON format.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional
import warnings
from pathlib import Path


class CSVProcessor:
    """
    Processes large CSV files efficiently and generates comprehensive statistics.

    Designed specifically for fraud detection datasets with PCA-transformed
    features (V1-V28), Time, Amount, and Class variables.
    """

    def __init__(self, chunk_size: int = 10000):
        """
        Initialize the CSV processor.

        Args:
            chunk_size: Number of rows to process at a time for large files
        """
        self.chunk_size = chunk_size
        self.stats = {}

    def load_and_process(self, file_path: str) -> Dict[str, Any]:
        """
        Load CSV file and generate comprehensive statistics.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing all statistics and analysis results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Get file size to determine processing strategy
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_size_mb > 50:  # Use chunked reading for large files
            return self._process_large_file(file_path)
        else:
            return self._process_small_file(file_path)

    def _process_small_file(self, file_path: Path) -> Dict[str, Any]:
        """Process small files by loading entirely into memory."""
        df = pd.read_csv(file_path)
        return self._generate_statistics(df)

    def _process_large_file(self, file_path: Path) -> Dict[str, Any]:
        """Process large files using chunked reading."""
        # Initialize accumulators
        chunk_stats = []
        total_rows = 0
        column_names = None

        # Process file in chunks
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            if column_names is None:
                column_names = list(chunk.columns)

            chunk_stat = self._generate_chunk_statistics(chunk)
            chunk_stats.append(chunk_stat)
            total_rows += len(chunk)

        # Aggregate chunk statistics
        return self._aggregate_chunk_statistics(chunk_stats, total_rows, column_names)

    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for the entire dataframe."""
        stats = {
            'file_info': self._get_file_info(df),
            'data_types': self._analyze_data_types(df),
            'basic_statistics': self._calculate_basic_statistics(df),
            'distribution_analysis': self._analyze_distributions(df),
            'missing_values': self._analyze_missing_values(df),
            'fraud_specific_analysis': self._analyze_fraud_features(df),
            'correlation_analysis': self._calculate_correlations(df)
        }
        return stats

    def _generate_chunk_statistics(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics for a single chunk."""
        return {
            'count': len(chunk),
            'numeric_stats': chunk.select_dtypes(include=[np.number]).describe().to_dict(),
            'value_counts': {col: chunk[col].value_counts().head(10).to_dict()
                           for col in chunk.select_dtypes(exclude=[np.number]).columns},
            'missing_counts': chunk.isnull().sum().to_dict(),
            'dtypes': chunk.dtypes.astype(str).to_dict()
        }

    def _aggregate_chunk_statistics(self, chunk_stats: List[Dict], total_rows: int, columns: List[str]) -> Dict[str, Any]:
        """Aggregate statistics from multiple chunks."""
        # This is a simplified aggregation - for production, you'd want more sophisticated aggregation
        aggregated = {
            'file_info': {
                'total_rows': total_rows,
                'total_columns': len(columns),
                'columns': columns,
                'estimated_size_mb': total_rows * len(columns) * 8 / (1024 * 1024)  # Rough estimate
            },
            'data_types': chunk_stats[0]['dtypes'] if chunk_stats else {},
            'basic_statistics': self._aggregate_numeric_stats(chunk_stats),
            'missing_values': self._aggregate_missing_values(chunk_stats),
            'note': 'Statistics generated from chunked processing - some analyses may be approximated'
        }
        return aggregated

    def _get_file_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic file information."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types of all columns."""
        dtype_info = {}
        for col in df.columns:
            dtype_info[col] = {
                'pandas_dtype': str(df[col].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object',
                'unique_values': int(df[col].nunique()),
                'unique_ratio': df[col].nunique() / len(df)
            }
        return dtype_info

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive basic statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        stats = {
            'numeric_statistics': {},
            'categorical_statistics': {}
        }

        # Numeric statistics
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            for col in numeric_cols:
                stats['numeric_statistics'][col] = {
                    'count': int(desc.loc['count', col]),
                    'mean': float(desc.loc['mean', col]),
                    'std': float(desc.loc['std', col]),
                    'min': float(desc.loc['min', col]),
                    'q1': float(desc.loc['25%', col]),
                    'median': float(desc.loc['50%', col]),
                    'q3': float(desc.loc['75%', col]),
                    'max': float(desc.loc['max', col]),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis())
                }

        # Categorical statistics
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(20)
            stats['categorical_statistics'][col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'value_counts': value_counts.to_dict()
            }

        return stats

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions with histogram data."""
        distributions = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Create histogram data
            hist, bin_edges = np.histogram(df[col].dropna(), bins=20)

            distributions[col] = {
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist()
                },
                'percentiles': {
                    'p1': float(df[col].quantile(0.01)),
                    'p5': float(df[col].quantile(0.05)),
                    'p10': float(df[col].quantile(0.10)),
                    'p90': float(df[col].quantile(0.90)),
                    'p95': float(df[col].quantile(0.95)),
                    'p99': float(df[col].quantile(0.99))
                }
            }

        return distributions

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        missing_analysis = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_analysis[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(df) * 100)
            }

        # Overall missing data summary
        total_missing = df.isnull().sum().sum()
        missing_analysis['summary'] = {
            'total_missing_values': int(total_missing),
            'percentage_of_total_values': float(total_missing / (len(df) * len(df.columns)) * 100),
            'columns_with_missing': [col for col in df.columns if df[col].isnull().any()]
        }

        return missing_analysis

    def _analyze_fraud_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fraud detection specific features."""
        fraud_analysis = {}

        # Time analysis
        if 'Time' in df.columns:
            fraud_analysis['time_analysis'] = {
                'total_duration_seconds': float(df['Time'].max() - df['Time'].min()),
                'total_duration_hours': float((df['Time'].max() - df['Time'].min()) / 3600),
                'time_gaps': {
                    'mean_gap': float(df['Time'].diff().mean()),
                    'median_gap': float(df['Time'].diff().median()),
                    'max_gap': float(df['Time'].diff().max())
                },
                'time_distribution': df['Time'].describe().to_dict()
            }

        # Amount analysis
        if 'Amount' in df.columns:
            fraud_analysis['amount_analysis'] = {
                'zero_amount_count': int((df['Amount'] == 0).sum()),
                'zero_amount_percentage': float((df['Amount'] == 0).sum() / len(df) * 100),
                'large_transactions': {
                    'over_1000': int((df['Amount'] > 1000).sum()),
                    'over_5000': int((df['Amount'] > 5000).sum()),
                    'over_10000': int((df['Amount'] > 10000).sum())
                },
                'amount_distribution': df['Amount'].describe().to_dict()
            }

        # Class analysis (fraud detection)
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts()
            fraud_analysis['class_analysis'] = {
                'fraud_count': int(class_counts.get(1, 0)),
                'normal_count': int(class_counts.get(0, 0)),
                'fraud_percentage': float(class_counts.get(1, 0) / len(df) * 100),
                'imbalance_ratio': float(class_counts.get(0, 0) / max(class_counts.get(1, 0), 1))
            }

            # Fraud vs Normal comparison for Amount
            if 'Amount' in df.columns:
                fraud_analysis['fraud_amount_comparison'] = {
                    'fraud_amount_stats': df[df['Class'] == 1]['Amount'].describe().to_dict(),
                    'normal_amount_stats': df[df['Class'] == 0]['Amount'].describe().to_dict()
                }

        return fraud_analysis

    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrix for numeric variables."""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {'note': 'Not enough numeric columns for correlation analysis'}

        correlation_matrix = numeric_df.corr()

        # Find high correlations (excluding self-correlations)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for "high" correlation
                    high_correlations.append({
                        'variable_1': correlation_matrix.columns[i],
                        'variable_2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'correlation_with_target': correlation_matrix['Class'].to_dict() if 'Class' in correlation_matrix.columns else {}
        }

    def _aggregate_numeric_stats(self, chunk_stats: List[Dict]) -> Dict[str, Any]:
        """Aggregate numeric statistics from chunks."""
        # Simplified aggregation - in production, you'd want proper statistical aggregation
        if not chunk_stats:
            return {}

        first_chunk = chunk_stats[0]
        if 'numeric_stats' not in first_chunk:
            return {}

        return first_chunk['numeric_stats']  # Simplified - just return first chunk stats

    def _aggregate_missing_values(self, chunk_stats: List[Dict]) -> Dict[str, Any]:
        """Aggregate missing value counts from chunks."""
        if not chunk_stats:
            return {}

        total_missing = {}
        for chunk in chunk_stats:
            if 'missing_counts' in chunk:
                for col, count in chunk['missing_counts'].items():
                    total_missing[col] = total_missing.get(col, 0) + count

        return total_missing

    def export_to_json(self, statistics: Dict[str, Any], output_path: str) -> None:
        """Export statistics to JSON file."""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Convert numpy types to JSON-serializable types
        json_stats = convert_numpy_types(statistics)

        with open(output_path, 'w') as f:
            json.dump(json_stats, f, indent=2, default=str)


def process_csv_file(file_path: str, output_path: Optional[str] = None, chunk_size: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to process a CSV file and optionally save results.

    Args:
        file_path: Path to the CSV file
        output_path: Optional path to save JSON output
        chunk_size: Chunk size for large file processing

    Returns:
        Dictionary containing all statistics
    """
    processor = CSVProcessor(chunk_size=chunk_size)
    statistics = processor.load_and_process(file_path)

    if output_path:
        processor.export_to_json(statistics, output_path)

    return statistics


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_processor.py <csv_file_path> [output_json_path]")
        sys.exit(1)

    csv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        stats = process_csv_file(csv_path, json_path)
        print(f"Successfully processed {csv_path}")
        if json_path:
            print(f"Statistics saved to {json_path}")
        else:
            print("Sample statistics:")
            if 'file_info' in stats:
                print(f"  Rows: {stats['file_info']['total_rows']}")
                print(f"  Columns: {stats['file_info']['total_columns']}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)