"""
Pattern Analysis Module for Large CSV Files

This module performs advanced pattern analysis on fraud detection datasets,
including temporal pattern detection, outlier analysis, clustering, and
fraud pattern identification. Results are exported to structured JSON format.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class PatternAnalyzer:
    """
    Performs comprehensive pattern analysis on fraud detection datasets.

    Analyzes temporal patterns, outliers, clusters, correlations, and fraud-specific
    patterns. Optimized for datasets with PCA-transformed features (V1-V28).
    """

    def __init__(self, max_samples: int = 50000):
        """
        Initialize the pattern analyzer.

        Args:
            max_samples: Maximum number of samples to use for computationally intensive analyses
        """
        self.max_samples = max_samples
        self.patterns = {}

    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing all pattern analysis results
        """
        # Sample data if too large for computationally intensive operations
        sampled_df = self._sample_data(df)

        analysis_results = {
            'dataset_info': {
                'original_size': len(df),
                'analyzed_size': len(sampled_df),
                'sampling_applied': len(df) > self.max_samples
            }
        }

        # Temporal pattern detection
        if 'Time' in df.columns:
            analysis_results['temporal_patterns'] = self.detect_temporal_patterns(sampled_df)

        # Outlier detection
        analysis_results['outlier_analysis'] = self.detect_outliers(sampled_df)

        # Clustering analysis
        analysis_results['clustering_analysis'] = self.perform_clustering_analysis(sampled_df)

        # Correlation analysis
        analysis_results['correlation_analysis'] = self.analyze_correlations(sampled_df)

        # Fraud pattern identification
        if 'Class' in df.columns:
            analysis_results['fraud_patterns'] = self.identify_fraud_patterns(sampled_df)

        # Pattern summary
        analysis_results['pattern_summary'] = self._generate_pattern_summary(analysis_results)

        return analysis_results

    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample data if dataset is too large, maintaining fraud ratio."""
        if len(df) > self.max_samples:
            if 'Class' in df.columns:
                # Stratified sampling to preserve fraud ratio
                fraud_df = df[df['Class'] == 1]
                normal_df = df[df['Class'] == 0]

                fraud_ratio = len(fraud_df) / len(df)
                fraud_sample_size = min(len(fraud_df), int(self.max_samples * fraud_ratio))
                normal_sample_size = min(len(normal_df), self.max_samples - fraud_sample_size)

                fraud_sample = fraud_df.sample(n=fraud_sample_size, random_state=42)
                normal_sample = normal_df.sample(n=normal_sample_size, random_state=42)

                return pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                return df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
        return df

    def detect_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect temporal patterns in the Time variable.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing temporal pattern analysis
        """
        if 'Time' not in df.columns:
            return {'error': 'Time column not found'}

        time_data = df['Time'].copy()
        results = {}

        # Basic temporal statistics
        results['basic_stats'] = {
            'duration_seconds': float(time_data.max() - time_data.min()),
            'duration_hours': float((time_data.max() - time_data.min()) / 3600),
            'duration_days': float((time_data.max() - time_data.min()) / 86400),
            'median_time': float(time_data.median()),
            'mean_time': float(time_data.mean())
        }

        # Time gaps analysis
        time_diffs = time_data.diff().dropna()
        results['time_gaps'] = {
            'mean_gap_seconds': float(time_diffs.mean()),
            'median_gap_seconds': float(time_diffs.median()),
            'std_gap_seconds': float(time_diffs.std()),
            'min_gap_seconds': float(time_diffs.min()),
            'max_gap_seconds': float(time_diffs.max()),
            'gaps_over_1hour': int((time_diffs > 3600).sum()),
            'gaps_over_1day': int((time_diffs > 86400).sum())
        }

        # Hourly patterns (convert seconds to hours of day)
        hours = (time_data % 86400) / 3600  # Hour of day (0-23)
        hour_counts = pd.cut(hours, bins=24, labels=range(24)).value_counts().sort_index()

        results['hourly_patterns'] = {
            'peak_hour': int(hour_counts.idxmax()),
            'lowest_hour': int(hour_counts.idxmin()),
            'peak_count': int(hour_counts.max()),
            'lowest_count': int(hour_counts.min()),
            'hourly_distribution': hour_counts.to_dict()
        }

        # Weekly patterns (if duration > 7 days)
        if results['basic_stats']['duration_days'] > 7:
            days = (time_data / 86400) % 7  # Day of week (0-6)
            day_counts = pd.cut(days, bins=7, labels=range(7)).value_counts().sort_index()

            results['weekly_patterns'] = {
                'peak_day': int(day_counts.idxmax()),
                'lowest_day': int(day_counts.idxmin()),
                'peak_count': int(day_counts.max()),
                'lowest_count': int(day_counts.min()),
                'daily_distribution': day_counts.to_dict()
            }

        # Fraud timing patterns (if Class column exists)
        if 'Class' in df.columns:
            fraud_times = df[df['Class'] == 1]['Time']
            normal_times = df[df['Class'] == 0]['Time']

            if len(fraud_times) > 0 and len(normal_times) > 0:
                # Statistical test for different time distributions
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(fraud_times, normal_times)
                    results['fraud_timing'] = {
                        'fraud_mean_time': float(fraud_times.mean()),
                        'normal_mean_time': float(normal_times.mean()),
                        'fraud_median_time': float(fraud_times.median()),
                        'normal_median_time': float(normal_times.median()),
                        'ks_test_statistic': float(ks_stat),
                        'ks_test_pvalue': float(ks_pvalue),
                        'significantly_different': ks_pvalue < 0.05
                    }
                except:
                    results['fraud_timing'] = {'error': 'Could not perform statistical test'}

        return results

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using statistical methods (IQR, Z-score).

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing outlier analysis for each numerical column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            col_results = {}

            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = (col_data < lower_bound) | (col_data > upper_bound)
            col_results['iqr_method'] = {
                'outlier_count': int(iqr_outliers.sum()),
                'outlier_percentage': float(iqr_outliers.sum() / len(col_data) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values_sample': col_data[iqr_outliers].head(10).tolist()
            }

            # Z-score method (threshold = 3)
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = z_scores > 3

            col_results['zscore_method'] = {
                'outlier_count': int(z_outliers.sum()),
                'outlier_percentage': float(z_outliers.sum() / len(col_data) * 100),
                'max_zscore': float(z_scores.max()),
                'mean_zscore': float(z_scores.mean()),
                'outlier_values_sample': col_data[z_outliers].head(10).tolist()
            }

            # Modified Z-score method (using median)
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (col_data - median) / mad
                modified_z_outliers = np.abs(modified_z_scores) > 3.5

                col_results['modified_zscore_method'] = {
                    'outlier_count': int(modified_z_outliers.sum()),
                    'outlier_percentage': float(modified_z_outliers.sum() / len(col_data) * 100),
                    'outlier_values_sample': col_data[modified_z_outliers].head(10).tolist()
                }

            # Fraud correlation for outliers (if Class exists)
            if 'Class' in df.columns:
                # Check if outliers are more likely to be fraud
                fraud_mask = df['Class'] == 1
                iqr_outlier_fraud_rate = 0
                if iqr_outliers.sum() > 0:
                    iqr_outlier_fraud_rate = df.loc[iqr_outliers, 'Class'].mean()

                col_results['fraud_correlation'] = {
                    'overall_fraud_rate': float(df['Class'].mean()),
                    'iqr_outlier_fraud_rate': float(iqr_outlier_fraud_rate),
                    'outliers_more_likely_fraud': iqr_outlier_fraud_rate > df['Class'].mean()
                }

            outlier_results[col] = col_results

        # Summary statistics
        total_outliers_iqr = sum([outlier_results[col]['iqr_method']['outlier_count']
                                 for col in outlier_results])
        total_outliers_zscore = sum([outlier_results[col]['zscore_method']['outlier_count']
                                    for col in outlier_results])

        outlier_results['summary'] = {
            'total_outliers_iqr_method': total_outliers_iqr,
            'total_outliers_zscore_method': total_outliers_zscore,
            'columns_analyzed': len(outlier_results) - 1,  # Exclude summary itself
            'most_outliers_column_iqr': max(outlier_results.keys(),
                                           key=lambda x: outlier_results[x]['iqr_method']['outlier_count']
                                           if x != 'summary' else 0)
        }

        return outlier_results

    def perform_clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform clustering analysis using K-means on PCA components.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing clustering analysis results
        """
        # Get V features (PCA components)
        v_features = [col for col in df.columns if col.startswith('V')]

        if len(v_features) < 2:
            return {'error': 'Not enough V features for clustering analysis'}

        # Prepare data
        X = df[v_features].dropna()
        if len(X) < 10:
            return {'error': 'Not enough data points for clustering'}

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        # Determine optimal number of clusters using elbow method
        max_clusters = min(10, len(X) // 10)
        if max_clusters < 2:
            max_clusters = 2

        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)

            if k <= 8:  # Silhouette score is expensive for large k
                try:
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                except:
                    silhouette_scores.append(0)

        # Find optimal k using elbow method (simple heuristic)
        if len(inertias) >= 3:
            # Calculate rate of change
            rate_changes = []
            for i in range(1, len(inertias) - 1):
                rate_change = (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1])
                rate_changes.append(rate_change)

            optimal_k = cluster_range[np.argmax(rate_changes) + 1] if rate_changes else 3
        else:
            optimal_k = 3

        # Perform final clustering with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(X_scaled)

        results['optimal_clustering'] = {
            'optimal_k': int(optimal_k),
            'cluster_centers': kmeans_final.cluster_centers_.tolist(),
            'inertia': float(kmeans_final.inertia_),
            'silhouette_score': float(silhouette_score(X_scaled, final_labels)) if len(set(final_labels)) > 1 else 0
        }

        # Cluster characteristics
        cluster_stats = {}
        for cluster_id in range(optimal_k):
            cluster_mask = final_labels == cluster_id
            cluster_data = X[cluster_mask]

            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(X) * 100),
                'mean_features': cluster_data.mean().to_dict(),
                'std_features': cluster_data.std().to_dict()
            }

        results['cluster_characteristics'] = cluster_stats

        # Fraud distribution across clusters (if Class exists)
        if 'Class' in df.columns:
            X_with_class = df[v_features + ['Class']].dropna()
            if len(X_with_class) == len(X):
                fraud_cluster_analysis = {}
                for cluster_id in range(optimal_k):
                    cluster_mask = final_labels == cluster_id
                    cluster_fraud_rate = X_with_class.loc[cluster_mask, 'Class'].mean()

                    fraud_cluster_analysis[f'cluster_{cluster_id}'] = {
                        'fraud_rate': float(cluster_fraud_rate),
                        'fraud_count': int(X_with_class.loc[cluster_mask, 'Class'].sum()),
                        'normal_count': int((X_with_class.loc[cluster_mask, 'Class'] == 0).sum())
                    }

                # Find cluster with highest fraud rate
                fraud_rates = [fraud_cluster_analysis[f'cluster_{i}']['fraud_rate'] for i in range(optimal_k)]
                highest_fraud_cluster = np.argmax(fraud_rates)

                results['fraud_clustering'] = {
                    'overall_fraud_rate': float(X_with_class['Class'].mean()),
                    'cluster_fraud_analysis': fraud_cluster_analysis,
                    'highest_fraud_cluster': int(highest_fraud_cluster),
                    'highest_fraud_rate': float(fraud_rates[highest_fraud_cluster]),
                    'clustering_separates_fraud': max(fraud_rates) > 2 * X_with_class['Class'].mean()
                }

        # Elbow method results
        results['elbow_analysis'] = {
            'k_values': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores if silhouette_scores else []
        }

        return results

    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between variables.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing correlation analysis results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        results = {}

        # High correlations (excluding self-correlations)
        high_correlations = []
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]

                if abs(corr_value) > 0.5:
                    high_correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': float(corr_value),
                        'strength': 'high'
                    })

                if abs(corr_value) > 0.8:
                    strong_correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': float(corr_value),
                        'strength': 'very_high'
                    })

        results['high_correlations'] = {
            'high_correlations_count': len(high_correlations),
            'strong_correlations_count': len(strong_correlations),
            'high_correlations': high_correlations[:20],  # Top 20
            'strong_correlations': strong_correlations
        }

        # V feature correlations specifically
        v_features = [col for col in numeric_cols if col.startswith('V')]
        if len(v_features) > 1:
            v_corr_matrix = df[v_features].corr()
            v_high_corr = []

            for i in range(len(v_corr_matrix.columns)):
                for j in range(i+1, len(v_corr_matrix.columns)):
                    corr_value = v_corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:  # Lower threshold for V features
                        v_high_corr.append({
                            'variable_1': v_corr_matrix.columns[i],
                            'variable_2': v_corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })

            results['v_feature_correlations'] = {
                'notable_v_correlations': sorted(v_high_corr, key=lambda x: abs(x['correlation']), reverse=True)[:10]
            }

        # Target correlations (if Class exists)
        if 'Class' in df.columns:
            target_correlations = corr_matrix['Class'].drop('Class').sort_values(key=abs, ascending=False)

            results['target_correlations'] = {
                'strongest_positive': {
                    'variable': target_correlations.idxmax(),
                    'correlation': float(target_correlations.max())
                },
                'strongest_negative': {
                    'variable': target_correlations.idxmin(),
                    'correlation': float(target_correlations.min())
                },
                'top_10_absolute': [
                    {'variable': var, 'correlation': float(corr)}
                    for var, corr in target_correlations.head(10).items()
                ]
            }

        # Correlation summary statistics
        corr_values = corr_matrix.values
        # Remove diagonal (self-correlations)
        corr_values = corr_values[np.triu_indices_from(corr_values, k=1)]

        results['correlation_summary'] = {
            'mean_correlation': float(np.mean(np.abs(corr_values))),
            'max_correlation': float(np.max(np.abs(corr_values))),
            'correlations_over_05': int(np.sum(np.abs(corr_values) > 0.5)),
            'correlations_over_08': int(np.sum(np.abs(corr_values) > 0.8)),
            'total_correlation_pairs': len(corr_values)
        }

        return results

    def identify_fraud_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify fraud patterns by analyzing Class vs other variables.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing fraud pattern analysis
        """
        if 'Class' not in df.columns:
            return {'error': 'Class column not found for fraud pattern analysis'}

        fraud_df = df[df['Class'] == 1]
        normal_df = df[df['Class'] == 0]

        if len(fraud_df) == 0 or len(normal_df) == 0:
            return {'error': 'Need both fraud and normal transactions for comparison'}

        results = {}

        # Basic fraud statistics
        results['basic_fraud_stats'] = {
            'total_transactions': len(df),
            'fraud_count': len(fraud_df),
            'normal_count': len(normal_df),
            'fraud_rate': float(len(fraud_df) / len(df)),
            'imbalance_ratio': float(len(normal_df) / len(fraud_df)) if len(fraud_df) > 0 else float('inf')
        }

        # Amount patterns
        if 'Amount' in df.columns:
            fraud_amounts = fraud_df['Amount']
            normal_amounts = normal_df['Amount']

            # Statistical tests
            try:
                # Mann-Whitney U test (non-parametric)
                mw_stat, mw_pvalue = stats.mannwhitneyu(fraud_amounts, normal_amounts, alternative='two-sided')

                results['amount_patterns'] = {
                    'fraud_amount_stats': {
                        'mean': float(fraud_amounts.mean()),
                        'median': float(fraud_amounts.median()),
                        'std': float(fraud_amounts.std()),
                        'min': float(fraud_amounts.min()),
                        'max': float(fraud_amounts.max())
                    },
                    'normal_amount_stats': {
                        'mean': float(normal_amounts.mean()),
                        'median': float(normal_amounts.median()),
                        'std': float(normal_amounts.std()),
                        'min': float(normal_amounts.min()),
                        'max': float(normal_amounts.max())
                    },
                    'statistical_difference': {
                        'mann_whitney_statistic': float(mw_stat),
                        'mann_whitney_pvalue': float(mw_pvalue),
                        'significantly_different': mw_pvalue < 0.05
                    },
                    'zero_amount_analysis': {
                        'fraud_zero_amounts': int((fraud_amounts == 0).sum()),
                        'normal_zero_amounts': int((normal_amounts == 0).sum()),
                        'fraud_zero_percentage': float((fraud_amounts == 0).mean() * 100),
                        'normal_zero_percentage': float((normal_amounts == 0).mean() * 100)
                    }
                }
            except Exception as e:
                results['amount_patterns'] = {'error': f'Could not perform statistical analysis: {str(e)}'}

        # Time patterns
        if 'Time' in df.columns:
            fraud_times = fraud_df['Time']
            normal_times = normal_df['Time']

            try:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(fraud_times, normal_times)

                results['time_patterns'] = {
                    'fraud_time_stats': {
                        'mean': float(fraud_times.mean()),
                        'median': float(fraud_times.median()),
                        'std': float(fraud_times.std()),
                        'min': float(fraud_times.min()),
                        'max': float(fraud_times.max())
                    },
                    'normal_time_stats': {
                        'mean': float(normal_times.mean()),
                        'median': float(normal_times.median()),
                        'std': float(normal_times.std()),
                        'min': float(normal_times.min()),
                        'max': float(normal_times.max())
                    },
                    'statistical_difference': {
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'significantly_different': ks_pvalue < 0.05
                    }
                }
            except Exception as e:
                results['time_patterns'] = {'error': f'Could not perform statistical analysis: {str(e)}'}

        # V feature patterns
        v_features = [col for col in df.columns if col.startswith('V')]
        if v_features:
            v_feature_analysis = {}

            for feature in v_features[:10]:  # Analyze first 10 V features
                fraud_feature = fraud_df[feature]
                normal_feature = normal_df[feature]

                try:
                    # T-test for means
                    t_stat, t_pvalue = stats.ttest_ind(fraud_feature, normal_feature)

                    v_feature_analysis[feature] = {
                        'fraud_mean': float(fraud_feature.mean()),
                        'normal_mean': float(normal_feature.mean()),
                        'fraud_std': float(fraud_feature.std()),
                        'normal_std': float(normal_feature.std()),
                        'mean_difference': float(fraud_feature.mean() - normal_feature.mean()),
                        't_statistic': float(t_stat),
                        't_pvalue': float(t_pvalue),
                        'significantly_different': t_pvalue < 0.05
                    }
                except:
                    v_feature_analysis[feature] = {'error': 'Could not perform t-test'}

            # Find features with largest differences
            significant_features = []
            for feature, analysis in v_feature_analysis.items():
                if 'significantly_different' in analysis and analysis['significantly_different']:
                    significant_features.append({
                        'feature': feature,
                        'mean_difference': analysis['mean_difference'],
                        'p_value': analysis['t_pvalue']
                    })

            significant_features.sort(key=lambda x: abs(x['mean_difference']), reverse=True)

            results['v_feature_patterns'] = {
                'analyzed_features': list(v_feature_analysis.keys()),
                'significantly_different_features': significant_features[:5],
                'feature_analysis': v_feature_analysis
            }

        return results

    def _generate_pattern_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all pattern findings."""
        summary = {
            'analysis_completed': True,
            'modules_analyzed': list(analysis_results.keys()),
            'key_findings': []
        }

        # Temporal findings
        if 'temporal_patterns' in analysis_results:
            temporal = analysis_results['temporal_patterns']
            if 'fraud_timing' in temporal and temporal['fraud_timing'].get('significantly_different'):
                summary['key_findings'].append('Fraud transactions have significantly different timing patterns')

        # Outlier findings
        if 'outlier_analysis' in analysis_results:
            outlier = analysis_results['outlier_analysis']
            if 'summary' in outlier:
                total_outliers = outlier['summary']['total_outliers_iqr_method']
                if total_outliers > 0:
                    summary['key_findings'].append(f'Detected {total_outliers} outliers across all features')

        # Clustering findings
        if 'clustering_analysis' in analysis_results:
            clustering = analysis_results['clustering_analysis']
            if 'fraud_clustering' in clustering and clustering['fraud_clustering'].get('clustering_separates_fraud'):
                summary['key_findings'].append('Clustering effectively separates fraud from normal transactions')

        # Correlation findings
        if 'correlation_analysis' in analysis_results:
            corr = analysis_results['correlation_analysis']
            if 'high_correlations' in corr:
                high_count = corr['high_correlations']['high_correlations_count']
                if high_count > 0:
                    summary['key_findings'].append(f'Found {high_count} high correlations between variables')

        # Fraud pattern findings
        if 'fraud_patterns' in analysis_results:
            fraud = analysis_results['fraud_patterns']
            if 'basic_fraud_stats' in fraud:
                fraud_rate = fraud['basic_fraud_stats']['fraud_rate']
                summary['key_findings'].append(f'Dataset has {fraud_rate:.2%} fraud rate')

        if not summary['key_findings']:
            summary['key_findings'].append('No significant patterns detected in the analysis')

        return summary

    def export_to_json(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Export analysis results to JSON file."""
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
        json_results = convert_numpy_types(analysis_results)

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)


def analyze_patterns_for_csv(file_path: str, output_path: Optional[str] = None,
                            max_samples: int = 50000) -> Dict[str, Any]:
    """
    Convenience function to analyze patterns for a CSV file.

    Args:
        file_path: Path to CSV file
        output_path: Optional path to save JSON output
        max_samples: Maximum samples to use for analysis

    Returns:
        Dictionary containing all pattern analysis results
    """
    # Load data
    df = pd.read_csv(file_path)

    # Analyze patterns
    analyzer = PatternAnalyzer(max_samples=max_samples)
    results = analyzer.analyze_all_patterns(df)

    # Save results if output path specified
    if output_path:
        analyzer.export_to_json(results, output_path)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pattern_analyzer.py <csv_file_path> [output_json_path]")
        sys.exit(1)

    csv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        results = analyze_patterns_for_csv(csv_path, json_path)

        print(f"Pattern analysis completed for {csv_path}")
        print(f"Dataset size: {results['dataset_info']['original_size']} rows")

        if results['dataset_info']['sampling_applied']:
            print(f"Sampling applied: Analyzed {results['dataset_info']['analyzed_size']} samples")

        print("Analysis modules completed:")
        for module in results.get('pattern_summary', {}).get('modules_analyzed', []):
            print(f"  - {module}")

        print("Key findings:")
        for finding in results.get('pattern_summary', {}).get('key_findings', []):
            print(f"  - {finding}")

        if json_path:
            print(f"Results saved to: {json_path}")

    except Exception as e:
        print(f"Error analyzing patterns: {e}")
        sys.exit(1)