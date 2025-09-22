"""
Statistics Display Interface for AI Analytics Chatbot

This module implements comprehensive statistics display functionality with expandable sections,
interactive charts, data overview, and download options for generated reports.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import io
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StatsDisplay:
    """
    Comprehensive statistics display interface with expandable sections,
    interactive charts, and download functionality.
    """

    def __init__(self):
        """Initialize the StatsDisplay component."""
        self.charts_cache = {}

    def display_data_overview(self, stats_data: Dict[str, Any]) -> None:
        """
        Display data overview including shape, types, and missing values.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'file_info' not in stats_data:
            st.warning("No data overview available")
            return

        file_info = stats_data['file_info']

        with st.expander("📊 Data Overview", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Rows", f"{file_info.get('rows', 0):,}")
                st.metric("Total Columns", file_info.get('columns', 0))

            with col2:
                if 'memory_usage_mb' in file_info:
                    st.metric("Memory Usage", f"{file_info['memory_usage_mb']:.1f} MB")
                if 'file_size_mb' in file_info:
                    st.metric("File Size", f"{file_info['file_size_mb']:.1f} MB")

            with col3:
                missing_info = stats_data.get('missing_values', {})
                total_missing = missing_info.get('total_missing_values', 0)
                missing_percent = missing_info.get('missing_percentage', 0)
                st.metric("Missing Values", f"{total_missing:,}")
                st.metric("Missing %", f"{missing_percent:.2f}%")

    def display_data_types(self, stats_data: Dict[str, Any]) -> None:
        """
        Display data types analysis in an expandable section.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'data_types' not in stats_data:
            return

        data_types = stats_data['data_types']

        with st.expander("🔍 Data Types Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Numeric Columns")
                numeric_cols = data_types.get('numeric_columns', [])
                if numeric_cols:
                    for col in numeric_cols:
                        st.write(f"• {col}")
                else:
                    st.write("No numeric columns found")

            with col2:
                st.subheader("Categorical Columns")
                categorical_cols = data_types.get('categorical_columns', [])
                if categorical_cols:
                    for col in categorical_cols:
                        st.write(f"• {col}")
                else:
                    st.write("No categorical columns found")

            # Data type summary
            type_summary = data_types.get('type_summary', {})
            if type_summary:
                st.subheader("Type Distribution")
                type_df = pd.DataFrame(list(type_summary.items()),
                                     columns=['Data Type', 'Count'])
                st.dataframe(type_df, use_container_width=True)

    def display_basic_statistics(self, stats_data: Dict[str, Any]) -> None:
        """
        Display basic statistics tables in expandable sections.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'basic_statistics' not in stats_data:
            return

        basic_stats = stats_data['basic_statistics']

        with st.expander("📈 Basic Statistics"):
            # Numeric statistics
            if 'numeric_stats' in basic_stats:
                st.subheader("Numeric Variables Summary")
                numeric_df = pd.DataFrame(basic_stats['numeric_stats']).T
                if not numeric_df.empty:
                    # Format numbers for better readability
                    numeric_df = numeric_df.round(4)
                    st.dataframe(numeric_df, use_container_width=True)
                else:
                    st.write("No numeric statistics available")

            # Categorical statistics
            if 'categorical_stats' in basic_stats:
                st.subheader("Categorical Variables Summary")
                cat_stats = basic_stats['categorical_stats']
                if cat_stats:
                    for col, stats in cat_stats.items():
                        with st.container():
                            st.write(f"**{col}:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Unique Values", stats.get('unique_count', 0))
                            with col2:
                                st.metric("Most Frequent", stats.get('top_value', 'N/A'))
                            with col3:
                                st.metric("Frequency", stats.get('top_frequency', 0))
                else:
                    st.write("No categorical statistics available")

    def display_distribution_analysis(self, stats_data: Dict[str, Any]) -> None:
        """
        Display distribution analysis with interactive charts.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'distribution_analysis' not in stats_data:
            return

        dist_data = stats_data['distribution_analysis']

        with st.expander("📊 Distribution Analysis"):
            # Histogram data
            if 'histograms' in dist_data:
                st.subheader("Variable Distributions")
                histograms = dist_data['histograms']

                # Create tabs for different variable groups
                if histograms:
                    # Group variables
                    pca_vars = [k for k in histograms.keys() if k.startswith('V')]
                    other_vars = [k for k in histograms.keys() if not k.startswith('V')]

                    if pca_vars or other_vars:
                        tabs = []
                        if other_vars:
                            tabs.append("Key Variables")
                        if pca_vars:
                            tabs.append("PCA Components")

                        tab_objects = st.tabs(tabs)
                        tab_idx = 0

                        if other_vars:
                            with tab_objects[tab_idx]:
                                self._display_histogram_charts(other_vars, histograms)
                            tab_idx += 1

                        if pca_vars:
                            with tab_objects[tab_idx]:
                                self._display_histogram_charts(pca_vars[:12], histograms)  # Limit to first 12

            # Value counts for categorical variables
            if 'value_counts' in dist_data:
                st.subheader("Categorical Variable Distributions")
                value_counts = dist_data['value_counts']
                for col, counts in value_counts.items():
                    if counts:
                        st.write(f"**{col} Distribution:**")
                        counts_df = pd.DataFrame(list(counts.items()),
                                               columns=['Value', 'Count'])
                        fig = px.bar(counts_df, x='Value', y='Count',
                                   title=f"{col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)

    def _display_histogram_charts(self, variables: List[str], histograms: Dict[str, Any]) -> None:
        """Display histogram charts for a list of variables."""
        # Display in grid format
        cols_per_row = 2
        for i in range(0, len(variables), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                var_idx = i + j
                if var_idx < len(variables):
                    var_name = variables[var_idx]
                    if var_name in histograms:
                        with col:
                            self._create_histogram_chart(var_name, histograms[var_name])

    def _create_histogram_chart(self, var_name: str, hist_data: Dict[str, Any]) -> None:
        """Create a histogram chart for a variable."""
        try:
            bins = hist_data.get('bins', [])
            counts = hist_data.get('counts', [])

            if bins and counts and len(bins) == len(counts) + 1:
                # Create bin centers for plotting
                bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

                fig = go.Figure(data=go.Bar(x=bin_centers, y=counts, name=var_name))
                fig.update_layout(
                    title=f"{var_name} Distribution",
                    xaxis_title=var_name,
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"Unable to display histogram for {var_name}")
        except Exception as e:
            st.write(f"Error displaying {var_name}: {str(e)}")

    def display_correlation_analysis(self, stats_data: Dict[str, Any]) -> None:
        """
        Display correlation analysis with interactive heatmap.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'correlation_analysis' not in stats_data:
            return

        corr_data = stats_data['correlation_analysis']

        with st.expander("🔗 Correlation Analysis"):
            if 'correlation_matrix' in corr_data:
                corr_matrix = corr_data['correlation_matrix']
                if corr_matrix:
                    st.subheader("Correlation Heatmap")

                    # Convert to DataFrame for plotting
                    corr_df = pd.DataFrame(corr_matrix)

                    # Create interactive heatmap
                    fig = px.imshow(corr_df,
                                  title="Variable Correlation Matrix",
                                  color_continuous_scale="RdBu_r",
                                  aspect="auto")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

            # High correlations table
            if 'high_correlations' in corr_data:
                high_corrs = corr_data['high_correlations']
                if high_corrs:
                    st.subheader("High Correlations (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(high_corrs)
                    if not high_corr_df.empty:
                        st.dataframe(high_corr_df, use_container_width=True)
                    else:
                        st.write("No high correlations found")

    def display_fraud_analysis(self, stats_data: Dict[str, Any]) -> None:
        """
        Display fraud-specific analysis.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'fraud_specific_analysis' not in stats_data:
            return

        fraud_data = stats_data['fraud_specific_analysis']

        with st.expander("🚨 Fraud Detection Analysis"):
            # Class distribution
            if 'class_distribution' in fraud_data:
                class_dist = fraud_data['class_distribution']
                st.subheader("Fraud vs Normal Transactions")

                col1, col2 = st.columns(2)
                with col1:
                    if isinstance(class_dist, dict):
                        normal_count = class_dist.get(0, 0)
                        fraud_count = class_dist.get(1, 0)
                        total = normal_count + fraud_count

                        st.metric("Normal Transactions", f"{normal_count:,}")
                        st.metric("Fraud Transactions", f"{fraud_count:,}")
                        if total > 0:
                            fraud_rate = (fraud_count / total) * 100
                            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")

                with col2:
                    # Create pie chart
                    if isinstance(class_dist, dict) and class_dist:
                        labels = ['Normal', 'Fraud']
                        values = [class_dist.get(0, 0), class_dist.get(1, 0)]
                        fig = px.pie(values=values, names=labels,
                                   title="Transaction Distribution")
                        st.plotly_chart(fig, use_container_width=True)

            # Time analysis
            if 'time_analysis' in fraud_data:
                time_analysis = fraud_data['time_analysis']
                st.subheader("Temporal Patterns")
                for key, value in time_analysis.items():
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")

            # Amount analysis
            if 'amount_analysis' in fraud_data:
                amount_analysis = fraud_data['amount_analysis']
                st.subheader("Transaction Amount Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Normal Transactions:**")
                    normal_stats = amount_analysis.get('normal_transactions', {})
                    for stat, value in normal_stats.items():
                        if isinstance(value, (int, float)):
                            st.metric(stat.replace('_', ' ').title(), f"${value:.2f}")

                with col2:
                    st.write("**Fraud Transactions:**")
                    fraud_stats = amount_analysis.get('fraud_transactions', {})
                    for stat, value in fraud_stats.items():
                        if isinstance(value, (int, float)):
                            st.metric(stat.replace('_', ' ').title(), f"${value:.2f}")

    def display_missing_values_analysis(self, stats_data: Dict[str, Any]) -> None:
        """
        Display missing values analysis.

        Args:
            stats_data: Statistics dictionary from CSV processor
        """
        if not stats_data or 'missing_values' not in stats_data:
            return

        missing_data = stats_data['missing_values']

        with st.expander("❓ Missing Values Analysis"):
            # Overall missing values summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Missing", missing_data.get('total_missing_values', 0))
                st.metric("Missing %", f"{missing_data.get('missing_percentage', 0):.2f}%")

            # Per-column missing values
            if 'missing_by_column' in missing_data:
                missing_by_col = missing_data['missing_by_column']
                if missing_by_col:
                    st.subheader("Missing Values by Column")
                    missing_df = pd.DataFrame(list(missing_by_col.items()),
                                            columns=['Column', 'Missing Count'])
                    missing_df['Missing %'] = (missing_df['Missing Count'] /
                                             stats_data['file_info']['rows'] * 100).round(2)
                    missing_df = missing_df[missing_df['Missing Count'] > 0]

                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("No missing values found in any column!")

    def create_download_options(self, stats_data: Dict[str, Any],
                              analysis_report: Optional[str] = None) -> None:
        """
        Create download options for statistics and reports.

        Args:
            stats_data: Statistics dictionary from CSV processor
            analysis_report: Optional markdown analysis report
        """
        with st.expander("💾 Download Options"):
            col1, col2, col3 = st.columns(3)

            with col1:
                # Download raw statistics as JSON
                if stats_data:
                    json_str = json.dumps(stats_data, indent=2, default=str)
                    st.download_button(
                        label="📄 Download Statistics (JSON)",
                        data=json_str,
                        file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

            with col2:
                # Download analysis report as markdown
                if analysis_report:
                    st.download_button(
                        label="📝 Download Report (MD)",
                        data=analysis_report,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

            with col3:
                # Download summary as CSV
                if stats_data and 'basic_statistics' in stats_data:
                    summary_data = self._create_summary_csv(stats_data)
                    st.download_button(
                        label="📊 Download Summary (CSV)",
                        data=summary_data,
                        file_name=f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    def _create_summary_csv(self, stats_data: Dict[str, Any]) -> str:
        """Create a summary CSV from statistics data."""
        try:
            summary_rows = []

            # File info
            file_info = stats_data.get('file_info', {})
            summary_rows.append(['Metric', 'Value'])
            summary_rows.append(['Total Rows', file_info.get('rows', 0)])
            summary_rows.append(['Total Columns', file_info.get('columns', 0)])
            summary_rows.append(['File Size (MB)', file_info.get('file_size_mb', 0)])

            # Missing values
            missing_info = stats_data.get('missing_values', {})
            summary_rows.append(['Total Missing Values', missing_info.get('total_missing_values', 0)])
            summary_rows.append(['Missing Percentage', missing_info.get('missing_percentage', 0)])

            # Fraud analysis if available
            fraud_info = stats_data.get('fraud_specific_analysis', {})
            if 'class_distribution' in fraud_info:
                class_dist = fraud_info['class_distribution']
                if isinstance(class_dist, dict):
                    summary_rows.append(['Normal Transactions', class_dist.get(0, 0)])
                    summary_rows.append(['Fraud Transactions', class_dist.get(1, 0)])

            # Convert to CSV string
            csv_buffer = io.StringIO()
            for row in summary_rows:
                csv_buffer.write(','.join(str(item) for item in row) + '\n')

            return csv_buffer.getvalue()
        except Exception:
            return "Error,Unable to generate summary CSV\n"

    def display_complete_statistics(self, stats_data: Dict[str, Any],
                                  analysis_report: Optional[str] = None) -> None:
        """
        Display all statistics sections in a comprehensive layout.

        Args:
            stats_data: Statistics dictionary from CSV processor
            analysis_report: Optional markdown analysis report
        """
        if not stats_data:
            st.warning("No statistics data available to display")
            return

        st.header("📊 Data Analysis Dashboard")

        # Display all sections
        self.display_data_overview(stats_data)
        self.display_data_types(stats_data)
        self.display_basic_statistics(stats_data)
        self.display_distribution_analysis(stats_data)
        self.display_correlation_analysis(stats_data)
        self.display_fraud_analysis(stats_data)
        self.display_missing_values_analysis(stats_data)
        self.create_download_options(stats_data, analysis_report)


# Streamlit app for testing (only runs when called directly)
if __name__ == "__main__":
    st.title("Statistics Display Component Test")

    # Sample data for testing
    sample_stats = {
        'file_info': {
            'rows': 284807,
            'columns': 31,
            'file_size_mb': 150.2,
            'memory_usage_mb': 67.8
        },
        'data_types': {
            'numeric_columns': ['Time', 'V1', 'V2', 'Amount', 'Class'],
            'categorical_columns': [],
            'type_summary': {'float64': 29, 'int64': 2}
        },
        'basic_statistics': {
            'numeric_stats': {
                'Time': {'count': 284807, 'mean': 94813.9, 'std': 47488.1},
                'Amount': {'count': 284807, 'mean': 88.35, 'std': 250.12}
            }
        },
        'missing_values': {
            'total_missing_values': 0,
            'missing_percentage': 0.0,
            'missing_by_column': {}
        },
        'fraud_specific_analysis': {
            'class_distribution': {0: 284315, 1: 492},
            'time_analysis': {'fraud_peak_hour': 12.5},
            'amount_analysis': {
                'normal_transactions': {'mean': 88.3, 'std': 250.1},
                'fraud_transactions': {'mean': 122.2, 'std': 256.7}
            }
        }
    }

    # Create and display stats
    stats_display = StatsDisplay()
    stats_display.display_complete_statistics(sample_stats)