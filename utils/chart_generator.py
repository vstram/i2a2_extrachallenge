"""
Chart Generator for Large CSV Files

This module generates various types of charts and visualizations for fraud detection
analysis, optimized for large datasets. All charts are encoded as base64 strings
for JSON embedding and easy integration with web interfaces.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('default')


class ChartGenerator:
    """
    Generates various charts and visualizations for fraud detection analysis.

    Optimized for large datasets with sampling capabilities and base64 encoding
    for JSON embedding.
    """

    def __init__(self, max_samples: int = 10000, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the chart generator.

        Args:
            max_samples: Maximum number of samples to use for large datasets
            figsize: Default figure size for matplotlib charts
        """
        self.max_samples = max_samples
        self.figsize = figsize
        self.charts = {}

    def generate_all_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all charts for the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with chart names as keys and base64 encoded images as values
        """
        # Sample data if too large
        sampled_df = self._sample_data(df)

        charts = {}

        # Generate histograms for numerical variables
        hist_charts = self.generate_histograms(sampled_df)
        charts.update(hist_charts)

        # Generate correlation heatmap
        corr_chart = self.generate_correlation_heatmap(sampled_df)
        if corr_chart:
            charts['correlation_heatmap'] = corr_chart

        # Generate scatter plots for key relationships
        scatter_charts = self.generate_scatter_plots(sampled_df)
        charts.update(scatter_charts)

        # Generate fraud class distribution
        if 'Class' in df.columns:
            class_charts = self.generate_fraud_distribution_charts(sampled_df)
            charts.update(class_charts)

        return charts

    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample data if dataset is too large."""
        if len(df) > self.max_samples:
            # Stratified sampling to preserve fraud ratio if Class column exists
            if 'Class' in df.columns:
                fraud_df = df[df['Class'] == 1]
                normal_df = df[df['Class'] == 0]

                fraud_sample_size = min(len(fraud_df), self.max_samples // 10)  # ~10% fraud
                normal_sample_size = min(len(normal_df), self.max_samples - fraud_sample_size)

                fraud_sample = fraud_df.sample(n=fraud_sample_size, random_state=42)
                normal_sample = normal_df.sample(n=normal_sample_size, random_state=42)

                return pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                return df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
        return df

    def generate_histograms(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate histogram plots for numerical variables.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with histogram chart names and base64 encoded images
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        charts = {}

        # Create individual histograms for key variables
        key_variables = ['Time', 'Amount'] + [col for col in numeric_cols if col.startswith('V')]

        for col in key_variables:
            if col in df.columns:
                charts[f'histogram_{col.lower()}'] = self._create_histogram(df, col)

        # Create a combined histogram grid for V1-V28
        v_columns = [col for col in df.columns if col.startswith('V')]
        if len(v_columns) > 0:
            charts['histogram_v_features_grid'] = self._create_histogram_grid(df, v_columns[:16])  # First 16 V features

        return charts

    def _create_histogram(self, df: pd.DataFrame, column: str) -> str:
        """Create a single histogram and return as base64 string."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create histogram with different colors for fraud vs normal if Class exists
        if 'Class' in df.columns and column != 'Class':
            fraud_data = df[df['Class'] == 1][column].dropna()
            normal_data = df[df['Class'] == 0][column].dropna()

            ax.hist(normal_data, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
            ax.hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
            ax.legend()
            ax.set_title(f'Distribution of {column} by Class')
        else:
            ax.hist(df[column].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {column}')

        ax.set_xlabel(column)
        ax.set_ylabel('Density' if 'Class' in df.columns else 'Frequency')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_histogram_grid(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Create a grid of histograms for multiple variables."""
        n_cols = 4
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                ax.hist(df[col].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                ax.set_title(f'{col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_correlation_heatmap(self, df: pd.DataFrame) -> Optional[str]:
        """
        Generate correlation heatmap for all numerical variables.

        Args:
            df: Input DataFrame

        Returns:
            Base64 encoded correlation heatmap image
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return None

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use a mask to show only lower triangle for better readability if many variables
        if len(corr_matrix.columns) > 10:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={"shrink": .8})
        else:
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={"shrink": .8})

        ax.set_title('Correlation Heatmap of Numerical Variables')
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def generate_scatter_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate scatter plots for key variable relationships.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with scatter plot names and base64 encoded images
        """
        charts = {}

        # Key relationships to visualize
        relationships = []

        # Amount vs Time
        if 'Amount' in df.columns and 'Time' in df.columns:
            relationships.append(('Time', 'Amount'))

        # Find high correlation pairs among V features
        v_columns = [col for col in df.columns if col.startswith('V')]
        if len(v_columns) >= 2:
            # Add a few interesting V feature combinations
            if 'V1' in df.columns and 'V2' in df.columns:
                relationships.append(('V1', 'V2'))
            if 'V3' in df.columns and 'V4' in df.columns:
                relationships.append(('V3', 'V4'))

        # Generate scatter plots
        for x_var, y_var in relationships:
            if x_var in df.columns and y_var in df.columns:
                chart_name = f'scatter_{x_var.lower()}_{y_var.lower()}'
                charts[chart_name] = self._create_scatter_plot(df, x_var, y_var)

        return charts

    def _create_scatter_plot(self, df: pd.DataFrame, x_var: str, y_var: str) -> str:
        """Create a scatter plot and return as base64 string."""
        fig, ax = plt.subplots(figsize=self.figsize)

        if 'Class' in df.columns:
            # Color by fraud class
            fraud_mask = df['Class'] == 1
            normal_mask = df['Class'] == 0

            ax.scatter(df.loc[normal_mask, x_var], df.loc[normal_mask, y_var],
                      alpha=0.6, label='Normal', c='blue', s=20)
            ax.scatter(df.loc[fraud_mask, x_var], df.loc[fraud_mask, y_var],
                      alpha=0.8, label='Fraud', c='red', s=20)
            ax.legend()
        else:
            ax.scatter(df[x_var], df[y_var], alpha=0.6, c='skyblue', s=20)

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f'{y_var} vs {x_var}')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def generate_fraud_distribution_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate fraud class distribution charts.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with fraud distribution chart names and base64 encoded images
        """
        charts = {}

        if 'Class' not in df.columns:
            return charts

        # Class distribution pie chart
        charts['fraud_class_distribution'] = self._create_class_pie_chart(df)

        # Class distribution bar chart
        charts['fraud_class_bar_chart'] = self._create_class_bar_chart(df)

        # Amount distribution by class
        if 'Amount' in df.columns:
            charts['amount_by_class_boxplot'] = self._create_amount_by_class_boxplot(df)

        # Time distribution by class
        if 'Time' in df.columns:
            charts['time_by_class_histogram'] = self._create_time_by_class_histogram(df)

        return charts

    def _create_class_pie_chart(self, df: pd.DataFrame) -> str:
        """Create fraud class distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))

        class_counts = df['Class'].value_counts()
        labels = ['Normal', 'Fraud']
        colors = ['lightblue', 'lightcoral']

        wedges, texts, autotexts = ax.pie(class_counts.values, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)

        ax.set_title('Fraud vs Normal Transaction Distribution')

        return self._fig_to_base64(fig)

    def _create_class_bar_chart(self, df: pd.DataFrame) -> str:
        """Create fraud class distribution bar chart."""
        fig, ax = plt.subplots(figsize=self.figsize)

        class_counts = df['Class'].value_counts()
        labels = ['Normal', 'Fraud']
        colors = ['skyblue', 'salmon']

        bars = ax.bar(labels, class_counts.values, color=colors)
        ax.set_title('Fraud vs Normal Transaction Counts')
        ax.set_ylabel('Count')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')

        return self._fig_to_base64(fig)

    def _create_amount_by_class_boxplot(self, df: pd.DataFrame) -> str:
        """Create amount distribution by class boxplot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Filter out extreme outliers for better visualization
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['Amount'] >= lower_bound) & (df['Amount'] <= upper_bound)]

        # Create boxplot
        fraud_amounts = filtered_df[filtered_df['Class'] == 1]['Amount']
        normal_amounts = filtered_df[filtered_df['Class'] == 0]['Amount']

        box_data = [normal_amounts, fraud_amounts]
        box = ax.boxplot(box_data, labels=['Normal', 'Fraud'], patch_artist=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title('Transaction Amount Distribution by Class')
        ax.set_ylabel('Amount')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_time_by_class_histogram(self, df: pd.DataFrame) -> str:
        """Create time distribution by class histogram."""
        fig, ax = plt.subplots(figsize=self.figsize)

        fraud_time = df[df['Class'] == 1]['Time']
        normal_time = df[df['Class'] == 0]['Time']

        ax.hist(normal_time, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        ax.hist(fraud_time, bins=50, alpha=0.7, label='Fraud', color='red', density=True)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Density')
        ax.set_title('Transaction Time Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)  # Free memory
        buffer.close()
        return image_base64

    def save_charts_to_files(self, charts: Dict[str, str], output_dir: str) -> None:
        """
        Save base64 encoded charts as PNG files.

        Args:
            charts: Dictionary of chart names and base64 encoded images
            output_dir: Directory to save the images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for chart_name, base64_image in charts.items():
            # Decode base64 and save as PNG
            image_data = base64.b64decode(base64_image)

            with open(output_path / f"{chart_name}.png", "wb") as f:
                f.write(image_data)

    def generate_charts_with_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate charts with additional metadata.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing charts and metadata
        """
        charts = self.generate_all_charts(df)

        metadata = {
            'total_charts_generated': len(charts),
            'original_dataset_size': len(df),
            'sampled_dataset_size': min(len(df), self.max_samples),
            'sampling_applied': len(df) > self.max_samples,
            'chart_names': list(charts.keys()),
            'generation_parameters': {
                'max_samples': self.max_samples,
                'figsize': self.figsize
            }
        }

        return {
            'charts': charts,
            'metadata': metadata
        }


def generate_charts_for_csv(file_path: str, output_dir: Optional[str] = None,
                           max_samples: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to generate charts for a CSV file.

    Args:
        file_path: Path to CSV file
        output_dir: Optional directory to save chart images
        max_samples: Maximum samples to use for large datasets

    Returns:
        Dictionary containing charts and metadata
    """
    # Load data
    df = pd.read_csv(file_path)

    # Generate charts
    generator = ChartGenerator(max_samples=max_samples)
    result = generator.generate_charts_with_metadata(df)

    # Save charts if output directory specified
    if output_dir:
        generator.save_charts_to_files(result['charts'], output_dir)
        result['metadata']['charts_saved_to'] = output_dir

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chart_generator.py <csv_file_path> [output_directory]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = generate_charts_for_csv(csv_path, output_dir)

        print(f"Successfully generated {result['metadata']['total_charts_generated']} charts")
        print(f"Dataset size: {result['metadata']['original_dataset_size']} rows")

        if result['metadata']['sampling_applied']:
            print(f"Sampling applied: Using {result['metadata']['sampled_dataset_size']} samples")

        print("Generated charts:")
        for chart_name in result['metadata']['chart_names']:
            print(f"  - {chart_name}")

        if output_dir:
            print(f"Charts saved to: {output_dir}")

    except Exception as e:
        print(f"Error generating charts: {e}")
        sys.exit(1)