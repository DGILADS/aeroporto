import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict

class CorrelationAnalyzer:
    """
    Service for correlation analysis between different airport operational metrics.
    """

    @staticmethod
    def calculate_correlations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate Pearson, Spearman, and Kendall correlations."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            "pearson": numeric_df.corr(method='pearson'),
            "spearman": numeric_df.corr(method='spearman'),
            "kendall": numeric_df.corr(method='kendall')
        }

    @staticmethod
    def plot_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Heatmap"):
        """Generate a heatmap for visually explaining correlations."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(title)
        return plt.gcf()

    @staticmethod
    def plot_pairplot(df: pd.DataFrame):
        """Generate a pairplot for overall distribution and correlation visualization."""
        return sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6})
