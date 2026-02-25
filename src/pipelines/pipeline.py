import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import os

class DataPipeline:
    """
    Data Engineering pipeline for airport operations data.
    
    Handles ETL, data cleaning, normalization, and outlier detection.
    """

    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tratamento de valores ausentes e duplicatas."""
        df = df.drop_duplicates()
        df = df.fillna(df.median(numeric_only=True))
        return df

    def normalize(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalização de dados (MinMax, ZScore, MaxAbs)."""
        df_numeric = df.select_dtypes(include=[np.number])
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_data = scaler.fit_transform(df_numeric)
        df_normalized = pd.DataFrame(normalized_data, columns=df_numeric.columns, index=df.index)
        
        # Merge back with non-numeric data
        for col in df.columns:
            if col not in df_numeric.columns:
                df_normalized[col] = df[col]
                
        return df_normalized

    def detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Identify outliers using Z-score."""
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        return df[(z_scores < threshold).all(axis=1)]

    def detect_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify outliers using IQR."""
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Geração de relatórios automáticos."""
        report = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": df.describe().to_dict()
        }
        return report

    def save_data(self, df: pd.DataFrame, filename: str, is_raw: bool = False):
        """Save data to csv files."""
        base_path = self.raw_path if is_raw else self.processed_path
        path = os.path.join(base_path, filename)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")
