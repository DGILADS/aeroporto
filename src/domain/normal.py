import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Any
from .base import DistributionModel

class NormalDistribution(DistributionModel):
    """
    Normal Distribution model for Check-in times.
    
    Used to model operations where most outcomes cluster around a mean,
    such as the time it takes for a passenger to complete check-in.
    """

    def __init__(self, mean: float = 0.0, std_dev: float = 1.0):
        self.mean = mean
        self.std_dev = std_dev

    def simulate(self, size: int) -> np.ndarray:
        """Simulate check-in times using the normal distribution."""
        return np.random.normal(self.mean, self.std_dev, size)

    def fit(self, data: np.ndarray) -> None:
        """Fit mean and standard deviation to the observed data."""
        self.mean = np.mean(data)
        self.std_dev = np.std(data)

    def probability(self, x: float) -> float:
        """Calculate the Probability Density Function (PDF) at x."""
        return stats.norm.pdf(x, self.mean, self.std_dev)

    def expected_value(self) -> float:
        """Return the mean of the distribution."""
        return self.mean

    def variance(self) -> float:
        """Return the variance (square of std deviation)."""
        return self.std_dev ** 2

    def z_score(self, x: float) -> float:
        """Calculate the Z-score for a given value."""
        return (x - self.mean) / self.std_dev

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        """Generate Histogram, Normal Curve, and Boxplot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot: Histogram + PDF
        if data is not None:
            sns.histplot(data, kde=True, ax=ax1, stat="density", label="Simulated Data", color="skyblue")
            
            # Theoretical curve
            x = np.linspace(min(data), max(data), 100)
            y = self.probability(x)
            ax1.plot(x, y, 'r-', lw=2, label="Theoretical Normal")
        
        ax1.set_title(f"Normal Distribution (μ={self.mean:.2f}, σ={self.std_dev:.2f})")
        ax1.legend()
        
        # Boxplot for outliers
        if data is not None:
            sns.boxplot(x=data, ax=ax2, color="lightgreen")
            ax2.set_xlabel("Values")
        
        plt.tight_layout()
        return fig
