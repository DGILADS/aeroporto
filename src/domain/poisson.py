import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Any
from .base import DistributionModel

class PoissonDistribution(DistributionModel):
    """
    Poisson Distribution model for Emergency Landings.
    
    Used to model the number of rare events occurring in a fixed interval of time.
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def simulate(self, size: int) -> np.ndarray:
        """Simulate number of emergency landings using Poisson distribution."""
        return np.random.poisson(self.lam, size)

    def fit(self, data: np.ndarray) -> None:
        """Fit the lambda parameter (mean) to observed data."""
        self.lam = np.mean(data)

    def probability(self, k: int) -> float:
        """Calculate the Probability Mass Function (PMF) for k events."""
        return stats.poisson.pmf(k, self.lam)

    def expected_value(self) -> float:
        """The expected value of a Poisson distribution is lambda."""
        return self.lam

    def variance(self) -> float:
        """The variance of a Poisson distribution is also lambda."""
        return self.lam

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        """Visualize the discrete distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if data is not None:
            # Empirical distribution
            counts = np.bincount(data)
            probs = counts / len(data)
            ax.bar(range(len(probs)), probs, alpha=0.5, label="Empirical (Simulated)", color="salmon")
            
            # Theoretical PMF
            x = np.arange(0, max(data) + 1)
            y = [self.probability(val) for val in x]
            ax.step(x, y, where='mid', color='red', lw=2, label=f"Theoretical Poisson (λ={self.lam:.2f})")
            ax.scatter(x, y, color='red')

        ax.set_title(f"Poisson Distribution of Rare Events")
        ax.set_xlabel("Number of Events")
        ax.set_ylabel("Probability")
        ax.legend()
        
        return fig
