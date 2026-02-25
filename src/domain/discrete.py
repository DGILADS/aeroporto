import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Any, List
from .base import DistributionModel

class BinomialDistribution(DistributionModel):
    """Model n inspections with probability p of success."""
    def __init__(self, n: int = 10, p: float = 0.5):
        self.n = n
        self.p = p

    def simulate(self, size: int) -> np.ndarray:
        return np.random.binomial(self.n, self.p, size)

    def fit(self, data: np.ndarray) -> None:
        # Assuming n is fixed, we estimate p
        self.p = np.mean(data) / self.n

    def probability(self, k: int) -> float:
        return stats.binom.pmf(k, self.n, self.p)

    def expected_value(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(0, self.n + 1)
        y = self.probability(x)
        ax.bar(x, y, color='teal', alpha=0.7, label=f'Theoretical Binomial (n={self.n}, p={self.p:.2f})')
        ax.set_title("Binomial Distribution: Security Inspections")
        ax.legend()
        return fig

class BernoulliDistribution(BinomialDistribution):
    """Special case of Binomial where n=1 (Success/Failure)."""
    def __init__(self, p: float = 0.5):
        super().__init__(n=1, p=p)

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [0, 1]
        y = [1 - self.p, self.p]
        ax.bar(['Failure', 'Success'], y, color=['#ff9999','#66b3ff'])
        ax.set_title(f"Bernoulli Distribution: Luggage Routing (p={self.p})")
        return fig

class GeometricDistribution(DistributionModel):
    """Model days until first system failure with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def simulate(self, size: int) -> np.ndarray:
        return np.random.geometric(self.p, size)

    def fit(self, data: np.ndarray) -> None:
        self.p = 1 / np.mean(data)

    def probability(self, k: int) -> float:
        return stats.geom.pmf(k, self.p)

    def expected_value(self) -> float:
        return 1 / self.p

    def variance(self) -> float:
        return (1 - self.p) / (self.p ** 2)

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, 20) # Showing first 20 days
        y = self.probability(x)
        ax.stem(x, y, linefmt='C1-', markerfmt='C1o', label=f'p={self.p:.2f}')
        ax.set_title("Geometric Distribution: Time to First Failure")
        ax.set_xlabel("Days")
        ax.set_ylabel("Probability")
        ax.legend()
        return fig

class MultinomialDistribution(DistributionModel):
    """Model allocation of n flights to k gates with probabilities p_list."""
    def __init__(self, n: int = 100, p_list: List[float] = None):
        self.n = n
        self.p_list = p_list if p_list else [0.4, 0.3, 0.2, 0.1]

    def simulate(self, size: int) -> np.ndarray:
        return np.random.multinomial(self.n, self.p_list, size)

    def fit(self, data: np.ndarray) -> None:
        # Sum counts across all simulations and normalize
        total_counts = np.sum(data, axis=0)
        self.p_list = (total_counts / np.sum(total_counts)).tolist()

    def probability(self, counts: List[int]) -> float:
        return stats.multinomial.pmf(counts, self.n, self.p_list)

    def expected_value(self) -> List[float]:
        return [self.n * p for p in self.p_list]

    def variance(self) -> List[float]:
        return [self.n * p * (1 - p) for p in self.p_list]

    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        fig, ax = plt.subplots(figsize=(10, 6))
        gates = [f"Gate {i+1}" for i in range(len(self.p_list))]
        expected = self.expected_value()
        
        ax.bar(gates, expected, color='mediumpurple', alpha=0.8)
        ax.set_title(f"Multinomial Distribution: Gate Allocation (n={self.n})")
        ax.set_ylabel("Expected Number of Flights")
        
        return fig
