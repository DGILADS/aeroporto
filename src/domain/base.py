from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class DistributionModel(ABC):
    """
    Abstract base class for all statistical distribution models in the airport operations system.
    
    This class follows the template method pattern and defines the interface for simulation,
    fitting, and probability calculations.
    """

    @abstractmethod
    def simulate(self, size: int) -> np.ndarray:
        """
        Simulate data points according to the distribution.

        Parameters
        ----------
        size : int
            The number of data points to generate.

        Returns
        -------
        np.ndarray
            Array of simulated values.
        """
        pass

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Fit distribution parameters to given data.

        Parameters
        ----------
        data : np.ndarray
            Data to fit the parameters to.
        """
        pass

    @abstractmethod
    def probability(self, x: float) -> float:
        """
        Calculate the probability (PDF or PMF) for a given value x.

        Parameters
        ----------
        x : float
            Value to evaluate.

        Returns
        -------
        float
            Probability value.
        """
        pass

    @abstractmethod
    def expected_value(self) -> float:
        """
        Calculate the expected value (mean) of the distribution.

        Returns
        -------
        float
            Expected value.
        """
        pass

    @abstractmethod
    def variance(self) -> float:
        """
        Calculate the variance of the distribution.

        Returns
        -------
        float
            Variance.
        """
        pass

    @abstractmethod
    def plot(self, data: Optional[np.ndarray] = None) -> Any:
        """
        Generate visualization for the distribution.

        Parameters
        ----------
        data : Optional[np.ndarray], optional
            Optional data to plot against the theoretical distribution.

        Returns
        -------
        Any
            Visualization object (e.g., matplotlib figure).
        """
        pass
