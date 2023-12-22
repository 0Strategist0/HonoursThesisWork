import numpy as np
import numpy.typing as npt

def weighted_mean(data: npt.ArrayLike, weights: npt.ArrayLike = 1, axis: int | None = None) -> np.ndarray:
    return np.sum(weights * data, axis=axis) / np.sum(weights, axis=axis)

def weighted_var(data: npt.ArrayLike, weights: npt.ArrayLike = np.array(1), ddof: int = 1, axis: int | None = None) -> np.ndarray:
    
    normed_weights = np.abs(weights) / np.max(np.abs(weights), axis=axis)
    return (np.sum(normed_weights * (data - weighted_mean(data, weights, axis)) ** 2.0, axis=axis) 
            / (np.sum(normed_weights, axis=axis) - ddof))

def weighted_std(data: npt.ArrayLike, weights: npt.ArrayLike = np.array(1), ddof: int = 1, axis: int | None = None) -> np.ndarray:
    return np.sqrt(weighted_var(data, ddof, weights, axis))