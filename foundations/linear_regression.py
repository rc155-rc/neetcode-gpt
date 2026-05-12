import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        n = len(y_true)
        ar1 = np.multiply(y_true,np.log(y_pred))
        ar2 = np.multiply(1-y_true,np.log(1-y_pred))

        s = ar1+ar2
        s = np.sum(s)/-n

        L = round(s,4)
        return L

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        n = y_true.shape[0]
        comb = np.multiply(y_true,np.log(y_pred))
        comb = np.sum(comb,axis=1)
        comb = np.sum(comb,axis=0)/-n
        comb = round(comb,4)
        return comb
