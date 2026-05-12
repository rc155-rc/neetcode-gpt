import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        den = np.sum(np.exp(z - np.max(z)))
        s_m = [(np.exp(a- np.max(z)))/den for a in z]
        s_m = np.round(s_m,4)
        return s_m
