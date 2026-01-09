import numpy as np

class LinearAlgebraEngine:
    """
    The mathematical basis of our Neural Network.
    """
    @staticmethod
    def dot_product(input_data, weights):
        """
        Matrix දෙකක් ගුණ කිරීම (Dot Product).
        Input shape: (m, n)
        Weights shape: (n, p)
        Result shape: (m, p)
        """
        return np.dot(input_data, weights)

    @staticmethod
    def add_bias(z, bias):
        """
        සෑම output එකකටම bias අගය එකතු කිරීම.
        """
        return z + bias


# (Testing)
# if __name__ == "__main__":
#     # උදාහරණයක්: Input data 3ක් තියෙනවා (Features)
#     X = np.array([1.0, 2.0, 3.0])
#
#     # Weight matrix එකක් (3 neurons වලට සම්බන්ධ)
#     W = np.array([[0.1, 0.2, 0.3],
#                   [0.4, 0.5, 0.6],
#                   [0.7, 0.8, 0.9]])
#
#     # Dot product එක ගමු
#     Z = LinearAlgebraEngine.dot_product(X, W)
#     print(f"Result after Dot Product: {Z}")