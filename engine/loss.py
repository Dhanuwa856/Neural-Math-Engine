import numpy as np

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error (MSE) ගණනය කිරීම.
        Formula: (1/n) * sum((y_true - y_pred)^2)

        """
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        MSE වල derivative එක.
        Backpropagation වලදී weights update කරන්න මේක ඕනේ.
        Formula: 2/n * (y_pred - y_true)
        """
        return 2 * (y_true - y_pred) / y_true.size