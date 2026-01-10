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
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def cross_entropy(y_true, y_pred):
        # 0 ලොග් (log) කිරීමෙන් වැළකීමට ඉතා කුඩා අගයක් එකතු කරමු (epsilon)
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        # Softmax සමඟ පාවිච්චි කරන විට gradient එක ඉතා සරලයි: y_pred - y_true
        return (y_pred - y_true) / y_true.shape[0]

