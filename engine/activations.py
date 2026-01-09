import numpy as np

class Activations:
    @staticmethod
    def sigmoid(x):
        """
        Returns any value between 0 and 1.
        Formula: 1 / (1 + e^-x)
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(sigmoid_output):
        """
        The derivative of the sigmoid.
        Formula: f(x) * (1 - f(x))
        """
        return sigmoid_output * (1 - sigmoid_output)

    @staticmethod
    def relu(x):
        """
        ධන අගයන් එලෙසම තබා ඍණ අගයන් 0 කරයි.
        සූත්‍රය: max(0, x)
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        ReLU වල derivative එක.
        x > 0 නම් 1, නැතිනම් 0.
        """
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(output):
        # Tanh වල derivative එක: 1 - tanh(x)^2
        # අපේ ActivationLayer එකේ අපි දෙන්නේ output එක නිසා මෙහෙම ලියමු
        return 1 - output ** 2