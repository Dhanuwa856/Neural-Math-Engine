import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        """
        Layer එකක් නිර්මාණය කිරීම.
        input_size: කලින් layer එකේ neurons ගණන
        output_size: මේ layer එකේ neurons ගණන
        """
        # Weights ආරම්භයේදී ඉතා කුඩා random අගයන් ලෙස (Mean 0, Std 1)
        self.weights = np.random.randn(input_size, output_size) * 0.01
        # Biases ආරම්භයේදී 0 ලෙස
        self.biases = np.zeros((1, output_size))

        # Backpropagation සඳහා දත්ත ගබඩා කිරීමට
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Z = X . W + b
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output