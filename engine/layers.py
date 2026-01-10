import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        """
        Layer එකක් නිර්මාණය කිරීම.
        input_size: කලින් layer එකේ neurons ගණන
        output_size: මේ layer එකේ neurons ගණන
        """
        # Weights ආරම්භයේදී ඉතා කුඩා random අගයන් ලෙස (Mean 0, Std 1)
        self.weights = np.random.randn(input_size, output_size) * 0.1
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

    def backward(self, output_error, learning_rate):
        """
        output_error: ඊළඟ layer එකෙන් ලැබෙන වැරැද්ද (Gradient)
        learning_rate: අපි කොච්චර වේගයෙන් ඉගෙන ගන්නවාද කියන අගය (Alpha)
        """

        # 1. Weights වලට ලැබෙන error එක (Input.T * Error)
        # මෙතනදී Transpose (.T) කරන්නේ Matrix shapes ගැලපෙන්න
        input_error = np.dot(output_error, self.weights.T)
        weight_error = np.dot(self.input.T, output_error)

        # 2. Weights සහ Biases update කිරීම (Gradient Descent)
        self.weights -= learning_rate * weight_error
        self.biases -= learning_rate * output_error

        # 3. කලින් layer එකට අවශ්‍ය error එක ආපසු යැවීම
        return input_error

class ActivationLayer:
    def __init__(self, activation, activation_derivative):
        """
        activation: පාවිච්චි කරන function එක (උදා: sigmoid)
        activation_derivative: එහි derivative එක
        """
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backpropagation: output_error * f'(input)
        මෙහිදී learning_rate එක පාවිච්චි වෙන්නේ නැහැ (මොකද මෙතන weights නැති නිසා),
        නමුත් engine එකේ loop එකට ගැලපෙන්න ඒක මෙතන තියෙන්න ඕනේ.
        """
        return output_error * self.activation_derivative(self.output)