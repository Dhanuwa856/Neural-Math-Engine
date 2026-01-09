import numpy as np

class NeuralNetwork:
    def __init__(self):
        # සියලුම layers ගබඩා කරගන්නා ලැයිස්තුව
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add(self, layer):
        """Network එකට අලුත් layer එකක් එකතු කිරීම"""
        self.layers.append(layer)

    def set_loss(self, loss_func, loss_derivative_func):
        """පාවිච්චි කරන Loss function එක තීරණය කිරීම"""
        self.loss = loss_func
        self.loss_derivative = loss_derivative_func

    def predict(self, input_data):
        """Input එකක් දීලා output එකක් ලබා ගැනීම (Forward Pass)"""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        """
        epochs: කී පාරක් දත්ත සියල්ල කියවිය යුතුද?
        """
        for i in range(epochs):
            display_error = 0
            for j in range(len(x_train)):
                # --- 1. Forward Pass (පුරෝකථනය) ---
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # --- 2. Loss ගණනය කිරීම ---
                display_error += self.loss(y_train[j], output)

                # --- 3. Backward Pass (ඉගෙන ගැනීම) ---
                # මුලින්ම loss එකේ derivative එක ගමු
                error = self.loss_derivative(y_train[j], output)

                # වැරැද්ද ආපස්සට යවමු (Reverse order)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # හැම 100 epoch එකකටම පාරක් output එක පෙන්වමු
            if (i + 1) % 100 == 0:
                print(f'Epoch {i + 1}/{epochs}  Error={display_error / len(x_train)}')
