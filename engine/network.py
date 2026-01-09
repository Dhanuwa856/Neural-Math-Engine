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

