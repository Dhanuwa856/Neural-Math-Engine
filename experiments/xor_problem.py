import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.network import NeuralNetwork
from engine.layers import DenseLayer, ActivationLayer
from engine.activations import Activations
from engine.loss import Loss

# 1. දත්ත සකස් කිරීම (XOR Data)
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# 2. Network එක නිර්මාණය කිරීම
net = NeuralNetwork()
net.add(DenseLayer(2, 4)) # Input -> Hidden
net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))
net.add(DenseLayer(4, 1)) # Hidden -> Output
net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))

net.set_loss(Loss.mse, Loss.mse_derivative)

# 3. පුහුණු කිරීම (Training) - මෙහිදී errors list එක ලැබෙනවා
print("--- Training Started ---")
errors = net.train(x_train, y_train, epochs= 5000, learning_rate= 0.2)
print("--- Training Completed ---")

# 4. පරීක්ෂා කිරීම (Final Testing)
print("\n" + "=" * 45)
print(f"{'INPUT':<15} | {'TARGET':<10} | {'PREDICTION':<12}")
print("-" * 45)

for x, y in zip(x_train, y_train):
    output = net.predict(x)
    pred_val = output[0][0]
    result = np.round(pred_val)
    print(f"{str(x[0]):<15} | {str(y[0]):<10} | {pred_val:.4f} (-> {int(result)})")

print("=" * 45)

# 5. ප්‍රස්තාරය ඇඳීම (Visualization)
plt.figure(figsize=(10, 5))
plt.plot(errors, color='blue', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Neural Network Learning Curve (XOR Problem)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()