import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.network import NeuralNetwork
from mnist_loader import load_mnist_data

# 1. පින්තූර 500ක් load කරගමු (Test කරන්න විතරක්)
_, (x_test, _), (_, y_test_raw) = load_mnist_data(limit=1000)

# 2. Save කරපු model එක load කරමු
net = NeuralNetwork.load('mnist_model.pkl')

# 3. සියලුම පින්තූර සඳහා Predictions ගමු
predictions = []
for x in x_test:
    output = net.predict(x)
    predictions.append(np.argmax(output))

# 4. Accuracy එක ගණනය කරමු
acc = accuracy_score(y_test_raw, predictions)
print(f"\n🔥 Total Accuracy: {acc * 100:.2f}%")

# 5. Confusion Matrix එක හදමු
cm = confusion_matrix(y_test_raw, predictions)

# ලස්සනට පෙන්වමු
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix: Where did the AI fail?')
plt.show()