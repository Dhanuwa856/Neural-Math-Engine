import pickle
import numpy as np
from mnist_loader import load_mnist_data

# Model එක load කිරීම
def load_model(filename='mnist_model.pkl'):
    with open(filename, 'rb') as f:
        print("✅ Model loaded successfully!")
        return pickle.load(f)


# 1. දත්ත load කරගමු (Test data විතරක් ඇති)
_, (x_test, y_test_enc), (_, y_test_raw) = load_mnist_data(limit=100)

# 2. Save කරපු model එක ගමු
loaded_net = load_model('mnist_model.pkl')

# 3. එකක් පරීක්ෂා කරලා බලමු
idx = 5
output = loaded_net.predict(x_test[idx])
prediction = np.argmax(output)

print(f"\n--- Testing Saved Model ---")
print(f"Actual Number: {y_test_raw[idx]}")
print(f"AI Prediction: {prediction}")