# Neural-Math-Engine 🧠🔥
### Building Deep Learning from Scratch with Pure Mathematics

This project is a high-level implementation of a Neural Network engine built using only **Python** and **NumPy**. It aims to demonstrate the underlying mathematics of AI, including Linear Algebra, Calculus, and Gradient Descent, without relying on frameworks like TensorFlow or PyTorch.

## 🚀 The Achievement: Solving XOR
The engine successfully solved the classic **XOR problem**, which is a non-linearly separable logic gate. 
- **Training Accuracy**: ~99%
- **Final Loss (MSE)**: 0.002
- **Architecture**: 2 Input Neurons → 4 Hidden Neurons (Sigmoid) → 1 Output Neuron (Sigmoid)



## 🧮 Mathematical Foundations
This project proves the core mechanics of Deep Learning:
- **Forward Propagation**: Implementing $Z = W \cdot X + b$ using vectorized Dot Products.
- **Activation Functions**: Manual implementation of Sigmoid, Tanh, and ReLU with their respective derivatives.
- **Backpropagation**: Applying the **Chain Rule** to calculate gradients and update weights.
- **Loss Functions**: Mean Squared Error (MSE) calculation and its derivation for optimization.



## 📂 Project Structure
```text
Neural-Math-Engine/
│
├── engine/             # The Core AI Engine
│   ├── matrix.py       # Linear Algebra operations
│   ├── activations.py  # Sigmoid, ReLU, Tanh
│   ├── loss.py         # MSE & Derivatives
│   ├── layers.py       # Dense & Activation Layers
│   └── network.py      # Training & Prediction Loop
│
├── experiments/        # Practical Applications
│   └── xor_problem.py  # First successful test
│
└── README.md
```
## 🛠️ How to Run
- **Clone the repository.**
- **Ensure you have NumPy and Matplotlib installed**: 
``` pip install numpy matplotlib. ```
- **Run the XOR experiment**:
````
python experiments/xor_problem.py
````
