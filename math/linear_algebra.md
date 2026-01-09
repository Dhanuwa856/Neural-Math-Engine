# Linear Algebra for Neural Networks

To build a neural network from scratch, we need to understand how data moves through the layers.

## 1. Data Structures
- **Scalar**: A single number (e.g., learning rate $\alpha = 0.01$).
- **Vector**: A 1D array of numbers (e.g., input features).
- **Matrix**: A 2D array of numbers (e.g., weights of a layer $W$).

## 2. Key Operations
### Dot Product
In a neural network, the forward pass is a series of dot products:
$$Z = X \cdot W + b$$
Where:
- $X$: Input matrix
- $W$: Weight matrix
- $b$: Bias vector