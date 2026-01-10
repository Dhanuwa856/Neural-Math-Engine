import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_data(limit=2000):
    print(f"Loading {limit} images from MNIST... (This may take a minute)")

    # 1. MNIST දත්ත ලබා ගැනීම
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]

    # 2. Normalization (0-255 අගයන් 0-1 අතරට ගැනීම)
    X = X.astype('float32') / 255.0

    # 3. Label ටික ඉලක්කම් (integers) බවට හරවා One-Hot Encode කිරීම
    y = y.astype(np.uint8)

    # 4. දත්ත බෙදා ගැනීම (Training & Testing)
    x_train, x_test = X[:limit], X[limit:limit + 500]
    y_train, y_test = y[:limit], y[limit:limit + 500]

    # 5. Shape එක සකස් කිරීම: (Samples, 1, 784)
    x_train = x_train.reshape(-1, 1, 784)
    x_test = x_test.reshape(-1, 1, 784)

    # 6. Labels ටික One-Hot Encode කිරීම
    y_train_encoded = np.eye(10)[y_train].reshape(-1, 1, 10)
    y_test_encoded = np.eye(10)[y_test].reshape(-1, 1, 10)

    print("Data loaded successfully!")
    return (x_train, y_train_encoded), (x_test, y_test_encoded), (y_train, y_test)



if __name__ == "__main__":
    # දත්ත 500ක් ලෝඩ් කරලා බලමු වැඩද කියලා
    (x_train, y_train_enc), (x_test, y_test_enc), (y_train, y_test) = load_mnist_data(limit=500)

    print("\n--- Summary ---")
    print(f"Training Data Shape: {x_train.shape}")  # (500, 1, 784) වෙන්න ඕනේ
    print(f"Training Labels Shape: {y_train_enc.shape}")  # (500, 1, 10) වෙන්න ඕනේ
    print(f"First Label (Number): {y_train[0]}")
    print(f"First Label (One-Hot): {y_train_enc[0]}")

    # පින්තූරයක් ඇත්තටම තියෙනවාද කියලා බලමු
    import matplotlib.pyplot as plt

    plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[0]}")
    plt.show()