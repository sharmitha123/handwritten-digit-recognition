import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from scipy.signal import convolve2d

# Download MNIST
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.astype("float32") / 255.0

def conv2d(img, filt):
    return convolve2d(img, filt, mode='valid')

def max_pool(img, size=2):
    h, w = img.shape
    img = img[:h-h%2, :w-w%2]
    return img.reshape(h//2,2,w//2,2).max(axis=(1,3))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def cnn_forward(img):
    filt = np.random.randn(3,3) * 0.1
    conv = conv2d(img, filt)
    conv = np.maximum(conv, 0)

    pooled = max_pool(conv)
    flat = pooled.ravel()

    w1 = np.random.randn(flat.size, 64) * 0.1
    b1 = np.zeros(64)
    fc1 = np.maximum(flat @ w1 + b1, 0)

    w2 = np.random.randn(64, 10) * 0.1
    b2 = np.zeros(10)
    out = fc1 @ w2 + b2

    return softmax(out)

probs = cnn_forward(X_test[0])
pred = int(np.argmax(probs))

print("Predicted:", pred)
print("Actual:", y_test[0])

plt.imshow(X_test[0], cmap='gray')
plt.title(f"Pred: {pred} | Actual: {y_test[0]}")
plt.axis("off")
plt.show()
