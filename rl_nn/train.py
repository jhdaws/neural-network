import numpy as np
import matplotlib.pyplot as plt
from nn import rlNeuralNetwork

# load dataset
data = np.load("races_dataset.npz")
X = data["images"]     # (num_samples, 64, 64, 3)
Y = data["labels"]     # (num_samples, 4)

# flatten images
num_samples = X.shape[0]
X_flat = X.reshape(num_samples, -1).T    # (12288, num_samples)
Y_onehot = Y.T                            # (4, num_samples)

# image parameters for array
input_size = 64 * 64 * 3  # 12288
hidden_size = 64
output_size = 4

net = rlNeuralNetwork(input_size, hidden_size, output_size, lr=0.1)
losses, accs = net.train(X_flat, Y_onehot, epochs=200)

# Save training weights for UI usage
np.savez("model_weights.npz", W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)
print("✅ Model weights saved to 'model_weights.npz'")

# plot loss and accuracy
epochs = range(len(losses))

plt.figure(figsize=(10, 4))

# plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, color='red')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accs, color='blue')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

