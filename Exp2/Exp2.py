# %% [markdown]
# # Build and train a fully connected neural network , without relying on deep learning libraries such as TensorFlow or PyTorch.

# %%
import os
import csv
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


# %% [markdown]
# # Utilities

# %%
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    # y: (N,)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    # logits: (N, C), y_true: (N,)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# %% [markdown]
# # Activations

# %%
def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    return (z > 0).astype(np.float32)

def sigmoid(z):
    # stable sigmoid
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid_from_a(a):
    # derivative using a = sigmoid(z)
    return a * (1.0 - a)

def tanh(z):
    return np.tanh(z)

def dtanh_from_a(a):
    # derivative using a = tanh(z)
    return 1.0 - a**2


def softmax(logits):
    # stable softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

# %% [markdown]
# # Neural Network (NumPy)

# %%

class NeuralNetwork:
    """
    MLP: [input_dim] -> hidden_layers... -> [num_classes]
    Hidden activations: relu/sigmoid/tanh
    Output activation: softmax (handled in loss for stability)
    """
    def __init__(self, layer_sizes, activation="relu", lr=0.01, weight_decay=0.0, seed=42):
        """
        layer_sizes: list like [784, 256, 128, 10]
        activation: "relu" | "sigmoid" | "tanh"
        lr: learning rate (GD)
        weight_decay: L2 regularization strength (0 disables)
        """
        assert len(layer_sizes) >= 2
        assert activation in ("relu", "sigmoid", "tanh")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.rng = np.random.default_rng(seed)

        # Parameters
        self.W = []
        self.b = []

        # Cache for forward pass
        self.Z = []  # pre-activations
        self.A = []  # activations (including input as A[0])

        # Gradients
        self.dW = []
        self.db = []

        self._init_params()

    def _init_params(self):
        self.W.clear()
        self.b.clear()

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            # He init for ReLU, Xavier for sigmoid/tanh
            if i < len(self.layer_sizes) - 2:  # hidden layer
                if self.activation_name == "relu":
                    scale = math.sqrt(2.0 / fan_in)
                else:
                    scale = math.sqrt(1.0 / fan_in)
            else:
                # output layer init: Xavier is fine
                scale = math.sqrt(1.0 / fan_in)

            W_i = (self.rng.standard_normal((fan_in, fan_out)).astype(np.float32)) * scale
            b_i = np.zeros((1, fan_out), dtype=np.float32)

            self.W.append(W_i)
            self.b.append(b_i)

        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(bb) for bb in self.b]

    def _hidden_activation(self, z):
        if self.activation_name == "relu":
            return relu(z)
        elif self.activation_name == "sigmoid":
            return sigmoid(z)
        else:
            return tanh(z)

    def _hidden_activation_derivative(self, z, a):
        # provide derivative wrt z; we can use either z or a
        if self.activation_name == "relu":
            return drelu(z)
        elif self.activation_name == "sigmoid":
            return dsigmoid_from_a(a)
        else:
            return dtanh_from_a(a)

    def forward(self, X):
        """
        Forward propagate through all layers.
        Stores intermediates in self.A and self.Z.
        X: (N, D)
        Returns logits: (N, C)
        """
        self.Z = []
        self.A = [X]  # A[0] is input

        out = X
        L = len(self.W)

        for i in range(L):
            z = out @ self.W[i] + self.b[i]     # (N, fan_out)
            self.Z.append(z)

            if i < L - 1:
                out = self._hidden_activation(z)
            else:
                # output layer returns logits; softmax will be used in loss/predict
                out = z

            self.A.append(out)

        logits = self.A[-1]
        return logits

    def compute_loss(self, logits, y_onehot):
        """
        Cross-Entropy loss with softmax, computed stably.
        logits: (N, C)
        y_onehot: (N, C)
        Returns scalar loss (float)
        """
        # softmax probs
        probs = softmax(logits)

        # cross-entropy
        eps = 1e-12
        ce = -np.sum(y_onehot * np.log(probs + eps)) / logits.shape[0]

        # L2 weight decay
        if self.weight_decay > 0:
            l2 = 0.5 * self.weight_decay * sum(np.sum(w * w) for w in self.W)
            ce = ce + l2

        return float(ce)

    def backward(self, logits, y_onehot):
        """
        Backprop through network to compute grads for W and b.
        logits: (N, C)
        y_onehot: (N, C)
        """
        N = logits.shape[0]
        L = len(self.W)

        # dLogits for softmax + cross-entropy:
        # If loss = -sum(y*log softmax(logits))/N then gradient is (softmax - y)/N
        probs = softmax(logits)
        dZ = (probs - y_onehot) / N  # (N, C)

        # Loop backward over layers
        for i in reversed(range(L)):
            A_prev = self.A[i]  # (N, fan_in)

            self.dW[i] = (A_prev.T @ dZ).astype(np.float32)  # (fan_in, fan_out)
            self.db[i] = np.sum(dZ, axis=0, keepdims=True).astype(np.float32)  # (1, fan_out)

            # add weight decay gradient
            if self.weight_decay > 0:
                self.dW[i] += (self.weight_decay * self.W[i]).astype(np.float32)

            if i > 0:
                # propagate to previous layer
                dA_prev = dZ @ self.W[i].T  # (N, fan_in)
                z_prev = self.Z[i - 1]
                a_prev = self.A[i]          # activation output of previous layer (after activation)
                dAct = self._hidden_activation_derivative(z_prev, a_prev)  # (N, fan_in)
                dZ = dA_prev * dAct

    def update_parameters(self):
        """
        Gradient descent update
        """
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.dW[i]
            self.b[i] -= self.lr * self.db[i]

    def predict(self, X):
        """
        Returns predicted classes (N,)
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def evaluate(self, X, y):
        """
        Returns accuracy on (X, y)
        """
        logits = self.forward(X)
        return accuracy_from_logits(logits, y)


# %% [markdown]
# # Data pipeline (Torch -> NumPy)

# %%
def make_mnist_loaders(batch_size=64, train_val_split=55000, seed=42):
    """
    Returns: train_loader, val_loader, test_loader
    Uses torch for loading, then you'll convert batches to numpy with .cpu().numpy().
    """
    transform = torchvision.transforms.ToTensor()  # yields float in [0,1]
    train_dataset_full = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # split train into train/val (required)
    total = len(train_dataset_full)  # 60000
    assert 0 < train_val_split < total
    val_size = total - train_val_split
    gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset_full, [train_val_split, val_size], generator=gen
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def torch_batch_to_numpy(images, labels):
    """
    Allowed ops: .cpu().numpy()
    images: torch.Tensor (B, 1, 28, 28)
    labels: torch.Tensor (B,)
    Returns:
      X: np.ndarray (B, 784) float32
      y: np.ndarray (B,) int64
      y_oh: np.ndarray (B, 10) float32
    """
    images = images.cpu()
    labels = labels.cpu()

    images_np = images.numpy().astype(np.float32)  # (B,1,28,28)
    labels_np = labels.numpy().astype(np.int64)    # (B,)

    # Flatten
    X = images_np.reshape(images_np.shape[0], -1)  # (B, 784)

    # Already normalized in [0,1] via ToTensor(), but keep it explicit:
    X = np.clip(X, 0.0, 1.0).astype(np.float32)

    y_oh = one_hot(labels_np, 10)
    return X, labels_np, y_oh


# %% [markdown]
# # Training / validation

# %%
def run_one_experiment(cfg, out_dir):
    """
    cfg fields:
      - hidden_layers: list[int]
      - activation: str
      - lr: float
      - weight_decay: float
      - batch_size: int
      - epochs: int
      - seed: int
    """
    ensure_dir(out_dir)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_loader, val_loader, test_loader = make_mnist_loaders(
        batch_size=cfg["batch_size"], train_val_split=cfg.get("train_val_split", 55000), seed=cfg["seed"]
    )

    layer_sizes = [784] + cfg["hidden_layers"] + [10]
    net = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=cfg["activation"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        seed=cfg["seed"],
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg["epochs"] + 1):
        # ---- Train ----
        train_losses = []
        train_accs = []

        for images, labels in train_loader:
            X, y, y_oh = torch_batch_to_numpy(images, labels)

            logits = net.forward(X)
            loss = net.compute_loss(logits, y_oh)
            net.backward(logits, y_oh)
            net.update_parameters()

            acc = accuracy_from_logits(logits, y)

            train_losses.append(loss)
            train_accs.append(acc)

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))

        # ---- Validate ----
        val_losses = []
        val_accs = []
        for images, labels in val_loader:
            X, y, y_oh = torch_batch_to_numpy(images, labels)
            logits = net.forward(X)
            loss = net.compute_loss(logits, y_oh)
            acc = accuracy_from_logits(logits, y)
            val_losses.append(loss)
            val_accs.append(acc)

        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{os.path.basename(out_dir)}] "
            f"Epoch {epoch:02d}/{cfg['epochs']} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

    # ---- Test accuracy ----
    test_accs = []
    test_losses = []
    for images, labels in test_loader:
        X, y, y_oh = torch_batch_to_numpy(images, labels)
        logits = net.forward(X)
        test_losses.append(net.compute_loss(logits, y_oh))
        test_accs.append(accuracy_from_logits(logits, y))
    test_loss = float(np.mean(test_losses))
    test_acc = float(np.mean(test_accs))

    # Save history CSV
    hist_path = os.path.join(out_dir, "history.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])

    # Save plots
    plot_metrics(history, out_dir)

    # Save final results
    final = {
        "best_val_acc": float(np.max(history["val_acc"])),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    with open(os.path.join(out_dir, "final_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    return final


def plot_metrics(history, out_dir):
    epochs = history["epoch"]

    # Loss plot
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()



# %% [markdown]
# # main

# %%
def main():
    set_seed(42)

    base_out = "experiments_mnist_numpy"
    ensure_dir(base_out)

    # Try different hyperparameter configurations:
    experiments = [
        # 1 hidden layer
        {"name": "mlp_1x128_relu", "hidden_layers": [128], "activation": "relu", "lr": 0.05, "weight_decay": 0.0, "batch_size": 64, "epochs": 10, "seed": 42},
        {"name": "mlp_1x256_relu", "hidden_layers": [256], "activation": "relu", "lr": 0.05, "weight_decay": 0.0, "batch_size": 64, "epochs": 10, "seed": 42},
        {"name": "mlp_1x256_tanh", "hidden_layers": [256], "activation": "tanh", "lr": 0.05, "weight_decay": 0.0, "batch_size": 64, "epochs": 10, "seed": 42},

        # 2 hidden layers
        {"name": "mlp_2x256_128_relu", "hidden_layers": [256, 128], "activation": "relu", "lr": 0.05, "weight_decay": 0.0, "batch_size": 64, "epochs": 12, "seed": 42},
        {"name": "mlp_2x256_128_sigmoid", "hidden_layers": [256, 128], "activation": "sigmoid", "lr": 0.05, "weight_decay": 0.0, "batch_size": 64, "epochs": 12, "seed": 42},

        # Regularization example
        {"name": "mlp_2x256_128_relu_l2", "hidden_layers": [256, 128], "activation": "relu", "lr": 0.05, "weight_decay": 1e-4, "batch_size": 64, "epochs": 12, "seed": 42},
    ]

    summary_rows = []
    for cfg in experiments:
        out_dir = os.path.join(base_out, cfg["name"])
        final = run_one_experiment(cfg, out_dir)

        summary_rows.append({
            "name": cfg["name"],
            "hidden_layers": str(cfg["hidden_layers"]),
            "activation": cfg["activation"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "best_val_acc": final["best_val_acc"],
            "test_acc": final["test_acc"],
        })

    # Save a global experiment summary table (CSV)
    summary_path = os.path.join(base_out, "experiment_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "hidden_layers", "activation", "lr", "weight_decay", "batch_size", "epochs", "best_val_acc", "test_acc"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nSaved experiment summary to:", summary_path)
    print("Each experiment folder contains:")
    print("  - config.json")
    print("  - history.csv")
    print("  - loss.png, accuracy.png")
    print("  - final_results.json")


if __name__ == "__main__":
    main()



