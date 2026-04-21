import re
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


from datasets import load_dataset

raw = load_dataset("imdb")
print(raw)
print("\nSample review:")
print(raw["train"][0]["text"][:300])
print("\nLabel:", raw["train"][0]["label"])


def tokenize(text: str):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9']", " ", text)
    return text.split()

MAX_VOCAB = 20_000
MIN_FREQ  = 2
MAX_LEN   = 256

print("Building vocabulary...")
counter = Counter()
for ex in raw["train"]:
    counter.update(tokenize(ex["text"]))

SPECIAL = ["<pad>", "<unk>"]
vocab   = SPECIAL + [w for w, c in counter.most_common(MAX_VOCAB) if c >= MIN_FREQ]
stoi    = {w: i for i, w in enumerate(vocab)}

PAD_IDX = stoi["<pad>"]
UNK_IDX = stoi["<unk>"]

print(f"Vocabulary size: {len(vocab):,}")


def encode(text: str, max_len: int = MAX_LEN):
    tokens = tokenize(text)[:max_len]
    return [stoi.get(t, UNK_IDX) for t in tokens]

class IMDBDataset(Dataset):
    def __init__(self, split):
        self.data = [
            (encode(ex["text"]), ex["label"])
            for ex in raw[split]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return (
            torch.tensor(ids,   dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts_padded, torch.stack(labels), lengths

BATCH_SIZE = 64

train_ds = IMDBDataset("train")
test_ds  = IMDBDataset("test")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_dl)} | Test batches: {len(test_dl)}")
x_s, y_s, l_s = next(iter(train_dl))
print("Batch shape:", x_s.shape, "| Labels:", y_s[:8], "| Lengths:", l_s[:8])


class BiLSTMSentiment(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.

    Parameters
    ----------
    vocab_size  : vocabulary size
    embed_dim   : embedding dimension
    hidden_size : LSTM hidden units per direction
    num_layers  : stacked LSTM layers
    dropout     : dropout probability
    num_classes : output classes (2 for binary)
    pad_idx     : padding token index
    pooling     : 'last' (final hidden states) or 'mean' (average all timesteps)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.5,
        num_classes: int   = 2,
        pad_idx:     int   = 0,
        pooling:     str   = "last",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.pooling     = pooling

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (hidden, cell) = self.lstm(packed)

        if self.pooling == "last":
            sentence = torch.cat([hidden[-2], hidden[-1]], dim=1)

        else:
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
            mask = (x != 0).unsqueeze(-1).float()
            output = output * mask
            sentence = output.sum(dim=1) / lengths.unsqueeze(1).float().to(output.device)

        return self.fc(self.dropout(sentence))

model = BiLSTMSentiment(
    vocab_size  = len(vocab),
    embed_dim   = 128,
    hidden_size = 128,
    num_layers  = 2,
    dropout     = 0.5,
    num_classes = 2,
    pad_idx     = PAD_IDX,
    pooling     = "last",
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss   = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, lengths in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss   = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total

def train_model(
    model, train_loader, test_loader,
    epochs=10, lr=1e-3, weight_decay=1e-4
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train loss={tr_loss:.4f}  acc={tr_acc:.4f}  |  "
            f"test  loss={te_loss:.4f}  acc={te_acc:.4f}  "
            f"[{time.time()-t0:.1f}s]"
        )

    return history


history = train_model(
    model, train_dl, test_dl,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-4,
)

print(f"\nBest test accuracy: {max(history['test_acc']):.4f}")


import matplotlib.pyplot as plt

epochs_range = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["test_loss"],  label="Test")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train")
axes[1].plot(epochs_range, history["test_acc"],  label="Test")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

fig.suptitle("Bidirectional LSTM on IMDB — Training Curves", fontsize=13)
plt.tight_layout()
plt.show()


class LSTMSentiment(nn.Module):
    """Unified LSTM classifier: unidirectional or bidirectional, last or mean pooling."""

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.5,
        num_classes:   int   = 2,
        pad_idx:       int   = 0,
        bidirectional: bool  = True,
        pooling:       str   = "last",
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling       = pooling
        self.num_dirs      = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * self.num_dirs, num_classes)

    def forward(self, x, lengths):
        embedded   = self.dropout(self.embedding(x))
        packed     = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed)

        if self.pooling == "last":
            if self.bidirectional:
                sentence = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                sentence = hidden[-1]
        else:
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
            mask      = (x != 0).unsqueeze(-1).float()
            output    = output * mask
            sentence  = output.sum(dim=1) / lengths.unsqueeze(1).float().to(output.device)

        return self.fc(self.dropout(sentence))

ablation_configs = [
    {"bidirectional": False, "pooling": "last", "num_layers": 2, "label": "Uni-LSTM (last)"},
    {"bidirectional": True,  "pooling": "last", "num_layers": 2, "label": "Bi-LSTM (last)"},
    {"bidirectional": True,  "pooling": "mean", "num_layers": 2, "label": "Bi-LSTM (mean)"},
    {"bidirectional": True,  "pooling": "mean", "num_layers": 3, "label": "Bi-LSTM-3L (mean)"},
]

ABLATION_EPOCHS = 7
ablation_results = []

for cfg in ablation_configs:
    print(f"\n--- {cfg['label']} ---")
    m = LSTMSentiment(
        vocab_size    = len(vocab),
        embed_dim     = 128,
        hidden_size   = 128,
        num_layers    = cfg["num_layers"],
        dropout       = 0.5,
        num_classes   = 2,
        pad_idx       = PAD_IDX,
        bidirectional = cfg["bidirectional"],
        pooling       = cfg["pooling"],
    ).to(device)
    h = train_model(m, train_dl, test_dl, epochs=ABLATION_EPOCHS, lr=1e-3)
    best_acc = max(h["test_acc"])
    ablation_results.append({"label": cfg["label"], "best_test_acc": best_acc, "history": h})
    print(f"Best test acc: {best_acc:.4f}")


print(f"{'Config':<22} {'Best Test Acc':>14}")
print("-" * 38)
for r in ablation_results:
    print(f"{r['label']:<22} {r['best_test_acc']:>14.4f}")

plt.figure(figsize=(9, 5))
for r in ablation_results:
    plt.plot(range(1, ABLATION_EPOCHS + 1), r["history"]["test_acc"], marker="o", label=r["label"])
plt.title("Ablation: Uni-LSTM vs Bi-LSTM Pooling Strategies")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


@torch.no_grad()
def get_predictions(m, loader):
    m.eval()
    all_preds, all_labels = [], []
    for x, y, lengths in loader:
        x = x.to(device)
        logits = m(x, lengths)
        preds  = logits.argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    return torch.cat(all_preds), torch.cat(all_labels)

preds, labels = get_predictions(model, test_dl)

for cls, name in enumerate(["Negative", "Positive"]):
    tp = ((preds == cls) & (labels == cls)).sum().item()
    fp = ((preds == cls) & (labels != cls)).sum().item()
    fn = ((preds != cls) & (labels == cls)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"{name:10s}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

overall_acc = (preds == labels).float().mean().item()
print(f"\nOverall Accuracy: {overall_acc:.4f}")


num_classes = 2
conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
for p, l in zip(preds, labels):
    conf_matrix[l, p] += 1

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf_matrix.numpy(), cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Neg", "Pred Pos"])
ax.set_yticklabels(["True Neg", "True Pos"])
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, conf_matrix[i, j].item(), ha="center", va="center", fontsize=14)
ax.set_title("Confusion Matrix — Bi-LSTM on IMDB Test Set")
plt.colorbar(im)
plt.tight_layout()
plt.show()


@torch.no_grad()
def predict(text: str, m=model):
    m.eval()
    ids = encode(text)
    if not ids:
        return "unknown", 0.0
    x       = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    logits  = m(x, lengths)
    probs   = F.softmax(logits, dim=1)[0]
    label   = int(logits.argmax(1).item())
    return ("positive" if label == 1 else "negative"), float(probs[label])

test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.",
    "Terrible film. Boring, predictable, and a complete waste of time. I walked out after 30 minutes.",
    "It was okay, nothing special. Some scenes were good but overall it felt a bit flat.",
    "One of the best movies I've seen in years. A masterpiece of storytelling and cinematography.",
    "The movie started slow but the ending completely redeemed it — absolutely worth watching.",
    "Despite the great cast, the script was dull and the pacing was painfully slow throughout.",
]

print(f"{'Review':<70} {'Sentiment':<12} {'Confidence'}")
print("-" * 97)
for review in test_reviews:
    sentiment, confidence = predict(review)
    print(f"{review[:68]:<70} {sentiment:<12} {confidence:.4f}")
