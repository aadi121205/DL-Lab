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

raw = load_dataset("yelp_review_full")
print(raw)

sample = raw["train"][0]
print("\nSample review text :", sample["text"][:200])
print("Label (0-indexed)  :", sample["label"], "→", sample["label"] + 1, "stars")


import matplotlib.pyplot as plt

all_labels_raw = [ex["label"] for ex in raw["train"]]
label_raw_counts = Counter(all_labels_raw)

stars_sorted = sorted(label_raw_counts.keys())
counts_sorted = [label_raw_counts[s] for s in stars_sorted]

plt.figure(figsize=(7, 4))
plt.bar([str(s + 1) + "★" for s in stars_sorted], counts_sorted, color="steelblue")
plt.title("Yelp Reviews — Rating Distribution (train split)")
plt.xlabel("Star Rating")
plt.ylabel("Count")
for i, c in enumerate(counts_sorted):
    plt.text(i, c + 1000, str(c), ha="center", fontsize=9)
plt.tight_layout()
plt.show()

print(f"Total train reviews: {len(all_labels_raw):,}")
print(f"Total test  reviews: {len(raw['test']):,}")


def label_to_sentiment(label: int) -> int:
    if label <= 1:
        return 0
    elif label == 2:
        return 1
    else:
        return 2

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
NUM_CLASSES  = len(LABEL_NAMES)

TRAIN_LIMIT = 60_000
TEST_LIMIT  = 10_000

rng = random.Random(SEED)

def build_split(hf_split, limit):
    data = [
        {"text": ex["text"].strip(), "label": label_to_sentiment(ex["label"])}
        for ex in hf_split
        if ex["text"] and ex["text"].strip()
    ]
    rng.shuffle(data)
    return data[:limit]

train_data = build_split(raw["train"], TRAIN_LIMIT)
test_data  = build_split(raw["test"],  TEST_LIMIT)

label_counts = Counter(ex["label"] for ex in train_data)
print("Train class distribution:")
for cls, name in enumerate(LABEL_NAMES):
    print(f"  {name:10s} (class {cls}): {label_counts[cls]:>6,}  ({label_counts[cls]/len(train_data)*100:.1f}%)")

print(f"\nTrain: {len(train_data):,}  |  Test: {len(test_data):,}")


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9']", " ", text)
    return text.split()

MAX_VOCAB = 20_000
MIN_FREQ  = 2
MAX_LEN   = 200

print("Building vocabulary from training data...")
counter = Counter()
for ex in train_data:
    counter.update(tokenize(ex["text"]))

SPECIAL = ["<pad>", "<unk>"]
vocab   = SPECIAL + [w for w, c in counter.most_common(MAX_VOCAB) if c >= MIN_FREQ]
stoi    = {w: i for i, w in enumerate(vocab)}

PAD_IDX = stoi["<pad>"]
UNK_IDX = stoi["<unk>"]

print(f"Vocabulary size : {len(vocab):,}")
print(f"Max sequence len: {MAX_LEN}")


def encode(text: str, max_len: int = MAX_LEN):
    tokens = tokenize(text)[:max_len]
    return [stoi.get(t, UNK_IDX) for t in tokens] or [UNK_IDX]

class AmazonDataset(Dataset):
    def __init__(self, data):
        self.samples = [
            (encode(ex["text"]), ex["label"])
            for ex in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return (
            torch.tensor(ids,   dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths       = torch.tensor([len(t) for t in texts], dtype=torch.long)
    texts_padded  = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts_padded, torch.stack(labels), lengths

BATCH_SIZE = 64

train_ds = AmazonDataset(train_data)
test_ds  = AmazonDataset(test_data)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

train_labels   = [ex["label"] for ex in train_data]
train_counts   = Counter(train_labels)
total_train    = len(train_labels)
class_weights  = torch.tensor(
    [total_train / (NUM_CLASSES * train_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float,
).to(device)

print("Class weights:", class_weights.tolist())
print(f"Train batches: {len(train_dl)} | Test batches: {len(test_dl)}")


class GRUSentiment(nn.Module):
    """
    GRU-based product review sentiment classifier.

    Parameters
    ----------
    vocab_size    : vocabulary size
    embed_dim     : embedding dimension
    hidden_size   : GRU hidden units per direction
    num_layers    : stacked GRU layers
    dropout       : dropout probability
    num_classes   : output classes (3: neg/neutral/pos)
    pad_idx       : padding token index
    bidirectional : use bidirectional GRU
    pooling       : 'last' or 'mean'
    """

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.4,
        num_classes:   int   = 3,
        pad_idx:       int   = 0,
        bidirectional: bool  = True,
        pooling:       str   = "mean",
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling       = pooling
        self.num_dirs      = 2 if bidirectional else 1
        self.hidden_size   = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
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
        embedded = self.dropout(self.embedding(x))

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.gru(packed)

        if self.pooling == "last":
            if self.bidirectional:
                sentence = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                sentence = hidden[-1]

        else:
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
            mask      = (x != 0).unsqueeze(-1).float()
            output    = output * mask
            lengths_f = lengths.unsqueeze(1).float().to(output.device)
            sentence  = output.sum(dim=1) / lengths_f

        return self.fc(self.dropout(sentence))

model = GRUSentiment(
    vocab_size    = len(vocab),
    embed_dim     = 128,
    hidden_size   = 128,
    num_layers    = 2,
    dropout       = 0.4,
    num_classes   = NUM_CLASSES,
    pad_idx       = PAD_IDX,
    bidirectional = True,
    pooling       = "mean",
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")

print("\nParameter breakdown:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  {name:<30} {str(tuple(p.shape)):<25} {p.numel():>10,}")


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
    epochs=10, lr=1e-3, weight_decay=1e-4,
    class_weights=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
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
    class_weights=class_weights,
)

print(f"\nBest test accuracy: {max(history['test_acc']):.4f}")


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

fig.suptitle("Bidirectional GRU on Amazon Reviews — Training Curves", fontsize=13)
plt.tight_layout()
plt.show()


class RecurrentSentiment(nn.Module):
    """Unified recurrent classifier: RNN / LSTM / GRU, uni or bidirectional."""

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.4,
        num_classes:   int   = 3,
        pad_idx:       int   = 0,
        cell_type:     str   = "GRU",
        bidirectional: bool  = True,
    ):
        super().__init__()
        self.cell_type     = cell_type.upper()
        self.bidirectional = bidirectional
        self.num_dirs      = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
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
        embedded = self.dropout(self.embedding(x))
        packed   = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.cell_type == "LSTM":
            _, (hidden, _) = self.rnn(packed)
        else:
            _, hidden = self.rnn(packed)

        if self.bidirectional:
            sentence = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            sentence = hidden[-1]

        return self.fc(self.dropout(sentence))

ablation_configs = [
    {"cell_type": "RNN",  "bidirectional": False, "label": "Vanilla RNN"},
    {"cell_type": "LSTM", "bidirectional": False, "label": "Uni-LSTM"},
    {"cell_type": "GRU",  "bidirectional": False, "label": "Uni-GRU"},
    {"cell_type": "LSTM", "bidirectional": True,  "label": "Bi-LSTM"},
    {"cell_type": "GRU",  "bidirectional": True,  "label": "Bi-GRU"},
]

ABLATION_EPOCHS = 5
ablation_results = []

for cfg in ablation_configs:
    print(f"\n--- {cfg['label']} ---")
    m = RecurrentSentiment(
        vocab_size    = len(vocab),
        embed_dim     = 128,
        hidden_size   = 128,
        num_layers    = 2,
        dropout       = 0.4,
        num_classes   = NUM_CLASSES,
        pad_idx       = PAD_IDX,
        cell_type     = cfg["cell_type"],
        bidirectional = cfg["bidirectional"],
    ).to(device)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    h = train_model(
        m, train_dl, test_dl,
        epochs=ABLATION_EPOCHS, lr=1e-3, class_weights=class_weights
    )
    best_acc = max(h["test_acc"])
    ablation_results.append({
        "label": cfg["label"],
        "best_test_acc": best_acc,
        "params": n_params,
        "history": h,
    })
    print(f"Best acc: {best_acc:.4f}  |  Params: {n_params:,}")


print(f"{'Config':<18} {'Params':>12} {'Best Test Acc':>14}")
print("-" * 46)
for r in ablation_results:
    print(f"{r['label']:<18} {r['params']:>12,} {r['best_test_acc']:>14.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for r in ablation_results:
    axes[0].plot(
        range(1, ABLATION_EPOCHS + 1), r["history"]["test_acc"],
        marker="o", label=r["label"]
    )
axes[0].set_title("Test Accuracy by Cell Type")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Test Accuracy")
axes[0].legend()

labels     = [r["label"]         for r in ablation_results]
best_accs  = [r["best_test_acc"] for r in ablation_results]
param_cnts = [r["params"] / 1e6  for r in ablation_results]

x = np.arange(len(labels))
w = 0.35
ax2 = axes[1]
bars1 = ax2.bar(x - w/2, best_accs,  w, label="Best Test Acc", color="steelblue")
ax2.set_ylabel("Best Test Accuracy", color="steelblue")
ax2.set_ylim(0, 1.1)
ax3 = ax2.twinx()
bars2 = ax3.bar(x + w/2, param_cnts, w, label="Params (M)",    color="coral", alpha=0.8)
ax3.set_ylabel("Parameters (Millions)", color="coral")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right")
ax2.set_title("Accuracy vs Parameter Count")

lines1, lab1 = ax2.get_legend_handles_labels()
lines2, lab2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper left")

plt.tight_layout()
plt.show()


@torch.no_grad()
def get_predictions(m, loader):
    m.eval()
    all_preds, all_labels = [], []
    for x, y, lengths in loader:
        x      = x.to(device)
        logits = m(x, lengths)
        preds  = logits.argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    return torch.cat(all_preds), torch.cat(all_labels)

preds, labels = get_predictions(model, test_dl)

print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 54)
for cls, name in enumerate(LABEL_NAMES):
    tp      = ((preds == cls) & (labels == cls)).sum().item()
    fp      = ((preds == cls) & (labels != cls)).sum().item()
    fn      = ((preds != cls) & (labels == cls)).sum().item()
    support = (labels == cls).sum().item()
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{name:<12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10,}")

overall_acc = (preds == labels).float().mean().item()
print(f"\nOverall Accuracy: {overall_acc:.4f}")


conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
for p, l in zip(preds, labels):
    conf[l, p] += 1

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf.numpy(), cmap="Blues")
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"Pred\n{n}" for n in LABEL_NAMES], fontsize=10)
ax.set_yticklabels([f"True\n{n}" for n in LABEL_NAMES], fontsize=10)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, conf[i, j].item(), ha="center", va="center", fontsize=12,
                color="white" if conf[i, j] > conf.max() * 0.6 else "black")
ax.set_title("Confusion Matrix — GRU on Amazon Reviews (Test Set)")
plt.colorbar(im)
plt.tight_layout()
plt.show()


@torch.no_grad()
def predict(text: str, m=model):
    m.eval()
    ids     = encode(text)
    x       = torch.tensor(ids,      dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    logits  = m(x, lengths)
    probs   = F.softmax(logits, dim=1)[0]
    label   = int(logits.argmax(1).item())
    return LABEL_NAMES[label], float(probs[label]), probs.tolist()

custom_reviews = [
    "This moisturizer is absolutely amazing. My skin has never felt so soft and hydrated!",
    "Terrible product. It broke me out badly and the smell is awful. Do not buy.",
    "It's okay for the price. Nothing special but gets the job done.",
    "I've repurchased this serum three times. Works perfectly for my dry skin.",
    "The packaging looked nice but the product itself was disappointing and watery.",
    "Decent shampoo, lathers well and smells nice. Hair feels clean but nothing extraordinary.",
]

print(f"{'Review':<62} {'Sentiment':<10} {'Conf':>6}   Neg / Neu / Pos")
print("-" * 100)
for review in custom_reviews:
    sentiment, confidence, probs = predict(review)
    prob_str = "  ".join(f"{p:.2f}" for p in probs)
    print(f"{review[:60]:<62} {sentiment:<10} {confidence:>6.4f}   {prob_str}")
