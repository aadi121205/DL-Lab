import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE    = 64
BLOCK_SIZE    = 128
D_MODEL       = 128
N_HEADS       = 4
N_LAYERS      = 3
D_FF          = 256
DROPOUT       = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS     = 3000
EVAL_INTERVAL = 300
EVAL_ITERS    = 50
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
torch.manual_seed(42)


URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"

with urllib.request.urlopen(URL) as r:
    text = r.read().decode('utf-8')

print(f"Total characters: {len(text):,}")
print(f"Sample:\n{text[:300]}")


chars   = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi    = {c: i for i, c in enumerate(chars)}
itos    = {i: c for c, i in stoi.items()}

encode  = lambda s: [stoi[c] for c in s]
decode  = lambda ids: ''.join(itos[i] for i in ids)

print(f"Vocabulary size: {VOCAB_SIZE}")

data      = torch.tensor(encode(text), dtype=torch.long)
split     = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")


def get_batch(split_name):
    """Return (src, tgt) tensors for a random batch.
    src = context tokens  [B, T]
    tgt = next-token targets [B, T]  (tgt[i] = src[i] shifted right by 1)
    """
    data_split = train_data if split_name == 'train' else val_data
    ix  = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    src = torch.stack([data_split[i : i + BLOCK_SIZE]     for i in ix])
    tgt = torch.stack([data_split[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return src.to(DEVICE), tgt.to(DEVICE)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention Is All You Need'."""

    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Parameters
    ----------
    d_model  : total embedding dimension
    n_heads  : number of heads  (d_model must be divisible by n_heads)
    dropout  : attention dropout probability
    causal   : if True, apply causal (look-ahead) mask  ← used in decoder self-attn
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        """
        q, k, v : (B, T, D)
        Returns : (B, T, D)
        """
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape

        def split_heads(x, T):
            return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(q), T_q)
        K = split_heads(self.W_k(k), T_k)
        V = split_heads(self.W_v(v), T_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal:
            mask = torch.triu(
                torch.ones(T_q, T_k, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf')
            )

        attn_weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = attn_weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.d_k)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Two-layer FFN with ReLU: FFN(x) = max(0, xW1+b1)W2+b2"""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Single Transformer Encoder layer.
    
    Sub-layers:
        1. Multi-Head Self-Attention
        2. Position-wise Feed-Forward
    Each wrapped with residual connection and Layer Norm.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.norm1(x + self.drop(
            self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        ))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x

class TransformerEncoder(nn.Module):
    """Stack of N EncoderLayers."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_key_padding_mask=None):
        x = self.pe(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Single Transformer Decoder layer.

    Sub-layers:
        1. Masked Multi-Head Self-Attention   (causal)
        2. Multi-Head Cross-Attention         (attends to encoder output)
        3. Position-wise Feed-Forward
    Each wrapped with residual connection and Layer Norm.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=True)
        self.cross_attn       = MultiHeadAttention(d_model, n_heads, dropout, causal=False)
        self.ffn              = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.norm1(x + self.drop(
            self.masked_self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask)
        ))
        x = self.norm2(x + self.drop(
            self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        ))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x

class TransformerDecoder(nn.Module):
    """Stack of N DecoderLayers."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.pe(self.embed(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, memory, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Seq2Seq Transformer with shared token embedding between encoder & decoder."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.encoder.embed.weight = self.decoder.embed.weight
        self.output_proj.weight   = self.decoder.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        """
        src : (B, T_src)  — encoder input tokens
        tgt : (B, T_tgt)  — decoder input tokens  (right-shifted targets)
        Returns logits: (B, T_tgt, vocab_size)
        """
        memory = self.encoder(src)
        out    = self.decoder(tgt, memory)
        return self.output_proj(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print(f"Model parameters: {model.count_parameters():,}")
print(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)

@torch.no_grad()
def estimate_loss():
    """Estimate mean loss over EVAL_ITERS batches for train and val."""
    model.eval()
    losses = {}
    for split_name in ('train', 'val'):
        batch_losses = []
        for _ in range(EVAL_ITERS):
            src, tgt = get_batch(split_name)
            logits = model(src, tgt)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), tgt.view(B * T))
            batch_losses.append(loss.item())
        losses[split_name] = np.mean(batch_losses)
    model.train()
    return losses

train_losses, val_losses = [], []
model.train()

for step in range(1, MAX_ITERS + 1):
    if step % EVAL_INTERVAL == 0 or step == 1:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"Step {step:4d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    src, tgt = get_batch('train')
    logits = model(src, tgt)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), tgt.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

print("Training complete.")


eval_steps = [1] + list(range(EVAL_INTERVAL, MAX_ITERS + 1, EVAL_INTERVAL))

plt.figure(figsize=(9, 4))
plt.plot(eval_steps, train_losses, label='Train Loss', marker='o')
plt.plot(eval_steps, val_losses,   label='Val Loss',   marker='s')
plt.xlabel('Training Step')
plt.ylabel('Cross-Entropy Loss')
plt.title('Transformer Training & Validation Loss (Shakespeare)')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int = 40):
    """Autoregressive generation with temperature scaling and top-k sampling.

    The encoder receives the full prompt; the decoder generates one token at a
    time, feeding its own outputs back as new decoder inputs.
    """
    model.eval()
    src = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    memory = model.encoder(src)

    dec_input = src.clone()

    generated = list(encode(prompt))

    for _ in range(max_new_tokens):
        dec_ctx = dec_input[:, -BLOCK_SIZE:]

        dec_out = model.decoder(dec_ctx, memory)
        logits  = model.output_proj(dec_out[:, -1, :])

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        generated.append(next_tok.item())
        dec_input = torch.cat([dec_input, next_tok], dim=1)

    return decode(generated)

prompt_text = "ROMEO:"
output = generate(model, prompt_text, max_new_tokens=300, temperature=0.8, top_k=40)
print("=" * 60)
print(output)
print("=" * 60)
