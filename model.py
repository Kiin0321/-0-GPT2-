import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Embedding Layer
# -----------------------------
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        input_embeddings = self.embedding(input_ids)  # (B, T, D)

        positions = torch.arange(T, device=input_ids.device)
        pos_emb = self.position_embedding(positions)  # (T, D)
        return input_embeddings + pos_emb
        

# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # causal mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), 1))

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) * self.scale  # (B, nh, T, T)
        causal_mask = self.mask[:T, :T].bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # attention_mask: (B, T)
        if attention_mask is not None:
            # 扩展到 (B, 1, 1, T)
            am = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~am, float("-inf"))

        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# -----------------------------
# Transformer Layer
# -----------------------------
class GPTTransformerLayer(nn.Module):
    def __init__(self, dim, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.attn = MultiHeadAttention(dim, num_heads, context_length, dropout, qkv_bias)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), attention_mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# -----------------------------
# GPT Model (支持 attention_mask)
# -----------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(cfg["vocab_size"], cfg["context_length"], cfg["emb_dim"])
        self.embed_drop = nn.Dropout(cfg["dropout"])
        self.layers = nn.ModuleList([
            GPTTransformerLayer(
                dim=cfg["emb_dim"],
                context_length=cfg["context_length"],
                dropout=cfg["dropout"],
                num_heads=cfg["n_heads"],
                qkv_bias=cfg["qkv_bias"],
            )
            for _ in range(cfg["n_layers"])
        ])
        self.ln_f = nn.LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding_layer(input_ids)
        x = self.embed_drop(x)
        for block in self.layers:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        return self.output_layer(x)

class LMDataset(Dataset):
    def __init__(self, data_dir):
        import os
        import glob
        import datasets
        self._pylist = None
        try:
            ds = datasets.load_from_disk(data_dir)
            if isinstance(ds, datasets.DatasetDict):
                ds = ds.get("train", list(ds.values())[0])
            self.ds = ds
        except Exception:
            pattern = os.path.join(data_dir, "data-*.arrow")
            files = sorted(glob.glob(pattern))
            if not files:
                raise
            try:
                ds = datasets.Dataset.from_file(files[0])
                self.ds = ds
            except Exception:
                import pyarrow.ipc as pa_ipc
                try:
                    reader = pa_ipc.open_file(files[0])
                    table = reader.read_all()
                except Exception:
                    reader = pa_ipc.open_stream(files[0])
                    table = reader.read_all()
                col = table.column("input_ids")
                self._pylist = col.to_pylist()
                self.ds = None
    def __len__(self):
        return len(self._pylist) if self._pylist is not None else len(self.ds)
    def __getitem__(self, idx):
        if self._pylist is not None:
            return {"input_ids": self._pylist[idx]}
        ex = self.ds[idx]
        return {"input_ids": ex["input_ids"]}

def load_tokenizer(tokenizer_dir):
    import os
    from transformers import AutoTokenizer, GPT2TokenizerFast
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        vocab = os.path.join(tokenizer_dir, "vocab.json")
        merges = os.path.join(tokenizer_dir, "merges.txt")
        tok = GPT2TokenizerFast(vocab_file=vocab, merges_file=merges)
    if tok.pad_token_id is None:
        try:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        except Exception:
            pass
    return tok

class LMDataCollator:
    def __init__(self, context_length, pad_id):
        self.context_length = context_length
        self.pad_id = pad_id
    def __call__(self, batch):
        input_list = []
        attn_list = []
        label_list = []
        for item in batch:
            orig = item["input_ids"]
            L = min(len(orig), self.context_length)
            x = torch.tensor(orig[:L], dtype=torch.long)
            if L < self.context_length:
                pad = torch.full((self.context_length - L,), self.pad_id, dtype=torch.long)
                x = torch.cat([x, pad], dim=0)
            input_list.append(x)
            am = torch.zeros(self.context_length, dtype=torch.long)
            if L > 0:
                am[:L] = 1
            attn_list.append(am)
            y = torch.full((self.context_length,), -100, dtype=torch.long)
            if L >= 2:
                y[:L-1] = torch.tensor(orig[1:L], dtype=torch.long)
            label_list.append(y)
        input_ids = torch.stack(input_list, dim=0)
        attention_mask = torch.stack(attn_list, dim=0)
        labels = torch.stack(label_list, dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def build_lm_dataloader(data_dir, tokenizer_dir, batch_size, context_length, shuffle=True, num_workers=0):
    tok = load_tokenizer(tokenizer_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ds = LMDataset(data_dir)
    collate = LMDataCollator(context_length, pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)

def get_vocab_size(tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    try:
        base = int(getattr(tok, 'vocab_size', 0) or 0)
        added = int(len(getattr(tok, 'get_added_vocab')() if hasattr(tok, 'get_added_vocab') else {}))
        total = base + added
        return int(total) if total > 0 else int(len(tok))
    except Exception:
        return int(getattr(tok, 'vocab_size', len(tok)))

def compute_lm_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
