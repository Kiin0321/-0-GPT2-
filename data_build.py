import os
os.environ.setdefault('TRANSFORMERS_NO_TF','1')
import re
import hashlib
import random
import datasets
from tqdm.auto import tqdm
import numpy as _np

def get_tokenizer(tokenizer_dir):
    import os as _os
    _os.environ.setdefault('TRANSFORMERS_NO_TF','1')
    try:
        from transformers import AutoTokenizer as _AT
        tok = _AT.from_pretrained(tokenizer_dir)
    except Exception:
        from tokenizers import ByteLevelBPETokenizer as _BBPE
        tok_raw = _BBPE(
            vocab=_os.path.join(tokenizer_dir, 'vocab.json'),
            merges=_os.path.join(tokenizer_dir, 'merges.txt')
        )
        # ensure special tokens exist in vocab
        try:
            tok_raw.add_special_tokens(['<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>'])
        except Exception:
            pass
        class _Tok:
            def __init__(self, t):
                self._t = t
                self.pad_token = '<|pad|>'
                self.eos_token = '<|eos|>'
                self.bos_token = '<|bos|>'
                self.unk_token = '<|unk|>'
                self.pad_token_id = self._t.token_to_id(self.pad_token)
                self.eos_token_id = self._t.token_to_id(self.eos_token)
                self.bos_token_id = self._t.token_to_id(self.bos_token)
                self.unk_token_id = self._t.token_to_id(self.unk_token)
            def __len__(self):
                return self._t.get_vocab_size()
            def __call__(self, texts, add_special_tokens=False, return_attention_mask=False, return_length=False):
                if isinstance(texts, str):
                    texts = [texts]
                encs = [self._t.encode(t) for t in texts]
                out = {"input_ids": [e.ids for e in encs]}
                if return_length:
                    out["length"] = [len(e.ids) for e in encs]
                return out
            def token_to_id(self, t):
                return self._t.token_to_id(t)
        tok = _Tok(tok_raw)
    if getattr(tok, 'pad_token_id', None) is None:
        try:
            tok.pad_token = '<|pad|>'
        except Exception:
            pass
    if getattr(tok, 'eos_token_id', None) is None:
        try:
            tok.eos_token = '<|eos|>'
        except Exception:
            pass
    if getattr(tok, 'bos_token_id', None) is None:
        try:
            tok.bos_token = '<|bos|>'
        except Exception:
            pass
    if getattr(tok, 'unk_token_id', None) is None:
        try:
            tok.unk_token = '<|unk|>'
        except Exception:
            pass
    return tok

def _ensure_dataset(ds):
    if isinstance(ds, datasets.DatasetDict):
        ds = ds['train'] if 'train' in ds else list(ds.values())[0]
    return ds

def _pick_text_col(ds):
    for c in ds.column_names:
        if c.lower() in ("text","content","raw","body"):
            return c
    return ds.column_names[0]

def _to_text_list(ds, text_col="text"):
    xs = ds[text_col]
    try:
        xs = list(xs)
    except Exception:
        pass
    out = []
    for t in xs:
        if t is None:
            continue
        if isinstance(t, bytes):
            t = t.decode("utf-8","ignore")
        elif not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if t:
            out.append(t)
    return out

def _normalize_text(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _is_chinese_ratio_ok(s, min_ratio=0.5):
    total = len(s)
    if total == 0:
        return False
    cn = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return (cn / total) >= min_ratio

def _drop_noisy(s):
    if not _is_chinese_ratio_ok(s, 0.3):
        return True
    if len(s) < 5:
        return True
    return False

def _dedup_texts(texts):
    seen = set()
    out = []
    for t in tqdm(texts, desc="去重", unit="条"):
        key = hashlib.md5(_normalize_text(t).encode("utf-8", "ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

def _segment_long_text(text, max_chars=8000):
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"[\n\r]+|(?<=[。！？；])", text)
    out = []
    buf = []
    cur = 0
    for p in parts:
        if cur + len(p) > max_chars and buf:
            out.append("".join(buf))
            buf = [p]
            cur = len(p)
        else:
            buf.append(p)
            cur += len(p)
    if buf:
        out.append("".join(buf))
    return out








def encode_batch_user(texts, tokenizer):
    enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    out = []
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    for ids in enc["input_ids"]:
        if len(ids) > 0:
            out.append([bos] + ids + [eos])
    return out

def encode_corpus_user(texts, tokenizer, workers=8, batch_size=64):
    encoded = []
    for i in tqdm(range(0, len(texts), batch_size), desc="分词", unit="批"):
        batch = texts[i:i+batch_size]
        encoded.extend(encode_batch_user(batch, tokenizer))
    return encoded

def bucket_sampling_user(encoded, target_tokens, ratios=(0.15, 0.35, 0.35, 0.15)):
    buckets = {0: [], 1: [], 2: [], 3: []}
    for seq in encoded:
        L = len(seq)
        if L < 32:
            buckets[0].append(seq)
        elif L < 128:
            buckets[1].append(seq)
        elif L < 512:
            buckets[2].append(seq)
        else:
            buckets[3].append(seq)
    totals = [int(target_tokens * r) for r in ratios]
    sampled = []
    for b in range(4):
        random.shuffle(buckets[b])
        total = 0
        for seq in buckets[b]:
            if total >= totals[b]:
                break
            sampled.append(seq)
            total += len(seq)
    return sampled

def build_token_stream_user(seqs):
    stream = []
    for s in seqs:
        stream.extend(s)
    return _np.array(stream, dtype=_np.int32)

def build_chunks_user(stream, ctx_len=1024, random_offset=False):
    chunks = []
    n = len(stream)
    offset = random.randint(0, ctx_len - 1) if random_offset else 0
    pos = offset
    while pos + ctx_len <= n:
        chunks.append(stream[pos:pos+ctx_len])
        pos += ctx_len
    return chunks

def write_bin_idx_user(chunks, out_prefix):
    dirp = os.path.dirname(out_prefix)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    if not chunks:
        raise ValueError("no chunks to write: build_chunks_user produced empty list")
    with open(out_prefix + ".bin", "wb") as fbin, open(out_prefix + ".idx", "wb") as fidx:
        offset = 0
        for ch in tqdm(chunks, desc="写入", unit="块"):
            arr = _np.array(ch, dtype=_np.int32)
            arr.tofile(fbin)
            fidx.write(_np.int64(offset))
            fidx.write(_np.int32(len(arr)))
            offset += len(arr)



def mix_packed_bins(prefixes, out_prefix, shuffle=True, seed=42, max_chunks=None):
    dirp = os.path.dirname(out_prefix)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    entries = []
    for si, pref in enumerate(prefixes):
        idx_path = f"{pref}.idx"
        bin_path = f"{pref}.bin"
        with open(idx_path, 'rb') as f:
            i = 0
            while True:
                rec = f.read(12)
                if not rec:
                    break
                off = int.from_bytes(rec[:8], 'little', signed=True)
                ln = int.from_bytes(rec[8:], 'little', signed=True)
                entries.append((si, off, ln, bin_path))
                i += 1
                if max_chunks and i >= max_chunks:
                    break
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(entries)
    with open(f"{out_prefix}.bin", 'wb') as fbin, open(f"{out_prefix}.idx", 'wb') as fidx:
        out_off = 0
        for si, off, ln, bin_path in tqdm(entries, desc="混合写入", unit="块"):
            with open(bin_path, 'rb') as src:
                src.seek(off * 4)
                buf = src.read(ln * 4)
                fbin.write(buf)
                fidx.write(_np.int64(out_off))
                fidx.write(_np.int32(ln))
                out_off += ln

def build_pack_from_arrow_buckets(
    arrow_dir,
    tokenizer_dir,
    out_prefix,
    ctx_len=1024,
    target_tokens=1_000_000_000,
    bucket_ratios=(0.15, 0.35, 0.35, 0.15),
    seed=42,
    min_chars=8,
    max_doc_chars=8000,
    batch_size=64
):
    tok = get_tokenizer(tokenizer_dir)
    rnd = random.Random(seed)
    ds = datasets.load_from_disk(arrow_dir)
    ds = _ensure_dataset(ds)
    col = _pick_text_col(ds)
    if col != "text":
        ds = ds.rename_column(col, "text")
    texts = _to_text_list(ds, "text")
    _tmp = []
    for t in tqdm(texts, desc="清洗", unit="条"):
        if len(t) >= min_chars and not _drop_noisy(t):
            _tmp.append(t)
    texts = _tmp
    texts = _dedup_texts(texts)
    _seg = []
    for t in tqdm(texts, desc="分段", unit="文"):
        _seg.extend(_segment_long_text(t, max_doc_chars))
    texts = _seg
    rnd.shuffle(texts)
    encoded = encode_corpus_user(texts, tok, batch_size=batch_size)
    sampled = bucket_sampling_user(encoded, target_tokens, ratios=bucket_ratios)
    stream = build_token_stream_user(sampled)
    chunks = build_chunks_user(stream, ctx_len=ctx_len, random_offset=False)
    dirp = os.path.dirname(out_prefix)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    write_bin_idx_user(chunks, out_prefix)

def build_pack_from_arrow_buckets_streaming(
    arrow_dir,
    tokenizer_dir,
    out_prefix,
    ctx_len=1024,
    target_tokens=1_000_000_000,
    bucket_ratios=(0.15, 0.35, 0.35, 0.15),
    seed=42,
    min_chars=8,
    max_doc_chars=8000,
    batch_size=64
):
    tok = get_tokenizer(tokenizer_dir)
    rnd = random.Random(seed)
    ds = datasets.load_from_disk(arrow_dir)
    ds = _ensure_dataset(ds)
    col = _pick_text_col(ds)
    if col != "text":
        ds = ds.rename_column(col, "text")
    dirp = os.path.dirname(out_prefix)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    fbin = open(out_prefix + ".bin", "wb")
    fidx = open(out_prefix + ".idx", "wb")
    totals = [int(target_tokens * r) for r in bucket_ratios]
    used = [0, 0, 0, 0]
    acc = []
    out_off = 0
    total_used = 0
    pbar = tqdm(total=target_tokens, desc="打包", unit="tok")
    i = 0
    n = len(ds)
    while total_used < target_tokens and i < n:
        j = min(i + batch_size, n)
        batch = ds[i:j]["text"]
        texts = []
        for t in batch:
            if t is None:
                continue
            if isinstance(t, bytes):
                t = t.decode("utf-8", "ignore")
            t = t.strip()
            if len(t) >= min_chars and not _drop_noisy(t):
                texts.extend(_segment_long_text(t, max_doc_chars))
        if texts:
            enc = tok(texts, add_special_tokens=False, return_attention_mask=False)
            for ids in enc["input_ids"]:
                if not ids:
                    continue
                seq = [tok.bos_token_id] + ids + [tok.eos_token_id]
                L = len(seq)
                if L < 32:
                    b = 0
                elif L < 128:
                    b = 1
                elif L < 512:
                    b = 2
                else:
                    b = 3
                if used[b] >= totals[b]:
                    continue
                acc.extend(seq)
                used[b] += L
                total_used += L
                pbar.update(L)
                while len(acc) >= ctx_len:
                    ch = acc[:ctx_len]
                    arr = _np.asarray(ch, dtype=_np.int32)
                    arr.tofile(fbin)
                    fidx.write(_np.int64(out_off))
                    fidx.write(_np.int32(ctx_len))
                    out_off += ctx_len
                    acc = acc[ctx_len:]
                if total_used >= target_tokens:
                    break
        i = j
    fbin.close()
    fidx.close()
