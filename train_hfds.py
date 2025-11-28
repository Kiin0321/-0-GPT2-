import os, math, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from model import build_lm_dataloader, GPTModel, compute_lm_loss, get_vocab_size, load_tokenizer

def notebook_train_hfds(
    data_dir,
    tokenizer_dir,
    ctx_len=1024,
    emb_dim=768, n_heads=12, n_layers=12, dropout=0.1, qkv_bias=False, tie_weights=True,
    batch_size=8, num_workers=0,
    lr=3e-4, weight_decay=0.1, warmup_steps=3000, max_steps=0, epochs=1, grad_accum=1,
    log_every=50, save_every=1000, save_dir=r"G:\Anaconda\Kaggle\gpt2\checkpoints", resume=""
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = build_lm_dataloader(data_dir, tokenizer_dir, batch_size, ctx_len, shuffle=True, num_workers=num_workers)
    b = next(iter(dl))
    max_id = int(b["input_ids"].max().item())
    vocab_nominal = get_vocab_size(tokenizer_dir)
    vocab_eff = max(vocab_nominal, max_id + 1)
    cfg = {
        "vocab_size": vocab_eff,
        "context_length": ctx_len,
        "emb_dim": emb_dim,
        "dropout": dropout,
        "n_heads": n_heads,
        "qkv_bias": qkv_bias,
        "n_layers": n_layers,
    }
    model = GPTModel(cfg).to(device)
    if tie_weights:
        model.output_layer.weight = model.embedding_layer.embedding.weight
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias") or ("ln" in n.lower()) or ("layernorm" in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)
    total_steps = max_steps if (max_steps and max_steps > 0) else (epochs * len(dl))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = LambdaLR(optim, lr_lambda)
    try:
        scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu')
    except Exception:
        from torch.cuda.amp import GradScaler as _GS
        scaler = _GS()
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        try:
            optim.load_state_dict(ckpt.get("optimizer", {}))
        except Exception:
            pass
        try:
            scaler.load_state_dict(ckpt.get("scaler", {}))
        except Exception:
            pass
        start_step = int(ckpt.get("step", 0))
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    step = start_step
    running, count = 0.0, 0
    p_epoch = tqdm(total=total_steps, desc="训练", unit="step")
    for _ in range(epochs):
        for batch in dl:
            step += 1
            elapsed = step - start_step
            if device == "cuda":
                try:
                    from torch.amp import autocast as _ac
                    _cuda_ctx = _ac('cuda')
                except Exception:
                    from torch.cuda.amp import autocast as _ac
                    _cuda_ctx = _ac()
                with _cuda_ctx:
                    logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                    loss = compute_lm_loss(logits, batch["labels"].to(device)) / grad_accum
            else:
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = compute_lm_loss(logits, batch["labels"]) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item()
            count += 1
            if log_every and elapsed % log_every == 0:
                avg = running * grad_accum / max(1, count)
                ppl = math.exp(avg) if avg < 700 else float('inf')
                lr_now = scheduler.get_last_lr()[0]
                valid_tokens = int((batch["labels"] != -100).sum().item())
                p_epoch.set_postfix_str(f"loss {avg:.4f} | ppl {ppl:.2f} | lr {lr_now:.6f} | valid {valid_tokens}")
                running, count = 0.0, 0
            if save_every and elapsed % save_every == 0:
                path = os.path.join(save_dir, f"ckpt_step_{step}.pt")
                _obj = {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "cfg": cfg,
                }
                try:
                    torch.save(_obj, path)
                except RuntimeError:
                    torch.save(_obj, path, _use_new_zipfile_serialization=False)
            p_epoch.update(1)
            if elapsed >= total_steps:
                break
    p_epoch.close()
    return model

def save_final_model(model, save_dir=r'G:\Anaconda\Kaggle\gpt2\checkpoints', file_name="final_model.pt"):
    os.makedirs(save_dir, exist_ok=True)
    final_ckpt = os.path.join(save_dir, file_name)
    sd_cpu = {}
    for k, v in tqdm(model.state_dict().items(), desc="准备最终权重", unit="param"):
        sd_cpu[k] = v.detach().cpu()
    try:
        torch.save({'model': sd_cpu}, final_ckpt)
    except RuntimeError:
        torch.save({'model': sd_cpu}, final_ckpt, _use_new_zipfile_serialization=False)
    return final_ckpt

_CKPT_CACHE = {}

def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        kth = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, -float('inf')), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))
    return logits

def qa_generate(
    question,
    ckpt_path,
    tokenizer_dir,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    stop_on_eos=True,
    device=None,
    use_half=True,
    cfg_override=None,
    tie_weights=True,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
    qkv_bias=False,
    ctx_len=1024,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = load_tokenizer(tokenizer_dir)
    eos_id = tok.eos_token_id
    vocab_nominal = get_vocab_size(tokenizer_dir)
    cfg = cfg_override or {
        'vocab_size': vocab_nominal,
        'context_length': ctx_len,
        'emb_dim': emb_dim,
        'dropout': dropout,
        'n_heads': n_heads,
        'qkv_bias': qkv_bias,
        'n_layers': n_layers,
    }
    key = (ckpt_path, device, str(use_half), str(cfg))
    cached = _CKPT_CACHE.get(key)
    if cached is None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = GPTModel(cfg)
        if tie_weights:
            try:
                model.output_layer.weight = model.embedding_layer.embedding.weight
            except Exception:
                pass
        try:
            model.load_state_dict(ckpt.get('model', {}), strict=False)
        except Exception:
            model.load_state_dict(ckpt['model'])
        if device == 'cuda' and use_half:
            model.to(dtype=torch.float16)
        model.to(device)
        model.eval()
        _CKPT_CACHE[key] = model
    model = _CKPT_CACHE[key]
    prompt = ("提问: " + str(question)).strip() + "\n\n回答: "
    enc = tok(prompt, add_special_tokens=False, return_attention_mask=False)
    input_ids = torch.tensor(enc['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
    orig_len = input_ids.shape[1]
    try:
        from torch.amp import autocast as _ac
        cuda_ctx = _ac('cuda') if device == 'cuda' else torch.no_grad()
    except Exception:
        from torch.cuda.amp import autocast as _ac
        cuda_ctx = _ac() if device == 'cuda' else torch.no_grad()
    with torch.inference_mode():
        with cuda_ctx:
            for _ in range(max_new_tokens):
                x = input_ids[:, -cfg['context_length']:]
                attn = torch.ones_like(x, dtype=torch.long)
                logits = model(x, attention_mask=attn)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                nid = int(next_id.item())
                input_ids = torch.cat([input_ids, next_id], dim=1)
                if stop_on_eos and eos_id is not None and nid == eos_id:
                    break
    ans_ids = input_ids[0, orig_len:].tolist()
    return tok.decode(ans_ids, skip_special_tokens=True)
