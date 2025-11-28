import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from model import GPTModel, build_lm_dataloader, get_vocab_size, compute_lm_loss
import math

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=r'G:\Anaconda\Kaggle\gpt2\data\lm_thu_wiki_124m')
    p.add_argument('--tokenizer_dir', default=r'G:\Anaconda\Kaggle\gpt2\tokenizer\bytebpe_zh')
    p.add_argument('--context_length', type=int, default=1024)
    p.add_argument('--emb_dim', type=int, default=768)
    p.add_argument('--n_heads', type=int, default=12)
    p.add_argument('--n_layers', type=int, default=12)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--qkv_bias', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--save_dir', default=r'G:\Anaconda\Kaggle\gpt2\checkpoints')
    p.add_argument('--resume', default='')
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--tie_weights', action='store_true')
    args, _ = p.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size = get_vocab_size(args.tokenizer_dir)
    cfg = {
        'vocab_size': vocab_size,
        'context_length': args.context_length,
        'emb_dim': args.emb_dim,
        'dropout': args.dropout,
        'n_heads': args.n_heads,
        'qkv_bias': args.qkv_bias,
        'n_layers': args.n_layers,
    }
    model = GPTModel(cfg).to(device)
    loader = build_lm_dataloader(args.data_dir, args.tokenizer_dir, args.batch_size, args.context_length, shuffle=True, num_workers=0)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('bias') or 'ln' in n or 'LayerNorm' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=args.lr)
    scaler = GradScaler()

    if args.tie_weights:
        model.output_layer.weight = model.embedding_layer.embedding.weight

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt.get('model', {}))
        optim.load_state_dict(ckpt.get('optimizer', {}))
        scaler.load_state_dict(ckpt.get('scaler', {}))
        start_step = ckpt.get('step', 0)

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    step = start_step
    running = 0.0
    count = 0
    total_steps = args.max_steps if args.max_steps else (args.epochs * len(loader))
    def lr_lambda(s):
        if s < args.warmup_steps:
            return s / max(1, args.warmup_steps)
        return max(0.0, (total_steps - s) / max(1, total_steps - args.warmup_steps))
    scheduler = LambdaLR(optim, lr_lambda)
    for epoch in range(args.epochs):
        for batch in loader:
            step += 1
            with autocast(enabled=(device == 'cuda')):
                logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                loss = compute_lm_loss(logits, batch['labels'].to(device)) / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item()
            count += 1
            if args.log_every and step % args.log_every == 0:
                v = running * args.grad_accum / max(1, count)
                try:
                    ppl = math.exp(v) if v < 700 else float('inf')
                except Exception:
                    ppl = float('nan')
                print('step', step, 'loss', f'{v:.4f}', 'ppl', f'{ppl:.2f}', 'lr', f'{scheduler.get_last_lr()[0]:.6f}')
                running = 0.0
                count = 0
            if args.save_every and step % args.save_every == 0:
                path = os.path.join(args.save_dir, f'ckpt_step_{step}.pt')
                torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict(), 'scaler': scaler.state_dict(), 'step': step, 'cfg': cfg}, path)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

if __name__ == '__main__':
    main()
