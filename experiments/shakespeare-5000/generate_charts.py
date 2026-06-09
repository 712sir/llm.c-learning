"""
Generate wandb-style charts from Shakespeare 5000-step training log.

Run: python generate_charts.py
Output: wandb-screenshots/ (6 PNGs + 1 run_info.txt)
"""

import re
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 1. Parse training log ──
log_path = os.path.join(os.path.dirname(__file__), 'train_5000.log')
out_dir = os.path.join(os.path.dirname(__file__), 'wandb-screenshots')
os.makedirs(out_dir, exist_ok=True)

with open(log_path, 'r') as f:
    content = f.read()

# Extract config params
model_params = float(re.search(r'number of parameters: ([\d.]+)M', content).group(1))
learning_rate = float(re.search(r'learning_rate = ([\d.e+-]+)', content).group(1))
min_lr = float(re.search(r'min_lr = ([\d.e+-]+)', content).group(1))
warmup_iters = int(re.search(r'warmup_iters = (\d+)', content).group(1))
lr_decay_iters = int(re.search(r'lr_decay_iters = (\d+)', content).group(1))
max_iters = int(re.search(r'max_iters = (\d+)', content).group(1))
batch_size = int(re.search(r'batch_size = (\d+)', content).group(1))
block_size = int(re.search(r'block_size = (\d+)', content).group(1))
tokens_per_iter = int(re.search(r'tokens per iteration will be: ([\d,]+)', content).group(1).replace(',', ''))

# Parse iteration lines
iter_pattern = r'iter (\d+): loss ([\d.]+), time ([\d.]+)ms, mfu ([\d.\-]+)%'
iters = []
for m in re.finditer(iter_pattern, content):
    iters.append({
        'iter': int(m.group(1)),
        'loss': float(m.group(2)),
        'time_ms': float(m.group(3)),
        'mfu': float(m.group(4)),
    })

# Parse eval step lines
step_pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
steps = []
for m in re.finditer(step_pattern, content):
    steps.append({
        'step': int(m.group(1)),
        'train_loss': float(m.group(2)),
        'val_loss': float(m.group(3)),
    })

iter_nums = np.array([x['iter'] for x in iters])
iter_loss = np.array([x['loss'] for x in iters])
iter_time = np.array([x['time_ms'] for x in iters])
iter_mfu = np.array([x['mfu'] for x in iters])

step_nums = np.array([s['step'] for s in steps])
train_losses = np.array([s['train_loss'] for s in steps])
val_losses = np.array([s['val_loss'] for s in steps])

# Compute LR schedule (cosine decay with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

lr_vals = np.array([get_lr(i) for i in iter_nums])

# Compute exponential moving average for loss
alpha = 0.1
loss_ema = np.zeros_like(iter_loss)
loss_ema[0] = iter_loss[0]
for i in range(1, len(iter_loss)):
    loss_ema[i] = alpha * iter_loss[i] + (1 - alpha) * loss_ema[i-1]

# Exclude eval iters from timing (time > 100ms = eval step)
is_normal = iter_time < 100
normal_time = iter_time[is_normal]
avg_time = np.mean(normal_time)
normal_mfu = iter_mfu[is_normal]
avg_mfu = np.mean(normal_mfu[normal_mfu > 0])

# ── Style ──
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})

COLOR_TRAIN = '#1f77b4'
COLOR_VAL = '#ff7f0e'
COLOR_LR = '#2ca02c'
COLOR_TIME = '#9467bd'
COLOR_MFU = '#d62728'

# ── Chart 1: Train/Val Loss Curves ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(step_nums, train_losses, 'o-', color=COLOR_TRAIN, markersize=4, linewidth=1.5, label='Train Loss')
ax.plot(step_nums, val_losses, 's-', color=COLOR_VAL, markersize=4, linewidth=1.5, label='Val Loss')
ax.axvline(x=1750, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate('Best val loss: 1.4704 (step 1750)', xy=(1750, 1.4704),
            xytext=(2500, 2.5), arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
            fontsize=8, color='green')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Shakespeare 5000-step Training: Train & Val Loss')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '01_loss_curves.png'), bbox_inches='tight')
plt.close(fig)

# ── Chart 2: Iteration Loss + EMA Smoothing ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iter_nums, iter_loss, alpha=0.15, color=COLOR_TRAIN, linewidth=0.5, label='Raw iter loss')
ax.plot(iter_nums, loss_ema, color=COLOR_TRAIN, linewidth=1.2, label=f'EMA (α={alpha})')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Shakespeare 5000-step Training: Iteration Loss + Smoothing')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '02_iter_loss.png'), bbox_inches='tight')
plt.close(fig)

# ── Chart 3: LR Schedule ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iter_nums, lr_vals * 1000, color=COLOR_LR, linewidth=1.5)  # ×1000 for readability
ax.set_xlabel('Iteration')
ax.set_ylabel('Learning Rate (×10⁻³)')
ax.set_title(f'Cosine LR Schedule (warmup={warmup_iters}, decay={lr_decay_iters})')
ax.axvline(x=warmup_iters, color='gray', linestyle=':', alpha=0.5, label=f'Warmup end ({warmup_iters})')
ax.axhline(y=learning_rate * 1000, color='gray', linestyle=':', alpha=0.3)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '03_lr_schedule.png'), bbox_inches='tight')
plt.close(fig)

# ── Chart 4: Iteration Time ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iter_nums[is_normal], normal_time, '.', color=COLOR_TIME, markersize=2, alpha=0.5, label=f'Per-iter time')
ax.axhline(y=avg_time, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_time:.1f} ms')
ax.set_xlabel('Iteration')
ax.set_ylabel('Time (ms)')
ax.set_title(f'Shakespeare 5000-step Training: Iteration Time (avg={avg_time:.1f}ms)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '04_iter_time.png'), bbox_inches='tight')
plt.close(fig)

# ── Chart 5: MFU ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iter_nums[is_normal], normal_mfu[normal_mfu > 0] if len(normal_mfu[normal_mfu > 0]) > 0 else normal_mfu,
        '.', color=COLOR_MFU, markersize=2, alpha=0.5, label='Per-iter MFU')
ax.axhline(y=avg_mfu, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_mfu:.1f}%')
ax.set_xlabel('Iteration')
ax.set_ylabel('MFU (%)')
ax.set_title(f'Shakespeare 5000-step Training: Model FLOPs Utilization (avg={avg_mfu:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '05_mfu.png'), bbox_inches='tight')
plt.close(fig)

# ── Chart 6: Combined Dashboard ──
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: Loss curves
ax = axes[0, 0]
ax.plot(step_nums, train_losses, 'o-', color=COLOR_TRAIN, markersize=3, linewidth=1, label='Train')
ax.plot(step_nums, val_losses, 's-', color=COLOR_VAL, markersize=3, linewidth=1, label='Val')
ax.set_title('Loss')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Top-right: Iteration time
ax = axes[0, 1]
ax.plot(iter_nums[is_normal], normal_time, '.', color=COLOR_TIME, markersize=1, alpha=0.5)
ax.axhline(y=avg_time, color='red', linestyle='--', linewidth=1)
ax.set_title(f'Iter Time (avg={avg_time:.1f}ms)')
ax.grid(True, alpha=0.3)

# Bottom-left: MFU
ax = axes[1, 0]
ax.plot(iter_nums[is_normal], normal_mfu, '.', color=COLOR_MFU, markersize=1, alpha=0.5)
ax.axhline(y=avg_mfu, color='red', linestyle='--', linewidth=1)
ax.set_title(f'MFU (avg={avg_mfu:.1f}%)')
ax.grid(True, alpha=0.3)

# Bottom-right: LR schedule
ax = axes[1, 1]
ax.plot(iter_nums, lr_vals * 1000, color=COLOR_LR, linewidth=1.5)
ax.set_title('LR Schedule (×10⁻³)')
ax.grid(True, alpha=0.3)

fig.suptitle(f'Shakespeare 5000-step Training Dashboard | {model_params}M params | RTX 4090',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '06_combined_dashboard.png'), bbox_inches='tight')
plt.close(fig)

# ── Run Info ──
final_train = train_losses[-1]
final_val = val_losses[-1]
best_val = min(val_losses)
best_step = step_nums[np.argmin(val_losses)]

info = f"""WandB Run Summary
====================
Run: shakespeare-5000
Project: shakespeare-char
Platform: AutoDL RTX 4090 24GB (WANDB_MODE=offline)

Training Config:
  dataset: shakespeare_char (1,003,854 train / 111,540 val tokens)
  model: {model_params}M params (6 layers, 6 heads, 384 dim)
  batch_size: {batch_size}
  block_size: {block_size}
  tokens_per_iter: {tokens_per_iter:,}
  max_iters: {max_iters}
  learning_rate: {learning_rate} (cosine to {min_lr})
  warmup_iters: {warmup_iters}
  dropout: 0.2
  compile: False

Final Metrics:
  train_loss (final): {final_train:.4f}
  val_loss (final): {final_val:.4f}
  best_val_loss: {best_val:.4f} (step {best_step})
  iter_time (avg): {avg_time:.1f} ms
  MFU (avg): {avg_mfu:.1f}%
  total_time: ~3 min

Key Insight:
  Overfitting starts at step 1750-2000. Best model is at step {best_step}
  with val loss {best_val:.4f}. Final model at step 5000 has train loss
  {final_train:.4f} but val loss worsened to {final_val:.4f}.
  Use step {best_step} checkpoint for generation, not step 5000.
"""

with open(os.path.join(out_dir, '00_run_info.txt'), 'w') as f:
    f.write(info)

print(f'Done! Generated {len(os.listdir(out_dir))} files in {out_dir}')
for fname in sorted(os.listdir(out_dir)):
    fpath = os.path.join(out_dir, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {fname} ({size_kb:.0f} KB)')
