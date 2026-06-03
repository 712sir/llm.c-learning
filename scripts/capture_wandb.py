"""Parse nanoGPT training output.log and create wandb-style plots."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

SAVE_DIR = "d:/study/llm.c-learning/experiments/gpt2-wikitext/wandb-screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Parse output.log
log_path = "D:/study/Project-nanoGPT/wandb/run-20260601_150851-1ugkht12/files/output.log"
with open(log_path, "r") as f:
    lines = f.readlines()

# Parse step lines: step N: train loss X, val loss Y
step_data = []  # (step, train_loss, val_loss)
for line in lines:
    m = re.search(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
    if m:
        step_data.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

# Parse iter lines: iter N: loss X, time Yms, mfu Z%
iter_data = []  # (iter, loss, time_ms, mfu_pct)
for line in lines:
    m = re.search(r"iter (\d+): loss ([\d.]+), time ([\d.]+)ms, mfu ([\d.\-]+)%", line)
    if m:
        iter_data.append((int(m.group(1)), float(m.group(2)),
                          float(m.group(3)), float(m.group(4))))

steps, train_losses, val_losses = zip(*step_data) if step_data else ([], [], [])
iters, losses, times, mfus = zip(*iter_data) if iter_data else ([], [], [], [])

# Filter eval steps from iter times (eval steps take ~3.4s vs normal ~17ms)
normal_iters = [(i, t) for i, t in zip(iters, times) if t < 100]
eval_iters_list = [(i, t) for i, t in zip(iters, times) if t >= 100]

# Compute MFU as percentage (already in %)
valid_mfu = [(i, m) for i, m in zip(iters, mfus) if m > 0]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 5.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

WANDB_BLUE = "#1f77b4"
WANDB_ORANGE = "#ff7f0e"
WANDB_GREEN = "#2ca02c"
WANDB_RED = "#d62728"
WANDB_PURPLE = "#9467bd"

# === Chart 1: Train Loss & Val Loss ===
fig, ax = plt.subplots()
ax.plot(steps, train_losses, label="train_loss", color=WANDB_BLUE,
        linewidth=1.5, alpha=0.85, marker=".", markersize=3)
ax.scatter(steps, val_losses, label="val_loss", color=WANDB_ORANGE,
           s=30, zorder=5, edgecolors="white", linewidth=0.5)

for step, tloss, vloss in zip(steps, train_losses, val_losses):
    ax.annotate(f"{vloss:.2f}", (step, vloss), textcoords="offset points",
                xytext=(0, 10), fontsize=7, color=WANDB_ORANGE, ha="center")

ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Shakespeare Char · Training & Validation Loss (500 steps)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "01_loss_curves.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: 01_loss_curves.png  (steps={len(steps)}, train_loss: {train_losses[0]:.2f} -> {train_losses[-1]:.2f})")

# === Chart 2: Iteration Loss (smoothed) ===
fig, ax = plt.subplots()
ax.plot(iters, losses, color="#d3d3d3", linewidth=0.5, alpha=0.5, label="raw loss")
# Smooth with rolling average
window = 20
if len(losses) > window:
    smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
    ax.plot(iters[window-1:], smoothed, color=WANDB_BLUE, linewidth=1.5, label=f"smoothed (window={window})")
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Shakespeare Char · Training Loss per Iteration", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "02_iter_loss.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: 02_iter_loss.png")

# === Chart 3: Learning Rate Schedule ===
# nanoGPT uses cosine decay: lr decays from 1e-3 to 1e-4 over 5000 steps
# We ran 500 steps, so lr decayed slightly
total_steps = 5000
min_lr = 1e-4
max_lr = 1e-3
warmup = 100
lr_steps = np.arange(0, 501)
lrs = []
for s in lr_steps:
    if s < warmup:
        lrs.append(max_lr * s / warmup)
    elif s > total_steps:
        lrs.append(min_lr)
    else:
        decay_ratio = (s - warmup) / (total_steps - warmup)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        lrs.append(min_lr + coeff * (max_lr - min_lr))

fig, ax = plt.subplots()
ax.plot(lr_steps, lrs, color=WANDB_GREEN, linewidth=1.8)
ax.axvline(x=500, color=WANDB_RED, linestyle="--", linewidth=1, alpha=0.7)
ax.annotate(f"  stopped here\n  (step 500, lr={lrs[500]:.2e})",
            xy=(500, lrs[500]), fontsize=8, color=WANDB_RED)
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Shakespeare Char · Cosine LR Schedule (warmup=100)", fontsize=13, fontweight="bold")
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "03_lr_schedule.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: 03_lr_schedule.png")

# === Chart 4: Iteration Time (normal iters only, exclude eval) ===
fig, ax = plt.subplots()
if normal_iters:
    ni_steps, ni_times = zip(*normal_iters)
    ax.plot(ni_steps, ni_times, color=WANDB_RED, linewidth=1.0, alpha=0.8)
    ax.axhline(y=np.mean(ni_times), color="gray", linestyle="--", linewidth=0.8,
               label=f"mean = {np.mean(ni_times):.1f} ms")
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.legend(fontsize=9)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_title("Shakespeare Char · Iteration Time (normal steps)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "04_iter_time.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: 04_iter_time.png  (mean={np.mean(ni_times):.1f}ms)")

# === Chart 5: MFU ===
fig, ax = plt.subplots()
if valid_mfu:
    mfu_steps, mfu_vals = zip(*valid_mfu)
    ax.plot(mfu_steps, mfu_vals, color=WANDB_PURPLE, linewidth=1.0, alpha=0.8)
    ax.axhline(y=np.mean(mfu_vals), color="gray", linestyle="--", linewidth=0.8,
               label=f"mean = {np.mean(mfu_vals):.2f}%")
    ax.set_ylabel("MFU (%)", fontsize=11)
    ax.legend(fontsize=9)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_title("Shakespeare Char · Model FLOPs Utilization", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "05_mfu.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: 05_mfu.png  (mean={np.mean(mfu_vals):.2f}%)")

# === Chart 6: Combined Dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Loss
ax = axes[0, 0]
ax.plot(steps, train_losses, label="train_loss", color=WANDB_BLUE, linewidth=1.5, marker=".", markersize=3)
ax.scatter(steps, val_losses, label="val_loss", color=WANDB_ORANGE, s=25, zorder=5, edgecolors="white", linewidth=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Loss Curves")
ax.legend(fontsize=9)

# Top-right: Iteration Loss
ax = axes[0, 1]
ax.plot(iters, losses, color="#d3d3d3", linewidth=0.5, alpha=0.4)
if len(losses) > window:
    ax.plot(iters[window-1:], smoothed, color=WANDB_BLUE, linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Raw & Smoothed Iter Loss")

# Bottom-left: LR
ax = axes[1, 0]
ax.plot(lr_steps, lrs, color=WANDB_GREEN, linewidth=1.8)
ax.axvline(x=500, color=WANDB_RED, linestyle="--", linewidth=1)
ax.set_xlabel("Step")
ax.set_ylabel("LR")
ax.set_title("Cosine LR Schedule")
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

# Bottom-right: MFU + Iter Time
ax = axes[1, 1]
ax2 = ax.twinx()
if valid_mfu:
    ax.plot(mfu_steps, mfu_vals, color=WANDB_PURPLE, linewidth=1.0, alpha=0.8, label="MFU")
if normal_iters:
    ax2.plot(ni_steps, ni_times, color=WANDB_RED, linewidth=0.6, alpha=0.5, label="time (ms)")
ax.set_xlabel("Iteration")
ax.set_ylabel("MFU (%)", color=WANDB_PURPLE)
ax2.set_ylabel("Time (ms)", color=WANDB_RED)
ax.set_title("MFU + Iter Time")

fig.suptitle("Shakespeare Char · WandB Dashboard (500 steps, GTX 1650 4GB)", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(SAVE_DIR, "06_combined_dashboard.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: 06_combined_dashboard.png")

# === Chart 7: Final Summary ===
output = f"""WandB Run Summary
====================
Run: week1-demo
URL: https://wandb.ai/models-hefei-university-of-technology/shakespeare-char/runs/1ugkht12
Project: shakespeare-char
Platform: GTX 1650 4GB

Training Config:
  dataset: shakespeare_char
  model: 10.65M params (6 layers, 6 heads, 384 dim)
  batch_size: 4
  block_size: 64
  max_iters: 500
  learning_rate: 1e-3 (cosine to 1e-4)
  warmup_iters: 100
  compile: False (Windows no Triton)

Final Metrics:
  train_loss: {train_losses[-1]:.4f}  (started at {train_losses[0]:.2f}, down {train_losses[0]-train_losses[-1]:.2f})
  val_loss: {val_losses[-1]:.4f}  (started at {val_losses[0]:.2f})
  iter_time (avg): {np.mean(ni_times):.1f} ms
  MFU (avg): {np.mean(mfu_vals):.2f}%
  total_time: ~{sum(times)/1000:.0f}s
"""
with open(os.path.join(SAVE_DIR, "00_run_info.txt"), "w", encoding="utf-8") as f:
    f.write(output)
print("Saved: 00_run_info.txt")

print(f"\nAll {len(os.listdir(SAVE_DIR))} files saved to: {SAVE_DIR}")
print("Done!")
