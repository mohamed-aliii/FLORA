#!/usr/bin/env python3
"""
Generate a Colab notebook for PEFT Fed-MAE experiments.

This creates a self-contained .ipynb that runs all three fine-tuning modes
(full_ft, lora_naive, lora_fednmc) on Retina and/or COVID-FL with
configurable hyperparameters.

Usage:
    python create_peft_notebook.py
"""

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


cells = []

# ==============================================================
# Cell 0: Title
# ==============================================================
cells.append(md_cell("""
# PEFT Fed-MAE: Parameter-Efficient Federated Fine-Tuning

> Experiments for paper: LoRA + FedNMC vs Full Fine-Tuning on Fed-MAE pretrained ViT-B/16.

**Three modes:**
| Mode | Trainable | Classifier | Aggregation | Expected |
|---|---|---|---|---|
| `full_ft` | 100% (85.8M) | Linear + CE | FedAvg | ~91% (baseline) |
| `lora_naive` | <5% (LoRA + head) | Linear + CE | FedAvg | Collapse under non-IID |
| `lora_fednmc` | <5% (LoRA only) | Prototype Cosine | EMA centroids | Recovery |

**Datasets:** Retina (Split-1, 5 clients) and COVID-FL (split_real, 12 clients)
"""))

# ==============================================================
# Cell 1: Mount Drive & Install
# ==============================================================
cells.append(md_cell("## 1. Environment Setup"))

cells.append(code_cell("""
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys

# Install exact timm version required by SSL-FL
subprocess.check_call([sys.executable, "-m", "pip", "install", "timm==0.3.2", "-q"])

# Verify
import timm
print(f"timm version: {timm.__version__}")
assert timm.__version__ == "0.3.2", f"Expected timm 0.3.2, got {timm.__version__}"

# Install other deps
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "matplotlib", "scikit-learn", "pandas", "-q"])
print("Environment ready.")
"""))

# ==============================================================
# Cell 2: Configuration (USER EDITS THIS)
# ==============================================================
cells.append(md_cell("""
## 2. Configuration

**Edit the cell below** to match your Drive layout and choose the experiment.
"""))

cells.append(code_cell("""
# ============================================================
# MASTER CONFIGURATION — EDIT THESE
# ============================================================

# --- Paths ---
DRIVE_REPO    = "/content/drive/MyDrive/fedmamba_salt"     # Where SSL-FL-main lives
DRIVE_RESULTS = "/content/drive/MyDrive/peft_results"      # Results saved here

# --- Dataset selection ---
# Options: "retina" or "covidfl"
DATASET = "covidfl"

# --- Dataset-specific settings (auto-configured below) ---
DATASET_CONFIGS = {
    "retina": {
        "data_path":   "/content/drive/MyDrive/data/Retina",
        "data_set":    "Retina",
        "nb_classes":  2,
        "n_clients":   5,
        "split_type":  "split_1",
        "ckpt_path":   "/content/drive/MyDrive/original/retina/mae_vit_base.pth",
        "mask_ratio":  0.6,
        "ft_blr":      "3e-3",
        "ft_rounds":   100,
    },
    "covidfl": {
        "data_path":   "/content/drive/MyDrive/COVID-FL",
        "data_set":    "COVID-FL",
        "nb_classes":  3,
        "n_clients":   12,
        "split_type":  "split_real",
        "ckpt_path":   "/content/drive/MyDrive/original/covidfl/mae_vit_base.pth",
        "mask_ratio":  0.3,
        "ft_blr":      "3e-3",
        "ft_rounds":   100,
    },
}

CFG = DATASET_CONFIGS[DATASET]

# --- Fine-tuning mode ---
# Options: "full_ft", "lora_naive", "lora_fednmc"
PEFT_MODE = "lora_fednmc"

# --- LoRA Hyperparameters (only used if PEFT_MODE != "full_ft") ---
LORA_RANK    = 32        # Rank. Higher = more expressive. Try 16, 32, 64
LORA_ALPHA   = 64.0      # Scaling. Usually 2*rank is a good default
LORA_TARGETS = "qkv_proj"  # "qkv", "qkv_proj", or "all"

# Advanced Baseline Improvements
# Class weights counteract extreme class scarcity.
if DATASET == "covidfl":
    CLASS_WEIGHTS = "1.0 1.0 20.0"  # Heavy penalty for missing COVID-19
elif DATASET == "retina":
    CLASS_WEIGHTS = "1.0 5.0"  # Adjust ratio based on Retina's class imbalance
else:
    CLASS_WEIGHTS = None

# --- FedNMC Hyperparameters (only used if PEFT_MODE == "lora_fednmc") ---
PROTO_TAU      = 0.1     # Minimum (final) temperature for cosine classifier
PROTO_MOMENTUM = 0.9     # EMA momentum for prototype updates
PROTO_WARMUP_ROUNDS = 1  # Number of initial rounds to use low momentum for cold-start
PROTO_WARMUP_MOMENTUM = 0.3 # Lower EMA momentum used during warmup rounds

# --- Dynamic Temperature Annealing (lora_fednmc only) ---
TAU_INIT        = 1.0    # Starting temperature (1.0 = soft, set == PROTO_TAU to disable)
TAU_DECAY_ROUNDS = 60    # Rounds over which tau anneals to PROTO_TAU
TAU_STAB_THRESH  = 0.85  # Prototype stability threshold to advance annealing

# --- MAB Client Selection (lora_fednmc only) ---
USE_MAB      = True   # Enabled to actively hunt for COVID clients
MAB_C        = 1.0    # Exploration constant (higher = more exploration)
MAB_EMA      = 0.1    # EMA smoothing for reward estimates

# --- Training ---
FT_ROUNDS    = CFG["ft_rounds"]  # Communication rounds
# Automatically lower learning rate for LoRA to prevent catastrophic client drift & NaN overflow
FT_BLR       = "5e-4" if PEFT_MODE != "full_ft" else CFG["ft_blr"]
FT_BATCH_SIZE = 16               # Batch size per client (16 for A100/L4)
NUM_LOCAL_CLIENTS = 5            # Reduced from -1 to 5 to allow MAB to filter out Normal clients
E_EPOCH       = 1                # Local epochs per round
WARMUP_EPOCHS = 5                # LR warmup epochs
MIN_LR        = 5e-5             # Minimum learning rate for cosine decay
LAYER_DECAY   = 0.75             # Layer-wise LR decay (full_ft only)
WEIGHT_DECAY  = 0.05
DROP_PATH     = 0.1

# --- Augmentation (full_ft only) ---
MIXUP    = 0.8
CUTMIX   = 1.0
REPROB   = 0.25
SMOOTHING = 0.1

# --- Misc ---
NUM_WORKERS   = 4
SAVE_FREQ     = 10
SEED          = 0

print(f"Dataset:   {DATASET} ({CFG['data_set']})")
print(f"Mode:      {PEFT_MODE}")
print(f"Clients:   {CFG['n_clients']}, Split: {CFG['split_type']}")
print(f"Rounds:    {FT_ROUNDS}, Batch: {FT_BATCH_SIZE}")
if PEFT_MODE != "full_ft":
    print(f"LoRA:      rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
if PEFT_MODE == "lora_fednmc":
    print(f"FedNMC:    tau={PROTO_TAU}, momentum={PROTO_MOMENTUM}")
    print(f"Tau anneal: {TAU_INIT} -> {PROTO_TAU} over {TAU_DECAY_ROUNDS} stable rounds")
    print(f"MAB:       {'enabled' if USE_MAB else 'disabled'} (c={MAB_C}, ema={MAB_EMA})")
"""))

# ==============================================================
# Cell 3: Setup paths & verify
# ==============================================================
cells.append(md_cell("## 3. Path Verification"))

cells.append(code_cell("""
import os, sys, shutil

# Setup code path
CODE_DIR = os.path.join(DRIVE_REPO, "SSL-FL-main", "code")
FEDMAE_DIR = os.path.join(CODE_DIR, "fed_mae")

# Copy repo to local for faster I/O
LOCAL_REPO = "/content/SSL-FL-code"
if os.path.exists(LOCAL_REPO):
    shutil.rmtree(LOCAL_REPO)
shutil.copytree(CODE_DIR, LOCAL_REPO)
print(f"Copied code to {LOCAL_REPO}")

# Verify key files exist
for f in ["run_peft_finetune_FedAvg.py", "peft_lora.py", "models_vit.py"]:
    fp = os.path.join(LOCAL_REPO, "fed_mae", f)
    assert os.path.isfile(fp), f"Missing: {fp}"
    print(f"  ✓ {f}")

# Verify checkpoint
assert os.path.isfile(CFG["ckpt_path"]), f"Checkpoint not found: {CFG['ckpt_path']}"
ckpt_size = os.path.getsize(CFG["ckpt_path"]) / 1e6
print(f"  ✓ Checkpoint: {ckpt_size:.0f} MB")

# Verify dataset
assert os.path.isdir(CFG["data_path"]), f"Dataset not found: {CFG['data_path']}"
split_dir = os.path.join(CFG["data_path"], f"{CFG['n_clients']}_clients", CFG["split_type"])
if os.path.isdir(split_dir):
    clients = sorted(f for f in os.listdir(split_dir) if f.endswith(".csv"))
    print(f"  ✓ Dataset: {len(clients)} client CSVs in {split_dir}")
else:
    print(f"  ⚠ Split dir not found: {split_dir}")

# Create output directory
RUN_NAME = f"{DATASET}_{PEFT_MODE}_r{LORA_RANK}_class_cond_proto_tau_anneal"
if PEFT_MODE == "full_ft":
    RUN_NAME = f"{DATASET}_{PEFT_MODE}"
OUTPUT_DIR = os.path.join(DRIVE_RESULTS, RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  ✓ Output: {OUTPUT_DIR}")
"""))

# ==============================================================
# Cell 4: Copy dataset locally (optional, for speed)
# ==============================================================
cells.append(md_cell("## 4. Copy Dataset Locally (Optional — Faster I/O)"))

cells.append(code_cell("""
import time

LOCAL_DATA = f"/content/{DATASET}_local"
USE_LOCAL_DATA = True  # Set False to use Drive directly

if USE_LOCAL_DATA and not os.path.exists(LOCAL_DATA):
    print(f"Copying dataset to {LOCAL_DATA}...")
    start = time.time()
    shutil.copytree(CFG["data_path"], LOCAL_DATA)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.0f}s")
    DATA_PATH = LOCAL_DATA
elif USE_LOCAL_DATA:
    DATA_PATH = LOCAL_DATA
    print(f"Local dataset already exists: {LOCAL_DATA}")
else:
    DATA_PATH = CFG["data_path"]
    print(f"Using Drive dataset directly: {DATA_PATH}")

print(f"DATA_PATH = {DATA_PATH}")
"""))

# ==============================================================
# Cell 5: Inspect data distribution
# ==============================================================
cells.append(md_cell("## 5. Data Distribution Inspection"))

cells.append(code_cell("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

split_dir = os.path.join(DATA_PATH, f"{CFG['n_clients']}_clients", CFG["split_type"])
labels_path = os.path.join(DATA_PATH, "labels.csv")

# Load global labels
labels = {}
with open(labels_path) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                labels[parts[0]] = int(float(parts[1]))
            except ValueError:
                pass

# Per-client class distribution
client_csvs = sorted(f for f in os.listdir(split_dir) if f.endswith(".csv"))
client_dists = {}

for csv_file in client_csvs:
    client_name = csv_file.replace(".csv", "")
    with open(os.path.join(split_dir, csv_file)) as f:
        fnames = [line.strip().split(",")[0] for line in f if line.strip()]
    
    class_counts = [0] * CFG["nb_classes"]
    for fname in fnames:
        if fname in labels:
            class_counts[labels[fname]] += 1
    client_dists[client_name] = class_counts

# Plot
fig, ax = plt.subplots(figsize=(max(10, len(client_csvs)), 6))
x = np.arange(len(client_dists))
width = 0.8 / CFG["nb_classes"]
colors = plt.cm.Set2(np.linspace(0, 1, CFG["nb_classes"]))

for c in range(CFG["nb_classes"]):
    counts = [client_dists[k][c] for k in client_dists]
    ax.bar(x + c * width, counts, width, label=f"Class {c}", color=colors[c])

ax.set_xlabel("Client", fontsize=12)
ax.set_ylabel("# Samples", fontsize=12)
ax.set_title(f"{CFG['data_set']} — {CFG['split_type']} Data Distribution", fontsize=14)
ax.set_xticks(x + width * (CFG["nb_classes"] - 1) / 2)
ax.set_xticklabels(list(client_dists.keys()), rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "data_distribution.png"), dpi=150)
plt.show()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'data_distribution.png')}")
"""))

# ==============================================================
# Cell 6: Run PEFT Fine-Tuning
# ==============================================================
cells.append(md_cell("""
## 6. Run PEFT Fine-Tuning

This cell runs the selected fine-tuning mode. Output is printed live and saved to the output directory.
"""))

cells.append(code_cell("""
import subprocess

os.chdir(os.path.join(LOCAL_REPO, "fed_mae"))
sys.path.insert(0, LOCAL_REPO)

# Build command
cmd = [
    sys.executable, "run_peft_finetune_FedAvg.py",
    "--peft_mode", PEFT_MODE,
    "--data_path", DATA_PATH,
    "--data_set", CFG["data_set"],
    "--finetune", CFG["ckpt_path"],
    "--nb_classes", str(CFG["nb_classes"]),
    "--n_clients", str(CFG["n_clients"]),
    "--split_type", CFG["split_type"],
    "--output_dir", OUTPUT_DIR,
    "--model", "vit_base_patch16",
    "--batch_size", str(FT_BATCH_SIZE),
    "--blr", str(FT_BLR),
    "--min_lr", str(MIN_LR),
    "--max_communication_rounds", str(FT_ROUNDS),
    "--E_epoch", str(E_EPOCH),
    "--clip_grad", "1.0",
    "--warmup_epochs", str(WARMUP_EPOCHS),
    "--weight_decay", str(WEIGHT_DECAY),
    "--drop_path", str(DROP_PATH),
    "--save_ckpt_freq", str(SAVE_FREQ),
    "--num_workers", str(NUM_WORKERS),
    "--seed", str(SEED),
    "--smoothing", str(SMOOTHING),
    "--reprob", str(REPROB),
    "--num_local_clients", str(NUM_LOCAL_CLIENTS),
]

# Add mode-specific args
if PEFT_MODE != "full_ft":
    cmd += [
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_targets", LORA_TARGETS,
    ]

if CLASS_WEIGHTS is not None:
    # Disable smoothing and mixup to ensure strict weighted cross entropy applies
    cmd.extend(["--class_weights"] + CLASS_WEIGHTS.split())
    # Override smoothing directly in cmd
    try:
        idx = cmd.index("--smoothing")
        cmd[idx+1] = "0.0"
    except ValueError:
        pass

if PEFT_MODE == "full_ft":
    cmd += [
        "--mixup", str(MIXUP),
        "--cutmix", str(CUTMIX),
    ]

if PEFT_MODE == "lora_fednmc":
    cmd += [
        "--proto_tau", str(PROTO_TAU),
        "--proto_momentum", str(PROTO_MOMENTUM),
        "--proto_warmup_rounds", str(PROTO_WARMUP_ROUNDS),
        "--proto_warmup_momentum", str(PROTO_WARMUP_MOMENTUM),
        "--tau_init", str(TAU_INIT),
        "--tau_decay_rounds", str(TAU_DECAY_ROUNDS),
        "--tau_stability_thresh", str(TAU_STAB_THRESH),
    ]
    if USE_MAB:
        cmd += [
            "--use_mab",
            "--mab_c", str(MAB_C),
            "--mab_ema_alpha", str(MAB_EMA),
        ]

print("=" * 60)
print("Running command:")
print(" ".join(cmd))
print("=" * 60)

# Run with live output
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()

if process.returncode != 0:
    print(f"\\n*** Process exited with code {process.returncode} ***")
else:
    print(f"\\n*** Training complete! Results in: {OUTPUT_DIR} ***")
"""))

# ==============================================================
# Cell 7: Load & visualize results
# ==============================================================
cells.append(md_cell("## 7. Results Visualization"))

cells.append(code_cell("""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load log
log_path = os.path.join(OUTPUT_DIR, "log.txt")
assert os.path.isfile(log_path), f"No log found at {log_path}. Did training complete?"

records = []
with open(log_path) as f:
    for line in f:
        records.append(json.loads(line))

rounds = [r["epoch"] for r in records]
train_loss = [r["train_loss"] for r in records]
test_acc = [r["test_acc"] for r in records]
max_acc = [r["max_acc"] for r in records]

# Per-class recall
per_class_recalls = {c: [] for c in range(CFG["nb_classes"])}
for r in records:
    pcr = r.get("per_class_recall", {})
    for c in range(CFG["nb_classes"]):
        per_class_recalls[c].append(pcr.get(str(c), pcr.get(c, 0)))

# Summary
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
if os.path.isfile(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    print("=" * 50)
    print(f"  Mode:            {summary['mode']}")
    print(f"  Dataset:         {summary['dataset']}")
    print(f"  Best Accuracy:   {summary['best_accuracy']:.2f}%")
    print(f"  Trainable Params: {summary['trainable_params']:,}")
    print(f"  Total Params:    {summary['total_params']:,}")
    print(f"  Param Ratio:     {100*summary['trainable_params']/summary['total_params']:.2f}%")
    print(f"  Training Time:   {summary['training_time']}")
    print(f"  Final Per-Class Recall: {summary.get('per_class_recall_final', 'N/A')}")
    print("=" * 50)

# ---- Publication-quality plots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Test accuracy
axes[0, 0].plot(rounds, test_acc, 'b-', linewidth=2, label='Test Acc')
axes[0, 0].plot(rounds, max_acc, 'b--', linewidth=1, alpha=0.5, label='Best Acc')
axes[0, 0].set_xlabel("Communication Round", fontsize=12)
axes[0, 0].set_ylabel("Test Accuracy (%)", fontsize=12)
axes[0, 0].set_title(f"{PEFT_MODE}: Test Accuracy", fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Training loss
axes[0, 1].plot(rounds, train_loss, 'r-', linewidth=2)
axes[0, 1].set_xlabel("Communication Round", fontsize=12)
axes[0, 1].set_ylabel("Training Loss", fontsize=12)
axes[0, 1].set_title(f"{PEFT_MODE}: Training Loss", fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Per-class recall
colors = plt.cm.Set1(np.linspace(0, 1, CFG["nb_classes"]))
for c in range(CFG["nb_classes"]):
    axes[1, 0].plot(rounds, per_class_recalls[c], color=colors[c],
                   linewidth=2.5, label=f"Class {c}")
axes[1, 0].set_xlabel("Communication Round", fontsize=12)
axes[1, 0].set_ylabel("Recall", fontsize=12)
axes[1, 0].set_title(f"{PEFT_MODE}: Per-Class Recall", fontsize=14)
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_ylim(-0.05, 1.05)
axes[1, 0].grid(True, alpha=0.3)

# 4. Confusion matrix (final round)
confusion = np.array(records[-1].get("confusion_matrix", []))
if confusion.size > 0:
    im = axes[1, 1].imshow(confusion, interpolation='nearest', cmap='Blues')
    axes[1, 1].set_xlabel("Predicted", fontsize=12)
    axes[1, 1].set_ylabel("True", fontsize=12)
    axes[1, 1].set_title(f"{PEFT_MODE}: Confusion Matrix (Round {rounds[-1]})", fontsize=14)
    axes[1, 1].set_xticks(range(CFG["nb_classes"]))
    axes[1, 1].set_yticks(range(CFG["nb_classes"]))
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axes[1, 1].text(j, i, str(confusion[i, j]),
                          ha="center", va="center", fontsize=14,
                          color="white" if confusion[i, j] > confusion.max()/2 else "black")
    plt.colorbar(im, ax=axes[1, 1])

plt.suptitle(f"PEFT Fed-MAE: {PEFT_MODE} on {CFG['data_set']} ({CFG['split_type']})",
            fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "results_overview.png"), dpi=200)
plt.show()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'results_overview.png')}")
"""))

# ==============================================================
# Cell 8: Comparison across modes
# ==============================================================
cells.append(md_cell("""
## 8. Multi-Mode Comparison

Run this cell after completing experiments with multiple modes. It loads results from the Drive results directory and creates comparison plots for the paper.
"""))

cells.append(code_cell("""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Scan all results dirs
result_dirs = {}
for d in os.listdir(DRIVE_RESULTS):
    full = os.path.join(DRIVE_RESULTS, d)
    summary_p = os.path.join(full, "summary.json")
    if os.path.isfile(summary_p) and d.startswith(DATASET):
        with open(summary_p) as f:
            result_dirs[d] = json.load(f)

if len(result_dirs) < 2:
    print(f"Only {len(result_dirs)} result(s) found for {DATASET}. Run more experiments first.")
else:
    print(f"Found {len(result_dirs)} experiments for {DATASET}:")
    for name, s in result_dirs.items():
        print(f"  {name}: {s['best_accuracy']:.2f}% | {s['trainable_params']:,} params")

    # --- Comparison plot: test accuracy over rounds ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mode_colors = {"full_ft": "#2196F3", "lora_naive": "#F44336", "lora_fednmc": "#4CAF50"}
    mode_labels = {"full_ft": "Full Fine-Tune", "lora_naive": "LoRA + CE (Naive)",
                   "lora_fednmc": "LoRA + FedNMC (Ours)"}

    for name, summary in sorted(result_dirs.items()):
        log_path = os.path.join(DRIVE_RESULTS, name, "log.txt")
        if not os.path.isfile(log_path):
            continue
        records = [json.loads(l) for l in open(log_path)]
        mode = summary["mode"]
        color = mode_colors.get(mode, "gray")
        label = mode_labels.get(mode, mode)

        rounds = [r["epoch"] for r in records]
        test_acc = [r["test_acc"] for r in records]
        axes[0].plot(rounds, test_acc, color=color, linewidth=2.5, label=f"{label} ({summary['best_accuracy']:.1f}%)")

    axes[0].set_xlabel("Communication Round", fontsize=14)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=14)
    axes[0].set_title(f"Test Accuracy Comparison — {CFG['data_set']}", fontsize=16)
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # --- Bar chart: best accuracy vs param count ---
    modes = sorted(result_dirs.keys())
    best_accs = [result_dirs[m]["best_accuracy"] for m in modes]
    param_ratios = [100 * result_dirs[m]["trainable_params"] / result_dirs[m]["total_params"]
                    for m in modes]
    bar_colors = [mode_colors.get(result_dirs[m]["mode"], "gray") for m in modes]
    bar_labels = [mode_labels.get(result_dirs[m]["mode"], result_dirs[m]["mode"]) for m in modes]

    x = np.arange(len(modes))
    bars = axes[1].bar(x, best_accs, color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=11)
    axes[1].set_ylabel("Best Accuracy (%)", fontsize=14)
    axes[1].set_title(f"Best Accuracy & Param Efficiency — {CFG['data_set']}", fontsize=16)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, acc, pr in zip(bars, best_accs, param_ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{acc:.1f}%\\n({pr:.1f}% params)", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(DRIVE_RESULTS, f"{DATASET}_comparison.png"), dpi=200)
    plt.show()
    print(f"Saved: {os.path.join(DRIVE_RESULTS, f'{DATASET}_comparison.png')}")
"""))

# ==============================================================
# Cell 9: Per-class recall comparison (paper figure)
# ==============================================================
cells.append(md_cell("## 9. Per-Class Recall Comparison (Paper Figure)"))

cells.append(code_cell("""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mode_colors = {"full_ft": "#2196F3", "lora_naive": "#F44336", "lora_fednmc": "#4CAF50"}
mode_labels = {"full_ft": "Full Fine-Tune", "lora_naive": "LoRA + CE (Naive)",
               "lora_fednmc": "LoRA + FedNMC (Ours)"}
mode_styles = {"full_ft": "-", "lora_naive": "--", "lora_fednmc": "-"}

fig, axes = plt.subplots(1, CFG["nb_classes"], figsize=(6 * CFG["nb_classes"], 5))
if CFG["nb_classes"] == 1:
    axes = [axes]

for name in sorted(os.listdir(DRIVE_RESULTS)):
    full = os.path.join(DRIVE_RESULTS, name)
    summary_p = os.path.join(full, "summary.json")
    log_p = os.path.join(full, "log.txt")
    if not (os.path.isfile(summary_p) and os.path.isfile(log_p) and name.startswith(DATASET)):
        continue

    with open(summary_p) as f:
        summary = json.load(f)
    records = [json.loads(l) for l in open(log_p)]
    mode = summary["mode"]
    color = mode_colors.get(mode, "gray")
    label = mode_labels.get(mode, mode)
    style = mode_styles.get(mode, "-")

    rounds = [r["epoch"] for r in records]
    for c in range(CFG["nb_classes"]):
        recalls = [r.get("per_class_recall", {}).get(str(c), r.get("per_class_recall", {}).get(c, 0))
                   for r in records]
        axes[c].plot(rounds, recalls, color=color, linestyle=style,
                    linewidth=2.5, label=label)

for c in range(CFG["nb_classes"]):
    axes[c].set_xlabel("Communication Round", fontsize=12)
    axes[c].set_ylabel("Recall", fontsize=12)
    axes[c].set_title(f"Class {c} Recall", fontsize=14)
    axes[c].set_ylim(-0.05, 1.05)
    axes[c].legend(fontsize=10)
    axes[c].grid(True, alpha=0.3)

plt.suptitle(f"Per-Class Recall Comparison — {CFG['data_set']} ({CFG['split_type']})",
            fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(DRIVE_RESULTS, f"{DATASET}_per_class_recall.png"), dpi=200)
plt.show()
print(f"Saved: {DATASET}_per_class_recall.png")
"""))

# ==============================================================
# Cell 10: Generate paper table (LaTeX)
# ==============================================================
cells.append(md_cell("## 10. Generate Paper Table"))

cells.append(code_cell("""
import json, os

mode_labels = {"full_ft": "Full Fine-Tune (Baseline)",
               "lora_naive": "LoRA + Linear + CE",
               "lora_fednmc": "LoRA + FedNMC (Ours)"}

rows = []
for name in sorted(os.listdir(DRIVE_RESULTS)):
    full = os.path.join(DRIVE_RESULTS, name)
    summary_p = os.path.join(full, "summary.json")
    if not (os.path.isfile(summary_p) and name.startswith(DATASET)):
        continue
    with open(summary_p) as f:
        s = json.load(f)

    trainable_pct = 100 * s["trainable_params"] / s["total_params"]
    comm_per_round = s["trainable_params"]  # Only trainable params communicated
    total_comm = comm_per_round * s["rounds"]

    pcr = s.get("per_class_recall_final", {})
    recalls = [f"{pcr.get(str(c), pcr.get(c, 0)):.2f}" for c in range(CFG["nb_classes"])]

    rows.append({
        "mode": mode_labels.get(s["mode"], s["mode"]),
        "acc": s["best_accuracy"],
        "params": f"{s['trainable_params']/1e6:.2f}M",
        "pct": f"{trainable_pct:.1f}\\\\%",
        "comm": f"{total_comm/1e9:.2f}B",
        "recalls": recalls,
        "rank": s.get("lora_rank", "-"),
    })

# Print LaTeX table
print("\\\\begin{table}[t]")
print(f"\\\\caption{{Federated fine-tuning comparison on {CFG['data_set']} ({CFG['split_type']}).}}")
print(f"\\\\label{{tab:{DATASET}_results}}")
recall_cols = " ".join([f"c" for _ in range(CFG["nb_classes"])])
print(f"\\\\begin{{tabular}}{{l c c c c {recall_cols}}}")
print("\\\\toprule")
recall_headers = " & ".join([f"R$_{{{c}}}$" for c in range(CFG["nb_classes"])])
print(f"Method & Acc (\\\\%) & Params & \\\\% Total & Comm. & {recall_headers} \\\\\\\\")
print("\\\\midrule")
for r in rows:
    recall_str = " & ".join(r["recalls"])
    print(f"{r['mode']} & {r['acc']:.2f} & {r['params']} & {r['pct']} & {r['comm']} & {recall_str} \\\\\\\\")
print("\\\\bottomrule")
print("\\\\end{tabular}")
print("\\\\end{table}")

# Also save as markdown table
print("\\n\\n--- Markdown Version ---\\n")
recall_headers_md = " | ".join([f"R_{c}" for c in range(CFG["nb_classes"])])
print(f"| Method | Acc (%) | Params | % Total | Comm. | {recall_headers_md} |")
print(f"|---|---|---|---|---|{'|'.join(['---'] * CFG['nb_classes'])}|")
for r in rows:
    recall_str = " | ".join(r["recalls"])
    pct = r["pct"].replace("\\\\\\\\%", "%")
    print(f"| {r['mode']} | {r['acc']:.2f} | {r['params']} | {pct} | {r['comm']} | {recall_str} |")
"""))

# ==============================================================
# Cell 11: Ablation sweep helper
# ==============================================================
cells.append(md_cell("""
## 11. Ablation Sweep (Optional)

Run multiple LoRA rank values to generate an ablation table. Edit the list below and run.
"""))

cells.append(code_cell("""
# ---- ABLATION CONFIG ----
# Uncomment to run an ablation sweep over LoRA ranks
# WARNING: This will run MULTIPLE full training runs sequentially!

ABLATION_RANKS = [16, 32, 64]
ABLATION_MODE = "lora_fednmc"  # or "lora_naive"

RUN_ABLATION = False  # Set to True to execute

if RUN_ABLATION:
    for rank in ABLATION_RANKS:
        abl_name = f"{DATASET}_{ABLATION_MODE}_r{rank}_{LORA_TARGETS}"
        abl_output = os.path.join(DRIVE_RESULTS, abl_name)
        os.makedirs(abl_output, exist_ok=True)

        os.chdir(os.path.join(LOCAL_REPO, "fed_mae"))
        cmd = [
            sys.executable, "run_peft_finetune_FedAvg.py",
            "--peft_mode", ABLATION_MODE,
            "--data_path", DATA_PATH,
            "--data_set", CFG["data_set"],
            "--finetune", CFG["ckpt_path"],
            "--nb_classes", str(CFG["nb_classes"]),
            "--n_clients", str(CFG["n_clients"]),
            "--split_type", CFG["split_type"],
            "--output_dir", abl_output,
            "--model", "vit_base_patch16",
            "--batch_size", str(FT_BATCH_SIZE),
            "--blr", str(FT_BLR),
            "--max_communication_rounds", str(FT_ROUNDS),
            "--E_epoch", str(E_EPOCH),
            "--warmup_epochs", str(WARMUP_EPOCHS),
            "--weight_decay", str(WEIGHT_DECAY),
            "--drop_path", str(DROP_PATH),
            "--save_ckpt_freq", str(SAVE_FREQ),
            "--num_workers", str(NUM_WORKERS),
            "--seed", str(SEED),
            "--smoothing", str(SMOOTHING),
            "--reprob", str(REPROB),
            "--num_local_clients", "-1",
            "--lora_rank", str(rank),
            "--lora_alpha", str(rank * 2),
            "--lora_targets", LORA_TARGETS,
        ]
        if ABLATION_MODE == "lora_fednmc":
            cmd += ["--proto_tau", str(PROTO_TAU), "--proto_momentum", str(PROTO_MOMENTUM)]

        print(f"\\n{'='*60}")
        print(f"  Ablation: rank={rank}")
        print(f"  Output: {abl_output}")
        print(f"{'='*60}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        print(f"rank={rank} done, return code: {process.returncode}")
else:
    print("Ablation sweep disabled. Set RUN_ABLATION = True to execute.")
"""))

# ==============================================================
# Build notebook
# ==============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_path = Path("notebooks/PEFT_FedMAE_Experiments.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"[OK] Notebook written to: {out_path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
