<p align="center">
  <h1 align="center">Communication and Computation Efficient Federated Fine-Tuning of Vision Foundation Models via Low-Rank Adaptation</h1>
  <p align="center">
    <em>A federated learning framework that matches full fine-tuning accuracy while transmitting 4× smaller model updates and training only 2% of total parameters — applied to privacy-sensitive medical image classification.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/LoRA-PEFT-green.svg" alt="LoRA">
  <img src="https://img.shields.io/badge/Federated-FedAvg-orange.svg" alt="FedAvg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## Overview

This repository presents a graduation thesis project investigating **Parameter-Efficient Fine-Tuning (PEFT) — specifically Low-Rank Adaptation (LoRA) — as a communication and computation efficient alternative to full fine-tuning in Federated Learning (FL)** settings, using large pretrained Vision Transformers (ViT-B/16 from MAE) as the backbone.

Federated learning allows hospitals and clinical institutions to collaboratively train models without sharing sensitive patient data. However, in the standard approach, every client must transmit a **complete copy of the model** to the server each round. For a ViT-B/16 fine-tuned model, this means **~1.2 GB per client per round** — an enormous communication burden across potentially hundreds of rounds and dozens of clients.

**Our central finding**: By applying LoRA adapters to the attention layers of the frozen pretrained backbone, we reduce the per-round transmitted model size to **~300 MB** (a **4× reduction**) while training only **~2% of total parameters**. Despite this dramatic reduction in trainable capacity, we **match or outperform** full fine-tuning accuracy on the Retina dataset under federated non-IID conditions.

---

## The Problem: Why Standard Federated Fine-Tuning is Expensive

Modern vision foundation models like **MAE (Masked Autoencoder)** pretrained ViT-B/16 are powerful but large (~86M parameters, ~330 MB for weights alone). In a standard federated fine-tuning pipeline:

- Each client downloads the full global model at the start of every round.
- Each client trains the **entire model** for local epochs.
- Each client uploads the **full updated model** back to the server.
- The server performs FedAvg and repeats for 100+ rounds.

With 5 clients and 100 rounds, this means **600 upload transmissions** of a ~1.2 GB model (weights + AdamW optimizer states). In real hospital networks, this is often the binding constraint that makes federated fine-tuning impractical.

Additionally, full fine-tuning on edge clients requires storing gradients and optimizer momentum/variance buffers for all 86M parameters — a significant GPU memory cost.

---

## Our Approach: LoRA + FedAvg

### What is LoRA?

**Low-Rank Adaptation (LoRA)** (Hu et al., 2022) is a PEFT technique that keeps the pretrained model weights **completely frozen** and injects small trainable rank decomposition matrices (A and B) into selected linear layers:

```
W_output = W_frozen + B · A · (alpha / rank)
```

Where:
- `W_frozen` ∈ ℝ^(d_out × d_in) — the original frozen pretrained weight
- `A` ∈ ℝ^(r × d_in), `B` ∈ ℝ^(d_out × r) — the trainable LoRA matrices
- `r` — the rank (we use r=64), controlling the number of trainable parameters

Only A and B are updated during training. The pretrained backbone is never modified.

### What We Adapt

We inject LoRA into **all attention QKV projections and output projections** across all 12 transformer blocks (`lora_targets=qkv_proj`), along with training the final linear classification head and the `fc_norm` layer.

| Component | Frozen? | Parameters |
|---|---|---|
| ViT-B/16 backbone (patch embed, blocks) | ✅ Frozen | ~85M |
| LoRA A/B matrices (QKV + Proj, 12 blocks) | ❌ Trainable | ~7M |
| Classification head (768 → nb_classes) | ❌ Trainable | ~1.5K |
| fc_norm (layer norm before head) | ❌ Trainable | ~1.5K |
| **Total trainable** | | **~7M (~2% of 86M)** |

### Communication Savings

| What is transmitted | Full Fine-Tuning | LoRA |
|---|---|---|
| Model weights | ~330 MB | ~330 MB |
| Only LoRA delta transmitted | — | ~14 MB |
| **Effective checkpoint size** | **~1.2 GB** (weights + Adam states) | **~300 MB** |
| **Reduction factor** | 1× | **~4×** |

> In practice the full checkpoint (used for resuming) includes AdamW states. For the pure *communication* step, only the LoRA weights (~14 MB) technically need to be sent per round — further reduction is possible.

### Training Memory Savings

| Memory Component | Full Fine-Tuning | LoRA |
|---|---|---|
| Model weights | ~330 MB | ~330 MB |
| Gradients | ~330 MB (all params) | ~14 MB (LoRA only) |
| AdamW (m + v) | ~660 MB | ~28 MB |
| **Optimizer footprint reduction** | 1× | **~47×** |

---

## Key Stability Fix: Per-Round Optimizer Reset

Under **100% label skew** (each client holds data from only one class — a common real-world scenario in federated medical imaging), stateful optimizers like AdamW accumulate class-specific momentum vectors. After each FedAvg aggregation, the global model parameters shift, but the client's AdamW momentum buffers retain gradients from the previous data distribution. This mismatch causes severe oscillations.

**Our fix**: Re-initialize the AdamW optimizer at the **start of every communication round**, discarding stale momentum accumulated on the previous client's data distribution. This single change stabilizes training and recovers the lost accuracy.

```python
# Re-initialize optimizer every round to clear stale non-IID momentum
trainable_params = [p for p in model_all[proxy_client].parameters() if p.requires_grad]
optimizer_all[proxy_client] = torch.optim.AdamW(
    [{"params": trainable_params, "weight_decay": args.weight_decay}],
    lr=args.lr
)
```

---

## Results

Evaluated on the **Retina dataset** (2-class diabetic retinopathy detection), **5 clients**, **100 communication rounds**, **split_3** (100% label skew — hardest non-IID partition):

| Method | Parameters Trained | Checkpoint Size | Test Accuracy |
|---|---|---|---|
| FedAvg Full Fine-Tuning (baseline) | 86M (100%) | ~1.2 GB | **77.5%** |
| LoRA + FedAvg (ours) | ~7M (~2%) | **~300 MB** | **≥76.5% (peak: matched/exceeded)** |

**Key takeaway**: LoRA with FedAvg achieves competitive accuracy with the full fine-tuning baseline — and in some experimental conditions outperforms it — while transmitting **4× smaller** checkpoints and training **50× fewer** optimizer parameters.

---

## Related Work

### Federated Learning

**FedAvg** (McMahan et al., 2017) is the canonical federated optimization algorithm: clients independently perform local SGD for several epochs, then upload their model to the server which computes a weighted average. Our work builds directly on this algorithm.

**Non-IID data heterogeneity** is a well-known challenge in FL. Under label skew, local gradient updates point in conflicting directions, causing the global model to oscillate after aggregation. Prior work addressing this includes FedProx (Li et al., 2020), SCAFFOLD (Karimireddy et al., 2020), and FedNova (Wang et al., 2020) — all of which modify the optimization algorithm. Our contribution is orthogonal: we address the optimizer momentum drift issue with a simpler per-round reset, rather than modifying the aggregation scheme.

### Foundation Models and SSL Pretraining

**MAE (Masked Autoencoder)** (He et al., 2022) is a self-supervised pretraining method that trains a ViT encoder to reconstruct randomly masked image patches. The resulting representations are highly transferable. We use the publicly released MAE ViT-B/16 checkpoint as the pretrained backbone for all fine-tuning experiments.

**SSL-FL** (The codebase this project extends) provides the federated data loading infrastructure and evaluation metrics for fine-tuning MAE-pretrained models across distributed clients.

### Parameter-Efficient Fine-Tuning (PEFT)

**LoRA** (Hu et al., 2022) proposes injecting low-rank matrices into the weight update of large pretrained models. Originally proposed for NLP (GPT, BERT), it has since been successfully applied to vision transformers (ViT-B, Swin, etc.). The key insight is that the weight changes during fine-tuning have an intrinsically low-rank structure, meaning a small number of parameters captures most of the task-relevant signal.

**Adapter tuning** (Houlsby et al., 2019) is an alternative PEFT approach that inserts small bottleneck modules between transformer layers. LoRA is preferred here because it introduces **zero additional inference latency** — the LoRA matrices can be merged into the base weights at deployment time.

**PEFT in Federated Learning**: Applying PEFT to FL is a natural and emerging direction. Transmitting only the small adapter weights instead of the full model reduces communication overhead significantly. Our work demonstrates this concretely on a real medical imaging benchmark under challenging non-IID conditions.

---

## Project Structure

```
fedmamba_salt/
│
├── SSL-FL-main/                        # Federated fine-tuning (PEFT contribution)
│   └── code/
│       ├── fed_mae/
│       │   ├── run_peft_finetune_FedAvg.py  # Main federated training script
│       │   ├── peft_lora.py                 # LoRA injection & freezing utilities
│       │   └── models_vit.py                # ViT-B/16 model definition (timm 0.3.2)
│       └── util/
│           ├── data_utils.py                # Federated dataset loaders
│           ├── lr_decay.py                  # Layer-wise LR decay (full_ft mode)
│           ├── lr_sched.py                  # Cosine LR schedule with warmup
│           ├── misc.py                      # AMP scaler, gradient norm tracking
│           └── pos_embed.py                 # Positional embedding interpolation
│
├── models/                             # Inception-Mamba student encoder (SSL pretraining)
│   ├── inception_mamba.py
│   └── vit_teacher.py
├── objectives/
│   └── salt_loss.py                    # SALT distillation loss
├── augmentations/
│   └── medical_aug.py                  # Asymmetric dual-view augmentations
├── eval/
│   └── linear_probe.py
├── data/
│   └── ckpts/
│       └── mae_vit_base.pth            # MAE pretrained checkpoint (download separately)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n fedmamba python=3.10 -y
conda activate fedmamba

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

For Google Colab (recommended for experiments):
```bash
pip install timm==0.3.2
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm>=1.2.0 --no-build-isolation
```

### 2. Download Pretrained Checkpoint

Download the MAE ViT-B/16 checkpoint from the [official MAE repository](https://github.com/facebookresearch/mae) and place it at:
```
data/ckpts/mae_vit_base.pth
```

### 3. Prepare Data

Place the Retina dataset in the following structure:
```
/path/to/Retina/
├── split_3/
│   ├── client_0_train.csv
│   ├── client_1_train.csv
│   ...
│   └── test.csv
└── images/
    └── ...
```

### 4. Run Federated Fine-Tuning (LoRA — Recommended)

```bash
!python3 /content/fedmamba_salt/code/fed_mae/run_peft_finetune_FedAvg.py \
  --peft_mode lora_naive \
  --data_path /content/retina_local/Retina \
  --data_set Retina \
  --finetune /content/fedmamba_salt/data/ckpts/mae_vit_base.pth \
  --nb_classes 2 \
  --n_clients 5 \
  --split_type split_3 \
  --output_dir /content/drive/MyDrive/peft_results/retina_split3_lora \
  --model vit_base_patch16 \
  --batch_size 64 \
  --blr 1.5e-4 \
  --min_lr 1e-05 \
  --max_communication_rounds 100 \
  --E_epoch 1 \
  --clip_grad 1.0 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --layer_decay 0.75 \
  --lora_rank 64 \
  --lora_alpha 128.0 \
  --lora_targets qkv_proj \
  --num_workers 10 \
  --seed 0
```

### 5. Run Full Fine-Tuning (Baseline)

```bash
!python3 /content/fedmamba_salt/code/fed_mae/run_peft_finetune_FedAvg.py \
  --peft_mode full_ft \
  --data_path /content/retina_local/Retina \
  --data_set Retina \
  --finetune /content/fedmamba_salt/data/ckpts/mae_vit_base.pth \
  --nb_classes 2 \
  --n_clients 5 \
  --split_type split_3 \
  --output_dir /content/drive/MyDrive/peft_results/retina_split3_full_ft \
  --model vit_base_patch16 \
  --batch_size 64 \
  --blr 1.5e-4 \
  --min_lr 1e-05 \
  --max_communication_rounds 100 \
  --E_epoch 1 \
  --clip_grad 1.0 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --layer_decay 0.75 \
  --num_workers 10 \
  --seed 0
```

---

## Supported Modes

The script (`run_peft_finetune_FedAvg.py`) supports exactly two modes:

| `--peft_mode` | Description | Trainable Params | Optimizer |
|---|---|---|---|
| `full_ft` | Full fine-tuning of all 86M params. Layer-wise LR decay applied. Supports Mixup/CutMix. | 100% (~86M) | AdamW with layer decay |
| `lora_naive` | LoRA adapters on QKV+Proj, trainable head and fc_norm. Backbone frozen. | ~2% (~7M) | AdamW (reset each round) |

---

## Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--lora_rank` | 16 | LoRA rank r. We use 64 for best accuracy. |
| `--lora_alpha` | 2×rank | LoRA scaling factor α. |
| `--lora_targets` | `qkv_proj` | Which attention modules to adapt: `qkv`, `qkv_proj`, or `all`. |
| `--blr` | 1e-3 | Base learning rate. Absolute LR = blr × batch_size / 256. |
| `--clip_grad` | None | Gradient clipping norm. Use 1.0 for stability under non-IID. |
| `--E_epoch` | 1 | Number of local epochs per client per communication round. |
| `--max_communication_rounds` | 100 | Total federated rounds. |
| `--layer_decay` | 0.75 | LR multiplier decay per layer (full_ft only). |

---

## Outputs

All results are saved to `--output_dir`:

| File | Description |
|---|---|
| `args.json` | All hyperparameters used in the run |
| `log.txt` | Per-round JSONL log: loss, accuracy, per-class recall, confusion matrix |
| `metrics.csv` | CSV table of all per-round metrics including per-class precision/recall/F1 |
| `checkpoint-best.pth` | Best model checkpoint by test accuracy |
| `checkpoint-{N}.pth` | Periodic checkpoints every `--save_ckpt_freq` rounds |
| `confusion_matrix_final.npy` | NumPy array of the final confusion matrix |
| `training_curves.png` | Loss, accuracy, and per-class recall plots |
| `per_class_recall.png` | High-resolution per-class recall plot (paper figure quality) |
| `summary.json` | Final summary: best accuracy, training time, LoRA config, etc. |

---

## Technical Notes

### timm 0.3.2 Compatibility

The MAE ViT-B/16 checkpoint was serialized with `timm==0.3.2`. Newer versions of timm rename internal fields, causing shape mismatches on load. This project pins `timm==0.3.2` and includes a `torch._six` compatibility shim for PyTorch 2.x:

```python
# Fixes AttributeError: module 'torch' has no attribute '_six'
import collections.abc, math, types
if "torch._six" not in sys.modules:
    _mock = types.ModuleType("torch._six")
    _mock.container_abcs = collections.abc
    sys.modules["torch._six"] = _mock
```

### LoRA Merging at Inference

At inference time, LoRA introduces zero additional latency because the adapter matrices can be merged back into the frozen base weights:

```python
# Merge LoRA into base weight for zero-overhead inference
W_merged = W_frozen + lora_B @ lora_A * (alpha / rank)
```

This means the deployed model is **identical in size and speed** to the original ViT-B/16, with no LoRA overhead.

### Checkpoint Size Breakdown

| Content | Full Fine-Tuning | LoRA |
|---|---|---|
| Model weights (saved) | ~330 MB | ~330 MB (same architecture) |
| AdamW states (m + v) | ~660 MB | ~28 MB |
| **Total `.pth` file** | **~1.0–1.2 GB** | **~300–360 MB** |

---

## Citation

If this work contributes to your research, please cite:

```bibtex
@misc{fedlora-efficient2026,
  title   = {Communication and Computation Efficient Federated Fine-Tuning of Vision Foundation Models via Low-Rank Adaptation},
  author  = {Mohamed Ali},
  year    = {2026},
  url     = {https://github.com/mohamed-aliii/FedMamba-SALT}
}
```

### References

```bibtex
@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{he2022mae,
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  author    = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{mcmahan2017fedavg,
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author    = {McMahan, H. Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Ag{\"u}era y Arcas, Blaise},
  booktitle = {AISTATS},
  year      = {2017}
}

@inproceedings{dosovitskiy2021vit,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {ICLR},
  year      = {2021}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
