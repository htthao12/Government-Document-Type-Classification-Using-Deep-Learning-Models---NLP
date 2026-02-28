"""
train.py
────────
Trainer, hàm vẽ biểu đồ, pipeline chính.

Chạy:
    python train.py

Thay dữ liệu thực:
    Sửa hàm load_data() ở dưới để đọc CSV thay vì dùng synthetic data.
"""

import os, time, json, warnings, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report, roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── import từ các file cùng thư mục ──────────────────────
from dataset import (
    CFG, CLASSES, NUM_CLASSES, CLASS2ID,
    RAW_CSV, CLEAN_CSV,
    SimpleTokenizer, ScratchDataset, PretrainedDataset,
    split_data, make_loaders,
)
from models import (
    DeepTextCNN, DeepBiLSTMAttention, DeepRCNN,
    RoBERTaClassifier, XLMRClassifier, count_params,
)

print(f"[INFO] Device: {CFG['device']}")


# ═══════════════════════════════════════════════════════════
# HELPER: đường dẫn local cho pretrained (dùng offline)
# ═══════════════════════════════════════════════════════════
def get_local_pretrained_path(hf_id: str) -> str:
    """
    Chuyển HF id (vd: 'FacebookAI/roberta-base') thành đường dẫn local
    bên trong CFG['pretrained_dir'], vd: 'pretrained/FacebookAI_roberta-base'.
    """
    safe = hf_id.replace("/", "_")
    return os.path.join(CFG.get("pretrained_dir", "pretrained"), safe)


# ═══════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 name: str, lr: float, is_pretrained: bool = False):
        self.model        = model.to(CFG["device"])
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.name         = name
        self.is_pretrained = is_pretrained

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=CFG["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=CFG["epochs"])
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.history         = defaultdict(list)
        self.best_val_f1     = 0.0
        self.patience_counter = 0
        self.best_state      = None
        self.train_time      = 0.0

    # ── forward một batch (xử lý cả scratch lẫn pretrained) ──
    def _forward(self, batch):
        if self.is_pretrained:
            inputs, labels = batch
            inputs = {k: v.to(CFG["device"]) for k, v in inputs.items()}
            labels = labels.to(CFG["device"])
            logits = self.model(**inputs)
        else:
            ids, labels = batch
            ids    = ids.to(CFG["device"])
            labels = labels.to(CFG["device"])
            logits = self.model(ids)
        return logits, labels

    # ── 1 epoch train ──
    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss, preds_all, labels_all = 0.0, [], []

        pbar = tqdm(
            self.train_loader,
            desc=f"  [{self.name}] Ep {epoch+1:03d}/{CFG['epochs']} TRAIN",
            leave=False, ncols=115,
        )
        for batch in pbar:
            self.optimizer.zero_grad()
            logits, labels = self._forward(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds_all.extend(logits.argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        acc      = accuracy_score(labels_all, preds_all)
        f1       = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        return avg_loss, acc, f1

    # ── eval (val hoặc test) ──
    @torch.no_grad()
    def _eval(self, loader, split: str = "VAL"):
        self.model.eval()
        total_loss, preds_all, labels_all, probs_all = 0.0, [], [], []

        pbar = tqdm(loader, desc=f"  [{self.name}] {split}", leave=False, ncols=115)
        for batch in pbar:
            logits, labels = self._forward(batch)
            total_loss += self.criterion(logits, labels).item()
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds_all.extend(probs.argmax(-1))
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs)

        n        = len(loader)
        avg_loss = total_loss / n
        acc      = accuracy_score(labels_all, preds_all)
        f1_mac   = f1_score(labels_all, preds_all, average="macro",    zero_division=0)
        f1_wt    = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
        return avg_loss, acc, f1_mac, f1_wt, \
               np.array(preds_all), np.array(labels_all), np.array(probs_all)

    # ── huấn luyện đầy đủ với Early Stopping ──
    def fit(self):
        total, trainable = count_params(self.model)
        print(f"\n{'='*65}")
        print(f"  Model : {self.name}")
        print(f"  Params: {total:,} total  |  {trainable:,} trainable "
              f"({trainable/1e6:.1f}M)")
        print(f"{'='*65}")

        t0 = time.time()
        for epoch in range(CFG["epochs"]):
            tr_loss, tr_acc, tr_f1 = self._train_epoch(epoch)
            val_loss, val_acc, val_f1, val_wf1, _, _, _ = self._eval(
                self.val_loader, "VAL")
            self.scheduler.step()

            # lưu history
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["train_f1"].append(tr_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["val_wf1"].append(val_wf1)

            improved = val_f1 > self.best_val_f1
            mark = " ✓ BEST" if improved else ""
            print(
                f"  Ep {epoch+1:3d}/{CFG['epochs']} | "
                f"Tr  Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} F1:{tr_f1:.4f} | "
                f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f}{mark}"
            )

            if improved:
                self.best_val_f1 = val_f1
                self.best_state  = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= CFG["patience"]:
                print(f"  [Early Stop] patience={CFG['patience']} reached at epoch {epoch+1}.")
                break

        self.train_time = time.time() - t0
        print(f"  Training time: {self.train_time:.1f}s\n")

        if self.best_state:
            self.model.load_state_dict(self.best_state)

    # ── đánh giá trên test set ──
    def evaluate_test(self) -> dict:
        t0 = time.time()
        _, acc, f1_mac, f1_wt, preds, labels, probs = self._eval(
            self.test_loader, "TEST")
        inf_time = time.time() - t0
        n_test   = len(labels)

        # AUC-ROC (One-vs-Rest)
        lb = label_binarize(labels, classes=list(range(NUM_CLASSES)))
        try:
            auc = roc_auc_score(lb, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = 0.0

        return {
            "model":           self.name,
            "accuracy":        acc,
            "macro_f1":        f1_mac,
            "weighted_f1":     f1_wt,
            "auc_roc":         auc,
            "train_time_s":    self.train_time,
            "inference_time_s": inf_time,
            "throughput_sps":  n_test / inf_time,
            "best_val_f1":     self.best_val_f1,
            "epochs_trained":  len(self.history["train_loss"]),
            # raw arrays – dùng cho biểu đồ
            "preds":           preds,
            "labels":          labels,
            "probs":           probs,
            "cm":              confusion_matrix(labels, preds),
            "report":          classification_report(
                                   labels, preds,
                                   target_names=CLASSES, output_dict=True),
        }


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════
def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")


def plot_training_curves(history: dict, model_name: str, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves – {model_name}", fontsize=14, fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)

    for ax, (tr_key, val_key), title in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc"), ("train_f1","val_f1")],
        ["Loss", "Accuracy", "Macro-F1"],
    ):
        ax.plot(ep, history[tr_key],  label="Train", color="steelblue")
        ax.plot(ep, history[val_key], label="Val",   color="tomato")
        if title == "Macro-F1":
            ax.plot(ep, history["val_wf1"], label="Val Weighted-F1",
                    color="orange", linestyle="--")
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, f"{model_name}_training_curves.png"))


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_dir: str):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")

    for ax, mat, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw Count", "Row-Normalized"],
        ["d", ".2f"],
    ):
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES)
        ax.set_title(title)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))


def plot_roc_auc(probs: np.ndarray, labels: np.ndarray,
                 model_name: str, save_dir: str):
    lb     = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(9, 7))

    aucs = []
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
        score = sk_auc(fpr, tpr)
        aucs.append(score)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves – {model_name}\nMacro AUC = {np.mean(aucs):.4f}",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, f"{model_name}_roc_auc.png"))


def plot_all_roc(all_results: list, save_dir: str):
    """Overlay ROC micro-average của tất cả model lên 1 hình."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for res, color in zip(all_results, colors):
        lb = label_binarize(res["labels"], classes=list(range(NUM_CLASSES)))
        fpr_all, tpr_all = [], []
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(lb[:, i], res["probs"][:, i])
            fpr_all.append(fpr); tpr_all.append(tpr)
        fpr_cat = np.concatenate(fpr_all)
        tpr_cat = np.concatenate(tpr_all)
        idx     = np.argsort(fpr_cat)
        score   = sk_auc(fpr_cat[idx], tpr_cat[idx])
        ax.plot(fpr_cat[idx], tpr_cat[idx], color=color, lw=2.5,
                label=f"{res['model']} (micro AUC={score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Comparison – All Models", fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "all_models_roc.png"))


def plot_summary(all_results: list, save_dir: str):
    """Bar chart so sánh Accuracy / Macro-F1 / Weighted-F1 / AUC / Time / Throughput."""
    models  = [r["model"] for r in all_results]
    colors  = plt.cm.Set2(np.linspace(0, 1, len(models)))
    metrics = [
        ("accuracy",      "Accuracy"),
        ("macro_f1",      "Macro-F1"),
        ("weighted_f1",   "Weighted-F1"),
        ("auc_roc",       "AUC-ROC"),
    ]

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Model Comparison – Government Document Classification",
                 fontsize=16, fontweight="bold")

    # 4 metric bars
    for idx, (key, label) in enumerate(metrics):
        ax   = fig.add_subplot(gs[idx // 2, idx % 2])
        vals = [r[key] for r in all_results]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(3)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.12); ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)

    # Train time
    ax5  = fig.add_subplot(gs[2, 0])
    times = [r["train_time_s"] for r in all_results]
    bars = ax5.bar(models, times, color=colors, edgecolor="white")
    for bar, t in zip(bars, times):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{t:.0f}s", ha="center", va="bottom", fontsize=9)
    ax5.set_title("Training Time (s)", fontsize=12, fontweight="bold")
    ax5.tick_params(axis="x", rotation=25, labelsize=8); ax5.grid(axis="y", alpha=0.3)

    # Throughput
    ax6 = fig.add_subplot(gs[2, 1])
    tps = [r["throughput_sps"] for r in all_results]
    bars = ax6.bar(models, tps, color=colors, edgecolor="white")
    for bar, t in zip(bars, tps):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{t:.0f}", ha="center", va="bottom", fontsize=9)
    ax6.set_title("Inference Throughput (samples/s)", fontsize=12, fontweight="bold")
    ax6.tick_params(axis="x", rotation=25, labelsize=8); ax6.grid(axis="y", alpha=0.3)

    _savefig(fig, os.path.join(save_dir, "summary_comparison.png"))


def save_results_table(all_results: list, save_dir: str) -> pd.DataFrame:
    rows = [
        {
            "Model":             r["model"],
            "Accuracy":          f"{r['accuracy']:.4f}",
            "Macro-F1":          f"{r['macro_f1']:.4f}",
            "Weighted-F1":       f"{r['weighted_f1']:.4f}",
            "AUC-ROC":           f"{r['auc_roc']:.4f}",
            "Train Time (s)":    f"{r['train_time_s']:.1f}",
            "Throughput (sps)":  f"{r['throughput_sps']:.1f}",
            "Epochs":            r["epochs_trained"],
        }
        for r in all_results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "results_table.csv"), index=False)
    print("\n" + "=" * 75)
    print("FINAL RESULTS TABLE")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75 + "\n")
    return df


# ═══════════════════════════════════════════════════════════
# DATA LOADING – đọc từ CSV đã làm sạch (cột 'input', 'output')
# ═══════════════════════════════════════════════════════════
def load_data():
    csv_path = CLEAN_CSV if os.path.exists(CLEAN_CSV) else RAW_CSV
    print(f"[STEP 1] Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    texts = df["input"].astype(str).tolist()
    # ánh xạ nhãn string → id theo CLASS2ID từ dataset.py
    labels = df["output"].astype(str).map(CLASS2ID).tolist()
    print(f"  Total: {len(texts)} samples  |  Classes: {CLASSES}")
    return texts, labels


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    # ── 1. Data ──────────────────────────────────────────────
    texts, labels = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(texts, labels)
    print(f"  Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")

    splits = (X_tr, X_val, X_te, y_tr, y_val, y_te)

    # ── 2. Scratch tokenizer ─────────────────────────────────
    print("\n[STEP 2] Building vocabulary...")
    simple_tok = SimpleTokenizer(CFG["vocab_size"])
    simple_tok.fit(X_tr)
    print(f"  Vocab size: {len(simple_tok.word2id)}")

    def scratch_loaders():
        return make_loaders(ScratchDataset, simple_tok, splits,
                            CFG["max_len"], CFG["batch_size"])

    # ── 3. Pretrained tokenizers ─────────────────────────────
    print("\n[STEP 3] Loading pretrained tokenizers...")
    roberta_tok, xlmr_tok = None, None

    for name, hf_id, attr in [
        ("RoBERTa", "FacebookAI/roberta-base",    "roberta_tok"),
        ("XLM-R",   "FacebookAI/xlm-roberta-base","xlmr_tok"),
    ]:
        local_dir = get_local_pretrained_path(hf_id)
        try:
            tok = AutoTokenizer.from_pretrained(local_dir)
            if attr == "roberta_tok":
                roberta_tok = tok
            else:
                xlmr_tok = tok
            print(f"  {name} tokenizer: OK (loaded from '{local_dir}')")
        except Exception as e:
            print(f"  {name} tokenizer: FAILED when loading from '{local_dir}' ({e})")
            print("    -> Hãy chạy 'python prepare_pretrained.py' trên máy có mạng "
                  "và copy thư mục 'pretrained/' sang HPC.")

    def pretrained_loaders(tok):
        return make_loaders(PretrainedDataset, tok, splits,
                            CFG["max_len"], CFG["batch_size"])

    all_results = []

    # ════════════════════════════════════════════════════════
    # STEP 4 – Train from scratch
    # ════════════════════════════════════════════════════════
    print("\n[STEP 4] Training scratch models...")
    scratch_specs = [
        ("DeepTextCNN",     DeepTextCNN),
        ("DeepBiLSTM-Attn", DeepBiLSTMAttention),
        ("DeepRCNN",        DeepRCNN),
    ]
    for name, ModelCls in scratch_specs:
        model = ModelCls(
            vocab_size=len(simple_tok.word2id),
            embed_dim=CFG["embed_dim"],
            num_classes=NUM_CLASSES,
            max_len=CFG["max_len"],
        )
        tr_l, val_l, te_l = scratch_loaders()
        trainer = Trainer(model, tr_l, val_l, te_l,
                          name=name, lr=CFG["lr_scratch"])
        trainer.fit()
        res = trainer.evaluate_test()
        all_results.append(res)
        plot_training_curves(trainer.history, name, CFG["output_dir"])
        plot_roc_auc(res["probs"], res["labels"], name, CFG["output_dir"])
        print(f"  [{name}] Acc={res['accuracy']:.4f} | "
              f"Macro-F1={res['macro_f1']:.4f} | AUC={res['auc_roc']:.4f}")

    # ════════════════════════════════════════════════════════
    # STEP 5 – Fine-tune pretrained
    # ════════════════════════════════════════════════════════
    print("\n[STEP 5] Training pretrained models...")
    pretrained_specs = [
        ("RoBERTa-CustomHead", RoBERTaClassifier, roberta_tok, "FacebookAI/roberta-base"),
        ("XLMR-LabelAttn",     XLMRClassifier,    xlmr_tok,    "FacebookAI/xlm-roberta-base"),
    ]
    for name, ModelCls, tok, hf_id in pretrained_specs:
        if tok is None:
            print(f"  [{name}] Skipped (tokenizer unavailable)")
            continue
        try:
            model_path = get_local_pretrained_path(hf_id)
            model = ModelCls(NUM_CLASSES, model_name=model_path)
            tr_l, val_l, te_l = pretrained_loaders(tok)
            trainer = Trainer(model, tr_l, val_l, te_l,
                              name=name, lr=CFG["lr_pretrained"],
                              is_pretrained=True)
            trainer.fit()
            res = trainer.evaluate_test()
            all_results.append(res)
            plot_training_curves(trainer.history, name, CFG["output_dir"])
            plot_roc_auc(res["probs"], res["labels"], name, CFG["output_dir"])
            print(f"  [{name}] Acc={res['accuracy']:.4f} | "
                  f"Macro-F1={res['macro_f1']:.4f} | AUC={res['auc_roc']:.4f}")
        except Exception as e:
            print(f"  [{name}] Training failed: {e}")

    # ════════════════════════════════════════════════════════
    # STEP 6 – Plots & reports
    # ════════════════════════════════════════════════════════
    print("\n[STEP 6] Generating evaluation plots & reports...")
    save_results_table(all_results, CFG["output_dir"])
    plot_summary(all_results, CFG["output_dir"])
    plot_all_roc(all_results, CFG["output_dir"])

    # Confusion matrix cho mọi model
    for res in all_results:
        plot_confusion_matrix(res["cm"], res["model"], CFG["output_dir"])

    # Highlight best model
    best = max(all_results, key=lambda r: r["macro_f1"])
    print(f"\n[★ BEST MODEL] {best['model']}  "
          f"Macro-F1={best['macro_f1']:.4f}  AUC={best['auc_roc']:.4f}")
    plot_confusion_matrix(best["cm"], best["model"] + "_BEST", CFG["output_dir"])

    # JSON summary (không lưu arrays lớn)
    skip = {"preds", "labels", "probs", "cm", "report"}
    summary = [{k: v for k, v in r.items() if k not in skip}
               for r in all_results]
    with open(os.path.join(CFG["output_dir"], "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] All outputs in '{CFG['output_dir']}/'")
    for fname in sorted(os.listdir(CFG["output_dir"])):
        print(f"  • {fname}")


if __name__ == "__main__":
    main()