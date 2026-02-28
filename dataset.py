"""
dataset.py
──────────
Chứa: CFG, hằng số lớp, SimpleTokenizer, ScratchDataset,
PretrainedDataset, hàm sinh dữ liệu tổng hợp & chia tập.
"""

import os
import re
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# CONFIG  (chỉnh tại đây)
# ─────────────────────────────────────────────────────────
CFG = {
    "seed":          42,
    # Phân bố độ dài câu: đỉnh ~12–24 từ, hầu hết < 30 từ → 128 token đủ, giảm padding
    "max_len":       128,
    "batch_size":    64,
    "epochs":        100,
    "patience":      10,
    "lr_pretrained": 2e-5,
    "lr_scratch":    3e-4,
    "weight_decay":  1e-2,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size":    30_000,
    "embed_dim":     128,
    "output_dir":    "results",
    # Thư mục lưu model/tokenizer pretrained tải sẵn (dùng offline trên HPC)
    "pretrained_dir": "pretrained",
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ─────────────────────────────────────────────────────────
# ĐƯỜNG DẪN CSV GỐC & CSV ĐÃ LÀM SẠCH
# ─────────────────────────────────────────────────────────
RAW_CSV   = "dataset.csv"
CLEAN_CSV = "dataset_clean.csv"


def _clean_text_version(s: str) -> str:
    """
    Bỏ các đoạn '(Version ...)' trong chuỗi.
    """
    s = re.sub(r"\(Version[^)]*\)", "", str(s))
    return " ".join(s.split()).strip()


# Paraphrase: thay các cụm dễ leak nhãn bằng cách diễn đạt khác (random 1 trong list)
_PARAPHRASE_MAP = [
    (re.compile(r"\bA bill\b", re.I), ["Legislation to", "An act to", "A proposed law to", "Draft law to"]),
    (re.compile(r"\bA BILL\b"), ["Legislation to", "An act to", "A proposed law to"]),
    (re.compile(r"\bExecutive Order\b", re.I), ["Presidential directive", "Order by the President", "White House order", "Cabinet order"]),
    (re.compile(r"\bhearing on\b", re.I), ["session regarding", "meeting concerning", "proceedings on", "panel on"]),
    (re.compile(r"\bhearing to\b", re.I), ["session to", "meeting to", "proceedings to", "panel to"]),
    (re.compile(r"\bannual report\b", re.I), ["yearly report", "year-end report", "periodic report", "fiscal-year report"]),
    (re.compile(r"\bGAO report\b", re.I), ["Government Accountability Office report", "audit report", "watchdog report", "GAO assessment"]),
    (re.compile(r"\bGao report\b"), ["Government Accountability Office report", "audit report", "watchdog report"]),
    (re.compile(r"\breport on\b", re.I), ["assessment of", "review of", "document on", "analysis of"]),
    (re.compile(r"\boversight\b", re.I), ["monitoring", "supervision", "review", "scrutiny"]),
    (re.compile(r"\ba measure\b", re.I), ["a piece of legislation", "a proposal", "a bill", "draft legislation"]),
    (re.compile(r"\ba mesure\b", re.I), ["a piece of legislation", "a proposal", "a bill"]),
]


def _paraphrase_text(s: str) -> str:
    """Thay các cụm trong _PARAPHRASE_MAP bằng một cách diễn đạt khác (random)."""
    s = str(s)
    for pattern, alternatives in _PARAPHRASE_MAP:
        s = pattern.sub(lambda m: np.random.choice(alternatives), s)
    return " ".join(s.split()).strip()


def _load_or_create_clean_csv() -> pd.DataFrame:
    """
    Nếu 'dataset_clean.csv' đã tồn tại -> đọc luôn.
    Ngược lại: đọc 'dataset.csv', làm sạch cột 'input' rồi lưu ra 'dataset_clean.csv'.
    """
    if os.path.exists(CLEAN_CSV):
        return pd.read_csv(CLEAN_CSV)

    df = pd.read_csv(RAW_CSV)
    if "input" in df.columns:
        df["input"] = (
            df["input"]
            .astype(str)
            .apply(_clean_text_version)
            .apply(_paraphrase_text)
        )
    df.to_csv(CLEAN_CSV, index=False)
    return df


# ─────────────────────────────────────────────────────────
# NHÃN LỚP – đọc từ CSV đã làm sạch (cột 'output')
# ─────────────────────────────────────────────────────────
_df_labels = _load_or_create_clean_csv()
CLASSES = sorted(_df_labels["output"].astype(str).unique().tolist())
NUM_CLASSES = len(CLASSES)
CLASS2ID    = {c: i for i, c in enumerate(CLASSES)}
ID2CLASS    = {i: c for c, i in CLASS2ID.items()}




def split_data(texts, labels, seed=CFG["seed"]):
    """Chia 70 / 10 / 20 (train / val / test) stratified."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.30, random_state=seed, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.67, random_state=seed, stratify=y_tmp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─────────────────────────────────────────────────────────
# TOKENIZER ĐƠN GIẢN  (word-level, dùng cho scratch models)
# ─────────────────────────────────────────────────────────
class SimpleTokenizer:
    """Word-level tokenizer với vocab cố định."""

    def __init__(self, vocab_size: int = 30_000):
        self.vocab_size = vocab_size
        self.word2id: dict = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts: list[str]):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 2):
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.word2id.get(w, 1) for w in text.lower().split()]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids


# ─────────────────────────────────────────────────────────
# PYTORCH DATASETS
# ─────────────────────────────────────────────────────────
class ScratchDataset(Dataset):
    """Dataset cho các mô hình train from scratch."""

    def __init__(self, texts, labels, tokenizer: SimpleTokenizer, max_len: int):
        self.data = [
            (tokenizer.encode(t, max_len), l)
            for t, l in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, label = self.data[i]
        return (
            torch.tensor(ids,   dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


class PretrainedDataset(Dataset):
    """Dataset cho các mô hình pretrained (HuggingFace tokenizer)."""

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]


# ─────────────────────────────────────────────────────────
# HELPER: tạo DataLoader
# ─────────────────────────────────────────────────────────
def make_loaders(dataset_cls, tokenizer, splits, max_len, batch_size, **ds_kwargs):
    """
    splits = (X_tr, X_val, X_te, y_tr, y_val, y_te)
    Trả về (train_loader, val_loader, test_loader)
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits
    loaders = []
    for X, y, shuffle in [(X_tr, y_tr, True), (X_val, y_val, False), (X_te, y_te, False)]:
        ds = dataset_cls(X, y, tokenizer, max_len, **ds_kwargs)
        loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=2, pin_memory=True)
        )
    return tuple(loaders)