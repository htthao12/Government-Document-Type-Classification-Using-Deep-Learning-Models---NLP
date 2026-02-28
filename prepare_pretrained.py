
import os

from transformers import AutoModel, AutoTokenizer

from dataset import CFG


MODELS = [
    ("RoBERTa", "FacebookAI/roberta-base"),       # English
    ("XLM-R",   "FacebookAI/xlm-roberta-base"),  # Multilingual
]


def get_local_pretrained_path(hf_id: str) -> str:
    """
    Dùng cùng quy ước với train.py:
    'FacebookAI/roberta-base' → '<pretrained_dir>/FacebookAI_roberta-base'
    """
    base = CFG.get("pretrained_dir", "pretrained")
    safe = hf_id.replace("/", "_")
    return os.path.join(base, safe)


def download_one(name: str, hf_id: str):
    local_dir = get_local_pretrained_path(hf_id)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[{name}] Downloading '{hf_id}' -> '{local_dir}'")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"[{name}] Saved model & tokenizer to: {local_dir}\n")


def main():
    for name, hf_id in MODELS:
        try:
            download_one(name, hf_id)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")


if __name__ == "__main__":
    main()

