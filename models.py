

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ═══════════════════════════════════════════════════════════
# MODEL 1 – Deep TextCNN  (~50M tham số)
# ═══════════════════════════════════════════════════════════
class DeepTextCNN(nn.Module):
    """
    Multi-scale CNN với 5 kích thước filter, mỗi scale 3 Conv layers.
    Sau đó max-pool → ghép → deep FC head 4 tầng.
    ~50M params với vocab=30k, embed=512, filter=512.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 num_classes: int, max_len: int):
        super().__init__()
        num_filters  = 64
        filter_sizes = [2, 3, 4]

        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.3)

        # 3 × (Conv1d × 2 + BN + GELU)
        self.conv_blocks = nn.ModuleList()
        for fs in filter_sizes:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(embed_dim,  num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters), nn.GELU(),
                nn.Conv1d(num_filters, num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters), nn.GELU(),
            ))

        fc_in = num_filters * len(filter_sizes)  # 192
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, L)
        x = self.embed_dropout(self.embedding(x))   # (B, L, E)
        x = x.transpose(1, 2)                        # (B, E, L)
        pooled = [
            F.adaptive_max_pool1d(blk(x), 1).squeeze(-1)
            for blk in self.conv_blocks
        ]
        return self.fc(torch.cat(pooled, dim=1))


# ═══════════════════════════════════════════════════════════
# MODEL 2 – Deep BiLSTM + Multi-head Attention  (~50M)
# ═══════════════════════════════════════════════════════════
class DeepBiLSTMAttention(nn.Module):
    """
    4 tầng BiLSTM xếp chồng (có residual khi shape khớp) +
    Multi-head Self-Attention + Attentive Pooling → FC.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 num_classes: int, max_len: int):
        super().__init__()
        hidden = 64

        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.3)
        self.pos_enc       = nn.Embedding(max_len, embed_dim)

        # 2-layer Stacked BiLSTM
        self.lstms = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_size = embed_dim
        for i in range(2):
            self.lstms.append(
                nn.LSTM(in_size, hidden, batch_first=True,
                        bidirectional=True,
                        dropout=0.2 if i < 1 else 0.0)
            )
            self.norms.append(nn.LayerNorm(hidden * 2))
            in_size = hidden * 2

        # Multi-head Self-Attention
        self.attn      = nn.MultiheadAttention(hidden * 2, num_heads=4,
                                               dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden * 2)

        # Attentive Pooling
        self.attn_pool = nn.Linear(hidden * 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        out  = self.embed_dropout(self.embedding(x) + self.pos_enc(pos))

        for lstm, norm in zip(self.lstms, self.norms):
            residual = out if out.shape[-1] == lstm.hidden_size * 2 else None
            out, _   = lstm(out)
            out      = norm(out)
            if residual is not None:
                out = out + residual

        # Self-Attention + residual
        attn_out, _ = self.attn(out, out, out)
        out = self.attn_norm(out + attn_out)

        # Attentive Pooling
        weights = F.softmax(self.attn_pool(out), dim=1)   # (B, L, 1)
        pooled  = (out * weights).sum(dim=1)               # (B, H)
        return self.fc(pooled)


# ═══════════════════════════════════════════════════════════
# MODEL 3 – Deep RCNN  (~50M)
# ═══════════════════════════════════════════════════════════
class DeepRCNN(nn.Module):
    """
    3-layer BiGRU trích xuất ngữ cảnh → concat với embedding →
    3 scale CNN (mỗi scale 3 layers) → Adaptive MaxPool →
    ghép → deep FC.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 num_classes: int, max_len: int):
        super().__init__()
        hidden      = 64
        num_filters = 64

        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.3)

        self.rnn = nn.GRU(embed_dim, hidden, num_layers=2,
                  batch_first=True, bidirectional=True, dropout=0.2)

        rcnn_in = embed_dim + hidden * 2   # concat(embed, rnn_out)
        self.conv_blocks = nn.ModuleList()
        for fs in [3, 5]:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(rcnn_in,     num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters), nn.GELU(),
                nn.Conv1d(num_filters, num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters), nn.GELU(),
            ))

        fc_in = num_filters * 2   # 128
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        emb      = self.embed_dropout(self.embedding(x))  # (B, L, E)
        rnn_out, _ = self.rnn(emb)                          # (B, L, 2H)
        combined = torch.cat([emb, rnn_out], dim=-1)        # (B, L, E+2H)
        combined = combined.transpose(1, 2)                 # (B, E+2H, L)

        pooled = [
            F.adaptive_max_pool1d(blk(combined), 1).squeeze(-1)
            for blk in self.conv_blocks
        ]
        return self.fc(torch.cat(pooled, dim=1))


# ═══════════════════════════════════════════════════════════
# MODEL 4 – RoBERTa (English) + Custom Head  (fine-tune 5 layers cuối)
# ═══════════════════════════════════════════════════════════
class RoBERTaClassifier(nn.Module):
    """
    Backbone: RoBERTa-base (English) đã tải sẵn về local.
    Head: Attentive Pooling + Multi-sample Dropout (5 dropout rates).
    """

    def __init__(self, num_classes: int, model_name: str):
        """
        model_name: đường dẫn local đến model RoBERTa (vd: pretrained/FacebookAI_roberta-base).
        """
        super().__init__()
        self.bert   = AutoModel.from_pretrained(model_name,
                                                ignore_mismatched_sizes=True)
        hidden      = self.bert.config.hidden_size  # 768

        # --- Freeze toàn bộ ---
        for p in self.bert.parameters():
            p.requires_grad = False

        # --- Unfreeze 5 transformer layers cuối ---
        n_layers = len(self.bert.encoder.layer)
        for i in range(n_layers - 5, n_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

        # --- Custom head ---
        self.attn_pool = nn.Linear(hidden, 1)
        self.dropouts  = nn.ModuleList(
            [nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024,   512), nn.LayerNorm(512),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        seq = self.bert(**kwargs).last_hidden_state   # (B, L, H)

        # Attentive Pooling (mask padding)
        mask    = attention_mask.unsqueeze(-1).float()
        weights = self.attn_pool(seq).masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim=1)
        pooled  = (seq * weights).sum(dim=1)           # (B, H)

        # Multi-sample Dropout ensemble
        logits = sum(self.head(dp(pooled)) for dp in self.dropouts) / len(self.dropouts)
        return logits


# ═══════════════════════════════════════════════════════════
# MODEL 5 – XLM-R + Label-aware Attention Head  (fine-tune 5 layers cuối)
# ═══════════════════════════════════════════════════════════
class XLMRClassifier(nn.Module):
    """
    Backbone: model XLM-R đã được tải sẵn về local (vd: thư mục trong 'pretrained/').
    Head: Label-aware Cross-Attention (mỗi lớp học query riêng) +
          Multi-sample Dropout.
    """

    def __init__(self, num_classes: int, model_name: str):
        """
        model_name: đường dẫn local đến model XLM-R (không dùng trực tiếp HF id).
        """
        super().__init__()
        self.bert        = AutoModel.from_pretrained(model_name,
                                                     ignore_mismatched_sizes=True)
        self.num_classes = num_classes
        hidden           = self.bert.config.hidden_size  # 768

        # --- Freeze toàn bộ ---
        for p in self.bert.parameters():
            p.requires_grad = False

        # --- Unfreeze 5 transformer layers cuối ---
        n_layers = len(self.bert.encoder.layer)
        for i in range(n_layers - 5, n_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

        # --- Label-aware Cross-Attention head ---
        self.label_query = nn.Embedding(num_classes, hidden)
        self.cross_attn  = nn.MultiheadAttention(hidden, num_heads=8,
                                                  batch_first=True, dropout=0.1)
        self.norm        = nn.LayerNorm(hidden)

        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * num_classes, 2048), nn.LayerNorm(2048),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        seq    = self.bert(**kwargs).last_hidden_state   # (B, L, H)
        B      = seq.size(0)

        # Label-aware Cross-Attention
        idx     = torch.arange(self.num_classes, device=seq.device).unsqueeze(0).expand(B, -1)
        queries = self.label_query(idx)                       # (B, C, H)
        attn_out, _ = self.cross_attn(
            queries, seq, seq,
            key_padding_mask=(attention_mask == 0),
        )
        feat = self.norm(attn_out + queries).reshape(B, -1)  # (B, C*H)

        logits = sum(self.head(dp(feat)) for dp in self.dropouts) / len(self.dropouts)
        return logits


# ─────────────────────────────────────────────────────────
# Tiện ích: đếm tham số
# ─────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> tuple[int, int]:
    """Trả về (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable