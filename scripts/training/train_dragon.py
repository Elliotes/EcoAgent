#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DRAGON+ dual-encoder backbone for RATED (contrastive fine-tuning and inference)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


QUERY_MODEL_ID = "facebook/dragon-plus-query-encoder"
CONTEXT_MODEL_ID = "facebook/dragon-plus-context-encoder"


def tool_text_from_row(name: str, desc: str, tags: str = "") -> str:
    return f"{name} {desc} {tags}".strip()


class PairDataset(Dataset):
    def __init__(self, queries: List[str], tools: List[str]):
        self.queries = queries
        self.tools = tools

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        return self.queries[idx], self.tools[idx]


class DragonBackbone:
    """DRAGON+ query/context encoders with shared retrieval API for RATED."""

    def __init__(
        self,
        query_encoder: AutoModel,
        context_encoder: AutoModel,
        tokenizer,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.query_encoder = query_encoder
        self.context_encoder = context_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_encoder.to(self.device)
        self.context_encoder.to(self.device)
        self.query_encoder.eval()
        self.context_encoder.eval()
        self._dim: Optional[int] = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        max_length: int = 256,
        device: Optional[str] = None,
    ) -> "DragonBackbone":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_dir and Path(checkpoint_dir).exists():
            cfg_path = Path(checkpoint_dir) / "dragon_config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                q_id = cfg.get("query_model_id", QUERY_MODEL_ID)
                c_id = cfg.get("context_model_id", CONTEXT_MODEL_ID)
                max_length = cfg.get("max_length", max_length)
            else:
                q_id, c_id = QUERY_MODEL_ID, CONTEXT_MODEL_ID
            q_dir = Path(checkpoint_dir) / "query_encoder"
            c_dir = Path(checkpoint_dir) / "context_encoder"
            tokenizer = AutoTokenizer.from_pretrained(str(q_dir))
            query_encoder = AutoModel.from_pretrained(str(q_dir))
            context_encoder = AutoModel.from_pretrained(str(c_dir))
        else:
            tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_ID)
            query_encoder = AutoModel.from_pretrained(QUERY_MODEL_ID)
            context_encoder = AutoModel.from_pretrained(CONTEXT_MODEL_ID)
        return cls(query_encoder, context_encoder, tokenizer, max_length, device)

    def embedding_dim(self) -> int:
        if self._dim is None:
            with torch.no_grad():
                dummy = self.tokenizer(
                    ["test"],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.query_encoder(**dummy).last_hidden_state[:, 0, :]
                self._dim = int(out.shape[-1])
        return self._dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim()

    def _encode_batch(self, texts: List[str], encoder: AutoModel) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 1):
                batch = texts[i : i + 1]
                if len(batch) == 0:
                    continue
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                emb = encoder(**inputs).last_hidden_state[:, 0, :]
                emb = F.normalize(emb, p=2, dim=-1)
                out.append(emb.cpu().numpy())
        if not out:
            return np.zeros((0, self.embedding_dim()), dtype=np.float32)
        return np.vstack(out)

    def _encode_batched(
        self,
        texts: List[str],
        encoder: AutoModel,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        out: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding", leave=False)
        with torch.no_grad():
            for i in iterator:
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                emb = encoder(**inputs).last_hidden_state[:, 0, :]
                emb = F.normalize(emb, p=2, dim=-1)
                out.append(emb.cpu().numpy())
        if not out:
            return np.zeros((0, self.embedding_dim()), dtype=np.float32)
        return np.vstack(out)

    def encode_queries(
        self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> np.ndarray:
        return self._encode_batched(
            texts, self.query_encoder, batch_size, show_progress_bar=show_progress_bar
        )

    def encode_tools(
        self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> np.ndarray:
        return self._encode_batched(
            texts, self.context_encoder, batch_size, show_progress_bar=show_progress_bar
        )

    def encode(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """Backward-compatible alias: encodes tool/passage texts via context encoder."""
        return self.encode_tools(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        q_dir = output_dir / "query_encoder"
        c_dir = output_dir / "context_encoder"
        self.query_encoder.save_pretrained(q_dir)
        self.context_encoder.save_pretrained(c_dir)
        self.tokenizer.save_pretrained(q_dir)
        cfg = {
            "backbone": "dragon-plus",
            "query_model_id": QUERY_MODEL_ID,
            "context_model_id": CONTEXT_MODEL_ID,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim(),
            "normalize_embeddings": True,
        }
        (output_dir / "dragon_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def train_mode(self):
        self.query_encoder.train()
        self.context_encoder.train()

    def eval_mode(self):
        self.query_encoder.eval()
        self.context_encoder.eval()


def train_dragon_contrastive(
    pairs_df,
    tools_df,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    temperature: float = 0.05,
    max_seq_len: int = 256,
    use_fp16: bool = True,
    seed: int = 42,
) -> DragonBackbone:
    """InfoNCE-style contrastive fine-tuning on EcoAgent-Syn positive pairs."""
    import pandas as pd

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    backbone = DragonBackbone.from_pretrained(max_length=max_seq_len)
    backbone.train_mode()

    uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
    text_col = "tool_text" if "tool_text" in tools_df.columns else "description"
    uid2text = dict(zip(tools_df[uid_col].astype(str), tools_df[text_col].astype(str)))

    df = pairs_df[pairs_df["label"] == 1].copy()
    df["tool_text"] = df["tool_uid"].astype(str).map(uid2text)
    df = df.dropna(subset=["query", "tool_text"]).reset_index(drop=True)
    queries = df["query"].astype(str).tolist()
    tools = df["tool_text"].astype(str).tolist()
    print(f"[DRAGON] Contrastive training on {len(queries)} positive pairs")

    dataset = PairDataset(queries, tools)
    drop_last = len(queries) >= batch_size * 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    params = list(backbone.query_encoder.parameters()) + list(backbone.context_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(warmup_steps, total_steps // 2), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and backbone.device.startswith("cuda"))

    tau = max(temperature, 1e-4)
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for q_texts, t_texts in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            q_inputs = backbone.tokenizer(
                list(q_texts),
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(backbone.device)
            t_inputs = backbone.tokenizer(
                list(t_texts),
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(backbone.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_fp16 and backbone.device.startswith("cuda")):
                q_emb = backbone.query_encoder(**q_inputs).last_hidden_state[:, 0, :]
                t_emb = backbone.context_encoder(**t_inputs).last_hidden_state[:, 0, :]
                q_emb = F.normalize(q_emb, p=2, dim=-1)
                t_emb = F.normalize(t_emb, p=2, dim=-1)
                logits = (q_emb @ t_emb.T) / tau
                labels = torch.arange(logits.size(0), device=backbone.device)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1

        print(f"  Epoch {epoch + 1} avg loss: {epoch_loss / max(n_batches, 1):.4f}")

    backbone.eval_mode()
    save_path = output_dir / "encoder"
    backbone.save(save_path)
    print(f"[DRAGON] Saved fine-tuned encoders to {save_path}")
    return DragonBackbone.from_pretrained(save_path, max_length=max_seq_len)


def is_dragon_backbone(obj) -> bool:
    return isinstance(obj, DragonBackbone)
