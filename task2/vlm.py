"""Task 2 visual interface: cached SigLIP features -> connector -> visual tokens inside Qwen3's chat.

Manual, Step 1: "Implement the connector and insert its visual embeddings into the user message
before the question, preserving Qwen's chat structure. Construct matching labels, attention masks,
and position indices. [...] Calibrate connector output magnitude against text embeddings using
training data only, and record the rule and resulting scale."

Sequence layout (DECISIONS.md T2-7), thinking disabled:

  <|im_start|>user\\n | 196 visual tokens | \\n<text><|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n | <target><|im_end|>
  ------ pre ------                        ------------------------------- post ---------------------------------   ---- target ----

Only target tokens and <|im_end|> are supervised; visual tokens, prompt and padding are masked.
The no-image condition drops the 196 visual positions and keeps every text token id unchanged;
masks and positions are rebuilt from the shorter sequence. Text segments are tokenized once and
concatenated as token ids, so the text tokens are identical across all image conditions.
"""

import glob
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 6304
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # same checkpoint as Task 1
N_VIS, D_VIS, D_LM = 196, 768, 1024
IGNORE_INDEX = -100
PAD_ID = 151643  # <|endoftext|>, Qwen3's pad token
IM_END = "<|im_end|>"
IM_END_ID = 151645  # checked against the tokenizer in Formatter

CAPTION_TEXT = "Describe the image."
ANSWER_SUFFIX = "\nAnswer the question using a single word or number."
CAPS = {"question": 80, "caption": 64, "answer": 16}  # content tokens (manual: "Content-token caps")
_SLOT = "<<IMAGE-SLOT>>"  # only used to split the rendered chat template; never tokenized

LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}


# --------------------------------------------------------------------------- modules

class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024, then a fixed output scale (T2-8)."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(D_VIS, D_LM)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(D_LM, D_LM)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def unscaled(self, f):
        return self.fc2(self.act(self.fc1(f.to(self.fc1.weight.dtype))))

    def forward(self, f):  # [..., 196, 768] -> [..., 196, 1024]
        return self.unscaled(f) * self.scale


def new_connector():
    """FP32 connector with PyTorch's default Linear init drawn under seed 6304, without
    touching the global RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return Connector()


class LoRALinear(nn.Module):
    """W0 x + (alpha/r) B A dropout(x), as Task 1's LoRALinear without the OPLoRA projection."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(a.to(device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=device))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling


def add_lora(model):
    """Wrap q/k/v/o in every layer, in a fixed order, with A drawn from a seeded CPU generator."""
    gen = torch.Generator().manual_seed(SEED)
    for layer in model.model.layers:
        for name in LORA["targets"]:
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), LORA["r"], LORA["alpha"],
                                                      LORA["dropout"], gen))


def lora_state(model):
    return {f"{n}.{k}": getattr(m, k).detach().float().cpu().contiguous()
            for n, m in model.named_modules() if isinstance(m, LoRALinear) for k in ("lora_A", "lora_B")}


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)


def trainable(model, connector):
    """Named trainable parameters: the connector, plus LoRA factors once they exist."""
    return ([(f"connector.{n}", p) for n, p in connector.named_parameters()] +
            [(n, p) for n, p in model.named_parameters() if p.requires_grad])


# --------------------------------------------------------------------------- text formatting

class Formatter:
    """Token ids for each segment. The chat template is rendered with a placeholder where the
    image goes and split there, so the text around the image is exactly Qwen's template."""

    def __init__(self, tok):
        self.tok = tok
        assert tok.convert_tokens_to_ids(IM_END) == IM_END_ID and tok.pad_token_id == PAD_ID
        self._post = {}
        pre, post = self._render("x")
        assert pre == "<|im_start|>user\n", repr(pre)
        assert post.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(post)
        self.pre = self.ids(pre)

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _render(self, text):
        s = self.tok.apply_chat_template([{"role": "user", "content": f"{_SLOT}\n{text}"}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        pre, post = s.split(_SLOT)
        return pre, post

    def cap(self, text, kind):
        """Text cut to the first CAPS[kind] tokens of its standalone tokenization."""
        ids = self.ids(text)
        return text if len(ids) <= CAPS[kind] else self.tok.decode(ids[:CAPS[kind]])

    def post(self, text):
        if text not in self._post:
            self._post[text] = self.ids(self._render(text)[1])
        return self._post[text]

    def caption_example(self, image_id, caption):
        return {"image_id": image_id, "kind": "caption", "pre": self.pre, "post": self.post(CAPTION_TEXT),
                "target": self.ids(self.cap(caption, "caption"))}

    def qa_example(self, image_id, question, answer):
        return {"image_id": image_id, "kind": "answer", "pre": self.pre,
                "post": self.post(self.cap(question, "question") + ANSWER_SUFFIX),
                "target": self.ids(self.cap(answer, "answer"))}


# --------------------------------------------------------------------------- batches

def build_batch(examples, with_image, pad_side="right", supervise_end=True):
    """Token-level tensors for a list of examples (built by Formatter).

    with_image: one bool per example; False drops the 196 visual positions (no-image condition).
    Visual positions hold PAD_ID in input_ids and are flagged in vis_mask; their embeddings are
    replaced by connector outputs in embed(). Labels hold the target (and <|im_end|> when
    supervise_end) at the target positions and IGNORE_INDEX everywhere else; the next-token
    shift happens in the loss. position_ids count real tokens only, from 0, for either pad side.
    """
    rows = []
    for ex, img in zip(examples, with_image):
        n_vis = N_VIS if img else 0
        tgt = ex["target"] + ([IM_END_ID] if supervise_end else [])
        ids = ex["pre"] + [PAD_ID] * n_vis + ex["post"] + tgt
        vis = [0] * len(ex["pre"]) + [1] * n_vis + [0] * (len(ex["post"]) + len(tgt))
        lab = [IGNORE_INDEX] * (len(ex["pre"]) + n_vis + len(ex["post"])) + tgt
        rows.append((ids, vis, lab))
    T = max(len(r[0]) for r in rows)
    B = len(rows)
    out = {"input_ids": torch.full((B, T), PAD_ID, dtype=torch.long),
           "vis_mask": torch.zeros((B, T), dtype=torch.bool),
           "labels": torch.full((B, T), IGNORE_INDEX, dtype=torch.long),
           "attention_mask": torch.zeros((B, T), dtype=torch.long)}
    for k, (ids, vis, lab) in enumerate(rows):
        sl = slice(T - len(ids), T) if pad_side == "left" else slice(0, len(ids))
        out["input_ids"][k, sl] = torch.tensor(ids)
        out["vis_mask"][k, sl] = torch.tensor(vis, dtype=torch.bool)
        out["labels"][k, sl] = torch.tensor(lab)
        out["attention_mask"][k, sl] = 1
    out["position_ids"] = (out["attention_mask"].cumsum(1) - 1).clamp(min=0)
    out["image_ids"] = [ex["image_id"] for ex, img in zip(examples, with_image) if img]
    return out


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def embed(model, connector, batch, feats):
    """Text embeddings from Qwen's frozen table, with visual positions replaced, in order, by the
    connector outputs for `feats` ([n_images, 196, 768], one per example that has an image)."""
    emb = model.get_input_embeddings()(batch["input_ids"])
    if feats is not None and len(feats):
        vis = connector(feats).to(emb.dtype).reshape(-1, D_LM)
        assert vis.shape[0] == int(batch["vis_mask"].sum())
        emb = emb.masked_scatter(batch["vis_mask"].unsqueeze(-1), vis)
    return emb


def hidden_states(model, connector, batch, feats):
    return model.model(inputs_embeds=embed(model, connector, batch, feats), attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"]).last_hidden_state


def example_nll(model, connector, batch, feats):
    """Mean next-token cross-entropy over each example's supervised tokens -> [B], FP32.
    lm_head runs only on supervised positions."""
    h = hidden_states(model, connector, batch, feats)[:, :-1]
    y = batch["labels"][:, 1:]
    sel = y != IGNORE_INDEX
    ce = F.cross_entropy(model.lm_head(h[sel]).float(), y[sel], reduction="none")
    # scatter back to [B, T-1] and sum per row (deterministic, unlike an atomic index_add_)
    sums = torch.zeros(sel.shape, device=ce.device, dtype=torch.float32).masked_scatter(sel, ce).sum(1)
    return sums / sel.sum(1).clamp(min=1)


# --------------------------------------------------------------------------- features

class FeatureStore:
    """Cached SigLIP features ([N, 196, 768] FP16) for both splits, looked up by COCO image id."""

    def __init__(self, root, device="cpu"):
        from safetensors.torch import load_file
        self.feats, self.index = {}, {}
        for split in ["train", "val"]:
            d = load_file(str(Path(root) / f"siglip_{split}.safetensors"))
            self.feats[split] = d["features"].to(device)
            for row, i in enumerate(d["image_ids"].tolist()):
                assert i not in self.index
                self.index[i] = (split, row)

    def get(self, image_ids):
        return torch.stack([self.feats[s][r] for s, r in (self.index[i] for i in image_ids)]) if image_ids else None


def find_splits_dir(repo):
    """task2/splits in the repo, or the private Kaggle dataset `atml-pa3-task2-splits` (the same
    files; their hashes are recorded by each run)."""
    local = Path(repo) / "task2/splits"
    if (local / "grounding_train_pools.json").exists():
        return local
    hits = glob.glob("/kaggle/input/**/grounding_train_pools.json", recursive=True)
    assert hits, "task2/splits not found"
    return Path(sorted(hits)[0]).parent


def split_hashes(splits_dir):
    import hashlib
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(splits_dir).glob("*.json"))}


def find_feature_dir():
    hits = glob.glob("/kaggle/input/**/task2/features/siglip_train.safetensors", recursive=True)
    assert hits, "cached features not mounted under /kaggle/input"
    return Path(sorted(hits)[0]).parent


# --------------------------------------------------------------------------- calibration (T2-8)

@torch.no_grad()
def calibration_scale(model, connector, train_examples, train_feats, batch=256):
    """s = mean L2 norm of Qwen input embeddings over every text token of the training examples
    (prompt, target and <|im_end|>) / mean L2 norm of the initialized connector's outputs over
    every patch of every training image. FP32 throughout. Returns (s, record)."""
    table = model.get_input_embeddings().weight
    counts = torch.zeros(table.shape[0], dtype=torch.float64)
    for ex in train_examples:
        counts.index_add_(0, torch.tensor(ex["pre"] + ex["post"] + ex["target"] + [IM_END_ID]),
                          torch.ones(len(ex["pre"]) + len(ex["post"]) + len(ex["target"]) + 1, dtype=torch.float64))
    norms = table.float().norm(dim=-1).double().cpu()
    text_norm = float((counts * norms).sum() / counts.sum())
    total, n = 0.0, 0
    for s in range(0, len(train_feats), batch):
        f = train_feats[s:s + batch].to(connector.fc1.weight.device, torch.float32)
        total += float(connector.unscaled(f).float().norm(dim=-1).double().sum())
        n += f.shape[0] * f.shape[1]
    vis_norm = total / n
    return text_norm / vis_norm, {"text_mean_norm": text_norm, "text_tokens": int(counts.sum()),
                                  "connector_mean_norm_unscaled": vis_norm, "visual_tokens": n,
                                  "train_images": len(train_feats), "scale": text_norm / vis_norm}
