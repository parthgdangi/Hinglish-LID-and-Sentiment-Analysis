"""
Qwen3 (LoRA fine-tuned) + GNN (complementary) for Hinglish LID and Sentiment — full SentiMix
===========================================================================================
Variant with the Hugging Face `transformers` library REMOVED.

* Qwen3 is implemented directly in PyTorch (RMSNorm, GQA attention with
  q/k-norm, RoPE, SwiGLU MLP, KV-cache greedy generation) and loads the
  official safetensors weights.
* Qwen3 is now ALSO TRAINED: a hand-written LoRA (no `peft`, which depends on
  `transformers`) is fine-tuned on the train split for both tasks (sentiment
  and LID) with the same prompts used at inference. The adapters are merged
  into the base weights afterwards, so generation / saliency code is unchanged.
* Tokenisation uses the standalone `tokenizers` library (tokenizer.json).
* Weights/tokenizer files are fetched with `huggingface_hub` (plain file
  download only; no model code). If you place the files manually in
  QWEN_DIR, huggingface_hub is not needed for the model either.
* Everything else (GNN, evaluation, t-SNE, heatmaps) is unchanged.

Run:
    pip install torch tokenizers safetensors huggingface_hub torch-geometric scikit-learn pandas tqdm matplotlib
    python qwen3_gnn_finetune.py
"""

# ---------------------------------------------------------------------------
# 0. Dependencies
# ---------------------------------------------------------------------------
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

_pip(
    "tokenizers", "safetensors", "torch-geometric",
    "scikit-learn", "pandas", "tqdm", "matplotlib", "huggingface_hub",
)

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
import os, re, ast, csv, json, glob, random, bisect, copy, math
from types import SimpleNamespace
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tokenizers import Tokenizer
from safetensors.torch import load_file
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.nn import GCNConv, global_mean_pool

QWEN_NAME   = "Qwen/Qwen3-1.7B"        # swap for Qwen/Qwen3-4B or 8B with more VRAM
QWEN_DIR    = None                     # set to a local folder with config.json, tokenizer.json, *.safetensors to skip download
RAW_DIR     = r"C:\Users\hciii\OneDrive\Desktop\Parth_BCI_LLM_SMC_2026\SentiMix"     # download target for the dataset
PROC_DIR    = r"C:\Users\hciii\OneDrive\Desktop\Parth_BCI_LLM_SMC_2026\processed"    # parsed train/validation/test CSVs
ROMAN_ONLY  = False                    # True = keep only Latin-script rows
MAX_WORDS   = 64
GNN_HIDDEN  = 256
GNN_EPOCHS  = 30
GNN_LR      = 1e-3
GNN_BATCH   = 64
WINDOW      = 2                        # co-occurrence window for graph edges
GEN_BS      = 16                       # batch size for Qwen3 generation
SEED        = 42
CACHE       = "qwen_feature_cache"     # frozen-feature cache (skip re-extraction on reruns)

# --- Qwen3 fine-tuning (LoRA) ---
QWEN_TRAIN         = True    # fine-tune Qwen3 (LoRA) on train split
QWEN_EPOCHS        = 1
QWEN_LR            = 2e-4
QWEN_BS            = 8
QWEN_ACCUM         = 2       # grad accumulation -> effective batch 16
LORA_R             = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
QWEN_EVAL_ZEROSHOT = False   # True = also report the untuned Qwen3 as a baseline

PT_PATH            = "qwen3_gnn_demo.pt"    # self-contained demo bundle written after evaluation

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

LID_NAMES   = ["other", "eng", "hin"]           # label indices: 0 other, 1 eng, 2 hin
LID_MAP     = {"eng": 1, "hin": 2}
SENTI_NAMES = ["negative", "neutral", "positive"]
SENTI_MAP   = {n: i for i, n in enumerate(SENTI_NAMES)}

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(CACHE,    exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Full SentiMix: download + parse (CoNLL -> CSV) for train/val/test
# ---------------------------------------------------------------------------
SPLIT_ALIASES = {
    "train":      ["train"],
    "validation": ["validation", "valid", "dev", "val"],
    "test":       ["test"],
}

def _is_conll(path):
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(12):
                s = f.readline().strip()
                if s:
                    return s.lower().startswith("meta")
    except (UnicodeDecodeError, OSError):
        pass
    return False

def _answer_key():
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.lower().endswith((".txt", ".csv")):
            continue
        p = os.path.join(RAW_DIR, fn)
        try:
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip()
        except (UnicodeDecodeError, OSError):
            continue
        if "," in first and "sentiment" in first.lower():
            with open(p, encoding="utf-8", newline="") as f:
                rd = csv.reader(f)
                next(rd, None)
                m = {r[0].strip(): r[1].strip().lower() for r in rd if len(r) >= 2}
            if m:
                return m
    return {}

def _is_roman(words):
    import unicodedata
    for ch in " ".join(words):
        if ch.isalpha():
            try:
                if "LATIN" not in unicodedata.name(ch):
                    return False
            except ValueError:
                pass
    return True

def parse_split(split):
    out = os.path.join(PROC_DIR, f"{split}.csv")
    if os.path.exists(out):
        return out

    if not (os.path.isdir(RAW_DIR) and os.listdir(RAW_DIR)):
        from huggingface_hub import snapshot_download
        print("Downloading RTT1/SentiMix from Hugging Face …")
        snapshot_download(
            repo_id="RTT1/SentiMix",
            repo_type="dataset",
            local_dir=RAW_DIR,
        )

    files = []
    for alias in SPLIT_ALIASES[split]:
        files = [
            f for f in sorted(os.listdir(RAW_DIR))
            if alias in f.lower() and _is_conll(os.path.join(RAW_DIR, f))
        ]
        if files:
            break
    if not files:
        raise ValueError(
            f"No CoNLL file for split '{split}' in {RAW_DIR}: {os.listdir(RAW_DIR)}"
        )

    rows = []
    for fn in files:
        meta, senti, sent, tags = None, "unknown", [], []

        def flush():
            if meta is not None and sent:
                rows.append({"id": meta, "sentence": sent, "tags": tags, "sentiment": senti})

        for raw in open(os.path.join(RAW_DIR, fn), encoding="utf-8").read().split("\n"):
            s = raw.strip()
            if s.lower().startswith("meta"):
                flush()
                sent, tags = [], []
                p = s.split("\t")
                meta  = p[1].strip() if len(p) > 1 else None
                senti = p[2].strip().lower() if len(p) > 2 else "unknown"
            elif s:
                p = raw.rstrip("\r").split("\t")
                if len(p) >= 2:
                    sent.append(p[0].strip().lower())
                    tags.append(p[1].strip().lower())
            else:
                flush()
                meta, sent, tags = None, [], []
        flush()

    unk = [r for r in rows if r["sentiment"] == "unknown"]
    if unk:
        key = _answer_key()
        for r in unk:
            r["sentiment"] = key.get(str(r["id"]).strip(), "unknown")
        filled = sum(r["sentiment"] != "unknown" for r in unk)
        print(f"[{split}] {len(unk)} rows had no inline label; {filled} filled from answer key")

    if ROMAN_ONLY:
        rows = [r for r in rows if _is_roman(r["sentence"])]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sentence", "tags", "sentiment"])
        w.writeheader()
        w.writerows(rows)
    return out

def load_records(path):
    recs, dropped = [], 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                words = ast.literal_eval(row["sentence"])
                tags  = ast.literal_eval(row["tags"])
            except (ValueError, SyntaxError):
                dropped += 1
                continue
            s = row["sentiment"].strip().lower()
            if len(words) != len(tags) or not words or s not in SENTI_MAP:
                dropped += 1
                continue
            recs.append({
                "id":    row["id"],
                "words": words[:MAX_WORDS],
                "lid":   [LID_MAP.get(t, 0) for t in tags][:MAX_WORDS],
                "senti": SENTI_MAP[s],
            })
    return recs, dropped

train_recs, d1 = load_records(parse_split("train"))
val_recs,   d2 = load_records(parse_split("validation"))
test_recs,  d3 = load_records(parse_split("test"))
print(
    f"train {len(train_recs)} | validation {len(val_recs)} | test {len(test_recs)}"
    f"   (rows dropped: {d1}/{d2}/{d3})"
)
print(
    "Sentiment dist (test):",
    pd.Series([SENTI_NAMES[r["senti"]] for r in test_recs]).value_counts().to_dict(),
)

# ---------------------------------------------------------------------------
# 3a. Pure-PyTorch Qwen3 (no `transformers`)
#     Module names mirror the official checkpoint so weights load directly.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.d = c.heads, c.kv_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden, self.h  * self.d, bias=False)
        self.k_proj = nn.Linear(c.hidden, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.hidden, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden, bias=False)
        self.q_norm = RMSNorm(self.d, c.eps)     # Qwen3: per-head q/k RMSNorm
        self.k_norm = RMSNorm(self.d, c.eps)

    def forward(self, x, cos, sin, mask, cache):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h,  self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.kv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        kv_out = (k, v)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.h * self.d)), kv_out

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden, c.inter, bias=False)
        self.up_proj   = nn.Linear(c.hidden, c.inter, bias=False)
        self.down_proj = nn.Linear(c.inter, c.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden, c.eps)
        self.post_attention_layernorm = RMSNorm(c.hidden, c.eps)

    def forward(self, x, cos, sin, mask, cache):
        a, kv = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, kv

class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab, c.hidden)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.hidden, c.eps)

class Qwen3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.grad_ckpt = False          # gradient checkpointing (enabled during fine-tuning)
        self.model = Backbone(c)
        self.lm_head = nn.Linear(c.hidden, c.vocab, bias=False)

    # -- building blocks ------------------------------------------------------
    def embed(self, ids):
        return self.model.embed_tokens(ids)

    def logits(self, h):
        return self.lm_head(h)

    def _rope(self, pos, dtype):
        d = self.c.head_dim
        inv = 1.0 / (self.c.rope_theta ** (torch.arange(0, d, 2, device=pos.device, dtype=torch.float32) / d))
        fr  = pos[..., None].float() * inv                      # [B, T, d/2]
        emb = torch.cat([fr, fr], dim=-1)                       # [B, T, d]
        return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)

    def hidden(self, embeds, attn_mask=None, pos=None, caches=None, use_cache=False):
        """embeds [B,T,H]; attn_mask [B, past+T] (1 = real token); returns (final-normed hidden, caches)."""
        B, T, _ = embeds.shape
        dev   = embeds.device
        past  = 0 if caches is None else caches[0][0].size(2)
        S     = past + T
        if attn_mask is None:
            attn_mask = torch.ones(B, S, dtype=torch.bool, device=dev)
        if pos is None:
            pos = torch.arange(past, S, device=dev).unsqueeze(0).expand(B, -1)
        cos, sin = self._rope(pos, embeds.dtype)
        q_idx = torch.arange(past, S, device=dev)[:, None]
        k_idx = torch.arange(S, device=dev)[None, :]
        allowed = (k_idx <= q_idx)[None, None] & attn_mask.bool()[:, None, None, :]
        # float mask with a finite "-inf" avoids NaNs on fully-masked (left-pad) rows
        mask = torch.zeros(allowed.shape, dtype=embeds.dtype, device=dev)
        mask = mask.masked_fill(~allowed, torch.finfo(embeds.dtype).min)
        h, new = embeds, []
        for i, layer in enumerate(self.model.layers):
            if self.grad_ckpt and torch.is_grad_enabled() and caches is None:
                h, kv = torch.utils.checkpoint.checkpoint(
                    layer, h, cos, sin, mask, None, use_reentrant=False)
            else:
                h, kv = layer(h, cos, sin, mask, None if caches is None else caches[i])
            if use_cache:
                new.append(kv)
        return self.model.norm(h), (new if use_cache else None)

    @torch.no_grad()
    def generate(self, input_ids, attn_mask, max_new_tokens, eos_ids, pad_id):
        """Greedy decoding with KV cache; input is LEFT-padded."""
        B = input_ids.size(0)
        pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)
        h, caches = self.hidden(self.embed(input_ids), attn_mask, pos, None, use_cache=True)
        nxt = self.logits(h[:, -1]).argmax(-1)
        last = pos[:, -1]
        eos = torch.tensor(sorted(eos_ids), device=input_ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        out = []
        for _ in range(max_new_tokens):
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            out.append(nxt)
            done = done | torch.isin(nxt, eos)
            if done.all():
                break
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=attn_mask.dtype, device=attn_mask.device)], 1)
            last = last + 1
            h, caches = self.hidden(self.embed(nxt[:, None]), attn_mask, last[:, None], caches, use_cache=True)
            nxt = self.logits(h[:, -1]).argmax(-1)
        return torch.stack(out, 1)

def load_qwen(name, local_dir, dtype, dev):
    if local_dir is None:
        from huggingface_hub import snapshot_download
        print(f"Downloading {name} files (config, tokenizer, safetensors) …")
        local_dir = snapshot_download(
            repo_id=name,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
    cfg = json.load(open(os.path.join(local_dir, "config.json"), encoding="utf-8"))
    heads = cfg["num_attention_heads"]
    theta = cfg.get("rope_theta") or cfg.get("rope_parameters", {}).get("rope_theta", 1_000_000.0)
    c = SimpleNamespace(
        hidden   = cfg["hidden_size"],
        inter    = cfg["intermediate_size"],
        n_layers = cfg["num_hidden_layers"],
        heads    = heads,
        kv_heads = cfg["num_key_value_heads"],
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // heads),
        eps      = cfg["rms_norm_eps"],
        rope_theta = float(theta),
        vocab    = cfg["vocab_size"],
        tied     = cfg.get("tie_word_embeddings", False),
    )
    with torch.device("meta"):
        model = Qwen3(c)
    sd = {}
    for fn in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        sd.update({k: v.to(dtype) for k, v in load_file(fn).items()})
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"Unexpected weight keys: {unexpected[:5]}"
    if c.tied or "lm_head.weight" in missing:
        model.lm_head.weight = model.model.embed_tokens.weight
        missing = [m for m in missing if m != "lm_head.weight"]
    assert not missing, f"Missing weight keys: {missing[:5]}"
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)        # grads only needed w.r.t. input embeddings (saliency) / LoRA params
    tk = Tokenizer.from_file(os.path.join(local_dir, "tokenizer.json"))
    return model, tk, c

print(f"\nLoading Qwen3 ({QWEN_NAME}) with the pure-PyTorch implementation …")
qwen, tok, QCFG = load_qwen(
    QWEN_NAME, QWEN_DIR,
    torch.bfloat16 if device.type == "cuda" else torch.float32,
    device,
)
HIDDEN  = QCFG.hidden
PAD_ID  = tok.token_to_id("<|endoftext|>")
EOS_IDS = {tok.token_to_id("<|im_end|>"), PAD_ID}
print(f"Qwen3 hidden size: {HIDDEN}")

def encode(text):
    """Return (token ids list, char-offset list). No BOS is used by Qwen3."""
    e = tok.encode(text, add_special_tokens=False)
    return e.ids, e.offsets

# --- helpers for feature extraction ----------------------------------------

def word_layout(words):
    """Return (text, char_starts, char_ends) with space-separated words."""
    text, starts, ends, pos = "", [], [], 0
    for w in words:
        if text:
            text += " "
            pos  += 1
        starts.append(pos)
        ends.append(pos + len(w))
        text += w
        pos  += len(w)
    return text, starts, ends

def tok2word(offsets, starts, ends):
    """Map each token to the word containing its last character."""
    m = []
    for s, e in offsets:
        wi = -1
        if e > s:
            c = e - 1
            k = bisect.bisect_right(starts, c) - 1
            if k >= 0 and c < ends[k]:
                wi = k
        m.append(wi)
    return torch.tensor(m)

@torch.no_grad()
def word_features(words):
    """Frozen last-layer Qwen3 states, mean-pooled sub-tokens → words.
    Returns a tensor of shape [n_words, HIDDEN] in fp16.
    """
    text, st, en = word_layout(words)
    ids, offs = encode(text)
    ids, offs = ids[:512], offs[:512]
    ids_t = torch.tensor([ids], device=device)
    h = qwen.hidden(qwen.embed(ids_t))[0][0].float()
    widx = tok2word(offs, st, en).to(device)
    ok   = widx >= 0
    feats = torch.zeros(len(words), HIDDEN, device=device).index_add_(0, widx[ok], h[ok])
    cnt   = torch.bincount(widx[ok], minlength=len(words)).clamp(min=1).unsqueeze(1)
    return (feats / cnt).half().cpu()

# --- chat helpers -----------------------------------------------------------

def render(prompt):
    """Qwen3 chat template with thinking disabled (identical to the HF template)."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

SENT_PROMPT = lambda words: (
    "Classify the sentiment of this Hinglish (Hindi-English code-mixed, romanised) sentence.\n"
    "Answer with exactly one word: negative, neutral, or positive.\n\n"
    f"Sentence: {' '.join(words)}"
)

def lid_list(words):
    return "\n".join(f"{i+1}: {w}" for i, w in enumerate(words))

LID_PROMPT = lambda words: (
    "Assign a language tag to each word of this romanised Hinglish sentence.\n"
    "Tags: eng (English word), hin (Hindi word written in Latin script), "
    "other (names, numbers, symbols, emojis, etc.).\n"
    f"Reply with exactly {len(words)} lines in the format `index: tag` and nothing else.\n\n"
    f"{lid_list(words)}"
)

def encode_left_padded(texts):
    seqs = [encode(t)[0] for t in texts]
    L = max(len(s) for s in seqs)
    ids  = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):]  = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(device), mask.to(device)

@torch.no_grad()
def batch_chat(prompts, max_new_tokens):
    ids, mask = encode_left_padded([render(p) for p in prompts])
    out = qwen.generate(ids, mask, max_new_tokens, EOS_IDS, PAD_ID)
    return [
        tok.decode(o.tolist(), skip_special_tokens=True).strip().lower()
        for o in out
    ]

def qwen_sentiment_all(recs, bs=32):
    preds = []
    for b in tqdm(range(0, len(recs), bs), desc="Qwen3 sentiment"):
        for ans in batch_chat([SENT_PROMPT(r["words"]) for r in recs[b:b + bs]], 8):
            preds.append(
                next((SENTI_MAP[n] for n in SENTI_NAMES if n in ans), SENTI_MAP["neutral"])
            )
    return preds

def parse_lid(ans, n):
    pred = [0] * n
    for i, t in re.findall(r"(\d+)\s*[:\-]\s*(eng|hin|other)", ans):
        if 1 <= int(i) <= n:
            pred[int(i) - 1] = LID_MAP.get(t, 0)
    return pred

def qwen_lid_all(recs, bs=GEN_BS):
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["words"]))
    preds = [None] * len(recs)
    for b in tqdm(range(0, len(order), bs), desc="Qwen3 LID"):
        ids   = order[b:b + bs]
        n_max = max(len(recs[i]["words"]) for i in ids)
        answers = batch_chat([LID_PROMPT(recs[i]["words"]) for i in ids], 6 * n_max + 16)
        for i, a in zip(ids, answers):
            preds[i] = parse_lid(a, len(recs[i]["words"]))
    return preds

# ---------------------------------------------------------------------------
# 4. Extract frozen Qwen3 features for every sentence (cached to disk)
#    (GNN input = frozen, pre-fine-tuning Qwen3 states)
# ---------------------------------------------------------------------------
def build_graph(x, y_lid=None, y_senti=None):
    n  = x.size(0)
    e  = [
        [i, j]
        for i in range(n)
        for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1))
        if i != j
    ]
    ei = (
        torch.tensor(e, dtype=torch.long).t().contiguous()
        if e else
        torch.empty((2, 0), dtype=torch.long)
    )
    d = Data(x=x, edge_index=ei)
    if y_lid   is not None: d.y_lid   = torch.tensor(y_lid,    dtype=torch.long)
    if y_senti is not None: d.y_senti = torch.tensor([y_senti], dtype=torch.long)
    return d

def graphs_for(recs, name):
    cache_path = os.path.join(
        CACHE, f"{QWEN_NAME.replace('/', '_')}_{name}_{len(recs)}.pt"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached features for '{name}' split …")
        feats = torch.load(cache_path)
    else:
        feats = [
            word_features(r["words"])
            for r in tqdm(recs, desc=f"Extracting Qwen3 features ({name})")
        ]
        torch.save(feats, cache_path)
    return [build_graph(x, r["lid"], r["senti"]) for x, r in zip(feats, recs)]

print("\nBuilding graph datasets (this may take a while on first run) …")
train_g = graphs_for(train_recs, "train")
val_g   = graphs_for(val_recs,   "val")
test_g  = graphs_for(test_recs,  "test")

# ---------------------------------------------------------------------------
# 5. GNN: train on full train, select best epoch on full validation
# ---------------------------------------------------------------------------
class GNN(nn.Module):
    """2-layer GCN.
    task='lid'   → per-word logits
    task='senti' → per-sentence logits (mean-pooled over words)
    """
    def __init__(self, in_dim, hid, n_classes, task):
        super().__init__()
        self.task = task
        self.c1   = GCNConv(in_dim, hid)
        self.c2   = GCNConv(hid, hid)
        self.drop = nn.Dropout(0.3)
        self.out  = nn.Linear(hid, n_classes)

    def forward(self, data):
        x = self.drop(F.relu(self.c1(data.x.float(), data.edge_index)))
        h = F.relu(self.c2(x, data.edge_index))
        z = global_mean_pool(h, data.batch) if self.task == "senti" else h
        return self.out(self.drop(z)), z   # (logits, embeddings for t-SNE)

@torch.no_grad()
def gnn_run(model, graphs):
    """Return per-graph predictions and per-graph embeddings."""
    model.eval()
    preds, embs = [], []
    for b in GraphLoader(graphs, batch_size=128):
        b      = b.to(device)
        logits, z = model(b)
        p = logits.argmax(-1).cpu()
        z = z.cpu()
        if model.task == "lid":
            for k in range(b.num_graphs):
                s, e = b.ptr[k].item(), b.ptr[k + 1].item()
                preds.append(p[s:e].tolist())
                embs.append(z[s:e])
        else:
            preds.extend(p.tolist())
            embs.extend(z[k] for k in range(b.num_graphs))
    return preds, embs

def val_f1(model, graphs):
    p, _ = gnn_run(model, graphs)
    if model.task == "lid":
        return f1_score(
            [y for g in graphs for y in g.y_lid.tolist()],
            [x for q in p for x in q],
            average="macro",
        )
    return f1_score(
        [g.y_senti.item() for g in graphs], p, average="macro"
    )

def train_gnn(task, n_classes):
    model = GNN(HIDDEN, GNN_HIDDEN, n_classes, task).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
    best, best_state = -1, None

    for ep in range(GNN_EPOCHS):
        model.train()
        tot = nb = 0
        for b in GraphLoader(train_g, batch_size=GNN_BATCH, shuffle=True):
            b = b.to(device)
            logits, _ = model(b)
            y    = b.y_lid if task == "lid" else b.y_senti
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb  += 1

        f1 = val_f1(model, val_g)
        star = "  *" if f1 > best else ""
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
        print(
            f"[{task}] epoch {ep+1:2d}/{GNN_EPOCHS}  "
            f"train loss {tot/nb:.4f}  val macro-F1 {f1:.4f}{star}"
        )

    model.load_state_dict(best_state)
    print(f"[{task}] best val macro-F1 {best:.4f}")
    return model.eval()

print("\nTraining GNN for LID …")
gnn_lid   = train_gnn("lid",   len(LID_NAMES))
print("\nTraining GNN for Sentiment …")
gnn_senti = train_gnn("senti", len(SENTI_NAMES))

# ---------------------------------------------------------------------------
# 5b. Fine-tune Qwen3 with LoRA (pure PyTorch) on LID + Sentiment
#     One adapter, both tasks, same prompts as inference. Adapters are merged
#     into the base weights at the end, so generation / saliency code is unchanged.
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base  = base
        self.scale = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        lora = self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)
        return self.base(x) + lora * self.scale

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def _lora_sites(model):
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in LORA_TARGETS:
                if hasattr(parent, n):
                    yield parent, n

def add_lora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for parent, n in list(_lora_sites(model)):
        setattr(parent, n, LoRALinear(getattr(parent, n), LORA_R, LORA_ALPHA, LORA_DROPOUT))
    return [p for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def merge_lora(model):
    for parent, n in list(_lora_sites(model)):
        m = getattr(parent, n)
        if isinstance(m, LoRALinear):
            w = m.base.weight
            w += ((m.B @ m.A) * m.scale).to(w.dtype)
            setattr(parent, n, m.base)
    for p in model.parameters():
        p.requires_grad_(False)

def sft_examples(recs):
    """Same prompts as inference; targets are the label word / `i: tag` lines."""
    out = []
    for r in recs:
        w = r["words"]
        out.append((render(SENT_PROMPT(w)), SENTI_NAMES[r["senti"]]))
        out.append((render(LID_PROMPT(w)),
                    "\n".join(f"{i+1}: {LID_NAMES[t]}" for i, t in enumerate(r["lid"]))))
    return out

def sft_tokenize(pairs):
    im_end = tok.token_to_id("<|im_end|>")
    return [(encode(p)[0], encode(a)[0] + [im_end]) for p, a in pairs]

def sft_collate(items):
    B, L = len(items), max(len(p) + len(a) for p, a in items)
    ids  = torch.full((B, L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    lab  = torch.full((B, L), -100, dtype=torch.long)
    for i, (p, a) in enumerate(items):
        s = p + a
        ids[i, :len(s)]  = torch.tensor(s)
        mask[i, :len(s)] = 1
        lab[i, len(p):len(s)] = torch.tensor(a)       # loss on response tokens only
    return ids.to(device), mask.to(device), lab.to(device)

def finetune_qwen(recs):
    params = add_lora(qwen)
    print(f"LoRA trainable params: {sum(p.numel() for p in params)/1e6:.1f} M")
    path = os.path.join(CACHE, f"lora_{QWEN_NAME.replace('/', '_')}_{len(recs)}_e{QWEN_EPOCHS}.pt")
    if os.path.exists(path):
        print("Loading cached LoRA weights …")
        qwen.load_state_dict(torch.load(path, map_location=device), strict=False)
    else:
        data = sft_tokenize(sft_examples(recs))
        n_batches = math.ceil(len(data) / QWEN_BS)
        total = max(1, n_batches * QWEN_EPOCHS // QWEN_ACCUM)
        opt = torch.optim.AdamW(params, lr=QWEN_LR, weight_decay=0.0)
        warm = max(1, int(0.03 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
        qwen.grad_ckpt = True
        qwen.train()
        step = 0
        for ep in range(QWEN_EPOCHS):
            idx = list(range(len(data))); random.shuffle(idx)
            chunks = [sorted(idx[i:i + QWEN_BS * 50], key=lambda j: len(data[j][0]) + len(data[j][1]))
                      for i in range(0, len(idx), QWEN_BS * 50)]
            batches = [c[i:i + QWEN_BS] for c in chunks for i in range(0, len(c), QWEN_BS)]
            random.shuffle(batches)
            run = 0.0
            pbar = tqdm(batches, desc=f"Qwen3 LoRA epoch {ep+1}/{QWEN_EPOCHS}")
            for bi, b in enumerate(pbar):
                ids, mask, lab = sft_collate([data[j] for j in b])
                h, _ = qwen.hidden(qwen.embed(ids), mask)
                tgt  = lab[:, 1:]
                sel  = tgt != -100
                loss = F.cross_entropy(qwen.logits(h[:, :-1][sel]).float(), tgt[sel])
                (loss / QWEN_ACCUM).backward()
                run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
                if (bi + 1) % QWEN_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                pbar.set_postfix(loss=f"{run:.4f}")
        qwen.grad_ckpt = False
        qwen.eval()
        torch.save({k: v for k, v in qwen.state_dict().items()
                    if k.endswith(".A") or k.endswith(".B")}, path)
    # keep a CPU copy of the adapters: they go into the demo .pkl (merge_lora discards them)
    lora_sd = {k: v.detach().cpu().clone() for k, v in qwen.state_dict().items()
               if k.endswith(".A") or k.endswith(".B")}
    merge_lora(qwen)       # fold adapters into base weights -> generation/saliency code unchanged
    qwen.eval()
    return lora_sd

if QWEN_EVAL_ZEROSHOT:
    print("\nZero-shot Qwen3 baseline on test …")
    zs_lid   = qwen_lid_all(test_recs)
    zs_senti = qwen_sentiment_all(test_recs)

lora_sd = None
if QWEN_TRAIN:
    print("\nFine-tuning Qwen3 (LoRA) …")
    lora_sd = finetune_qwen(train_recs)

# ---------------------------------------------------------------------------
# 6. Evaluate the four conditions on the full test split
# ---------------------------------------------------------------------------
print("\nRunning inference on test split …")
g_lid_pred,   g_lid_emb   = gnn_run(gnn_lid,   test_g)
g_senti_pred, g_senti_emb = gnn_run(gnn_senti, test_g)
q_lid_pred   = qwen_lid_all(test_recs)
q_senti_pred = qwen_sentiment_all(test_recs)

true_lid   = [y for r in test_recs for y in r["lid"]]
true_senti = [r["senti"] for r in test_recs]
flat = lambda L: [x for s in L for x in s]

P = {
    1: {"lid": flat(g_lid_pred), "senti": q_senti_pred},   # GNN (LID) + Qwen3 (Sentiment)
    2: {"lid": flat(q_lid_pred), "senti": g_senti_pred},   # Qwen3 (LID) + GNN (Sentiment)
    3: {"lid": flat(q_lid_pred), "senti": q_senti_pred},   # Qwen3 only
    4: {"lid": flat(g_lid_pred), "senti": g_senti_pred},   # GNN only
}
COND = {
    1: "GNN (LID) + Qwen3 (Sentiment)",
    2: "GNN (Sentiment) + Qwen3 (LID)",
    3: "Qwen3 only",
    4: "GNN only",
}

rows = []
for c, name in COND.items():
    for task, y, key in (
        ("LID",       true_lid,   "lid"),
        ("Sentiment", true_senti, "senti"),
    ):
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, P[c][key], average="macro", zero_division=0
        )
        rows.append({
            "Condition": f"{c}. {name}",
            "Task": task,
            "Precision": pr,
            "Recall": rc,
            "F1": f1,
            "Accuracy": accuracy_score(y, P[c][key]),
        })

if QWEN_EVAL_ZEROSHOT:
    for task, y, pred in (("LID", true_lid, flat(zs_lid)), ("Sentiment", true_senti, zs_senti)):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        rows.append({"Condition": "5. Qwen3 zero-shot (untuned)", "Task": task,
                     "Precision": pr, "Recall": rc, "F1": f1,
                     "Accuracy": accuracy_score(y, pred)})

df = pd.DataFrame(rows)
pd.options.display.float_format = "{:.4f}".format
print(
    f"\nTest sentences: {len(test_recs)}  |  test words: {len(true_lid)}  (P/R/F1 macro-averaged; accuracy: LID per word, sentiment per sentence)\n"
)
print(df.to_string(index=False))
df.to_csv("qwen3_gnn_results.csv", index=False)
print("\nResults saved to qwen3_gnn_results.csv")

# ---------------------------------------------------------------------------
# 6b. Export a self-contained demo bundle (.pkl)
#     Holds: architecture description, Qwen3 config + LoRA adapters, both GNNs,
#     label maps, preprocessing settings and the test-set results table.
#     (The ~3 GB frozen Qwen3 base weights are NOT duplicated; the demo
#     re-downloads them by name.)  Written here rather than at the very end so
#     a failure in the plotting sections cannot lose it.
# ---------------------------------------------------------------------------
def _cpu_sd(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

def architecture_text():
    n_q  = sum(p.numel() for p in qwen.parameters())
    n_lo = sum(v.numel() for v in lora_sd.values()) if lora_sd else 0
    n_g1 = sum(p.numel() for p in gnn_lid.parameters())
    n_g2 = sum(p.numel() for p in gnn_senti.parameters())
    return "\n".join([
        "Qwen3 + GNN complementary architecture for Hinglish LID and Sentiment",
        "=" * 70,
        f"Input     : romanised Hinglish sentence -> words (max {MAX_WORDS})",
        "",
        f"[A] Qwen3 ({QWEN_NAME}), pure-PyTorch, {QCFG.n_layers} layers, hidden {QCFG.hidden}, "
        f"{QCFG.heads} heads / {QCFG.kv_heads} KV heads (GQA, q/k RMSNorm, RoPE, SwiGLU)",
        f"    base params      : {n_q/1e6:.0f} M (frozen)",
        f"    LoRA adapters    : r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}, {n_lo/1e6:.1f} M params"
        + ("" if lora_sd else "  [NOT TRAINED: base model only]"),
        "    Task 1 sentiment : prompt -> one word (negative / neutral / positive)",
        "    Task 2 LID       : prompt -> one `index: tag` line per word (eng / hin / other)",
        "",
        "[B] Frozen-Qwen3 word features (last layer, sub-tokens mean-pooled to words)",
        f"    -> word graph, edges between words within +/-{WINDOW} positions",
        "",
        "[C] GNN (2-layer GCN, ReLU, dropout 0.3)",
        f"    LID GNN       : GCN({HIDDEN}->{GNN_HIDDEN}) -> GCN({GNN_HIDDEN}->{GNN_HIDDEN}) -> Linear(->3), per word   [{n_g1/1e6:.2f} M params]",
        f"    Sentiment GNN : GCN({HIDDEN}->{GNN_HIDDEN}) -> GCN({GNN_HIDDEN}->{GNN_HIDDEN}) -> mean-pool -> Linear(->3)   [{n_g2/1e6:.2f} M params]",
        "",
        "Evaluated conditions: (1) GNN LID + Qwen3 sentiment, (2) Qwen3 LID + GNN sentiment,",
        "                      (3) Qwen3 only, (4) GNN only",
    ])

bundle = {
    "format_version": 1,
    "architecture":   architecture_text(),
    "qwen_name":      QWEN_NAME,
    "qwen_config":    vars(QCFG),
    "lora": {
        "r": LORA_R, "alpha": LORA_ALPHA, "targets": LORA_TARGETS,
        "state_dict": lora_sd,                    # None if QWEN_TRAIN = False
    },
    "gnn": {
        "in_dim": HIDDEN, "hidden": GNN_HIDDEN, "window": WINDOW,
        "lid_state":   _cpu_sd(gnn_lid),
        "senti_state": _cpu_sd(gnn_senti),
    },
    "labels":    {"lid": LID_NAMES, "senti": SENTI_NAMES},
    "max_words": MAX_WORDS,
    "results":   df.to_dict("records"),
}
torch.save(bundle, PT_PATH)
print(f"\nDemo bundle saved to {PT_PATH} ({os.path.getsize(PT_PATH)/1e6:.1f} MB)")
print(bundle["architecture"])

# ---------------------------------------------------------------------------
# 7. t-SNE scatterplots
# ---------------------------------------------------------------------------
def tsne2d(X):
    X = np.asarray(X, dtype=np.float32)
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    X = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)
    perp = min(30, max(5, X.shape[0] // 4))
    return TSNE(2, init="pca", perplexity=perp, random_state=SEED).fit_transform(X)

COLORS = ["#d95f02", "#1b9e77", "#7570b3"]

def scatter_figure(Xq, Xg, true, pq, pg, names, title, fname):
    cq, cg = tsne2d(Xq), tsne2d(Xg)
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    panels = [
        (cq, true, "Qwen3 representation — true label"),
        (cg, true, "GNN representation — true label"),
        (cq, pq,   "Qwen3 representation — Qwen3 prediction"),
        (cg, pg,   "GNN representation — GNN prediction"),
    ]
    for ax, (xy, lab, ttl) in zip(axs.ravel(), panels):
        lab = np.asarray(lab)
        for k, n in enumerate(names):
            m = lab == k
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, c=COLORS[k], label=f"{n} ({m.sum()})")
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.show()
    print(f"Saved {fname}")

rng = np.random.default_rng(SEED)

qf  = torch.cat([g.x for g in test_g]).float().numpy()
gf  = torch.cat(g_lid_emb).numpy()
idx = rng.choice(len(qf), size=min(4000, len(qf)), replace=False)
scatter_figure(
    qf[idx], gf[idx],
    np.array(true_lid)[idx],
    np.array(flat(q_lid_pred))[idx],
    np.array(flat(g_lid_pred))[idx],
    LID_NAMES,
    "Language identification (test words, t-SNE)",
    "scatter_lid.png",
)

qs  = np.stack([g.x.float().mean(0).numpy() for g in test_g])
gs  = torch.stack(g_senti_emb).numpy()
idx = rng.choice(len(qs), size=min(4000, len(qs)), replace=False)
scatter_figure(
    qs[idx], gs[idx],
    np.array(true_senti)[idx],
    np.array(q_senti_pred)[idx],
    np.array(g_senti_pred)[idx],
    SENTI_NAMES,
    "Sentiment (test sentences, t-SNE)",
    "scatter_sentiment.png",
)

# ---------------------------------------------------------------------------
# 8. Heatmaps — which words each model focuses on (gradient × input saliency)
# ---------------------------------------------------------------------------
def _pick(recs, pred=lambda r: True, k=1, lo=6, hi=18):
    return [r for r in recs if lo <= len(r["words"]) <= hi and pred(r)][:k]

# --- GNN saliency -----------------------------------------------------------
def _gnn_input(words_feats):
    g = build_graph(words_feats.float())
    x = g.x.to(device).requires_grad_(True)
    d = Data(x=x, edge_index=g.edge_index.to(device))
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    return d, x

def gnn_sent_saliency(feats):
    d, x    = _gnn_input(feats)
    logits, _ = gnn_senti(d)
    c = logits[0].argmax()
    g, = torch.autograd.grad(logits[0, c], x)
    s  = (g * x).sum(-1).abs().detach().cpu().numpy()
    return int(c), s / (s.max() + 1e-12)

def gnn_lid_saliency(feats):
    d, x    = _gnn_input(feats)
    logits, _ = gnn_lid(d)
    n, pred = x.size(0), logits.argmax(-1)
    M = np.zeros((n, n))
    for i in range(n):
        g, = torch.autograd.grad(logits[i, pred[i]], x, retain_graph=True)
        M[i] = (g * x).sum(-1).abs().detach().cpu().numpy()
    return pred.tolist(), M / (M.max(1, keepdims=True) + 1e-12)

# --- Qwen3 saliency (on the fine-tuned + merged model) -----------------------
CLS_TOK = [tok.encode(n, add_special_tokens=False).ids[0] for n in SENTI_NAMES]
assert len(set(CLS_TOK)) == 3, "Sentiment label first-tokens collide; saliency ambiguous"

def _qwen_prompt_tokens(text):
    ids, offs = encode(text)
    return torch.tensor([ids], device=device), offs

def _qwen_logits_from_embeds(emb):
    h, _ = qwen.hidden(emb)
    return qwen.logits(h)

def qwen_sent_saliency(words):
    text = render(SENT_PROMPT(words))
    sent = " ".join(words)
    base = text.rfind(sent)
    _, st, en = word_layout(words)
    st = [base + s for s in st]
    en = [base + e for e in en]

    ids, offs = _qwen_prompt_tokens(text)
    emb    = qwen.embed(ids).detach().requires_grad_(True)
    logits = _qwen_logits_from_embeds(emb)[0, -1].float()
    c      = int(torch.stack([logits[t] for t in CLS_TOK]).argmax())
    g, = torch.autograd.grad(logits[CLS_TOK[c]], emb)
    tok_s  = (g * emb).sum(-1)[0].abs().float().detach().cpu()
    w = tok2word(offs, st, en)
    s = np.zeros(len(words))
    for t, wi in enumerate(w.tolist()):
        if wi >= 0:
            s[wi] += tok_s[t].item()
    return c, s / (s.max() + 1e-12)

def qwen_lid_saliency(words, pred_tags):
    n    = len(words)
    text = render(LID_PROMPT(words))
    lst  = lid_list(words)
    base = text.rfind(lst)
    st, en, pos = [], [], base
    for i, w in enumerate(words):
        pref = f"{i+1}: "
        st.append(pos + len(pref))
        en.append(pos + len(pref) + len(w))
        pos += len(pref) + len(w) + 1

    resp, tag_pos = "", []
    for i, t in enumerate(pred_tags):
        pref = f"{i+1}: "
        tag_pos.append(len(text) + len(resp) + len(pref))
        resp += pref + LID_NAMES[t] + ("\n" if i < n - 1 else "")

    ids, offs = _qwen_prompt_tokens(text + resp)
    emb    = qwen.embed(ids).detach().requires_grad_(True)
    logits = _qwen_logits_from_embeds(emb)[0].float()
    w = tok2word(offs, st, en).tolist()
    M = np.zeros((n, n))
    for i in range(n):
        t = next(
            k for k, (s, e) in enumerate(offs)
            if s <= tag_pos[i] < e or (s >= tag_pos[i] and e > s)
        )
        g, = torch.autograd.grad(logits[t - 1, ids[0, t]], emb, retain_graph=True)
        ts = (g * emb).sum(-1)[0].abs().float().detach().cpu()
        for k, wi in enumerate(w):
            if wi >= 0:
                M[i, wi] += ts[k].item()
    return M / (M.max(1, keepdims=True) + 1e-12)

# --- Sentiment heatmap (2 examples per class) --------------------------------
print("\nGenerating sentiment heatmaps …")
ex_senti = [r for c in range(3) for r in _pick(test_recs, lambda r, c=c: r["senti"] == c, k=2)]
fig, axs = plt.subplots(len(ex_senti), 1, figsize=(11, 2.0 * len(ex_senti)), squeeze=False)
im = None
for ax, r in zip(axs[:, 0], ex_senti):
    i        = test_recs.index(r)
    qc, qs_  = qwen_sent_saliency(r["words"])
    gc, gs_  = gnn_sent_saliency(test_g[i].x)
    im = ax.imshow(np.stack([qs_, gs_]), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(r["words"])))
    ax.set_xticklabels(r["words"], rotation=40, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Qwen3", "GNN"])
    ax.set_title(
        f"true={SENTI_NAMES[r['senti']]} | Qwen3→{SENTI_NAMES[qc]} | GNN→{SENTI_NAMES[gc]}",
        fontsize=10,
    )
fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02,
             label="word influence (normalised)")
fig.suptitle("Sentiment: which words each model focuses on", y=1.0)
fig.savefig("heatmap_sentiment.png", dpi=160, bbox_inches="tight")
plt.show()
print("Saved heatmap_sentiment.png")

# --- LID heatmap (code-mixed examples) --------------------------------------
print("\nGenerating LID heatmaps …")
ex_lid = _pick(test_recs, lambda r: set(r["lid"]) >= {1, 2}, k=3, lo=6, hi=14)
fig, axs = plt.subplots(len(ex_lid), 2, figsize=(14, 5.2 * len(ex_lid)), squeeze=False)
for row, r in zip(axs, ex_lid):
    i  = test_recs.index(r)
    w  = r["words"]
    gp, gM = gnn_lid_saliency(test_g[i].x)
    qp     = parse_lid(batch_chat([LID_PROMPT(w)], 6 * len(w) + 16)[0], len(w))
    qM     = qwen_lid_saliency(w, qp)
    for ax, M, pr, nm in ((row[0], qM, qp, "Qwen3"), (row[1], gM, gp, "GNN")):
        ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels(w, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(
            [f"{x} [{LID_NAMES[p]}|{LID_NAMES[t]}]" for x, p, t in zip(w, pr, r["lid"])],
            fontsize=8,
        )
        ax.set_title(f"{nm}: row = word tagged [pred|true], column = influencing word")
        ax.set_xlabel("input word")
        ax.set_ylabel("word being tagged")
fig.suptitle("LID: which words each model focuses on (row-normalised)", y=1.0)
fig.tight_layout()
fig.savefig("heatmap_lid.png", dpi=160, bbox_inches="tight")
plt.show()
print("Saved heatmap_lid.png")
print("\nAll done.")