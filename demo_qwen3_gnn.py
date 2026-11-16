"""
Demo: Qwen3 (LoRA) + GNN for Hinglish sentiment and language identification
============================================================================
Loads the bundle written by qwen3_gnn_finetune.py (qwen3_gnn_demo.pkl), rebuilds
the architecture (pure PyTorch, no `transformers`) and runs it on sample
Hinglish statements.

Usage:
    python demo_qwen3_gnn.py                          # built-in sample sentences
    python demo_qwen3_gnn.py --text "yaar ye movie mast thi" --text "bakwas service"
    python demo_qwen3_gnn.py --file my_sentences.txt  # one sentence per line
    python demo_qwen3_gnn.py --interactive            # type sentences, empty line quits
    python demo_qwen3_gnn.py --show-arch              # print architecture + test results
    python demo_qwen3_gnn.py --qwen-dir <folder>      # local Qwen3 files (skip download)

Only load .pkl files you created yourself (pickle can execute code).

Needs: torch tokenizers safetensors huggingface_hub torch-geometric
"""
import os, re, json, glob, bisect, argparse, pickle, csv
from contextlib import contextmanager
from types import SimpleNamespace
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from safetensors.torch import load_file
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

SAMPLES = [
    "yaar ye movie bahut mast thi, totally paisa vasool",
    "kya bakwas service hai, bilkul waste of time",
    "kal meeting hai office mein 10 baje",
    "mujhe tumhara new phone bahut pasand aaya",
    "traffic ne aaj phir se mood kharab kar diya",
    "India ne match jeet liya, kya zabardast performance thi",
    "main kal market gaya tha sabzi lene",
    "tera behavior bilkul acha nahi tha, I am really disappointed",
]

LID_MAP = {"eng": 1, "hin": 2}

# --- prompts: MUST be identical to the ones used for training ----------------
def render(prompt):
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

def parse_lid(ans, n):
    pred = [0] * n
    for i, t in re.findall(r"(\d+)\s*[:\-]\s*(eng|hin|other)", ans):
        if 1 <= int(i) <= n:
            pred[int(i) - 1] = LID_MAP.get(t, 0)
    return pred

# ---------------------------------------------------------------------------
# Pure-PyTorch Qwen3 (same module names as the official checkpoint)
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
        self.q_norm = RMSNorm(self.d, c.eps)
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
        self.model = Backbone(c)
        self.lm_head = nn.Linear(c.hidden, c.vocab, bias=False)

    def embed(self, ids):
        return self.model.embed_tokens(ids)

    def logits(self, h):
        return self.lm_head(h)

    def _rope(self, pos, dtype):
        d = self.c.head_dim
        inv = 1.0 / (self.c.rope_theta ** (torch.arange(0, d, 2, device=pos.device, dtype=torch.float32) / d))
        fr  = pos[..., None].float() * inv
        emb = torch.cat([fr, fr], dim=-1)
        return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)

    def hidden(self, embeds, attn_mask=None, pos=None, caches=None, use_cache=False):
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
        mask = torch.zeros(allowed.shape, dtype=embeds.dtype, device=dev)
        mask = mask.masked_fill(~allowed, torch.finfo(embeds.dtype).min)
        h, new = embeds, []
        for i, layer in enumerate(self.model.layers):
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

# ---------------------------------------------------------------------------
# LoRA kept UN-merged so it can be switched off: the GNNs were trained on the
# features of the *frozen* base model, so features must be extracted with LoRA
# off, while the Qwen3 predictions use LoRA on.
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    enabled = True

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base  = base
        self.scale = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        y = self.base(x)
        if not LoRALinear.enabled:
            return y
        return y + (x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale

@contextmanager
def lora(on):
    prev, LoRALinear.enabled = LoRALinear.enabled, on
    try:
        yield
    finally:
        LoRALinear.enabled = prev

def attach_lora(model, lcfg):
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in lcfg["targets"]:
                if hasattr(parent, n):
                    setattr(parent, n, LoRALinear(getattr(parent, n), lcfg["r"], lcfg["alpha"]))
    res = model.load_state_dict(lcfg["state_dict"], strict=False)
    assert not res.unexpected_keys, f"Unexpected LoRA keys: {res.unexpected_keys[:5]}"
    missing = [k for k in res.missing_keys if k.endswith(".A") or k.endswith(".B")]
    assert not missing, f"Missing LoRA keys: {missing[:5]}"

def load_base(name, c, local_dir, dtype, dev):
    if local_dir is None:
        from huggingface_hub import snapshot_download
        print(f"Fetching {name} files (config, tokenizer, safetensors) …")
        local_dir = snapshot_download(repo_id=name, allow_patterns=["*.json", "*.safetensors", "*.txt"])
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
        p.requires_grad_(False)
    tk = Tokenizer.from_file(os.path.join(local_dir, "tokenizer.json"))
    return model, tk

# ---------------------------------------------------------------------------
# GNN + word graph (identical to training)
# ---------------------------------------------------------------------------
class GNN(nn.Module):
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
        return self.out(self.drop(z)), z

def build_graph(x, window):
    n = x.size(0)
    e = [[i, j] for i in range(n)
         for j in range(max(0, i - window), min(n, i + window + 1)) if i != j]
    ei = (torch.tensor(e, dtype=torch.long).t().contiguous()
          if e else torch.empty((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=ei)

def word_layout(words):
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

# ---------------------------------------------------------------------------
# The demo pipeline
# ---------------------------------------------------------------------------
class HinglishDemo:
    def __init__(self, pkl_path, qwen_dir=None):
        with open(pkl_path, "rb") as f:
            self.B = pickle.load(f)
        B = self.B
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
        print(f"Device: {self.dev}")

        c = SimpleNamespace(**B["qwen_config"])
        self.hidden_size = c.hidden
        print(f"Loading base {B['qwen_name']} …")
        self.qwen, self.tok = load_base(B["qwen_name"], c, qwen_dir, dtype, self.dev)
        if B["lora"]["state_dict"] is not None:
            attach_lora(self.qwen, B["lora"])
            print("LoRA adapters attached.")
        else:
            print("Bundle has no LoRA adapters: Qwen3 runs as the base model.")

        g = B["gnn"]
        self.window = g["window"]
        self.max_words = B["max_words"]
        self.gnn_lid = GNN(g["in_dim"], g["hidden"], len(B["labels"]["lid"]), "lid").to(self.dev)
        self.gnn_senti = GNN(g["in_dim"], g["hidden"], len(B["labels"]["senti"]), "senti").to(self.dev)
        self.gnn_lid.load_state_dict(g["lid_state"])
        self.gnn_senti.load_state_dict(g["senti_state"])
        self.gnn_lid.eval(); self.gnn_senti.eval()

        self.lid_names = B["labels"]["lid"]
        self.senti_names = B["labels"]["senti"]
        self.pad_id = self.tok.token_to_id("<|endoftext|>")
        self.eos_ids = {self.tok.token_to_id("<|im_end|>"), self.pad_id}
        self.cls_tok = [self.tok.encode(n, add_special_tokens=False).ids[0] for n in self.senti_names]
        assert len(set(self.cls_tok)) == len(self.cls_tok), "label first-tokens collide"

    # -- helpers --------------------------------------------------------------
    def encode(self, text):
        e = self.tok.encode(text, add_special_tokens=False)
        return e.ids, e.offsets

    def words_of(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())[: self.max_words]

    def left_pad(self, texts):
        seqs = [self.encode(t)[0] for t in texts]
        L = max(len(s) for s in seqs)
        ids  = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):]  = torch.tensor(s)
            mask[i, L - len(s):] = 1
        return ids.to(self.dev), mask.to(self.dev)

    @torch.no_grad()
    def word_features(self, words):
        """Frozen BASE Qwen3 features (LoRA off), as used to train the GNNs."""
        text, st, en = word_layout(words)
        ids, offs = self.encode(text)
        ids, offs = ids[:512], offs[:512]
        ids_t = torch.tensor([ids], device=self.dev)
        with lora(False):
            h = self.qwen.hidden(self.qwen.embed(ids_t))[0][0].float()
        widx = tok2word(offs, st, en).to(self.dev)
        ok = widx >= 0
        feats = torch.zeros(len(words), self.hidden_size, device=self.dev).index_add_(0, widx[ok], h[ok])
        cnt = torch.bincount(widx[ok], minlength=len(words)).clamp(min=1).unsqueeze(1)
        return (feats / cnt).half().cpu()

    @torch.no_grad()
    def gnn_predict(self, feats):
        d = build_graph(feats.float(), self.window).to(self.dev)
        d.batch = torch.zeros(feats.size(0), dtype=torch.long, device=self.dev)
        lid_logits, _ = self.gnn_lid(d)
        s_logits, _   = self.gnn_senti(d)
        return lid_logits.argmax(-1).tolist(), s_logits.softmax(-1)[0].cpu()

    @torch.no_grad()
    def qwen_sentiment(self, word_lists):
        """Probabilities over the label words, read from the first answer token."""
        ids, mask = self.left_pad([render(SENT_PROMPT(w)) for w in word_lists])
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with lora(True):
            h, _ = self.qwen.hidden(self.qwen.embed(ids), mask, pos)
        lg = self.qwen.logits(h[:, -1]).float()[:, self.cls_tok]
        return lg.softmax(-1).cpu()

    @torch.no_grad()
    def qwen_lid(self, word_lists):
        ids, mask = self.left_pad([render(LID_PROMPT(w)) for w in word_lists])
        n_max = max(len(w) for w in word_lists)
        with lora(True):
            out = self.qwen.generate(ids, mask, 6 * n_max + 16, self.eos_ids, self.pad_id)
        return [parse_lid(self.tok.decode(o.tolist(), skip_special_tokens=True).strip().lower(), len(w))
                for o, w in zip(out, word_lists)]

    # -- main entry -------------------------------------------------------------
    def analyze(self, texts, bs=8):
        texts = [t for t in texts if self.words_of(t)]
        words = [self.words_of(t) for t in texts]
        gnn_out = [self.gnn_predict(self.word_features(w)) for w in words]
        q_senti = torch.cat([self.qwen_sentiment(words[i:i + bs]) for i in range(0, len(words), bs)])
        q_lid = []
        for i in range(0, len(words), bs):
            q_lid += self.qwen_lid(words[i:i + bs])
        res = []
        for k, t in enumerate(texts):
            g_lid, g_p = gnn_out[k]
            q_p = q_senti[k]
            f_p = (q_p + g_p) / 2
            res.append({
                "text": t, "words": words[k],
                "qwen_sentiment": self.senti_names[int(q_p.argmax())], "qwen_conf": float(q_p.max()),
                "gnn_sentiment":  self.senti_names[int(g_p.argmax())], "gnn_conf":  float(g_p.max()),
                "fused_sentiment": self.senti_names[int(f_p.argmax())], "fused_conf": float(f_p.max()),
                "qwen_lid": [self.lid_names[x] for x in q_lid[k]],
                "gnn_lid":  [self.lid_names[x] for x in g_lid],
            })
        return res

def show(results):
    for i, r in enumerate(results, 1):
        tag = lambda tags: " ".join(f"{w}/{t}" for w, t in zip(r["words"], tags))
        print(f"\n[{i}] {r['text']}")
        print(f"    Qwen3 sentiment : {r['qwen_sentiment']:<9} ({r['qwen_conf']:.2f})")
        print(f"    GNN   sentiment : {r['gnn_sentiment']:<9} ({r['gnn_conf']:.2f})")
        print(f"    Fused (mean of probabilities, demo only) : {r['fused_sentiment']} ({r['fused_conf']:.2f})")
        print(f"    LID Qwen3 : {tag(r['qwen_lid'])}")
        print(f"    LID GNN   : {tag(r['gnn_lid'])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="qwen3_gnn_demo.pkl")
    ap.add_argument("--text", action="append", help="sentence to analyse (repeatable)")
    ap.add_argument("--file", help="text file, one sentence per line")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--show-arch", action="store_true")
    ap.add_argument("--qwen-dir", default=None, help="local folder with Qwen3 files (skips download)")
    ap.add_argument("--out", default="demo_predictions.csv")
    a = ap.parse_args()

    demo = HinglishDemo(a.pkl, a.qwen_dir)

    if a.show_arch:
        print("\n" + demo.B["architecture"])
        print("\nTest-set results stored in the bundle:")
        for r in demo.B.get("results", []):
            print(f"  {r['Condition']:<34} {r['Task']:<9} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f}")

    texts = list(a.text or [])
    if a.file:
        texts += [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    if not texts and not a.interactive:
        texts = SAMPLES

    if texts:
        results = demo.analyze(texts)
        show(results)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "qwen_sentiment", "qwen_conf", "gnn_sentiment", "gnn_conf",
                        "fused_sentiment", "fused_conf"])
            for r in results:
                w.writerow([r["text"], r["qwen_sentiment"], f"{r['qwen_conf']:.4f}",
                            r["gnn_sentiment"], f"{r['gnn_conf']:.4f}",
                            r["fused_sentiment"], f"{r['fused_conf']:.4f}"])
        print(f"\nSaved {a.out}")

    if a.interactive:
        print("\nType a Hinglish sentence (empty line to quit).")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                break
            if not s:
                break
            show(demo.analyze([s]))

if __name__ == "__main__":
    main()
