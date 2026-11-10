"""
forced_prefix_probe.py -- interventional test of where generation failures originate.

For each puzzle we regenerate level by level, but for the chosen levels we OVERWRITE the model's codes with the
encoder's codes (the real ground-truth codes), letting the model generate the remaining levels conditioned on them.
We then decode and score exact match.

How generation is reproduced without net.generate's internals:
  Level k's logits depend only on the inputs derived from codes of levels < k (the model is trained with one
  teacher-forced pass, which only works if attention is block-causal across levels). So we run the
  teacher-forced forward pass with the codes known so far (later levels zero-padded) and read the logits for level k.
  GATE: the unforced run ("free") must reproduce net.generate exactly. If it does not, this script's causal-masking
  assumption is wrong for your model; do not interpret the forced results.

Conditions (--conditions, comma separated):
  free      no forcing (validation)
  0         force Level 0 only
  0-3       force Levels 0..3
  2 / 3     force a single level
  5-11      force the late levels only (control: should NOT rescue early failures)
  0-10      force everything except the last level
  all       force everything (sanity: decoded ground-truth codes, expect ~100%)

Usage:
  python forced_prefix_probe.py --config config/eval/sudoku-hard.yaml --n 100 --out results/forced_prefix.pkl
"""
import argparse
import gc
import os
import pickle
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--bs", type=int, default=25, help="deprecated; samples are processed individually")
ap.add_argument("--conditions", default="free,0,0-1,0-2,0-3,2,3,5-11,0-10,all")
ap.add_argument("--no-eos", action="store_true")
ap.add_argument("--max-vram-gb", type=float, default=5.5,
                help="Abort cleanly if reserved VRAM exceeds this (default 5.5GB for 6GB-class GPUs).")
ap.add_argument("--out", default="results/forced_prefix.pkl")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
G = CFG.generation
if a.checkpoint_dir:
    ck = a.checkpoint_dir
else:
    ck = os.path.join("artifacts", CFG.task)
    if CFG.task.startswith("sudoku-"):
        sudoku_ck = os.path.join("artifacts", "sudoku", CFG.task[len("sudoku-"):])
        if not os.path.isfile(os.path.join(ck, CFG.checkpoints.generator)) and os.path.isfile(
            os.path.join(sudoku_ck, CFG.checkpoints.generator)
        ):
            ck = sudoku_ck
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])

tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
vq = tk.vqvae.vq
levels = list(vq.levels)
NL = len(levels)
bounds = np.cumsum([0] + levels)
SEQ = levels[-1]
P = int(CFG.model.prefix_len)
EOS = tk.pre_tokenizer.eos_token_id
PAD = tk.pre_tokenizer.pad_token_id if tk.pre_tokenizer.pad_token_id is not None else 0

net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()

temp = float(G.get("temperature", 1.0))
top_k = int(G.get("top_k", 1))
top_p = float(G.get("top_p", 0.0))
cfg_scale = float(G.get("cfg_scale", 0.0))

items = pickle.load(open(os.path.join(dd, CFG.data[a.split]), "rb"))[: a.n]
refs = pickle.load(open(os.path.join(dd, CFG.data[f"{a.split}_refs"]), "rb"))[: a.n]
assert len(items) == len(refs)
N = len(items)
src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


def dec(ids):
    ids = list(ids)
    if EOS is not None and EOS in ids:
        ids = ids[: ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def clear_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def abort_if_vram_exceeds(label, sample_idx=None):
    alloc, res = vram_gb()
    if res > a.max_vram_gb:
        if sample_idx is None:
            print(f"[forced-prefix][abort] {label}: reserved={res:.2f}GB > threshold {a.max_vram_gb:.2f}GB",
                  flush=True)
        else:
            print(f"[forced-prefix][abort] {label} sample {sample_idx}: reserved={res:.2f}GB > "
                  f"threshold {a.max_vram_gb:.2f}GB", flush=True)
        raise RuntimeError("VRAM threshold exceeded")


def teacher_forced_logits(codes, x):
    target_inputs = vq.idx_to_ncm_input(codes)
    if net.query_level0:
        l0 = net.build_l0_query_input(codes.shape[0], dtype=target_inputs.dtype)
        target_inputs = torch.cat([l0, target_inputs], dim=1)
    if net.use_ctx_encoder:
        return net(target_inputs, prefix_ids=x)
    if bool(getattr(net.config, "raw_token_prefix", False)):
        encoder = tk.vqvae.encoder
        if bool(getattr(net.config, "contextual_prefix", False)) and len(encoder.blocks) > 0:
            prefix_z = encoder.forward_nopos(x)
        else:
            prefix_z = encoder.wte(x)
    else:
        prefix_z = tk.vqvae.encode_z(x)
    return net(torch.cat([prefix_z[:, :P, :], target_inputs], dim=1))


@torch.no_grad()
def run_condition(x, enc, force):
    """Generate level by level; levels in `force` take the encoder's codes, the rest are the model's argmax."""
    codes = torch.zeros_like(enc)
    for k in range(NL):
        s = slice(int(bounds[k]), int(bounds[k + 1]))
        if k in force:
            codes[:, s] = enc[:, s]
        else:
            logits = teacher_forced_logits(codes, x)
            codes[:, s] = logits[:, s, :].argmax(-1)
    return codes


@torch.no_grad()
def decode_batch(codes):
    _, zl = vq.idx_to_zhats(codes, return_zlevels=True)
    ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
    return [dec(r).strip() for r in ids]


def parse_cond(s):
    s = s.strip()
    if s == "free":
        return frozenset()
    if s == "all":
        return frozenset(range(NL))
    if "-" in s:
        lo, hi = s.split("-")
        return frozenset(range(int(lo), int(hi) + 1))
    return frozenset([int(s)])


conds = [(c.strip(), parse_cond(c)) for c in a.conditions.split(",") if c.strip()]
if not any(len(f) == 0 for _, f in conds):
    conds.insert(0, ("free", frozenset()))

# ---- reference codes (validated path) and per-sample inputs
TEXT, X, ENC = [], [], []
with torch.no_grad():
    for i in range(N):
        text = refs[i].strip() if isinstance(refs[i], str) else dec(refs[i]).strip()
        ids = list(tk.pre_tokenizer.encode(text))
        if not a.no_eos and EOS is not None:
            ids = ids + [EOS]
        ids = (ids + [PAD] * SEQ)[:SEQ]
        ids_t = torch.tensor([ids], device="cuda")
        z = tk.vqvae.encode_z(ids_t)
        assert z.shape[1] == SEQ, f"encode_z length {z.shape[1]} != {SEQ}"
        ENC.append(vq.z_to_idx(z)[0].cpu())
        X.append(torch.tensor(prefix(items[i])))
        TEXT.append(text)
        del ids_t, z
        clear_cuda_cache()
X, ENC = torch.stack(X), torch.stack(ENC)
print(f"{N} puzzles; levels {levels}")
alloc0, res0 = vram_gb()
print(f"[forced-prefix] VRAM after load: allocated={alloc0:.2f}GB reserved={res0:.2f}GB", flush=True)
print(f"[forced-prefix] abort threshold: reserved > {a.max_vram_gb:.2f}GB", flush=True)

# ---- run all conditions
results = {}
for name, force in conds:
    codes_all, hyps = [], []
    try:
        for i in range(N):
            abort_if_vram_exceeds(f"condition {name}", i)
            clear_cuda_cache()
            x_i = X[i:i + 1].cuda()
            enc_i = ENC[i:i + 1].cuda()
            cb = run_condition(x_i, enc_i, force)
            codes_all.append(cb[0].cpu().numpy())
            hyps.append(decode_batch(cb)[0])
            del x_i, enc_i, cb
            clear_cuda_cache()
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        print(f"[forced-prefix][abort] stopping condition {name} early due to VRAM pressure; "
              f"saved partial results so far.", flush=True)
        break
    ok = np.array([h == t for h, t in zip(hyps, TEXT)])
    results[name] = {"codes": np.stack(codes_all), "ok": ok}
    print(f"ran {name:<6}: {int(ok.sum())}/{N} correct", flush=True)

if "free" not in results:
    print("[forced-prefix][abort] no valid free-run results; exiting before gate check.", flush=True)
    raise SystemExit(0)

# ---- gate: unforced loop vs net.generate
gen_codes = []
with torch.no_grad():
    try:
        for i in range(N):
            abort_if_vram_exceeds("generation gate", i)
            clear_cuda_cache()
            x_i = X[i:i + 1].cuda()
            gen = net.generate(None, x_i, temperature=temp, top_k=top_k, top_p=top_p,
                               cfg_scale=cfg_scale, cfg_null_eos=EOS)
            gen_codes.append(gen[0].cpu().numpy())
            del x_i, gen
            clear_cuda_cache()
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        print("[forced-prefix][abort] generation gate stopped early due to VRAM pressure.", flush=True)
        gen = np.stack(gen_codes)
        same = np.zeros(len(gen), dtype=bool) if len(gen) else np.array([], dtype=bool)
    else:
        gen = np.stack(gen_codes)
        same = (results["free"]["codes"] == gen).all(axis=1)
print(f"\nGATE: unforced level-by-level run reproduces net.generate on {int(same.sum())}/{N} puzzles "
      f"(must be {N}; otherwise the forced results below are not valid)")

# ---- groups defined by the free run
free_codes, E = results["free"]["codes"], ENC.cpu().numpy()
base_ok = results["free"]["ok"]
early = np.array([(free_codes[r, :bounds[4]] != E[r, :bounds[4]]).any() for r in range(N)])
groups = {
    "failures": ~base_ok,
    "early-div correct": base_ok & early,
    "late-div correct": base_ok & ~early,
}
print("group sizes:", {k: int(v.sum()) for k, v in groups.items()}, "(earlier analysis: 16 / 29 / 55)")

print("\n--- Forced-prefix results (levels listed are overwritten with the encoder's codes) ---")
print(f"{'forced':<8}{'all correct':<14}{'failures fixed':<18}{'early-div correct kept':<26}{'late-div correct kept'}")
for name, _ in conds:
    ok = results[name]["ok"]
    cells = [f"{int(ok.sum())}/{N}"]
    for g in ("failures", "early-div correct", "late-div correct"):
        m = groups[g]
        cells.append(f"{int(ok[m].sum())}/{int(m.sum())}")
    print(f"{name:<8}{cells[0]:<14}{cells[1]:<18}{cells[2]:<26}{cells[3]}")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump({"levels": levels, "enc": E, "results": results, "gate_same": same,
             "groups": {k: v for k, v in groups.items()}}, open(a.out, "wb"))
print(f"\nSaved {a.out}")