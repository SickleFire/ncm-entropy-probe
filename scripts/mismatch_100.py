import argparse
import gc
import os
import pickle
import sys
from collections import Counter

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--no-eos", action="store_true", help="do not append EOS to the reference before encoding")
ap.add_argument("--out", default="results/mismatch_probe.pkl")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])

tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
vq = tk.vqvae.vq
levels = list(vq.levels)
bounds = np.cumsum([0] + levels)
SEQ = levels[-1]
P = int(CFG.model.prefix_len)
EOS = tk.pre_tokenizer.eos_token_id
PAD = tk.pre_tokenizer.pad_token_id if tk.pre_tokenizer.pad_token_id is not None else 0

net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()

# same decoding settings as the eval script (config defaults), NOT temperature=0 / top_k=0
temp = float(G.get("temperature", 1.0))
top_k = int(G.get("top_k", 1))
top_p = float(G.get("top_p", 0.0))
cfg_scale = float(G.get("cfg_scale", 0.0))
print(f"generation settings: temp={temp} top_k={top_k} top_p={top_p} cfg_scale={cfg_scale}")

items = pickle.load(open(os.path.join(dd, CFG.data[a.split]), "rb"))[: a.n]
refs = pickle.load(open(os.path.join(dd, CFG.data[f"{a.split}_refs"]), "rb"))[: a.n]
assert len(items) == len(refs)
src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


def dec(ids):
    ids = list(ids)
    if EOS is not None and EOS in ids:
        ids = ids[: ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def teacher_forced_logits(enc, x):
    target_inputs = vq.idx_to_ncm_input(enc)
    if net.query_level0:
        l0 = net.build_l0_query_input(enc.shape[0], dtype=target_inputs.dtype)
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


E, Gn, T, OK = [], [], [], []
encode_agrees = None

with torch.no_grad():
    for i in range(len(refs)):
        text = refs[i].strip() if isinstance(refs[i], str) else dec(refs[i]).strip()

        # real ground-truth codes (validated path: 15/15 round-trip at full depth)
        ids = list(tk.pre_tokenizer.encode(text))
        if not a.no_eos and EOS is not None:
            ids = ids + [EOS]
        ids = (ids + [PAD] * SEQ)[:SEQ]
        ids_t = torch.tensor([ids], device="cuda")
        z = tk.vqvae.encode_z(ids_t)
        assert z.shape[1] == SEQ, f"encode_z length {z.shape[1]} != {SEQ}"
        enc = vq.z_to_idx(z)
        if encode_agrees is None:
            try:
                encode_agrees = bool(torch.equal(enc, tk.vqvae.encode(ids_t)))
            except Exception as e:
                encode_agrees = f"n/a ({type(e).__name__})"
            print(f"z_to_idx codes equal tk.vqvae.encode codes: {encode_agrees}")

        x = torch.tensor([prefix(items[i])], device="cuda")
        g = net.generate(None, x, temperature=temp, top_k=top_k, top_p=top_p,
                         cfg_scale=cfg_scale, cfg_null_eos=EOS)

        _, zl = vq.idx_to_zhats(g, return_zlevels=True)
        hyp = dec(tk.vqvae.decoder(zl[-1]).argmax(-1)[0].tolist()).strip()

        logits = teacher_forced_logits(enc, x)
        tf = torch.cat([logits[:, bounds[k]:bounds[k + 1], :].argmax(-1) for k in range(len(levels))], dim=1)

        E.append(enc[0].cpu().numpy())
        Gn.append(g[0].cpu().numpy())
        T.append(tf[0].cpu().numpy())
        OK.append(hyp == text)
        if (i + 1) % 10 == 0 or i + 1 == len(refs):
            print(f"Processed {i + 1}/{len(refs)}", flush=True)
        del ids_t, z, enc, x, g, _, zl, logits, tf
        gc.collect()
        torch.cuda.empty_cache()

E, Gn, T, OK = np.stack(E), np.stack(Gn), np.stack(T), np.array(OK)
n, n_ok = len(OK), int(OK.sum())
print(f"\nCorrect (free-running, exact match): {n_ok}/{n}   <-- must match the count used in your AUC analysis")

print("\n--- Per level ---")
print("level pos | gen!=enc per-position (all / correct) | teacher-forced acc vs enc | distinct level-codes gen / enc")
for k in range(len(levels)):
    s = slice(bounds[k], bounds[k + 1])
    mm_all = (Gn[:, s] != E[:, s]).mean() * 100
    mm_ok = (Gn[OK][:, s] != E[OK][:, s]).mean() * 100 if n_ok else float("nan")
    tf_acc = (T[:, s] == E[:, s]).mean() * 100
    d_gen = len(set(map(tuple, Gn[:, s].tolist())))
    d_enc = len(set(map(tuple, E[:, s].tolist())))
    print(f"{k:>5} {levels[k]:>3} | {mm_all:6.1f}% / {mm_ok:6.1f}%               | {tf_acc:6.1f}%                    | {d_gen:>3} / {d_enc:<3}")

first = []
for r in range(n):
    f = None
    for k in range(len(levels)):
        s = slice(bounds[k], bounds[k + 1])
        if (Gn[r, s] != E[r, s]).any():
            f = k
            break
    first.append(-1 if f is None else f)
first = np.array(first)
print("\n--- First level where generation diverges from the encoder codes (-1 = never) ---")
print("correct  :", dict(sorted(Counter(first[OK].tolist()).items())))
print("incorrect:", dict(sorted(Counter(first[~OK].tolist()).items())))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump({"levels": levels, "enc": E, "gen": Gn, "tf": T, "correct": OK, "first_divergence": first}, open(a.out, "wb"))
print(f"\nSaved {a.out}")
print("Level 0: if distinct gen codes == 1, the model emits one coarse code for every puzzle; then Level 0 carries no "
      "puzzle-specific information and its entropy/AUC mean something different from the mid-levels.")