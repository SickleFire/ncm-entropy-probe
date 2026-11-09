import argparse
import csv
import gc
import os
import pickle
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-vram-gb", type=float, default=5.5,
                    help="Abort cleanly if reserved VRAM exceeds this (default 5.5GB for 6GB cards).")
    ap.add_argument("--log-every", type=int, default=1,
                    help="Print per-sample progress every N samples.")
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

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    levels = list(net.levels)
    vocab_sizes = list(net.vocab_sizes)
    print(f"[probe] levels={levels}", flush=True)
    print(f"[probe] vocab_sizes={vocab_sizes}", flush=True)
    print(f"[probe] n_levels={len(levels)} -> expect {len(levels)} records per sample "
          f"(level 0 = coarse, levels 1..{len(levels)-1} = refinements)", flush=True)

    P = int(CFG.model.prefix_len)
    EOS = tk.pre_tokenizer.eos_token_id
    PAD = tk.pre_tokenizer.pad_token_id

    temp = float(G.get("temperature", 1.0))
    top_k = int(G.get("top_k", 1))
    top_p = float(G.get("top_p", 0.0))
    cfg_scale = float(G.get("cfg_scale", 0.0))
    print(f"[probe] temp={temp} top_k={top_k} top_p={top_p} cfg_scale={cfg_scale}", flush=True)

    alloc0, res0 = vram_gb()
    print(f"[probe] VRAM after load: allocated={alloc0:.2f}GB reserved={res0:.2f}GB", flush=True)
    print(f"[probe] abort threshold: reserved > {a.max_vram_gb:.2f}GB", flush=True)

    record = []
    orig_sample = net._sample_level

    def logged_sample_level(logits, temperature, top_k, top_p):
        record.append(logits.detach().float().cpu())
        return orig_sample(logits, temperature, top_k, top_p)

    net._sample_level = logged_sample_level

    items = pickle.load(open(os.path.join(dd, CFG.data[a.split]), "rb"))[: a.n]
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{a.split}_refs"]), "rb"))[: a.n]
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) \
        if isinstance(items[0], str) else None

    def prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [PAD] * P)[:P]

    def dec(ids):
        ids = list(ids)
        if EOS in ids:
            ids = ids[: ids.index(EOS)]
        return tk.pre_tokenizer.decode(
            [t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

    def is_correct(hyp, ref):
        return int(hyp.strip() == ref.strip())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "sample_id", "level", "L_cur", "V_k",
        "H_mean", "H_max", "C_mean", "C_min",
        "correct",
    ])

    n_total = len(items)
    for sidx in range(n_total):
        alloc, res = vram_gb()
        if res > a.max_vram_gb:
            print(f"[probe][abort] reserved={res:.2f}GB > threshold at sample {sidx}; "
                  f"writing partial CSV and exiting cleanly.", flush=True)
            break

        record.clear()
        torch.cuda.empty_cache()
        gc.collect()

        try:
            x = torch.tensor([prefix(items[sidx])], device="cuda")
            g = net.generate(None, x, temperature=temp, top_k=top_k, top_p=top_p,
                             cfg_scale=cfg_scale, cfg_null_eos=EOS)

            n_rec = len(record)
            if n_rec != len(levels):
                print(f"[probe][warn] sample {sidx}: got {n_rec} records, "
                      f"expected {len(levels)}", flush=True)

            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            dec_logits = tk.vqvae.decoder(zl[-1])
            ids = dec_logits.argmax(-1).tolist()
            hyp = dec(ids[0])
            ref = refs[sidx] if isinstance(refs[sidx], str) else dec(refs[sidx])
            correct = is_correct(hyp, ref)

            for lvl, lg in enumerate(record, start=0):
                p = F.softmax(lg, dim=-1)
                ent = -(p * (p + 1e-12).log()).sum(-1)
                conf = p.max(-1).values
                w.writerow([
                    sidx, lvl, int(lg.shape[1]), int(lg.shape[2]),
                    float(ent.mean()), float(ent.max()),
                    float(conf.mean()), float(conf.min()),
                    correct,
                ])

            if a.log_every > 0 and sidx % a.log_every == 0:
                parts = []
                for lvl, lg in enumerate(record, start=0):
                    p = F.softmax(lg, dim=-1)
                    h = float(-(p * (p + 1e-12).log()).sum(-1).mean())
                    parts.append(f"L{lvl} H={h:.3f}")
                alloc, res = vram_gb()
                print(f"[probe] sample {sidx}: correct={correct} | "
                      f"{' '.join(parts)} | VRAM alloc={alloc:.2f} res={res:.2f}", flush=True)

        except torch.cuda.OutOfMemoryError:
            print(f"[probe][oom] CUDA OOM at sample {sidx}; writing partial CSV and exiting.",
                  flush=True)
            break
        finally:
            for name in ("g", "x", "ids", "dec_logits", "zl"):
                if name in locals():
                    del locals()[name]
            record.clear()
            torch.cuda.empty_cache()
            gc.collect()

    f.close()
    alloc, res = vram_gb()
    print(f"[probe] final VRAM: allocated={alloc:.2f}GB reserved={res:.2f}GB", flush=True)
    print(f"[probe] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()