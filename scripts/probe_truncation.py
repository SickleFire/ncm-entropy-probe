import argparse
import gc
import os
import pickle
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    CFG = OmegaConf.load(a.config)
    ck = a.checkpoint_dir
    dd = a.data_dir

    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    sd = torch.load(os.path.join(ck, CFG.checkpoints.generator),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    P = int(CFG.model.prefix_len)
    EOS = tk.pre_tokenizer.eos_token_id
    PAD = tk.pre_tokenizer.pad_token_id
    temp = float(CFG.generation.get("temperature", 1.0))
    top_k = int(CFG.generation.get("top_k", 1))
    top_p = float(CFG.generation.get("top_p", 0.0))
    cfg_scale = float(CFG.generation.get("cfg_scale", 0.0))
    full_L = int(net.L)

    items = pickle.load(open(os.path.join(dd, CFG.data["test"]), "rb"))[: a.n]
    refs = pickle.load(open(os.path.join(dd, CFG.data["test_refs"]), "rb"))[: a.n]
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

    # Truncation points (level k means generate levels 0..k, skip k+1..end)
    TRUNC_LEVELS = [0, 8, 10, None]
    results = {lvl: 0 for lvl in TRUNC_LEVELS}

    print(f"[trunc] full sequence length L = {full_L}", flush=True)
    print(f"[trunc] truncation points: {TRUNC_LEVELS}", flush=True)
    print(f"[trunc] samples: {len(items)}", flush=True)

    alloc0 = torch.cuda.memory_allocated() / 1e9
    res0 = torch.cuda.memory_reserved() / 1e9
    print(f"[trunc] VRAM after load: alloc={alloc0:.2f}GB res={res0:.2f}GB", flush=True)

    for i in range(len(items)):
        x = torch.tensor([prefix(items[i])], device="cuda")
        ref = refs[i] if isinstance(refs[i], str) else dec(refs[i])

        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"[trunc] sample {i}/{len(items)}: VRAM alloc={alloc:.2f}GB res={res:.2f}GB",
              flush=True)

        for lvl in TRUNC_LEVELS:
            try:
                with torch.no_grad():
                    g = net.generate(None, x, temperature=temp, top_k=top_k,
                                     top_p=top_p, cfg_scale=cfg_scale,
                                     cfg_null_eos=EOS, stop_at_level=lvl)

                if g.shape[1] < full_L:
                    pad_len = full_L - g.shape[1]
                    g = torch.cat([
                        g,
                        torch.zeros((g.shape[0], pad_len), dtype=g.dtype, device=g.device),
                    ], dim=1)

                _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
                hyp = dec(ids[0])
                is_correct = int(hyp.strip() == ref.strip())
                results[lvl] += is_correct
            except torch.cuda.OutOfMemoryError:
                print(f"[trunc][oom] at sample {i} lvl={lvl}; stopping.", flush=True)
                lvl_err = True
            finally:
                for name in ("g", "zl", "ids", "dec_logits", "hyp"):
                    if name in locals():
                        del locals()[name]
                torch.cuda.empty_cache()
                gc.collect()

        for name in ("x", "ref"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()
        gc.collect()

    print("\n--- Truncation Results ---", flush=True)
    print(f"Total samples: {len(items)}", flush=True)
    for lvl in TRUNC_LEVELS:
        acc = results[lvl] / len(items) * 100
        name = f"Stop after level {lvl}" if lvl is not None else "Full generation"
        print(f"{name}: {results[lvl]}/{len(items)} correct ({acc:.1f}%)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"total_samples: {len(items)}\n")
        for lvl in TRUNC_LEVELS:
            acc = results[lvl] / len(items) * 100
            name = f"stop_after_level_{lvl}" if lvl is not None else "full"
            f.write(f"{name}: {results[lvl]}/{len(items)} ({acc:.1f}%)\n")
    print(f"[trunc] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()