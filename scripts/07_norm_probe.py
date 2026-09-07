"""Massive-activation probe: is the layer-12 collapse a norm artifact?

Observation 3 (prereg/DEVIATIONS.md) found 10/48 layers where spectral
clustering isolates <1% of states -- five of them a SINGLE token. That is the
massive-activation signature. This script measures it directly rather than
inferring it from cluster balance.

Reports, per layer, over the ALIGNED ANCHOR positions only (the states that
actually enter the analysis):
  - median and max L2 norm of the residual state
  - ratio max/median  (the massive-activation indicator)
  - max |single coordinate| and which dimension carries it
  - how concentrated the top dimension is (share of squared norm)

If a few dimensions dominate the squared norm from layer ~12 onward, cosine
geometry is governed by those dimensions and the k=2 partition is isolating
outliers, not linguistic structure.
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import ling.determinism as _det  # noqa: F401  (must precede torch/CUDA init)
import torch
from ling.model import load_model
from ling.hooks import ResidualCapture, forward_once, resolve_anchor_positions, anchor_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/problems.jsonl")
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--max-problems", type=int, default=20)
    ap.add_argument("--condition", default="nl_base")
    ap.add_argument("--out", default="results/norm_probe.json")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.corpus)][: a.max_problems]
    loaded = load_model(a.config)
    tok, model, L = loaded.tokenizer, loaded.model, loaded.n_layers

    per_layer = [[] for _ in range(L)]      # L2 norms of anchor states
    per_layer_max_coord = [0.0] * L
    per_layer_top_share = [[] for _ in range(L)]
    top_dim_counter = [{} for _ in range(L)]

    for rec in recs:
        real = rec["realizations"].get(a.condition)
        if not real or not real.get("text"):
            continue
        names = anchor_order(real["anchors"])
        with ResidualCapture(loaded) as cap:
            forward_once(loaded, real["text"])
            resid = cap.residuals()                      # [L][1, T, d]
        pos = resolve_anchor_positions(
            real["text"], real["anchors"], tok, expected_names=names)
        idx = [p.token_index for p in pos]
        for li in range(L):
            H = resid[li][0].float()                     # [T, d]
            for t in idx:
                v = H[t]
                n = torch.linalg.vector_norm(v).item()
                per_layer[li].append(n)
                mx, arg = torch.max(v.abs(), dim=0)
                mxv = mx.item()
                if mxv > per_layer_max_coord[li]:
                    per_layer_max_coord[li] = mxv
                sq = float(v.pow(2).sum())
                if sq > 0:
                    per_layer_top_share[li].append(float(mx.item() ** 2) / sq)
                d = int(arg.item())
                top_dim_counter[li][d] = top_dim_counter[li].get(d, 0) + 1

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    out = []
    print(f"{'layer':>5} {'med_norm':>10} {'max_norm':>11} {'max/med':>9} "
          f"{'max_coord':>11} {'top_dim_share':>14} {'dominant_dim':>13}")
    for li in range(L):
        ns = per_layer[li]
        m, mx = med(ns), max(ns) if ns else float("nan")
        ratio = mx / m if m else float("nan")
        share = med(per_layer_top_share[li])
        dom = max(top_dim_counter[li].items(), key=lambda kv: kv[1]) if top_dim_counter[li] else (None, 0)
        flag = "  <-- MASSIVE" if ratio > 10 else ""
        print(f"{li:5d} {m:10.2f} {mx:11.2f} {ratio:9.2f} "
              f"{per_layer_max_coord[li]:11.2f} {share:14.4f} "
              f"{str(dom[0]):>7}({dom[1]:3d}){flag}")
        out.append(dict(layer=li, median_norm=m, max_norm=mx, max_over_median=ratio,
                        max_abs_coord=per_layer_max_coord[li],
                        median_top_dim_share=share,
                        dominant_dim=dom[0], dominant_dim_count=dom[1],
                        n_states=len(ns)))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"condition": a.condition, "n_problems": len(recs), "layers": out},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
