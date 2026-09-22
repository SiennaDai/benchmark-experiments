#!/usr/bin/env python3
"""Offline Gram/eigenspace mechanism analysis of selected Muon quantizers."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_muon_int3_practical_scale import (  # noqa: E402
    codebook_for_rank, discover, factorized_topk, quantize_scales,
)
from analyze_muon_vector_int3_residual import eligible_items  # noqa: E402
from optim.muon_conditioned_int3_companding import INT3_CODEBOOK  # noqa: E402
from optim.muon_gram_perturbation import (  # noqa: E402
    active_band_indices, gram_perturbation_metrics, spectral_gap_proxy,
    subspace_angle_metrics,
)
from optim.muon_int3_scale_selection import block_scales  # noqa: E402
from optim.muon_spectral_sensitivity import decompose, quantize as production_quantize  # noqa: E402
from optim.muon_structural_decomposition import exact_polar  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_vector_int3 import pair_values, quantize_vectors, unpair_values  # noqa: E402

OUT = ROOT / "reports/muon_gram_perturbation_hypothesis"
ROOT_SNAPSHOT = ROOT / "reports/muon_update_fidelity_formal_s0_s1_results"
PRACTICAL_CSV = ROOT / "reports/muon_int3_practical_scale/tensor_level_results.csv"
DIRECT_STRUCT_CSV = ROOT / "reports/muon_structural_decomposition_mechanism/baseline_vs_structural.csv"
VQ_DIR = ROOT / "reports/muon_vector_int3_robustness"
LANDMARKS = (128, 512, 1024, 2048, 4096)
METHODS = (
    "direct_scalar_int4",
    "structural_scalar_int3_p98_k8",
    "structural_int4_k8",
    "structural_vq64_int3_k8",
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def optional_float(value):
    """Parse report values whose optional metrics are stored as empty cells."""
    return None if value is None or str(value).strip() == "" else float(value)


def key_of(row):
    seed = row.get("seed", row.get("evaluation_seed"))
    return int(seed), int(row["update"]), row.get("parameter_id", row.get("parameter_name"))


def cached_metric_indexes():
    practical = read_csv(PRACTICAL_CSV)
    direct_struct = read_csv(DIRECT_STRUCT_CSV)
    vq_forward = read_csv(VQ_DIR / "forward_split.csv")
    vq_reverse = read_csv(VQ_DIR / "reverse_split.csv")
    cache = {}
    for r in direct_struct:
        if r.get("kind") == "direct":
            cache[(key_of(r), "direct_scalar_int4")] = {
                "update_cosine": float(r["update_cosine"]),
                "update_relative_l2": float(r["update_relative_l2"]),
                "update_norm_ratio": float(r["update_norm_ratio"]),
                "exact_polar_cosine_cached": optional_float(r["exact_polar_cosine"]),
                "exact_polar_relative_l2_cached": optional_float(r["exact_polar_relative_l2"]),
                "raw_relative_l2_cached":float(r["raw_relative_l2"]),
                "source_report": "muon_structural_decomposition_mechanism/baseline_vs_structural.csv:kind=direct",
            }
    for r in practical:
        if int(r["k"]) != 8:
            continue
        if r["method"] == "percentile:p=98" and r["codebook_family"] == "global_lloyd_max":
            method = "structural_scalar_int3_p98_k8"
        elif r["method"] == "structural_int4_reference":
            method = "structural_int4_k8"
        else:
            continue
        cache[(key_of(r), method)] = {
            "update_cosine": float(r["update_cosine"]),
            "update_relative_l2": float(r["update_relative_l2"]),
            "update_norm_ratio": float(r["update_norm_ratio"]),
            "exact_polar_cosine_cached": optional_float(r["exact_polar_cosine"]),
            "exact_polar_relative_l2_cached": optional_float(r["exact_polar_relative_l2"]),
            "raw_relative_l2_cached":float(r["full_state_raw_relative_l2"]),
            "source_report": "muon_int3_practical_scale/tensor_level_results.csv",
        }
    for rows in (vq_forward, vq_reverse):
        for r in rows:
            if int(r["rank"]) != 8 or int(r["codewords"]) != 64 or r["split_role"] != "held_out":
                continue
            cache[(key_of(r), "structural_vq64_int3_k8")] = {
                "update_cosine": float(r["update_cosine"]),
                "update_relative_l2": float(r["update_relative_l2"]),
                "update_norm_ratio": float(r["update_norm_ratio"]),
                "exact_polar_cosine_cached": (float(r["exact_polar_cosine"]) if r.get("exact_polar_cosine") else None),
                "exact_polar_relative_l2_cached": (float(r["exact_polar_relative_l2"]) if r.get("exact_polar_relative_l2") else None),
                "raw_relative_l2_cached":float(r["full_state_raw_relative_l2"]),
                "source_report": f"muon_vector_int3_robustness/{'forward_split.csv' if int(r['evaluation_seed']) == 1 else 'reverse_split.csv'}:held_out",
            }
    return cache


def load_snapshot_index(root: Path):
    found = {}
    for seed, update, path in discover(root):
        snap = load_snapshot(path)
        found[(int(seed), int(update))] = snap
    expected = {(s, u) for s in (0, 1) for u in LANDMARKS}
    if set(found) != expected:
        raise FileNotFoundError(f"expected ten formal snapshots, found {sorted(found)}")
    return found


def matrix_metrics(m, mh, reference_spectral_norm=None):
    e = mh - m
    nf = torch.linalg.matrix_norm(m, ord="fro")
    ne = torch.linalg.matrix_norm(e, ord="fro")
    cosine = (m * mh).sum() / (nf * torch.linalg.matrix_norm(mh, ord="fro")).clamp_min(1e-30)
    e_gram = e.T @ e
    spec_e = torch.linalg.eigvalsh((e_gram+e_gram.T)*.5).max().clamp_min(0).sqrt()
    spec_m = (torch.as_tensor(float(reference_spectral_norm), dtype=m.dtype)
              if reference_spectral_norm is not None else torch.linalg.matrix_norm(m, ord=2))
    return {"raw_relative_fro": float(ne / nf.clamp_min(1e-30)),
            "raw_cosine": float(cosine),
            "raw_relative_spectral": float(spec_e / spec_m.clamp_min(1e-30))}


def reconstruction_methods(m, svd, seed, codebooks):
    # All side-information conditions use the canonical BF16 factorized rank-8
    # top-k representation. The FP32 residual is formed from the exact SVD.
    u, s, vh = svd.u, svd.singular_values, svd.vh
    c_exact = (u[:, :8] * s[:8]) @ vh[:8]
    c_hat = factorized_topk(u, s, vh, 8)
    residual = m - c_exact
    out = {"direct_scalar_int4": production_quantize(m, "int4-dynamic-b2048").float()}
    cb_info = codebook_for_rank(8)
    p98_scales = block_scales(residual, "percentile", percentile=98.0, block_size=2048)
    scalar_residual = quantize_scales(residual, p98_scales, cb_info)
    out["structural_scalar_int3_p98_k8"] = c_hat + scalar_residual
    q4_residual = production_quantize(residual, "int4-dynamic-b2048").float()
    out["structural_int4_k8"] = c_hat + q4_residual
    calibration_seed = 1 - int(seed)
    key = f"s{calibration_seed}_k8_w64_t8_v1200"
    if key not in codebooks:
        raise KeyError(f"canonical held-out VQ codebook not found: {key}")
    pairs, singles, pair_index, single_index = pair_values(residual, "contiguous")
    q_pairs, scales, labels = quantize_vectors(pairs, codebooks[key], scale_method="p98", block_size=2048)
    q_residual = unpair_values(q_pairs, singles, tuple(residual.shape), pair_index, single_index)
    out["structural_vq64_int3_k8"] = c_hat + q_residual
    return out, {"residual": residual, "c_hat": c_hat, "p98_scales": p98_scales,
                 "vq_scales": scales, "vq_labels": labels, "vq_codebook_id": key,
                 "calibration_seed": calibration_seed}


def subspace_metrics(ref_u, ref_s, ref_v, q_u, q_s, q_v):
    bands = active_band_indices(ref_s)
    result = {"reference_active_rank": sum(1 for x in ref_s if float(x / ref_s[0]) >= 1e-6) if float(ref_s[0]) > 0 else 0,
              "compressed_active_rank": sum(1 for x in q_s if float(x / q_s[0]) >= 1e-6) if float(q_s[0]) > 0 else 0}
    n = ref_s.numel()
    all_idx = list(range(n))
    groups = {**bands, "top8": list(range(min(8, n)))}
    sv = ref_s.clamp_min(torch.finfo(ref_s.dtype).tiny)
    svq = q_s.clamp_min(torch.finfo(q_s.dtype).tiny)
    result["spectrum_relative_l2"] = float(torch.linalg.vector_norm(q_s - ref_s) / torch.linalg.vector_norm(ref_s).clamp_min(1e-30))
    result["spectrum_rank_normalized_rms"] = result["spectrum_relative_l2"] / math.sqrt(max(n, 1))
    result["spectrum_log10_rmse"] = float(torch.log10(svq).sub(torch.log10(sv)).square().mean().sqrt())
    for band in ("head", "middle", "tail"):
        idx = bands[band]
        if idx:
            ii = torch.tensor(idx, dtype=torch.long)
            result[f"{band}_singular_relative_l2"] = float(torch.linalg.vector_norm(q_s[ii] - ref_s[ii]) / torch.linalg.vector_norm(ref_s[ii]).clamp_min(1e-30))
            result[f"{band}_singular_log10_rmse"] = float(torch.log10(svq[ii]).sub(torch.log10(sv[ii])).square().mean().sqrt())
        else:
            result[f"{band}_singular_relative_l2"] = float("nan")
            result[f"{band}_singular_log10_rmse"] = float("nan")
    for name, idx in groups.items():
        left = subspace_angle_metrics(ref_u, q_u, idx)
        right = subspace_angle_metrics(ref_v, q_v, idx)
        for side, metrics in (("left", left), ("right", right)):
            for metric in ("mean_sin", "max_sin", "fro_sin"):
                result[f"{name}_{side}_{metric}"] = metrics[metric]
        result[f"{name}_subspace_mean_sin"] = statistics.mean((left["mean_sin"], right["mean_sin"])) if idx else float("nan")
    for band in ("head", "middle", "tail"):
        result[f"{band}_eigengap"] = spectral_gap_proxy(ref_s, bands[band])
    return result


def corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "spearman":
        def ranks(a):
            order = np.argsort(a, kind="mergesort"); out = np.empty(len(a), dtype=float)
            i = 0
            while i < len(order):
                j = i + 1
                while j < len(order) and a[order[j]] == a[order[i]]: j += 1
                out[order[i:j]] = (i + j - 1) / 2.0
                i = j
            return out
        x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def predictor_rows(records, group_fields, outcome_fields=("update_distortion", "exact_polar_distortion")):
    predictors = (
        "raw_relative_fro", "raw_relative_spectral",
        "right_gram_relative_fro", "right_gram_relative_spectral",
        "left_gram_relative_fro", "left_gram_relative_spectral",
        "right_gram_linear_relative_fro", "spectrum_relative_l2",
        "tail_subspace_mean_sin", "tail_gap_proxy",
    )
    keys = sorted({tuple(r.get(k) for k in group_fields) for r in records}, key=lambda x: tuple(str(z) for z in x))
    out = []
    for key in keys:
        rows = [r for r in records if tuple(r.get(k) for k in group_fields) == key]
        for outcome in outcome_fields:
            for pred in predictors:
                xs = [float(r[pred]) for r in rows if r.get(pred) not in (None, "")]
                ys = [float(r[outcome]) for r in rows if r.get(pred) not in (None, "") and r.get(outcome) not in (None, "")]
                n = min(len(xs), len(ys))
                if n:
                    out.append({**dict(zip(group_fields, key)), "outcome": outcome, "predictor": pred,
                                "sample_count": n, "pearson": corr(xs[:n], ys[:n]), "spearman": corr(xs[:n], ys[:n], "spearman")})
    return out


def linear_fit_r2(train, test, feature_names, target="update_distortion", log_features=False):
    def matrix(rows):
        x = np.asarray([[float(r[f]) for f in feature_names] for r in rows], dtype=np.float64)
        y = np.asarray([float(r[target]) for r in rows], dtype=np.float64)
        if log_features: x = np.log10(np.maximum(x, 1e-12))
        return x, y
    if len(train) < len(feature_names) + 3 or len(test) < 3:
        return float("nan")
    x, y = matrix(train); xt, yt = matrix(test)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    x = (x - mu) / sd; xt = (xt - mu) / sd
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = np.column_stack((np.ones(len(xt)), xt)) @ beta
    denom = ((yt - yt.mean()) ** 2).sum()
    return float(1 - ((yt - pred) ** 2).sum() / denom) if denom > 0 else float("nan")


def regression_rows(records):
    models = {
        "raw_loglinear": (["raw_relative_fro"], True),
        "gram_fro_loglinear": (["right_gram_relative_fro"], True),
        "gram_spectral_loglinear": (["right_gram_relative_spectral"], True),
        "gram_plus_tail_gap": (["right_gram_relative_spectral", "tail_gap_proxy"], True),
    }
    output = []
    methods = sorted({r["method"] for r in records})
    for model, (features, logs) in models.items():
        output.append({"validation":"pooled_in_sample", "held_out_group":"all", "model":model,
                       "features":";".join(features), "train_n":len(records), "test_n":len(records),
                       "r2":linear_fit_r2(records, records, features, log_features=logs)})
        for train_seed, test_seed in ((0,1),(1,0)):
            tr=[r for r in records if r["seed"]==train_seed]; te=[r for r in records if r["seed"]==test_seed]
            output.append({"validation":"held_out_seed", "held_out_group":test_seed, "model":model,
                           "features":";".join(features), "train_n":len(tr), "test_n":len(te),
                           "r2":linear_fit_r2(tr, te, features, log_features=logs)})
        for method in methods:
            tr=[r for r in records if r["method"]!=method]; te=[r for r in records if r["method"]==method]
            output.append({"validation":"leave_one_method_out", "held_out_group":method, "model":model,
                           "features":";".join(features), "train_n":len(tr), "test_n":len(te),
                           "r2":linear_fit_r2(tr, te, features, log_features=logs)})
    return output


def matched_pairs(records):
    by=defaultdict(list)
    for row in records: by[(row["seed"],row["update"],row["parameter_id"])].append(row)
    pairs=[]
    for key, rr in by.items():
        for a,b in itertools.combinations(rr,2):
            x,y=a["raw_relative_fro"],b["raw_relative_fro"]
            rel=abs(x-y)/max((x+y)*0.5,1e-12)
            if rel <= 0.05:
                ga,gb=a["right_gram_relative_fro"],b["right_gram_relative_fro"]
                da,db=a["update_distortion"],b["update_distortion"]
                pairs.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"shape":a["shape"],
                    "method_a":a["method"],"method_b":b["method"],"raw_relative_difference":rel,
                    "raw_error_a":x,"raw_error_b":y,"gram_error_a":ga,"gram_error_b":gb,
                    "update_distortion_a":da,"update_distortion_b":db,
                    "smaller_gram_method":a["method"] if ga<gb else b["method"] if gb<ga else "tie",
                    "smaller_update_distortion_method":a["method"] if da<db else b["method"] if db<da else "tie",
                    "gram_orders_update":(ga-gb)*(da-db)>0})
    return pairs


def save_plots(out, records, correlations, matched, paired_sv, paired_int4):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plots unavailable: {exc}", flush=True); return
    colors={m:c for m,c in zip(METHODS,("#555555","#d95f02","#1b9e77","#7570b3"))}
    def scatter(x,y,name,xlabel,ylabel):
        plt.figure(figsize=(6,4.5))
        for m in METHODS:
            rr=[r for r in records if r["method"]==m]
            plt.scatter([r[x] for r in rr],[r[y] for r in rr],s=8,alpha=.35,label=m.replace("_"," "),color=colors[m])
        plt.xlabel(xlabel);plt.ylabel(ylabel);plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/name,dpi=150);plt.close()
    scatter("raw_relative_fro","update_distortion","raw_error_vs_update.png","raw relative Frobenius error","K=5 update cosine error")
    scatter("right_gram_relative_fro","update_distortion","right_gram_fro_vs_update.png","right Gram relative Frobenius perturbation","K=5 update cosine error")
    scatter("right_gram_relative_spectral","update_distortion","right_gram_spectral_vs_update.png","right Gram relative spectral perturbation","K=5 update cosine error")
    scatter("tail_subspace_mean_sin","update_distortion","tail_subspace_vs_update.png","mean tail principal-angle sine","K=5 update cosine error")
    scatter("tail_gap_proxy","update_distortion","gap_proxy_vs_update.png","tail eigengap perturbation proxy","K=5 update cosine error")
    scatter("raw_relative_fro","right_gram_relative_fro","raw_vs_gram_colored.png","raw relative Frobenius error","right Gram relative Frobenius perturbation")
    for rows,title,path in ((paired_sv,"scalar INT3 vs vector INT3","scalar_vs_vq_gram.png"),(paired_int4,"structural INT4 vs vector INT3","structural_int4_vs_vector_int3.png")):
        if rows:
            plt.figure(figsize=(6,4.5));plt.scatter([r["gram_ratio_vq_over_reference"] for r in rows],[r["update_distortion_ratio_vq_over_reference"] for r in rows],s=9,alpha=.35)
            plt.axvline(1,color="black",lw=.7);plt.axhline(1,color="black",lw=.7);plt.xlabel("VQ/reference Gram-error ratio");plt.ylabel("VQ/reference update-distortion ratio");plt.title(title);plt.tight_layout();plt.savefig(out/path,dpi=150);plt.close()
    agg=defaultdict(list)
    for r in correlations:
        if r.get("outcome")=="update_distortion" and r.get("scope")=="global":agg[r["predictor"]].append(float(r["spearman"]))
    if agg:
        labels=list(agg); values=[statistics.mean(agg[k]) for k in labels];plt.figure(figsize=(9,4));plt.bar(range(len(labels)),values);plt.xticks(range(len(labels)),labels,rotation=45,ha="right",fontsize=7);plt.ylabel("Spearman vs update distortion");plt.tight_layout();plt.savefig(out/"predictor_spearman.png",dpi=150);plt.close()
    models=read_csv(out/"predictive_models.csv") if (out/"predictive_models.csv").exists() else []
    selected=[r for r in models if r["validation"]=="held_out_seed"]
    if selected:
        plt.figure(figsize=(7,4));names=[f"{r['model']}→s{r['held_out_group']}" for r in selected];vals=[float(r["r2"]) for r in selected];plt.bar(range(len(names)),vals);plt.xticks(range(len(names)),names,rotation=45,ha="right",fontsize=7);plt.ylabel("held-out seed R²");plt.tight_layout();plt.savefig(out/"heldout_predictive_r2.png",dpi=150);plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--summary-existing",action="store_true",help="rebuild summary from existing result CSVs without recomputing matrices")
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    if args.summary_existing:
        names=("raw_error_metrics","gram_metrics","spectrum_metrics","subspace_metrics","update_metrics")
        merged={}
        for name in names:
            for row in read_csv(OUT/f"{name}.csv"):
                key=(int(row["seed"]),int(row["update"]),row["parameter_id"],row["method"])
                merged.setdefault(key,{}).update(row)
        records=list(merged.values())
        for row in records:
            for key in ("seed","update"):
                row[key]=int(row[key])
            for key,value in list(row.items()):
                if key not in ("seed","update","parameter_id","parameter_name","shape","method","cached_update_source") and value not in (None,""):
                    try: row[key]=float(value)
                    except (ValueError,TypeError): pass
        correlations=[]
        for filename,scope in (("correlation_summary.csv","global"),("per_method_correlations.csv","per_method"),("per_shape_correlations.csv","per_shape"),("per_seed_correlations.csv","per_seed")):
            for row in read_csv(OUT/filename):
                row["scope"]=scope
                correlations.append(row)
        prediction=read_csv(OUT/"predictive_models.csv")
        for row in prediction:
            row["r2"]=float(row["r2"])
        elapsed=float((OUT/"runtime_seconds.txt").read_text().split()[0])
        write_summary(records,correlations,prediction,read_csv(OUT/"matched_raw_error_pairs.csv"),read_csv(OUT/"scalar_vs_vq.csv"),read_csv(OUT/"int4_vs_vector_int3.csv"),OUT,elapsed)
        print("summary rebuilt from existing CSVs",flush=True)
        return
    snapshots=load_snapshot_index(ROOT_SNAPSHOT)
    cache=cached_metric_indexes()
    direct_raw_cache={key_of(r):float(r["raw_relative_l2"]) for r in read_csv(DIRECT_STRUCT_CSV) if r.get("kind")=="direct"}
    cb_blob=torch.load(VQ_DIR/"calibration_codebooks.pt",map_location="cpu",weights_only=False)
    codebooks=cb_blob["codebooks"]
    manifest=[
        {"method":"direct_scalar_int4","description":"production blockwise-dynamic INT4 b2048 Q(M)","rank":0,"source":"production_quantize(int4-dynamic-b2048)"},
        {"method":"structural_scalar_int3_p98_k8","description":"BF16 top-8 factors + p98 block scale + global Lloyd-Max 7-level scalar residual INT3","rank":8,"source":"prior practical-scale canonical recipe"},
        {"method":"structural_int4_k8","description":"BF16 top-8 factors + production blockwise-dynamic INT4 b2048 residual","rank":8,"source":"prior practical-scale canonical recipe"},
        {"method":"structural_vq64_int3_k8","description":"BF16 top-8 factors + 64-word 2-D MSE VQ, p98/2048, held-out opposite-seed codebook","rank":8,"source":"prior robustness codebook seed 2026, held-out split"},
    ]
    write_csv(OUT/"method_manifest.csv",manifest)
    raw_rows=[]; gram_rows=[]; decomp_rows=[]; spectrum_rows=[]; subspace_rows=[]; update_rows=[]; records=[]
    total=0
    for seed_update,(seed,update) in enumerate(sorted(snapshots)):
        snapshot=snapshots[(seed,update)]
        for item in eligible_items(snapshot):
            m=item["tensor"].detach().cpu().float()
            if m.ndim != 2: continue
            name=str(item.get("name",item.get("parameter_id")))
            pid=str(item.get("parameter_id",name))
            key=((seed,update,pid))
            svd=decompose(m)
            reconstructions,extra=reconstruction_methods(m,svd,seed,codebooks)
            ref_u,ref_s,ref_v=svd.u,svd.singular_values,svd.vh.T
            bands=active_band_indices(ref_s)
            eigengaps={band:spectral_gap_proxy(ref_s,idx) for band,idx in bands.items()}
            for method,mhat in reconstructions.items():
                cache_key=(key,method)
                if cache_key not in cache:
                    raise KeyError(f"no canonical cached update row for {cache_key}")
                cm=cache[cache_key]
                raw=matrix_metrics(m,mhat,float(ref_s[0]))
                gram=gram_perturbation_metrics(m,mhat,float(ref_s[0]))
                if max(gram["right_gram_identity_residual"],gram["left_gram_identity_residual"]) > 2e-4:
                    raise AssertionError(f"Gram perturbation identity failed for {method} {key}")
                q_u,q_s,q_vh=torch.linalg.svd(mhat.float(),full_matrices=False)
                q_v=q_vh.T
                spec=subspace_metrics(ref_u,ref_s,ref_v,q_u,q_s,q_v)
                p_ref=ref_u @ svd.vh
                p_hat=q_u @ q_vh
                p_delta=p_hat-p_ref
                p_ref_norm=torch.linalg.matrix_norm(p_ref,ord="fro")
                p_hat_norm=torch.linalg.matrix_norm(p_hat,ord="fro")
                pmet={"raw_cosine":float((p_ref*p_hat).sum()/(p_ref_norm*p_hat_norm).clamp_min(1e-30)),
                      "raw_relative_fro":float(torch.linalg.matrix_norm(p_delta,ord="fro")/p_ref_norm.clamp_min(1e-30))}
                exact_cos=pmet["raw_cosine"]
                exact_rel=pmet["raw_relative_fro"]
                e=mhat-m
                g_right_spec=gram["right_gram_relative_spectral"]
                tail_delta=eigengaps["tail"]
                gap_proxy=g_right_spec * (float(ref_s[0])**2) / max(tail_delta,1e-30) if float(ref_s[0])>0 else float("nan")
                row={"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":str(tuple(m.shape)),"method":method,
                     **raw,**gram,**spec,"exact_polar_cosine":exact_cos,"exact_polar_relative_l2":exact_rel,
                     "tail_gap_proxy":gap_proxy,"tail_eigengap":tail_delta,"head_eigengap":eigengaps["head"],"middle_eigengap":eigengaps["middle"],
                     "calibration_seed":extra["calibration_seed"] if method=="structural_vq64_int3_k8" else None,
                     "vq_codebook_id":extra["vq_codebook_id"] if method=="structural_vq64_int3_k8" else None,
                     "cached_update_source":cm["source_report"]}
                row["update_cosine"]=cm["update_cosine"]
                row["update_relative_l2"]=cm["update_relative_l2"]
                row["update_norm_ratio"]=cm["update_norm_ratio"]
                row["update_distortion"]=1.0-cm["update_cosine"]
                row["exact_polar_distortion"]=1.0-exact_cos
                cached_polar=cm.get("exact_polar_cosine_cached")
                if cached_polar is not None and abs(exact_cos-cached_polar) > 2e-3:
                    raise AssertionError(f"exact-polar mismatch for {method} {key}: {exact_cos} vs {cached_polar}")
                raw_diff=abs(raw["raw_relative_fro"]-cm["raw_relative_l2_cached"])
                if raw_diff > 3e-4:
                    raise AssertionError(f"reconstruction mismatch {method} {key}: raw L2 delta {raw_diff:.3g}")
                row["canonical_raw_relative_l2"] = cm["raw_relative_l2_cached"]
                row["reconstructed_raw_l2_abs_diff"] = raw_diff
                if method=="direct_scalar_int4":
                    row["direct_report_raw_relative_l2"] = direct_raw_cache[key]
                records.append(row)
                raw_rows.append({k:row[k] for k in ("seed","update","parameter_id","parameter_name","shape","method","raw_relative_fro","raw_cosine","raw_relative_spectral")})
                gram_rows.append({k:v for k,v in row.items() if k in ("seed","update","parameter_id","parameter_name","shape","method") or "gram_" in k})
                spectrum_rows.append({k:v for k,v in row.items() if k in ("seed","update","parameter_id","parameter_name","shape","method","reference_active_rank","compressed_active_rank","spectrum_relative_l2","spectrum_rank_normalized_rms","spectrum_log10_rmse") or "singular_" in k or "eigengap" in k})
                subspace_rows.append({k:v for k,v in row.items() if k in ("seed","update","parameter_id","parameter_name","shape","method") or "sin" in k})
                update_rows.append({k:row[k] for k in ("seed","update","parameter_id","parameter_name","shape","method","update_cosine","update_relative_l2","update_norm_ratio","update_distortion","exact_polar_cosine","exact_polar_relative_l2","exact_polar_distortion","cached_update_source")})
                decomp_rows.append({"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":str(tuple(m.shape)),"method":method,
                    "right_gram_linear_relative_fro":gram["right_gram_linear_relative_fro"],"right_gram_quadratic_relative_fro":gram["right_gram_quadratic_relative_fro"],
                    "right_gram_quadratic_over_delta_fro":gram["right_gram_quadratic_over_delta_fro"],"left_gram_linear_relative_fro":gram["left_gram_linear_relative_fro"],
                    "left_gram_quadratic_relative_fro":gram["left_gram_quadratic_relative_fro"],"left_gram_quadratic_over_delta_fro":gram["left_gram_quadratic_over_delta_fro"]})
                total+=1
        print(f"processed seed={seed} update={update}; method-instance rows={total}",flush=True)

    # Primary coverage is expected to be 300 matrix snapshots x four methods.
    if total != 1200:
        raise AssertionError(f"expected 1200 method/tensor instances, got {total}")
    write_csv(OUT/"raw_error_metrics.csv",raw_rows)
    write_csv(OUT/"gram_metrics.csv",gram_rows)
    write_csv(OUT/"gram_decomposition.csv",decomp_rows)
    write_csv(OUT/"spectrum_metrics.csv",spectrum_rows)
    write_csv(OUT/"subspace_metrics.csv",subspace_rows)
    write_csv(OUT/"update_metrics.csv",update_rows)

    correlation=[]
    for scope,fields in (("global",[]),("per_method",["method"]),("per_shape",["shape"]),("per_seed",["seed"])):
        for r in predictor_rows(records,fields):
            r["scope"]=scope
            correlation.append(r)
    write_csv(OUT/"correlation_summary.csv",[r for r in correlation if r["scope"]=="global"])
    write_csv(OUT/"per_method_correlations.csv",[r for r in correlation if r["scope"]=="per_method"])
    write_csv(OUT/"per_shape_correlations.csv",[r for r in correlation if r["scope"]=="per_shape"])
    write_csv(OUT/"per_seed_correlations.csv",[r for r in correlation if r["scope"]=="per_seed"])
    matched=matched_pairs(records);write_csv(OUT/"matched_raw_error_pairs.csv",matched)

    bykey=defaultdict(dict)
    for r in records: bykey[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    scalar_vq=[];int4_vq=[]
    for key,mm in bykey.items():
        for left,right,target in (("structural_scalar_int3_p98_k8","structural_vq64_int3_k8",scalar_vq),("structural_int4_k8","structural_vq64_int3_k8",int4_vq)):
            a,b=mm[left],mm[right]
            target.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"shape":a["shape"],"reference_method":left,"vq_method":right,
                "raw_ratio_vq_over_reference":b["raw_relative_fro"]/max(a["raw_relative_fro"],1e-30),
                "gram_ratio_vq_over_reference":b["right_gram_relative_fro"]/max(a["right_gram_relative_fro"],1e-30),
                "gram_spectral_ratio_vq_over_reference":b["right_gram_relative_spectral"]/max(a["right_gram_relative_spectral"],1e-30),
                "tail_subspace_ratio_vq_over_reference":b["tail_subspace_mean_sin"]/max(a["tail_subspace_mean_sin"],1e-30),
                "update_distortion_ratio_vq_over_reference":b["update_distortion"]/max(a["update_distortion"],1e-30),
                "exact_polar_distortion_ratio_vq_over_reference":b["exact_polar_distortion"]/max(a["exact_polar_distortion"],1e-30),
                "raw_error_delta_vq_minus_reference":b["raw_relative_fro"]-a["raw_relative_fro"],
                "gram_error_delta_vq_minus_reference":b["right_gram_relative_fro"]-a["right_gram_relative_fro"],
                "update_distortion_delta_vq_minus_reference":b["update_distortion"]-a["update_distortion"]})
    write_csv(OUT/"scalar_vs_vq.csv",scalar_vq);write_csv(OUT/"int4_vs_vector_int3.csv",int4_vq)
    pred=regression_rows(records);write_csv(OUT/"predictive_models.csv",pred)
    save_plots(OUT,records,correlation,matched,scalar_vq,int4_vq)
    write_methodology(OUT)
    write_summary(records,correlation,pred,matched,scalar_vq,int4_vq,OUT,time.perf_counter()-start)
    elapsed=time.perf_counter()-start
    (OUT/"runtime_seconds.txt").write_text(f"{elapsed:.3f} seconds\n")
    print(f"completed {total} Gram/spectrum method instances in {elapsed:.1f}s",flush=True)


def write_methodology(out):
    text="""# Methodology

## Scope and selected representations

This CPU-only diagnostic uses the ten formal FP32 snapshots (seed 0/1, updates 128/512/1024/2048/4096), all 30 eligible 2-D Muon matrices per snapshot, and four canonical representations. Direct INT4 is production blockwise-dynamic b2048. Structural methods use the exact rank-8 SVD residual and BF16-factorized top-8 component. Scalar INT3 is the previously selected p98 block rule with the frozen global Lloyd-Max 7-level codebook; structural INT4 applies the existing production dynamic INT4 quantizer to the residual; 2-D VQ is the prior 64-word MSE codebook, contiguous pairing, p98 scaling in b2048 blocks, with calibration from the opposite seed. No codebook is fit on its evaluation seed. K=5 update metrics are joined from canonical reports on `(seed, update, parameter_id)`; reconstructed raw errors and all Gram/subspace diagnostics are recomputed from FP32 snapshots.

## Error and Gram metrics

For each reconstruction, `E=Mhat-M`. The right Gram perturbation is computed as `Mhat.T@Mhat - M.T@M` and independently checked against `M.T@E + E.T@M + E.T@E`; the left-Gram analogue is `Mhat@Mhat.T - M@M.T`. Frobenius/spectral relative perturbations divide by the corresponding reference Gram norm; Gram cosine is Frobenius alignment, and trace distortion is normalized absolute trace change. The linear and quadratic Gram terms are reported separately, with quadratic/Frobenius-total ratio. Spectral norms of symmetric Gram matrices use the largest absolute eigenvalue.

The active spectrum uses `sigma_i/sigma_max >= 1e-6`. Head and tail are the strongest and weakest 10% of active singular indices; middle is the centered 10%; these match the established prior convention. Singular values are paired in descending order. Top-8, head, middle, and tail left/right singular subspaces use principal-angle sines from singular values of basis overlaps. The gap proxy divides the absolute Gram perturbation spectral norm by the minimum external eigengap of the selected band in eigenvalues `sigma^2`. This is a bound-inspired diagnostic only; no theorem assumptions or equality are asserted.

## Update geometry and statistical comparisons

Production K=5 update cosine/relative-L2 are reused from prior matched-state reports, rather than recomputed with a reimplemented transform. Exact-polar readouts are recomputed from the same SVDs used for subspace diagnostics and compared to cached values when available. Update distortion is `1-update_cosine`; exact-polar distortion is analogously `1-exact_polar_cosine`.

Pearson and Spearman correlations are descriptive and reported pooled, by method, by shape, and by seed. Linear/log-linear regression compares raw-only, Gram-only, and Gram-plus-tail-gap features using seed holdout in both directions and leave-one-method-out validation. Features are standardized using training data only. No significance tests are performed; snapshot landmarks and method variants are dependent. Matched raw-error pairs are within the same `(seed, update, parameter)` and require symmetric relative raw-error difference <= 5%; their selection does not inspect Gram or update outcomes.

Scalar INT3 vs VQ and structural INT4 vs VQ are paired by exact matrix identity and report ratios below 1 as improvements for the numerator method. Group-wise Muon metrics are not included in the primary regression dataset; the previous report is used only as context: finer groups raised within-group quantization cosine while lowering `C_opt`, and a small layer-0 perturbation test found 23–27% off-group output distortion energy under full-matrix Muon.

## Numerical and scope limitations

The per-matrix spectral-gap proxy can be tiny or zero for clustered/repeated modes, making ratios large; these are descriptive instability indicators. Principal-angle comparisons pair equal index ranges even when singular values cluster, so projectors are preferable to individual vector cosine but band boundaries themselves may be unstable. The two seeds/five landmarks are not independent samples. Reports contain metric tables rather than saved reconstructions. No new quantizer, optimizer, conditioner, or training behavior is implemented.
"""
    (out/"methodology.md").write_text(text)


def write_summary(records,corrs,pred,matched,sv,vq,out,elapsed):
    def avg(rows,key): return statistics.mean(float(r[key]) for r in rows)
    def med(rows,key): return statistics.median(float(r[key]) for r in rows)
    lines=["# Does Gram / eigenspace perturbation explain Muon low-bit error?","",
        f"CPU-only analysis of {len(records)//4} formal matrix snapshots and {len(records)} selected method instances. Runtime about {elapsed/60:.1f} minutes. No training was run; canonical quantizers and production K=5 outputs were reused.","",
        "## Methods and coverage","",
        "Compared exactly four representations: direct production dynamic INT4; BF16 top-8 structural scalar INT3 with fixed p98 scaling and the previously selected global Lloyd-Max codebook; BF16 top-8 structural INT4 using unchanged production dynamic INT4 on the residual; and BF16 top-8 structural 64-word 2-D INT3 VQ with p98/2048 normalization and the opposite-seed held-out codebook. All 300 formal 2-D matrix snapshots (2 seeds × 5 landmarks × 30 matrices) are covered. Prior update metrics are joined by seed/update/parameter ID; raw/Gram/spectrum readouts are recalculated from detached reconstructions.","",
        "## Pooled association with K=5 update distortion","",
        "See `correlation_summary.csv` for Pearson and Spearman associations; repeated matrices across landmarks and method variants are correlated observations, so these are descriptive only.","",
        "| predictor | Pearson | Spearman | n |","|:--|--:|--:|--:|"]
    for p in ("raw_relative_fro","raw_relative_spectral","right_gram_relative_fro","right_gram_relative_spectral","right_gram_linear_relative_fro","spectrum_relative_l2","tail_subspace_mean_sin","tail_gap_proxy"):
        row=next((r for r in corrs if r.get("scope")=="global" and r["outcome"]=="update_distortion" and r["predictor"]==p),None)
        if row:lines.append(f"| {p} | {float(row['pearson']):.3f} | {float(row['spearman']):.3f} | {row['sample_count']} |")
    global_corr={r["predictor"]:r for r in corrs if r.get("scope")=="global" and r.get("outcome")=="update_distortion"}
    def get_r2(model, validation, group):
        item=next((x for x in pred if x["model"]==model and x["validation"]==validation and str(x["held_out_group"])==str(group)),None)
        return float(item["r2"]) if item else float("nan")
    matched_rate=sum((v is True) or (isinstance(v,str) and v.strip().lower()=="true") for v in (r["gram_orders_update"] for r in matched))/max(len(matched),1)
    lines += ["","## Result and hypothesis assessment","",
        f"**Conclusion: partial support for eigenspace/gap geometry, but not for unnormalized global Gram perturbation magnitude as a standalone fidelity metric.** Update-distortion Spearman was {float(global_corr['raw_relative_fro']['spearman']):.3f} for raw relative-Frobenius error, {float(global_corr['right_gram_relative_fro']['spearman']):.3f} for right-Gram relative-Frobenius error, {float(global_corr['right_gram_relative_spectral']['spearman']):.3f} for right-Gram relative-spectral error, {float(global_corr['tail_subspace_mean_sin']['spearman']):.3f} for tail principal-angle sine, and {float(global_corr['tail_gap_proxy']['spearman']):.3f} for the gap-normalized Gram proxy. The raw/Gram norm associations are inverse and weak-to-modest; subspace and gap-normalized associations are strongly positive.","",
        f"A one-feature log-linear raw-error model gave held-out-seed R² {get_r2('raw_loglinear','held_out_seed',1):.3f}/{get_r2('raw_loglinear','held_out_seed',0):.3f} (seed 1/0); the Gram-Frobenius model gave {get_r2('gram_fro_loglinear','held_out_seed',1):.3f}/{get_r2('gram_fro_loglinear','held_out_seed',0):.3f}; the Gram-spectral model gave {get_r2('gram_spectral_loglinear','held_out_seed',1):.3f}/{get_r2('gram_spectral_loglinear','held_out_seed',0):.3f}. Adding tail eigengap to Gram spectral perturbation raised held-out R² to {get_r2('gram_plus_tail_gap','held_out_seed',1):.3f}/{get_r2('gram_plus_tail_gap','held_out_seed',0):.3f}. That combined model's leave-one-method-out R² ranges from {min(float(x['r2']) for x in pred if x['model']=='gram_plus_tail_gap' and x['validation']=='leave_one_method_out'):.3f} to {max(float(x['r2']) for x in pred if x['model']=='gram_plus_tail_gap' and x['validation']=='leave_one_method_out'):.3f}. The added signal is consistent with spectral separation/eigenspace sensitivity, not Gram magnitude alone.","",
        f"Among {len(matched)} pairs matched within 5% raw relative error, smaller Gram error ordered smaller update distortion in {matched_rate:.1%} of pairs, close to chance. This matched-pair check gives no reliable extra ordering from global Gram-Frobenius magnitude.","",
        f"For structural scalar INT3 → 64-word vector INT3, median raw-error, Gram-Frobenius, tail-angle, and update-distortion ratios were {med(sv,'raw_ratio_vq_over_reference'):.3f}, {med(sv,'gram_ratio_vq_over_reference'):.3f}, {med(sv,'tail_subspace_ratio_vq_over_reference'):.3f}, and {med(sv,'update_distortion_ratio_vq_over_reference'):.3f}. VQ improves all four in most/every paired case, but update distortion improves substantially more than Gram magnitude; this supports geometry sensitivity without establishing ΔG norm as the mediator.","",
        f"Structural INT4 and vector INT3 had average update cosine {avg([r for r in records if r['method']=='structural_int4_k8'],'update_cosine'):.3f} vs {avg([r for r in records if r['method']=='structural_vq64_int3_k8'],'update_cosine'):.3f}, and exact-polar cosine {avg([r for r in records if r['method']=='structural_int4_k8'],'exact_polar_cosine'):.3f} vs {avg([r for r in records if r['method']=='structural_vq64_int3_k8'],'exact_polar_cosine'):.3f}. Yet the median vector-INT3/structural-INT4 Gram-Frobenius perturbation ratio was {med(vq,'gram_ratio_vq_over_reference'):.3f}, with tail-angle ratio {med(vq,'tail_subspace_ratio_vq_over_reference'):.3f}. Similar update fidelity despite different ΔG size argues against a one-dimensional Gram-norm explanation; similar subspace movement is compatible with an eigenspace account.","",
        "Held-out-seed and leave-one-method-out R² are in `predictive_models.csv`. These models are descriptive; repeated tensor landmarks are not independent, and the minimum-gap proxy is sensitive to clustered spectra.","",
        "## Matched reconstruction-error comparison","",
        f"Found {len(matched)} within-instance method pairs whose raw relative-Frobenius errors differ by at most 5%. `matched_raw_error_pairs.csv` gives each pair and whether smaller Gram error orders smaller update distortion. Matched subsets are selected by raw-error proximity, not update results.","",
        "## Structural scalar INT3 vs 2-D vector INT3","",
        f"Across {len(sv)} matched instances, median VQ/scalar ratios are: raw error {med(sv,'raw_ratio_vq_over_reference'):.3f}, right-Gram Frobenius error {med(sv,'gram_ratio_vq_over_reference'):.3f}, tail-subspace angle {med(sv,'tail_subspace_ratio_vq_over_reference'):.3f}, update distortion {med(sv,'update_distortion_ratio_vq_over_reference'):.3f}. A ratio below one favors VQ. Exact-polar metrics and per-instance deltas are in `scalar_vs_vq.csv`.","",
        "## Structural INT4 vs vector INT3","",
        f"Across {len(vq)} matched instances, median VQ/structural-INT4 ratios are: raw error {med(vq,'raw_ratio_vq_over_reference'):.3f}, right-Gram Frobenius error {med(vq,'gram_ratio_vq_over_reference'):.3f}, tail-subspace angle {med(vq,'tail_subspace_ratio_vq_over_reference'):.3f}, update distortion {med(vq,'update_distortion_ratio_vq_over_reference'):.3f}. These methods have similar update fidelity at this rank but different residual representations; see `int4_vs_vector_int3.csv`.","",
        "## Interpretation","",
        "Overall, the evidence supports a narrower statement: singular-subspace displacement and its spectral-gap context track update distortion much more consistently than scalar raw-error or unnormalized Gram-norm summaries, but the Gram-norm matched-pair test is near chance and ΔG size is not sufficient. This does not yet justify implementing a Gram-aware compression objective. Stop the standalone-Gram-magnitude explanation branch; any next mechanism test should be specifically about mode/subspace geometry and gaps, with grouped Muon retained only as prior context. The existing group-wise study is cited in `methodology.md` but not rerun or pooled.","",
        "Only four required representative methods were evaluated; no new quantizer or Gram-aware objective was implemented. The stored head/middle/tail definition is active singular modes (sigma/sigma_max ≥ 1e-6), with each named band occupying 10% of active rank (middle centered). Gap-normalized quantities are descriptive perturbation-theory proxies, not rigorous bounds.","",
        f"Total CPU wall time: {elapsed:.1f} seconds. See `methodology.md` and CSVs for full definitions and subgroup outputs.",""]
    (out/"summary.md").write_text("\n".join(lines))


if __name__=="__main__": main()
