from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/root/seed/combo4_cand004")
RUN_ROOT = ROOT / "outputs" / "multiseed_final" / "cand004"
OUT_DIR = ROOT / "summaries_multiseed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def weighted_rmse(group: pd.DataFrame) -> float:
    return float(((group["RMSE"] ** 2 * group["n_points"]).sum() / group["n_points"].sum()) ** 0.5)


rows: list[pd.DataFrame] = []
for metrics_path in sorted(RUN_ROOT.glob("seed_*/metrics_summary.csv")):
    seed = int(metrics_path.parent.name.split("_", 1)[1])
    frame = pd.read_csv(metrics_path)
    frame.insert(0, "seed", seed)
    rows.append(frame)

if not rows:
    raise SystemExit(f"No metrics_summary.csv files found under {RUN_ROOT}")

all_metrics = pd.concat(rows, ignore_index=True)
all_metrics.to_csv(OUT_DIR / "cand004_multiseed_metrics_long.csv", index=False)

summary_rows = []
for (seed, split, model), group in all_metrics.groupby(["seed", "split", "model"], sort=True):
    summary_rows.append(
        {
            "seed": seed,
            "split": split,
            "model": model,
            "mean_per_well_nrmse": group["NRMSE_range"].mean(),
            "weighted_rmse": weighted_rmse(group),
            "mean_nse": group["NSE"].mean(),
            "n_wells": group["well"].nunique(),
            "n_rows": len(group),
        }
    )

per_seed = pd.DataFrame(summary_rows).sort_values(["split", "seed", "mean_per_well_nrmse"])
per_seed.to_csv(OUT_DIR / "cand004_multiseed_model_summary_by_seed.csv", index=False)

agg = (
    per_seed.groupby(["split", "model"], sort=True)
    .agg(
        mean_nrmse=("mean_per_well_nrmse", "mean"),
        std_nrmse=("mean_per_well_nrmse", "std"),
        mean_weighted_rmse=("weighted_rmse", "mean"),
        std_weighted_rmse=("weighted_rmse", "std"),
        mean_nse=("mean_nse", "mean"),
        std_nse=("mean_nse", "std"),
        n_seeds=("seed", "nunique"),
    )
    .reset_index()
)
agg = agg.sort_values(["split", "mean_nrmse"])
agg.to_csv(OUT_DIR / "cand004_multiseed_model_summary.csv", index=False)

rank_rows = []
for (seed, split), group in per_seed.groupby(["seed", "split"], sort=True):
    ranked = group.sort_values("mean_per_well_nrmse").reset_index(drop=True)
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        rank_rows.append(
            {
                "seed": seed,
                "split": split,
                "model": row.model,
                "rank_by_nrmse": rank,
                "mean_per_well_nrmse": row.mean_per_well_nrmse,
            }
        )

ranks = pd.DataFrame(rank_rows)
ranks.to_csv(OUT_DIR / "cand004_multiseed_model_ranks.csv", index=False)
rank_summary = (
    ranks.groupby(["split", "model"], sort=True)
    .agg(
        mean_rank=("rank_by_nrmse", "mean"),
        best_count=("rank_by_nrmse", lambda s: int((s == 1).sum())),
        n_seeds=("seed", "nunique"),
    )
    .reset_index()
    .sort_values(["split", "mean_rank", "model"])
)
rank_summary.to_csv(OUT_DIR / "cand004_multiseed_rank_summary.csv", index=False)

print(f"wrote {OUT_DIR}")
print(agg.to_string(index=False))
