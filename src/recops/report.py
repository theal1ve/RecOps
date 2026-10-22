"""Assemble artifacts/report.md from real run outputs (JSON files). No numbers are typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

from recops.config import Config


def _f(x, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_report(cfg: Config) -> Path:
    art = Path(cfg.paths.artifacts)
    meta = _load(Path(cfg.paths.data_processed) / "meta.json")
    sim = _load(art / "simulation.json")
    fin = _load(art / "final" / "results.json")
    lat = _load(art / "latency.json")
    prim = cfg.metrics.primary
    L: list[str] = ["# RecOps: results of this run", ""]
    L.append(
        "All numbers below are produced by the code in this run (profile settings in "
        "`configs/`). Offline metrics on a music dataset; see Limitations."
    )
    if meta:
        L += ["", "## Data", "", "```json", json.dumps(meta, indent=2), "```"]

    if fin:
        m = fin["metrics"]
        L += ["", "## Summary table (TEST window, discovery task: seen tracks excluded)", ""]
        L.append(
            f"Eval users: {fin['n_eval_users']}, target pairs: {fin['n_targets']}, "
            f"catalog: {fin['catalog_size']}, share of targets outside the catalog "
            f"(cold items, unreachable): {_f(fin['cold_item_target_share'])}"
        )
        L += [
            "",
            "| System | NDCG@10 | NDCG@20 | NDCG@50 | MAP@20 | Recall@20 | Coverage@20 | Entropy@20 | HeadShare@20 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, v in m.items():
            L.append(
                f"| {name} | {_f(v.get('ndcg@10'))} | {_f(v.get('ndcg@20'))} | {_f(v.get('ndcg@50'))} | "
                f"{_f(v.get('map@20'))} | {_f(v.get('recall@20'))} | {_f(v.get('coverage@20'))} | "
                f"{_f(v.get('entropy@20'))} | {_f(v.get('head_share@20'))} |"
            )
        cr = fin["candidate_recall"]
        L += [
            "",
            f"## Candidate stage: recall over candidate lists (N={cfg.candidates.per_source_k} per source)",
            "",
            "| Sources | Recall |",
            "|---|---|",
        ]
        for k, v in cr["recall"].items():
            L.append(f"| {k} | {_f(v)} |")
        L += ["", "Share of union hits found *only* by one source:", ""]
        for k, v in cr["unique_hit_share"].items():
            L.append(f"- {k}: {_f(v)}")
        cs = fin["cold_start_simulated"]
        L += [
            "",
            "## Cold start (simulated: history truncated to "
            f"{cs['history_positives']} positives, {cs['n_users']} users)",
            "",
            "| System | NDCG@20 | Recall@20 |",
            "|---|---|---|",
        ]
        for name in ("popularity_fallback", "als_on_tiny_history"):
            L.append(
                f"| {name} | {_f(cs[name].get('ndcg@20'))} | {_f(cs[name].get('recall@20'))} |"
            )
        ea = fin["error_analysis"]
        L += [
            "",
            "## Error analysis (best system, Recall@20)",
            "",
            "| Target popularity bucket | #targets | Recall@20 |",
            "|---|---|---|",
        ]
        for r in ea["by_target_popularity"]:
            L.append(f"| {r['bucket']} | {r['n_targets']} | {_f(r['recall@20'])} |")
        L += ["", "| User activity | #users | Recall@20 |", "|---|---|---|"]
        for r in ea["by_user_activity"]:
            L.append(f"| {r['tercile']} | {r['n_users']} | {_f(r['recall@20'])} |")
        fi = sorted(fin["feature_importance"].items(), key=lambda kv: -kv[1])[:10]
        L += ["", "## Top ranker features (CatBoost importance)", ""] + [
            f"- {k}: {v:.2f}" for k, v in fi
        ]

    if sim:
        L += [
            "",
            f"## Rolling simulation (primary metric: {prim})",
            "",
            "| Cycle | Drifted holdout | Corrupted candidate | Production | Candidate | Promoted | Alerts | Reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sim:
            alerts = "; ".join(r["monitoring"].get("alerts", [])) or "-"
            L.append(
                f"| {r['cycle']} | {r['drifted_holdout']} | {r['corrupted_candidate']} | "
                f"{_f(r['production'])} | {_f(r['candidate'])} | {r['promoted']} | {alerts} | {r['reason']} |"
            )
        L += ["", "PSI of monitored features/scores for the production model at each cycle:", ""]
        for r in sim:
            psi = r["monitoring"].get("psi")
            if psi:
                top = sorted(psi.items(), key=lambda kv: -kv[1])[:5]
                L.append(f"- cycle {r['cycle']}: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        L += [
            "",
            "Baselines on each cycle holdout (validation, candidate models):",
            "",
            "| Cycle | popularity | itemknn | als | rrf | ranker |",
            "|---|---|---|---|---|---|",
        ]
        for r in sim:
            s = r["systems_candidate"]
            L.append(
                f"| {r['cycle']} | "
                + " | ".join(
                    _f(s.get(n, {}).get(prim))
                    for n in ("popularity", "itemknn", "als", "rrf", "ranker")
                )
                + " |"
            )

    if lat:
        L += [
            "",
            "## API latency (in-process TestClient, single worker)",
            "",
            "```json",
            json.dumps(lat, indent=2),
            "```",
        ]

    L += [
        "",
        "## Limitations",
        "",
        "- Offline metrics are not an online effect; there was no A/B test.",
        "- The dataset is music (Yandex Music); results do not transfer to other domains.",
        "- Yambda-50M is itself a user subsample (10k users); this run additionally used profile settings shown above.",
        "- The drift is artificial (top-item permutation); the real-data PSI values above show natural shift only.",
        "- Cold start is simulated by truncating history; real Yambda test users are never cold.",
        "- Latency was measured in-process on the machine that ran the notebook, not on a production host.",
    ]
    out = art / "report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
