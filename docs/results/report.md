# RecOps: results of this run

All numbers below are produced by the code in this run (profile settings in `configs/`). Offline metrics on a music dataset; see Limitations.

## Data

```json
{
  "n_events": 46467212,
  "n_likes": 881456,
  "n_users": 9238,
  "n_items": 877168,
  "max_day": 300,
  "day0_shift": 0,
  "synthetic": false,
  "subsampled_users": null,
  "last_days": null
}
```

## Summary table (TEST window, discovery task: seen tracks excluded)

Eval users: 5000, target pairs: 193327, catalog: 189708, share of targets outside the catalog (cold items, unreachable): 0.1517

| System | NDCG@10 | NDCG@20 | NDCG@50 | MAP@20 | Recall@20 | Coverage@20 | Entropy@20 | HeadShare@20 |
|---|---|---|---|---|---|---|---|---|
| ranker | 0.0671 | 0.0624 | 0.0652 | 0.0243 | 0.0416 | 0.0132 | 0.5153 | 0.6654 |
| popularity | 0.0323 | 0.0324 | 0.0328 | 0.0124 | 0.0250 | 0.0001 | 0.2548 | 0.9395 |
| itemknn | 0.0295 | 0.0304 | 0.0342 | 0.0111 | 0.0220 | 0.0206 | 0.5661 | 0.6114 |
| als | 0.0329 | 0.0333 | 0.0381 | 0.0107 | 0.0235 | 0.0145 | 0.6068 | 0.9110 |
| rrf | 0.0367 | 0.0367 | 0.0417 | 0.0122 | 0.0260 | 0.0119 | 0.4824 | 0.9019 |
| ranker_als_only | 0.0582 | 0.0532 | 0.0530 | 0.0200 | 0.0335 | 0.0136 | 0.5933 | 0.8721 |

## Candidate stage: recall over candidate lists (N=100 per source)

| Sources | Recall |
|---|---|
| pop | 0.0332 |
| knn | 0.0424 |
| als | 0.0513 |
| pop+knn | 0.0653 |
| pop+als | 0.0768 |
| knn+als | 0.0747 |
| pop+knn+als | 0.0958 |

Share of union hits found *only* by one source:

- pop: 0.2202
- knn: 0.1978
- als: 0.3185

## Cold start (simulated: history truncated to 3 positives, 500 users)

| System | NDCG@20 | Recall@20 |
|---|---|---|
| popularity_fallback | 0.0237 | 0.0237 |
| als_on_tiny_history | 0.0072 | 0.0058 |

## Error analysis (best system, Recall@20)

| Target popularity bucket | #targets | Recall@20 |
|---|---|---|
| unseen_in_history | 11256 | 0.0000 |
| head | 63614 | 0.0495 |
| torso | 98382 | 0.0125 |
| tail | 20075 | 0.0000 |

| User activity | #users | Recall@20 |
|---|---|---|
| low_activity | 1667 | 0.0579 |
| mid_activity | 1666 | 0.0409 |
| high_activity | 1667 | 0.0260 |

## Top ranker features (CatBoost importance)

- i_n_users: 10.21
- i_org_share: 9.22
- i_pos_28d: 9.17
- pop_rank: 6.17
- i_age_days: 5.92
- i_n_pos: 5.60
- n_sources: 4.79
- i_pos_7d: 4.68
- als_rank: 4.19
- i_n_listens: 3.84

## Rolling simulation (primary metric: ndcg@20)

| Cycle | Drifted holdout | Corrupted candidate | Production | Candidate | Promoted | Alerts | Reason |
|---|---|---|---|---|---|---|---|
| 0 | False | False | n/a | 0.0673 | True | - | bootstrap: no production model yet |
| 1 | False | False | 0.0580 | 0.0683 | True | QUALITY alert: production ndcg@20 0.0580 vs reference 0.0673 | candidate 0.0683 >= required 0.0586 (production 0.0580) |
| 2 | True | False | 0.0295 | 0.0378 | True | QUALITY alert: production ndcg@20 0.0295 vs reference 0.0683 | candidate 0.0378 >= required 0.0298 (production 0.0295) |
| 3 | True | False | 0.0403 | 0.0496 | True | PSI alert: i_trend=0.613 | candidate 0.0496 >= required 0.0407 (production 0.0403) |
| 4 | True | True | 0.0407 | 0.0113 | False | PSI alert: i_trend=0.449; QUALITY alert: production ndcg@20 0.0407 vs reference 0.0496 | candidate 0.0113 < required 0.0411 (production 0.0407) |

PSI of monitored features/scores for the production model at each cycle:

- cycle 1: i_trend=0.050, pop_score=0.044, score=0.008, i_pos_7d=0.007, u_n_pos_7d=0.003
- cycle 2: pop_score=0.075, i_trend=0.062, score=0.018, i_pos_7d=0.007, u_skip_rate=0.002
- cycle 3: i_trend=0.613, pop_score=0.146, score=0.110, i_pos_7d=0.077, als_score=0.002
- cycle 4: i_trend=0.449, pop_score=0.130, i_pos_7d=0.073, score=0.030, u_n_pos_7d=0.014

Baselines on each cycle holdout (validation, candidate models):

| Cycle | popularity | itemknn | als | rrf | ranker |
|---|---|---|---|---|---|
| 0 | 0.0395 | 0.0331 | 0.0351 | 0.0418 | 0.0673 |
| 1 | 0.0305 | 0.0323 | 0.0349 | 0.0406 | 0.0683 |
| 2 | 0.0161 | 0.0162 | 0.0165 | 0.0164 | 0.0378 |
| 3 | 0.0462 | 0.0170 | 0.0182 | 0.0308 | 0.0496 |
| 4 | 0.0379 | 0.0076 | 0.0077 | 0.0144 | 0.0113 |

## API latency (in-process TestClient, single worker)

```json
{
  "n_requests": 300,
  "k": 20,
  "p50_ms": 48.43720900021253,
  "p95_ms": 60.78345524995257,
  "p99_ms": 65.0221713202427,
  "mean_ms": 49.64098984333001,
  "throughput_rps_sequential": 20.142710728168808,
  "fallback_ms": 3.178983999987395,
  "fallback_source": "fallback_popularity",
  "note": "sequential in-process TestClient; single worker; CPU of the host that ran it"
}
```

## Limitations

- Offline metrics are not an online effect; there was no A/B test.
- The dataset is music (Yandex Music); results do not transfer to other domains.
- Yambda-50M is itself a user subsample (10k users); this run additionally used profile settings shown above.
- The drift is artificial (top-item permutation); the real-data PSI values above show natural shift only.
- Cold start is simulated by truncating history; real Yambda test users are never cold.
- Latency was measured in-process on the machine that ran the notebook, not on a production host.
