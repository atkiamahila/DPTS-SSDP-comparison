# Consolidated Job Summaries

Notes:
- Dataset and budget come from the explicit job-id mapping.
- Model and method come from the summary/path tag-style inference.
- Reported Accuracy and No Match are parsed from the matching runlog Evaluation Result line.
- Reported budget_tokens column is budget_tokens + oom_error.
- All Stop Reasons preserves every stop reason found in stop_reason_counts.

============================================================================================================================================
Dataset: math | Model: llama-3b
============================================================================================================================================

| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1..10] | No Match | budget_tokens | oom_error | All Stop Reasons | Job ID |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|
| 3000 | DPTS | 23.20% | 3,937 | [23.80, 24.40, 24.80, 24.80, 24.80, 24.80, 24.80, 24.80, 24.80, 24.80] | 353 | 458 | 0 | {budget_tokens:458, early_stop_t_star:40, frontier_empty:2} | 6957015 |
| 3000 | SSDP | 40.20% | 2,049 | [40.40, 41.00, 41.00, 41.00, 41.00, 41.00, 41.00, 41.00, 41.00, 41.00] | 156 | 137 | 0 | {budget_tokens:137, early_stop_t_star:4, frontier_empty:359} | 6957015 |
| 5000 | DPTS | 35.80% | 5,936 | [35.40, 37.40, 38.00, 38.20, 38.20, 38.20, 38.20, 38.20, 38.20, 38.20] | 239 | 407 | 0 | {budget_tokens:407, early_stop_t_star:91, frontier_empty:2} | 6960340 |
| 5000 | SSDP | 41.80% | 2,325 | [42.60, 43.20, 43.20, 43.20, 43.20, 43.40, 43.40, 43.40, 43.40, 43.40] | 103 | 67 | 0 | {budget_tokens:67, early_stop_t_star:7, frontier_empty:426} | 6960340 |
| 8000 | DPTS | 43.40% | 7,606 | [42.40, 45.80, 46.80, 47.00, 47.20, 47.80, 48.40, 48.40, 48.40, 48.40] | 169 | 335 | 0 | {budget_tokens:335, early_stop_t_star:164, frontier_empty:1} | 6961480 |
| 8000 | SSDP | 43.20% | 2,800 | [42.60, 43.20, 43.40, 43.60, 43.60, 43.60, 43.60, 43.60, 43.60, 43.60] | 80 | 62 | 0 | {budget_tokens:62, early_stop_t_star:11, frontier_empty:421, max_rollout:6} | 6961480 |
| 10000 | DPTS | 45.60% | 8,606 | [45.20, 49.60, 50.00, 50.40, 51.20, 51.40, 51.40, 51.40, 51.40, 51.40] | 144 | 297 | 0 | {budget_tokens:297, early_stop_t_star:201, frontier_empty:2} | 6962936 |
| 10000 | SSDP | 43.00% | 2,989 | [42.80, 43.20, 44.00, 44.00, 44.20, 44.20, 44.20, 44.20, 44.20, 44.20] | 91 | 36 | 0 | {budget_tokens:36, early_stop_t_star:11, frontier_empty:422, max_rollout:31} | 6962936 |

============================================================================================================================================
Dataset: math | Model: llama-8b
============================================================================================================================================

| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1..10] | No Match | budget_tokens | oom_error | All Stop Reasons | Job ID |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|
| 3000 | DPTS | 21.20% | 3,891 | [21.60, 22.20, 22.20, 22.40, 22.40, 22.40, 22.40, 22.40, 22.40, 22.40] | 370 | 467 | 10 | {budget_tokens:467, early_stop_t_star:33, oom_error:10} | 6957015 |
| 3000 | SSDP | 46.80% | 2,180 | [47.00, 47.00, 47.20, 47.40, 47.40, 47.40, 47.40, 47.40, 47.40, 47.40] | 163 | 169 | 0 | {budget_tokens:169, early_stop_t_star:5, frontier_empty:326} | 6957015 |
| 5000 | DPTS | 33.80% | 6,034 | [33.60, 35.60, 36.20, 36.40, 36.40, 36.80, 37.00, 37.00, 37.20, 37.20] | 267 | 426 | 0 | {budget_tokens:426, early_stop_t_star:73, frontier_empty:1} | 6960340 |
| 5000 | SSDP | 48.80% | 2,688 | [49.20, 49.40, 49.40, 49.40, 49.40, 49.40, 49.40, 49.40, 49.40, 49.40] | 119 | 109 | 0 | {budget_tokens:109, early_stop_t_star:9, frontier_empty:382} | 6960340 |
| 8000 | DPTS | 43.00% | 7,639 | [41.80, 44.80, 45.80, 46.60, 47.00, 47.40, 47.60, 47.60, 47.80, 47.80] | 189 | 340 | 0 | {budget_tokens:340, early_stop_t_star:159, frontier_empty:1} | 6961480 |
| 8000 | SSDP | 50.60% | 3,186 | [49.60, 50.60, 50.80, 50.80, 50.80, 50.80, 50.80, 50.80, 50.80, 50.80] | 91 | 82 | 14 | {budget_tokens:82, early_stop_t_star:16, frontier_empty:396, max_rollout:6, oom_error:14} | 6961480 |
| 10000 | DPTS | 47.20% | 8,539 | [45.80, 49.00, 50.00, 51.60, 52.20, 52.60, 53.00, 53.00, 53.00, 53.00] | 156 | 286 | 10 | {budget_tokens:286, early_stop_t_star:213, frontier_empty:1, oom_error:10} | 6962936 |
| 10000 | SSDP | 50.20% | 3,252 | [49.60, 50.40, 50.80, 50.80, 50.80, 51.00, 51.00, 51.00, 51.00, 51.00] | 90 | 46 | 15 | {budget_tokens:46, early_stop_t_star:15, frontier_empty:398, max_rollout:41, oom_error:15} | 6962936 |

============================================================================================================================================
Dataset: gsm | Model: llama-3b
============================================================================================================================================

| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1..10] | No Match | budget_tokens | oom_error | All Stop Reasons | Job ID |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|
| 3000 | DPTS | 28.35% | 3,149 | [31.54, 35.10, 36.24, 37.38, 37.91, 38.06, 38.29, 38.29, 38.44, 38.51] | 660 | 793 | 0 | {budget_tokens:793, early_stop_t_star:524, frontier_empty:2} | 6956560 |
| 3000 | SSDP | 23.73% | 1,079 | [25.32, 25.93, 25.93, 25.93, 25.93, 25.93, 25.93, 25.93, 25.93, 25.93] | 756 | 10 | 0 | {budget_tokens:10, early_stop_t_star:1, frontier_empty:1308} | 6956560 |
| 5000 | DPTS | 35.10% | 3,871 | [40.94, 44.88, 46.10, 47.38, 47.76, 47.92, 47.92, 48.07, 48.14, 48.14] | 489 | 426 | 0 | {budget_tokens:426, early_stop_t_star:892, frontier_empty:1} | 6958666 |
| 5000 | SSDP | 22.97% | 1,099 | [24.64, 25.63, 25.63, 25.63, 25.63, 25.63, 25.63, 25.63, 25.63, 25.63] | 760 | 9 | 0 | {budget_tokens:9, early_stop_t_star:3, frontier_empty:1307} | 6958666 |
| 8000 | DPTS | 37.60% | 4,276 | [43.21, 47.54, 49.13, 50.19, 50.87, 51.18, 51.63, 51.86, 51.93, 52.01] | 427 | 170 | 0 | {budget_tokens:170, early_stop_t_star:1148, frontier_empty:1} | 6957664 |
| 8000 | SSDP | 22.37% | 1,100 | [24.03, 24.79, 24.87, 24.87, 24.87, 24.87, 24.87, 24.87, 24.87, 24.87] | 747 | 5 | 0 | {budget_tokens:5, early_stop_t_star:2, frontier_empty:1312} | 6957664 |
| 10000 | DPTS | 39.80% | 4,386 | [42.84, 47.16, 49.89, 51.02, 51.78, 52.31, 52.54, 52.62, 52.92, 52.92] | 399 | 88 | 0 | {budget_tokens:88, early_stop_t_star:1230, frontier_empty:1} | 6958465 |
| 10000 | SSDP | 22.37% | 1,105 | [24.34, 25.32, 25.40, 25.40, 25.40, 25.40, 25.40, 25.40, 25.40, 25.40] | 769 | 2 | 0 | {budget_tokens:2, early_stop_t_star:2, frontier_empty:1312, max_rollout:3} | 6958465 |

============================================================================================================================================
Dataset: gsm | Model: llama-8b
============================================================================================================================================

| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1..10] | No Match | budget_tokens | oom_error | All Stop Reasons | Job ID |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|
| 3000 | DPTS | NA | 3,386 | [43.28, 47.87, 50.44, 51.17, 51.97, 52.13, 52.21, 52.29, 52.53, 52.61] | NA | 889 | 15 | {budget_tokens:889, early_stop_t_star:354, oom_error:15} | 6956560 |
| 3000 | DPTS | 43.75% | 3,382 | [43.82, 48.45, 50.95, 52.01, 52.69, 53.07, 53.15, 53.30, 53.37, 53.37] | 430 | 950 | 10 | {budget_tokens:950, early_stop_t_star:369, oom_error:10} | 6956717 |
| 3000 | SSDP | 56.56% | 1,188 | [56.63, 58.30, 58.45, 58.45, 58.45, 58.45, 58.45, 58.45, 58.45, 58.45] | 77 | 9 | 0 | {budget_tokens:9, early_stop_t_star:1, frontier_empty:1309} | 6956560 |
| 5000 | DPTS | 57.39% | 4,370 | [59.06, 67.48, 70.66, 72.71, 73.69, 74.00, 74.45, 74.60, 74.98, 75.28] | 105 | 576 | 0 | {budget_tokens:576, early_stop_t_star:743} | 6958666 |
| 5000 | SSDP | 57.09% | 1,197 | [57.54, 58.61, 58.76, 58.76, 58.76, 58.76, 58.76, 58.76, 58.76, 58.76] | 85 | 3 | 0 | {budget_tokens:3, early_stop_t_star:2, frontier_empty:1314} | 6958666 |
| 8000 | DPTS | NA | 4,886 | [59.15, 68.54, 72.69, 74.73, 77.07, 78.56, 79.11, 79.66, 79.81, 80.05] | NA | 214 | 0 | {budget_tokens:214, early_stop_t_star:1064} | 6957664 |
| 8000 | DPTS | 61.26% | 4,872 | [49.05, 54.59, 56.86, 58.38, 59.82, 60.42, 61.26, 61.79, 62.24, 62.47] | 28 | 218 | 0 | {budget_tokens:218, early_stop_t_star:1101} | 7100414 |
| 8000 | SSDP | 56.33% | 1,214 | [56.71, 58.38, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53] | 84 | 3 | 0 | {budget_tokens:3, early_stop_t_star:3, frontier_empty:1311, max_rollout:2} | 6957664 |
| 10000 | DPTS | NA | 5,090 | [60.59, 69.99, 75.27, 77.49, 79.31, 79.97, 80.63, 81.04, 81.12, 81.45] | NA | 110 | 0 | {budget_tokens:110, early_stop_t_star:1103} | 6958465 |
| 10000 | DPTS | 62.24% | 5,048 | [49.58, 55.72, 58.45, 59.89, 61.33, 62.62, 63.31, 63.61, 63.91, 63.99] | 21 | 113 | 0 | {budget_tokens:113, early_stop_t_star:1205, max_rollout:1} | 7098282 |
| 10000 | SSDP | 56.79% | 1,205 | [57.32, 58.38, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53, 58.53] | 84 | 0 | 0 | {budget_tokens:0, early_stop_t_star:3, frontier_empty:1313, max_rollout:3} | 6958465 |

============================================================================================================================================
OOM Summary
============================================================================================================================================
- dataset=gsm, model=llama-8b, budget=3000, method=DPTS, oom_error=15, job_id=6956560
- dataset=gsm, model=llama-8b, budget=3000, method=DPTS, oom_error=10, job_id=6956717
- dataset=math, model=llama-8b, budget=3000, method=DPTS, oom_error=10, job_id=6957015
- dataset=math, model=llama-8b, budget=8000, method=SSDP, oom_error=14, job_id=6961480
- dataset=math, model=llama-8b, budget=10000, method=DPTS, oom_error=10, job_id=6962936
- dataset=math, model=llama-8b, budget=10000, method=SSDP, oom_error=15, job_id=6962936
- dataset=unknown, model=llama-3b, budget=None, method=DPTS, oom_error=1, job_id=6955732
- dataset=unknown, model=llama-3b, budget=None, method=DPTS, oom_error=99, job_id=6956211
- dataset=unknown, model=llama-3b, budget=None, method=DPTS, oom_error=38, job_id=6956478
- dataset=unknown, model=llama-3b, budget=None, method=SSDP, oom_error=7, job_id=6955732
- dataset=unknown, model=llama-3b, budget=None, method=SSDP, oom_error=20, job_id=6956211
- dataset=unknown, model=llama-3b, budget=None, method=SSDP, oom_error=12, job_id=6956478
- dataset=unknown, model=llama-8b, budget=None, method=DPTS, oom_error=200, job_id=6950854
- dataset=unknown, model=llama-8b, budget=None, method=DPTS, oom_error=323, job_id=6955732
- dataset=unknown, model=llama-8b, budget=None, method=DPTS, oom_error=347, job_id=6956211
- dataset=unknown, model=llama-8b, budget=None, method=DPTS, oom_error=340, job_id=6956478
- dataset=unknown, model=llama-8b, budget=None, method=DPTS, oom_error=167, job_id=6957143
- dataset=unknown, model=llama-8b, budget=None, method=SSDP, oom_error=34, job_id=6950854
- dataset=unknown, model=llama-8b, budget=None, method=SSDP, oom_error=133, job_id=6955732
- dataset=unknown, model=llama-8b, budget=None, method=SSDP, oom_error=115, job_id=6956211
- dataset=unknown, model=llama-8b, budget=None, method=SSDP, oom_error=124, job_id=6956478
