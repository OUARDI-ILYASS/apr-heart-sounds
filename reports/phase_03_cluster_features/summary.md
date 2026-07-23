# Phase Summary — `03_cluster_features`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T22:13:47.339004+00:00  •  **Duration:** 173.9 s  
**Data splits read:** `train`  

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C1 | The Normal/Abnormal distinction is recoverable by unsupervised clustering of the feature spaces. | ❌ contradicted | best ARI vs diagnosis at k=2 is 0.028, close to chance; ARI vs recording site reaches 0.201 |

## Key Findings

- **mfcc_silhouette_k2:** 0.3958
- **mfcc_ari_class_k2:** 0.0181
- **mfcc_ari_site_k2:** 0.2011
- **mfcc_pca_95pct_components:** `51`
- **logmel_silhouette_k2:** 0.1924
- **logmel_ari_class_k2:** 0.0280
- **logmel_ari_site_k2:** 0.0426
- **logmel_pca_95pct_components:** `37`
- **pwp_silhouette_k2:** 0.3877
- **pwp_ari_class_k2:** 0.0021
- **pwp_ari_site_k2:** 0.1591
- **pwp_pca_95pct_components:** `31`

## Table: clustering_k2_summary

| silhouette | domain | elbow_k | ari_vs_class | pca_95pct | ari_vs_site |
|---|---|---|---|---|---|
| 0.3958 | mfcc | 3 | 0.0181 | 51 | 0.2011 |
| 0.1924 | logmel | 4 | 0.0280 | 37 | 0.0426 |
| 0.3877 | pwp | 4 | 0.0021 | 31 | 0.1591 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/clustering/mfcc_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/results/clustering/logmel_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/results/clustering/pwp_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/figures/fig_feature_projections.pdf | — | — | 1.3950 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy` (shape=[30695, 234])
- `/workspace/apr-heart-sounds/data/processed/logmel/features_train.npy` (shape=[30695, 32, 188])
- `/workspace/apr-heart-sounds/data/processed/pwp/features_train.npy` (shape=[30695, 84])

## Notes

- log-Mel maps were flattened and PCA-reduced to 50 dimensions before k-means, because k-means in 6016-dimensional space suffers from distance concentration.

## Timing Breakdown

- `_total`: 173.90 s
- `mfcc_kmeans`: 43.20 s
- `pwp_kmeans`: 26.68 s
- `mfcc_projection`: 24.55 s
- `pwp_projection`: 20.41 s
- `logmel_projection`: 19.44 s
- `logmel_kmeans`: 19.34 s
- `logmel_pca_reduce`: 6.18 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `04_train_classical`
