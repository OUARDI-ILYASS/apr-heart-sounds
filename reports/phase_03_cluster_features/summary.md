# Phase Summary — `03_cluster_features`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `8ae8dd4e`  •  **Seed:** `42`  
**Started:** 2026-07-21T19:30:47.210487+00:00  •  **Duration:** 122.31 s  
**Data splits read:** `train`  

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C1 | The Normal/Abnormal distinction is recoverable by unsupervised clustering of the feature spaces. | ❌ contradicted | best ARI vs diagnosis at k=2 is 0.025, close to chance; ARI vs recording site reaches 0.189 |

## Key Findings

- **mfcc_silhouette_k2:** 0.3867
- **mfcc_ari_class_k2:** 0.0061
- **mfcc_ari_site_k2:** 0.1885
- **mfcc_pca_95pct_components:** `51`
- **logmel_silhouette_k2:** 0.1915
- **logmel_ari_class_k2:** 0.0249
- **logmel_ari_site_k2:** 0.0418
- **logmel_pca_95pct_components:** `37`
- **pwp_silhouette_k2:** 0.3640
- **pwp_ari_class_k2:** -0.0084
- **pwp_ari_site_k2:** 0.1402
- **pwp_pca_95pct_components:** `32`

## Table: clustering_k2_summary

| ari_vs_site | pca_95pct | ari_vs_class | domain | silhouette | elbow_k |
|---|---|---|---|---|---|
| 0.1885 | 51 | 0.0061 | mfcc | 0.3867 | 4 |
| 0.0418 | 37 | 0.0249 | logmel | 0.1915 | 4 |
| 0.1402 | 32 | -0.0084 | pwp | 0.3640 | 4 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/clustering/mfcc_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/results/clustering/logmel_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/results/clustering/pwp_clustering.json | — | — | 0.0070 |
| /workspace/apr-heart-sounds/figures/fig_feature_projections.pdf | — | — | 1.1790 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy` (shape=[25365, 234])
- `/workspace/apr-heart-sounds/data/processed/logmel/features_train.npy` (shape=[25365, 32, 188])
- `/workspace/apr-heart-sounds/data/processed/pwp/features_train.npy` (shape=[25365, 84])

## Notes

- log-Mel maps were flattened and PCA-reduced to 50 dimensions before k-means, because k-means in 6016-dimensional space suffers from distance concentration.

## Timing Breakdown

- `_total`: 122.31 s
- `mfcc_kmeans`: 30.75 s
- `pwp_kmeans`: 22.55 s
- `logmel_kmeans`: 19.98 s
- `logmel_pca_reduce`: 12.14 s
- `mfcc_projection`: 8.66 s
- `pwp_projection`: 6.84 s
- `logmel_projection`: 6.60 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `04_train_classical`
