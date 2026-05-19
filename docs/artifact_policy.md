# Artifact Policy

Training and evaluation create large intermediate files. Keep repository history focused on source, configuration, and curated result summaries.

## Keep Out Of Git

- Generated embeddings in `emb/`.
- Model checkpoints such as `.pth` files.
- Prediction arrays such as `*_y_pred_*.npy` and `*_y_true_*.npy`.
- Attention weight dumps unless they are small curated examples.
- Temporary CSV exports used for one-off analysis.

## Keep With Reports

For a reported experiment, preserve the command, configuration, metric table, prediction export, and checkpoint identifier in the same external run folder.

## Sharing Results

When a result is promoted into documentation, include the model variant, prompt-label setting, split name, and metric calculation script so the number can be traced later.
