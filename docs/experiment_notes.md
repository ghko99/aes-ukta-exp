# Experiment Notes

This repository compares KoBERT, GRU, and UKTA feature variants for Korean essay scoring. Record the exact run shape for each comparison.

## Variant Record

For each run, note:

- `config["mode"]` value.
- Whether topic labels are enabled.
- Feature file revision and UKTA feature count.
- Train, validation, and test split versions.
- Random seed and hardware used for embedding generation and training.

## Recommended Sequence

1. Generate or verify KoBERT embeddings.
2. Confirm feature dimensions match the selected model variant.
3. Train the configured model.
4. Run `performance.py` on the matching prediction files.
5. Archive predictions, labels, and attention weights together.

Keeping the variant metadata next to the result tables prevents accidental comparisons across different preprocessing states.
