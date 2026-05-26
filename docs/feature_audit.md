# Feature Audit Notes

Use this checklist when validating UKTA feature inputs for KoBERT-GRU scoring experiments.

## Input Checks

- Confirm the feature file matches the expected 294-dimensional schema.
- Verify feature column order before training.
- Check for missing values and constant columns.
- Record the feature extraction script revision.

## Model Checks

- Confirm the selected model variant expects UKTA features.
- Verify attention-enabled runs save attention weights with the prediction outputs.
- Compare runs with and without UKTA features on the same split.

## Reporting

When reporting a feature-based result, include the feature inventory version, split name, random seed, and metric script.
