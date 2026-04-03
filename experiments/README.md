# Experiments

This directory stores reproducible experiment records for this repository.

## Workflow

1. Keep a stable baseline commit or tag.
2. Create one branch per experiment, for example `exp/clip-topk-4`.
3. Save each experiment's config, command, metrics, and notes under `experiments/`.
4. Update `registry.yaml` so later experiments can reference earlier results.
5. Optionally mirror the summary into Notion using the same fields.

## Layout

```text
experiments/
  README.md
  registry.yaml
  templates/
    experiment_note.md
    notion_summary.md
  2026-04-02_baseline.md
  2026-04-03_example-experiment/
    config.yaml
    notes.md
    metrics.json
    stdout.log
```

## Required Fields Per Experiment

- Experiment id
- Name
- Date
- Branch
- Commit
- Baseline reference
- Command
- Config path or inline config
- Key metrics
- Summary
- Next step

## Notes

- Unless the user says otherwise, new experiments should start from the baseline setting and compare back to `exp-001`.
- Large artifacts should live outside git when needed; store their path here.
- Text logs can be saved here, but avoid committing huge raw outputs.
- If the working tree is dirty, record that explicitly in the notes.
