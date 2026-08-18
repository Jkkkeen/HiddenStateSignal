# Synthetic Hidden-State Fixture

Generate small response-only raw and pooled examples from the repository root:

```bash
python examples/synthetic_hidden/make_fixture.py \
  --output examples/synthetic_hidden/generated
```

The generated directory is ignored by Git. It contains Qwen3 Base, MiMo Base,
MiMo SFT, a pre-pooled NPZ example, and ready-to-run YAML configurations.

Use it to verify installation before pointing an adapter at real hidden-state
artifacts.
