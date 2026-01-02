# Day 01 — Environment, Smoke Tests, and Parity Checks

Day 1 was very much a setup day and a day to understand how TabPFN inference
is usually done via code. 

Setting up the environment and project layout I learned a couple things and I 
switched back to basic venv environments, which I find better. 

Setting up the TabPFN accounts to allow it to download weights and do cloud 
inference was a bit annoying, but once it is done it works every time which 
is nice.

I am a bit surprised of how slow it actually is, locally I was expecting that
since I run tests on the CPU, but on the cloud it is also somewhat slow given
the tiny datasats I am testing with. I hope this is due to connection latency
and not to inference, otherwise big experiments could be difficult to run.

## Day 1 Checklist
- ✅ Install and run TabPFN locally; run tabular toy dataset.
- ✅ Set up cloud inference (TabPFN API client or equivalent).
- ✅ Build reproducible harness script: split → run model → log metrics + latency.
- ✅ Deliverables: smoke-test notebook + baseline runner + “setup/gotchas” doc.

## What was set up
- Python environment (Windows)
- Local TabPFN package
- Prior Labs cloud inference (token-based, non-interactive)
- Minimal evaluation function (single split, fixed seed)

## Smoke tests performed
- Basic local smoke test
- Basic cloud smoke test
- Binary classification (breast_cancer)
- Multi-class classification (wine)
- Regression (diabetes)

## Local vs cloud parity
- Predictions are semantically equivalent
- Metrics differ only at ~1e-3 level
- Differences are far below split-to-split variance
- Conclusion: local and cloud can be treated as equivalent backends

## Frozen decisions
- Use fixed train/test split with seed=42 for sanity tests
- Treat local runs as behavioral reference
- Use cloud runs for scale benchmarks later

## Open questions (intentionally deferred)
- How does this thing work?
- Does this thing actually do what they say it does?
- Will I be able to do big tests given how slow it is?
- How do I calibrate this thing?
- How will it perform when faced with noisy market data?