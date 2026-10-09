# Zero-shot local-model benchmark (RQ4)

Standard-library runner. Gold labels are used only by `evaluate()`, never by `run()`.

```bash
python benchmark.py run --config config.example.json --out runs/trial
python benchmark.py evaluate --out runs/trial --bootstrap 2000
python audit_llm_results.py          # recompute Tables 11–13 from recorded runs
python -m unittest test_benchmark.py
```

Recorded comparison: Ollama 0.40.1, seven Q4_K_M snapshots, temperature 0, 32 completion tokens, three identical zero-shot repeats. See the repository root `README.md` and `AUDIT_REPORT_VI.md`.
