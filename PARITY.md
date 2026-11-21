# Harbor parity runner

This branch (`harbor-parity`) is upstream commit `86df98bf` plus an API-model inference script. It is
used for the parity experiments of the [Harbor SecureReviewer adapter](https://github.com/harbor-framework/adapters/pull/97).

- `infer/test_api.py` reviews selected items of `dataset/test.jsonl` with one Anthropic Messages API
  call each (temperature 0, `max_tokens` 2000). It parses the reply with upstream's
  `parse_security_review` and writes `result.jsonl` in the format `evaluate/SecureBleu.py` reads.
- `infer/single_call.py` is the API call, identical to the one the Harbor-side agent uses.
- `parity_indices.txt` holds the 25 parity items (0-based lines of `dataset/test.jsonl`), the same tasks
  as `uv run securereviewer --split parity`.

Nothing upstream is modified.

```bash
export ANTHROPIC_API_KEY=... ANTHROPIC_BASE_URL=...   # e.g. https://app.ppapi.ai/
pip install nltk==3.9.1 && python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger_eng
for r in 1 2; do
  python infer/test_api.py --model claude-haiku-4-5 --out runs/run$r --indices $(cat parity_indices.txt)
  (cd runs/run$r && python ../../evaluate/SecureBleu.py | tail -2)
done
```

Results (claude-haiku-4-5, 2026-11-20): `Average BLEU score` 25.96 and 26.25 (Harbor: 26.21 and 26.49).
