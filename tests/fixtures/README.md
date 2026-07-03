# Test fixtures

All fixtures are **synthetic**. There is **no real PII** and **no explicit
imagery** anywhere in this suite:

* Face/document/scene images are generated in-memory by `conftest.py` (solid
  PNG/JPEG frames) — DeepFace and OpenAI are mocked at their service boundaries,
  so no real inference or network call happens.
* PII values used in validator tests (SSNs, cards, emails) are synthetic test
  numbers (e.g. the `4242 4242 4242 4242` test card).
* Moderation tests mock the provider's category scores — no explicit content is
  ever loaded.

Drop synthetic sample documents here if you want to exercise the extract/PII
paths against a live (dev) OpenAI key manually; keep them free of real PII.
