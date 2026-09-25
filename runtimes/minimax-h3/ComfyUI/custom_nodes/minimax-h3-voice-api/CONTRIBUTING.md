# Contributing

1. Create a focused branch.
2. Install development dependencies with `pip install -e '.[dev]'`.
3. Run `pytest` and `ruff check minimax_voice_api tests`.
4. Do not commit checkpoints, generated media, voice samples, enrolled voices,
   API data, or personal `.env` settings.
5. Include a regression test for prompt construction, API contracts, or graph
   changes whenever practical.

Only submit voice-reference material when every affected person has consented to
its use and redistribution.
