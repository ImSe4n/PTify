"""PTify's rhythm model (Phase 33 C): notes as played -> rhythm as written.

Pure numpy here (`features`); torch only in `model` and `train`, so the
feature code runs anywhere the rest of PTify does. Design and its reasoning:
`docs/phase33-rhythm-model.md`.
"""
