# Ethics & Safety

- **Misinformation**: Classifier is probabilistic. Always provide a *confidence* score and encourage source verification.
- **Bias**: Myanmar language varieties (Unicode vs Zawgyi) and dialects can bias performance. Prefer Unicode normalization; consider Zawgyi conversion tools.
- **Privacy**: Do not log personal data in plaintext. Mask PII in datasets.
- **Human-in-the-loop**: Use predictions as decision support, not final judgement.
