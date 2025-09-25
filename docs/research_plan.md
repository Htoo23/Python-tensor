# Research Plan (Academic Rigor)

- **RQ1**: How well do lightweight neural models perform for Myanmar fake-news detection compared to classical baselines (SVM, LR)?
- **RQ2**: How robust is a small seq2seq chatbot trained on limited Myanmar data?
- **Datasets**: Start with toy data; replace with real Myanmar corpora (news, Facebook posts, OSCAR, CC100, custom annotations).
- **Metrics**:
  - Fake News: Accuracy, Precision/Recall/F1 (macro), ROC-AUC.
  - Chatbot: BLEU (automatic), plus human evaluation rubric (coherence, helpfulness, safety).
- **Methods**:
  - Tokenization via `TextVectorization` (baseline). Explore subword (SentencePiece) in ablations.
  - Models: BiLSTM baseline; try CNN/Transformer variants.
  - Regularization: dropout, early stopping, weight decay (via optimizer).
- **Reproducibility**: seed control, fixed splits, configuration logs.
- **Ethics**: See `docs/ethics.md` for bias/safety guidance.
