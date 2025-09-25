# Evaluation Protocol

## Fake News
- Compute Accuracy, Precision, Recall, F1 (macro) on `test.csv`.
- Provide confusion matrix and classification report.
- Threshold default 0.5; can be tuned on `dev.csv`.

## Chatbot
- Automatic: BLEU on held-out pairs (optional for small data).
- Human: 5-point Likert on Relevance, Fluency, Helpfulness, Safety.
- Safety filters: deny medical/financial decisions; encourage source-checking.
