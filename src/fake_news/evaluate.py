import os, json
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.common import utils

DATA_DIR = 'data/fake_news'
OUT_DIR = 'artifacts/fake_news'

def load_vectorizer():
    with open(os.path.join(OUT_DIR, 'vectorizer_vocab.txt'), encoding='utf-8') as f:
        vocab = [line.rstrip('\n') for line in f]
    with open(os.path.join(OUT_DIR, 'vectorizer_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    vec = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        standardize=cfg["standardize"],
        output_mode='int',
        output_sequence_length=cfg["seq_len"],
        split=cfg["split"],
    )
    vec.set_vocabulary(vocab)
    return vec

def main():
    model = tf.keras.models.load_model(os.path.join(OUT_DIR, 'final_model.keras'))
    vec = load_vectorizer()
    test_df = utils.load_csv(os.path.join(DATA_DIR, 'test.csv'))
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df['label'].values

    X = vec(tf.constant(texts))
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
    print('\nClassification Report:\n', classification_report(y_true, y_pred, digits=3))

if __name__ == '__main__':
    main()
