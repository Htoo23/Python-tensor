import os, pandas as pd, tensorflow as tf
from pathlib import Path
from src.common import utils
from src.common.data import dataset_from_df
from src.fake_news.model import build_bilstm_classifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'fake_news'
OUT_DIR = ROOT / 'artifacts' / 'fake_news'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[train] ROOT:", ROOT)
    print("[train] DATA_DIR:", DATA_DIR)
    print("[train] OUT_DIR:", OUT_DIR)
    utils.set_seeds(42)

    train_csv = DATA_DIR / 'train.csv'
    dev_csv = DATA_DIR / 'dev.csv'
    print("[train] Loading:", train_csv, dev_csv)
    train_df = utils.load_csv(str(train_csv))
    dev_df = utils.load_csv(str(dev_csv))
    print("[train] train/dev sizes:", len(train_df), len(dev_df))

    train_ds, vec = dataset_from_df(train_df, vectorizer=None, batch_size=16)
    dev_ds, _ = dataset_from_df(dev_df, vectorizer=vec, batch_size=16)
    vocab_size = len(vec.get_vocabulary())
    seq_len = train_ds.element_spec[0].shape[-1]
    print("[train] vocab_size:", vocab_size, "seq_len:", seq_len)

    model = build_bilstm_classifier(vocab_size=vocab_size, seq_len=seq_len)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(OUT_DIR / 'model.keras'), save_best_only=True, monitor='val_accuracy')
    ]
    print("[train] Starting fit...")
    history = model.fit(train_ds, validation_data=dev_ds, epochs=10, callbacks=cb, verbose=2)

    final_path = OUT_DIR / 'final_model.keras'
    model.save(str(final_path))
    print("[train] Saved model to:", final_path)

    import json
    vocab = vec.get_vocabulary()
    with open(OUT_DIR / 'vectorizer_vocab.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(vocab))
    vec_cfg = {"seq_len": int(seq_len), "standardize": "lower_and_strip_punctuation", "split": "whitespace"}
    with open(OUT_DIR / 'vectorizer_config.json', 'w', encoding='utf-8') as f:
        json.dump(vec_cfg, f, ensure_ascii=False, indent=2)
    print("[train] Wrote vectorizer files.")

    print('[train] Best val acc:', max(history.history['val_accuracy']))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print("[train] ERROR:", e)
        traceback.print_exc()
        raise
