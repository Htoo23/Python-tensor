import sys, json, numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'artifacts' / 'chatbot'
BOS = '<bos>'; EOS = '<eos>'

def must(p: Path):
    if not p.exists(): raise FileNotFoundError(f"Missing: {p}")

def load_vectorizer(prefix: str):
    vocab_path = OUT_DIR / f'{prefix}_vocab.txt'
    cfg_path   = OUT_DIR / f'{prefix}_config.json'
    must(vocab_path); must(cfg_path)
    vocab = vocab_path.read_text(encoding='utf-8').splitlines()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    vec = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        standardize=cfg["standardize"],
        output_mode='int',
        output_sequence_length=int(cfg["seq_len"]),
        split=cfg["split"],
    )
    vec.set_vocabulary(vocab)
    return vec

def load_artifacts():
    model_path = OUT_DIR / 'seq2seq.keras'
    must(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, load_vectorizer("src"), load_vectorizer("tgt")

def greedy_decode(model, src_vec, tgt_vec, text, max_len=None):
    vocab = tgt_vec.get_vocabulary()
    bos_id = vocab.index(BOS) if BOS in vocab else 1
    eos_id = vocab.index(EOS) if EOS in vocab else 2

    enc = tf.cast(src_vec(tf.constant([text])), tf.int32)
    target_len = int(model.inputs[1].shape[1])
    if max_len is None: max_len = target_len - 1

    dec = tf.fill([1, 1], tf.constant(bos_id, dtype=tf.int32))
    tokens = []
    for _ in range(max_len):
        pad_len = target_len - int(dec.shape[1])
        dec_padded = tf.pad(dec, [[0,0],[0,pad_len]])
        logits = model([enc, dec_padded], training=False)
        step = int(dec.shape[1]-1)
        next_id = int(tf.argmax(logits[0, step, :]))
        if next_id == eos_id: break
        tokens.append(vocab[next_id] if next_id < len(vocab) else "")
        dec = tf.concat([dec, [[next_id]]], axis=1)
    return " ".join(tokens).strip()

if __name__ == '__main__':
    model, sv, tv = load_artifacts()
    text = "မင်္ဂလာပါ" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", text)
    print("Bot:", greedy_decode(model, sv, tv, text) or "…")
