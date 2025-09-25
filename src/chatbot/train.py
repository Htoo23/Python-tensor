import tensorflow as tf
from pathlib import Path
from src.common import utils
from src.common.data import make_text_vectorizer
from src.chatbot.model import build_seq2seq

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / 'data' / 'chatbot' / 'pairs.tsv'
OUT_DIR = ROOT / 'artifacts' / 'chatbot'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOS = '<bos>'
EOS = '<eos>'
PAD_ID = 0

def load_pairs(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            src, tgt = line.split('\t')
            rows.append((src, f"{BOS} {tgt} {EOS}"))
    return rows

def save_vectorizer(vec: tf.keras.layers.TextVectorization, seq_len: int, prefix: str):
    import json
    vocab = vec.get_vocabulary()
    (OUT_DIR / f'{prefix}_vocab.txt').write_text("\n".join(vocab), encoding='utf-8')
    cfg = {"seq_len": int(seq_len), "standardize": None, "split": "whitespace"}
    (OUT_DIR / f'{prefix}_config.json').write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

def main():
    utils.set_seeds(42)
    pairs = load_pairs(DATA_FILE)
    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]

    src_vec = make_text_vectorizer(src_texts, vocab_size=6000, seq_len=32, standardize=None)
    tgt_vec = make_text_vectorizer(tgt_texts, vocab_size=6000, seq_len=32, standardize=None)

    X_enc = src_vec(tf.constant(src_texts))
    Y_full = tgt_vec(tf.constant(tgt_texts))

    X_enc  = tf.cast(X_enc,  tf.int32)
    Y_full = tf.cast(Y_full, tf.int32)
    pad_id = tf.constant(PAD_ID, dtype=tf.int32)

    Y_in = tf.concat([tf.fill([tf.shape(Y_full)[0], 1], pad_id), Y_full[:, :-1]], axis=1)

    model = build_seq2seq(
        vocab_size_src=len(src_vec.get_vocabulary()),
        vocab_size_tgt=len(tgt_vec.get_vocabulary()),
        seq_len_src=int(X_enc.shape[1]),
        seq_len_tgt=int(Y_in.shape[1])
    )

    model.fit([X_enc, Y_in], Y_full, epochs=50, batch_size=8, verbose=2)

    model.save(str(OUT_DIR / 'seq2seq.keras'))
    save_vectorizer(src_vec, int(X_enc.shape[1]), "src")
    save_vectorizer(tgt_vec, int(Y_in.shape[1]), "tgt")
    print("[chatbot] saved artifacts to", OUT_DIR)

if __name__ == '__main__':
    main()
