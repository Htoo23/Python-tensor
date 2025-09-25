from typing import Tuple
import tensorflow as tf

def make_text_vectorizer(texts, vocab_size=8000, seq_len=64, standardize='lower_and_strip_punctuation'):
    tv = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=standardize,
        output_mode='int',
        output_sequence_length=seq_len,
        split='whitespace',  
    )
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(64)
    tv.adapt(ds)
    return tv

def dataset_from_df(df, vectorizer=None, label_col="label", batch_size=32, shuffle=True):
    texts = df["text"].astype(str).tolist()
    labels = df[label_col].values
    if vectorizer is None:
        vectorizer = make_text_vectorizer(texts)
    X = vectorizer(tf.constant(texts))
    y = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, vectorizer

def seq2seq_dataset_from_pairs(pairs, src_vec=None, tgt_vec=None, vocab_size=8000, src_len=32, tgt_len=32, bos_token="<bos>", eos_token="<eos>"):
    src_texts = [p[0] for p in pairs]
    tgt_texts = [f"{bos_token} {p[1]} {eos_token}" for p in pairs]

    if src_vec is None:
        src_vec = make_text_vectorizer(src_texts, vocab_size=vocab_size, seq_len=src_len, standardize=None)
    if tgt_vec is None:
        tgt_vec = make_text_vectorizer(tgt_texts, vocab_size=vocab_size, seq_len=tgt_len, standardize=None)

    X_enc = src_vec(tf.constant(src_texts))
    X_dec_in = tgt_vec(tf.constant([t[:-len(eos_token)-1] for t in tgt_texts]))  # rough; not used
    Y_dec = tgt_vec(tf.constant(tgt_texts))

    ds = tf.data.Dataset.from_tensor_slices((X_enc, Y_dec)).shuffle(len(pairs), seed=42).batch(16).prefetch(tf.data.AUTOTUNE)
    return ds, src_vec, tgt_vec
