import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
FN_DIR = ARTIFACTS / "fake_news"
CB_DIR = ARTIFACTS / "chatbot"


def must_exist(path: Path, msg: str):
    if not path.exists():
        raise FileNotFoundError(f"{msg} Missing: {path}")


def load_vectorizer_from_files(dir_path: Path, prefix: str):
    """
    Rebuild a TextVectorization layer from UTF-8 vocab + JSON config.
    """
    vocab_path = dir_path / f"{prefix}_vocab.txt"
    cfg_path = dir_path / f"{prefix}_config.json"
    must_exist(vocab_path, f"[load_vectorizer {prefix}]")
    must_exist(cfg_path, f"[load_vectorizer {prefix}]")

    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    vec = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        standardize=cfg["standardize"],
        output_mode="int",
        output_sequence_length=int(cfg["seq_len"]),
        split=cfg["split"],
    )
    vec.set_vocabulary(vocab)
    return vec


def load_fake_news_artifacts():
    model_path = FN_DIR / "final_model.keras"
    must_exist(model_path, "[fake_news] Model not found.")
    model = tf.keras.models.load_model(model_path)

    vec = load_vectorizer_from_files(FN_DIR, "vectorizer")
    return model, vec


def classify_fake_news(text: str, model, vec):
    x = vec(tf.constant([text]))
    x = tf.cast(x, tf.int32)
    prob = model.predict(x, verbose=0)[0]
    label = int(np.argmax(prob))
    return label, float(prob[label])


BOS = "<bos>"
EOS = "<eos>"


def load_chatbot_artifacts():
    model_path = CB_DIR / "seq2seq.keras"
    must_exist(model_path, "[chatbot] Model not found.")
    model = tf.keras.models.load_model(model_path, compile=False)

    src_vec = load_vectorizer_from_files(CB_DIR, "src")
    tgt_vec = load_vectorizer_from_files(CB_DIR, "tgt")
    return model, src_vec, tgt_vec


def greedy_decode(model, src_vec, tgt_vec, text: str, max_len=None):
    vocab = tgt_vec.get_vocabulary()
    bos_id = vocab.index(BOS) if BOS in vocab else 1
    eos_id = vocab.index(EOS) if EOS in vocab else 2

    enc = tf.cast(src_vec(tf.constant([text])), tf.int32)

    target_len = int(model.inputs[1].shape[1])
    if max_len is None:
        max_len = target_len - 1

    dec = tf.fill([1, 1], tf.constant(bos_id, dtype=tf.int32))
    tokens = []
    for _ in range(max_len):
        pad_len = target_len - int(dec.shape[1])
        dec_padded = tf.pad(dec, [[0, 0], [0, pad_len]])
        logits = model([enc, dec_padded], training=False)  # (1, T_dec, V)
        next_id = int(tf.argmax(logits[0, dec.shape[1] - 1, :]))
        if next_id == eos_id:
            break
        tokens.append(vocab[next_id] if next_id < len(vocab) else "")
        dec = tf.concat([dec, [[next_id]]], axis=1)
    return " ".join(tokens).strip()


st.set_page_config(page_title=" NLP: Chatbot + Fake News", layout="centered")
st.title(" Low-Resource  NLP Demo")

tab1, tab2 = st.tabs(["🤖 Chatbot", "📰 Fake News Detector"])

with tab1:
    st.subheader("Seq2seq Chatbot (Toy)")
    st.caption(
        "Train with `python -m src.chatbot.train` first. "
        "Artifacts expected in `artifacts/chatbot`."
    )

    user_text = st.text_input("Say something in Myanmar:", "မင်္ဂလာပါ")
    if st.button("Reply"):
        try:
            model, src_vec, tgt_vec = load_chatbot_artifacts()
            reply = greedy_decode(model, src_vec, tgt_vec, user_text)
            st.success("Bot: " + (reply if reply else "…"))
        except FileNotFoundError as e:
            st.error(
                f"{e}\n\n"
                "Please run:\n"
                "  • `python -m src.chatbot.train`\n"
                "and ensure the following files exist:\n"
                "  • artifacts/chatbot/seq2seq.keras\n"
                "  • artifacts/chatbot/src_vocab.txt\n"
                "  • artifacts/chatbot/src_config.json\n"
                "  • artifacts/chatbot/tgt_vocab.txt\n"
                "  • artifacts/chatbot/tgt_config.json"
            )
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("Fake News Detection")
    st.caption(
        "Train with `python -m src.fake_news.train` first. "
        "Artifacts expected in `artifacts/fake_news`."
    )

    text = st.text_area(
        "Enter Myanmar news text:",
        "ကြက်ဆုံးရည် သောက်လျှင် Covid ကုသနိုင်သည်ဟု ...",
        height=150,
    )

    if st.button("Classify"):
        try:
            model, vec = load_fake_news_artifacts()
            label, conf = classify_fake_news(text, model, vec)
            cls = "FAKE" if label == 1 else "REAL/LEGIT"
            st.info(f"Prediction: **{cls}**  |  Confidence: {conf:.3f}")
            st.caption("Demo only — always verify sources.")
        except FileNotFoundError as e:
            st.error(
                f"{e}\n\n"
                "Please run:\n"
                "  • `python -m src.fake_news.train`\n"
                "and ensure the following files exist:\n"
                "  • artifacts/fake_news/final_model.keras\n"
                "  • artifacts/fake_news/vectorizer_vocab.txt\n"
                "  • artifacts/fake_news/vectorizer_config.json"
            )
        except Exception as e:
            st.error(f"Error: {e}")
