import os, json, numpy as np, tensorflow as tf

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

def load():
    model = tf.keras.models.load_model(os.path.join(OUT_DIR, 'final_model.keras'))
    vec = load_vectorizer()
    return model, vec

def predict(text: str):
    model, vec = load()
    x = vec(tf.constant([text]))
    prob = model.predict(x, verbose=0)[0]
    label = int(np.argmax(prob))
    return label, float(prob[label])

if __name__ == '__main__':
    lbl, conf = predict("ကြက်ဆုံးရည် သောက်ရင် Covid ကုသနိုင်တယ်")
    print("Prediction:", lbl, "Confidence:", conf)
