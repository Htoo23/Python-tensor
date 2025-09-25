import os, random, numpy as np, tensorflow as tf

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_unicode(text: str) -> str:
    return text

def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df

def save_keras(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_keras(path):
    return tf.keras.models.load_model(path)
