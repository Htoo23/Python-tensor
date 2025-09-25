from src.common.data import make_text_vectorizer

def test_text_vectorizer_basic():
    texts = ["မင်္ဂလာပါ", "မြန်မာစာ စမ်းကြည့်ပါ"]
    vec = make_text_vectorizer(texts, vocab_size=100, seq_len=8, standardize=None)
    V = vec.get_vocabulary()
    assert len(V) > 5
