import os, pandas as pd
from src.common import utils
from src.common.data import dataset_from_df
from src.fake_news.model import build_bilstm_classifier

def test_fake_news_forward():
    import pandas as pd
    df = pd.DataFrame({'text': ['မင်္ဂလာပါ', 'ကြက်ဆုံးရည် သောက်ပါ'], 'label':[0,1]})
    ds, vec = dataset_from_df(df, batch_size=2)
    vocab_size = len(vec.get_vocabulary())
    seq_len = ds.element_spec[0].shape[-1]
    model = build_bilstm_classifier(vocab_size=vocab_size, seq_len=seq_len)
    for x,y in ds.take(1):
        out = model(x)
        assert out.shape[-1] == 2
