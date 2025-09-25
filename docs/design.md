# System Design

## Components
- **Data layer**: `src/common/data.py` loads CSV/TSV and builds `tf.data.Dataset` via `TextVectorization`.
- **Fake News Model**: BiLSTM classifier with embeddings; trained with class weights if data is imbalanced.
- **Chatbot Model**: Seq2seq (GRU) with Bahdanau attention; greedy decoding for demo.
- **App**: Streamlit UI with two tabs (Chatbot, Fake News).

## Architecture Diagram (ASCII)

```
+---------------------+
|   Streamlit App     |
|  - Chatbot tab      |
|  - Fake News tab    |
+----------+----------+
           |
           v
+----------+----------+
|    Inference API    |
|  - load models      |
|  - preprocess       |
+-----+----------+----+
      |          |
      v          v
+-----+--+   +---+-----+
| Chatbot|   | FakeNews |
|  Seq2seq  |  BiLSTM   |
+-----+--+   +---+-----+
      |          |
      v          v
   TextVectorization
```

