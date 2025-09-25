import tensorflow as tf

def build_seq2seq(
    vocab_size_src=8000,
    vocab_size_tgt=8000,
    seq_len_src=32,
    seq_len_tgt=32,
    emb_dim=256,      
    enc_units=256,   
    dec_units=256,    
):
    """
    Simple, Keras-3 friendly seq2seq:
      - Encoder: Embedding + GRU -> final state
      - Decoder: Embedding + GRU(return_sequences=True) initialized with encoder state
      - Output: Dense over timesteps (3D)
    """

    enc_inputs = tf.keras.Input(shape=(seq_len_src,), dtype="int32", name="encoder_inputs")
    enc_emb = tf.keras.layers.Embedding(vocab_size_src, emb_dim, mask_zero=True, name="enc_emb")(enc_inputs)
    _, enc_state = tf.keras.layers.GRU(
        enc_units, return_sequences=False, return_state=True, name="enc_gru"
    )(enc_emb)

    dec_inputs = tf.keras.Input(shape=(seq_len_tgt,), dtype="int32", name="decoder_inputs")
    dec_emb = tf.keras.layers.Embedding(vocab_size_tgt, emb_dim, mask_zero=True, name="dec_emb")(dec_inputs)
    dec_outputs, _ = tf.keras.layers.GRU(
        dec_units, return_sequences=True, return_state=True, name="dec_gru"
    )(dec_emb, initial_state=enc_state)

    logits = tf.keras.layers.Dense(vocab_size_tgt, name="lm_head")(dec_outputs)  # (B, T_dec, V)

    model = tf.keras.Model([enc_inputs, dec_inputs], logits, name="seq2seq_gru")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
