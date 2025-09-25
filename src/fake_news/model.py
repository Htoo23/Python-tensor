import tensorflow as tf

def build_bilstm_classifier(vocab_size=8000, seq_len=64, emb_dim=128, lstm_units=128, num_classes=2):
    inputs = tf.keras.Input(shape=(seq_len,), dtype='int32')
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
