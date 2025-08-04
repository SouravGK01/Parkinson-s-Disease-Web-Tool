import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model

# --- Define the Transformer Encoder Block ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- Define the Hybrid Model Architecture ---
def build_hybrid_model(seq_input_shape, cnn_input_shape):
    # 1. Input and Branch for Sequential Data (e.g., MFCCs) -> Transformer
    seq_input = Input(shape=seq_input_shape, name="sequential_input")
    transformer_block = TransformerBlock(embed_dim=seq_input_shape[-1], num_heads=2, ff_dim=32)
    x1 = transformer_block(seq_input)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dropout(0.2)(x1)

    # 2. Input and Branch for Spectrogram/Waveform Data -> CNN
    cnn_input = Input(shape=cnn_input_shape, name="cnn_input")
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Conv1D(filters=64, kernel_size=3, activation='relu')(x2)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dropout(0.2)(x2)

    # 3. Fusion: Concatenate the outputs of both branches
    concatenated = Concatenate()([x1, x2])
    
    # 4. Final Classifier Head
    x = Dense(64, activation='relu')(concatenated)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid', name="output")(x)

    # Create the model with two inputs and one output
    model = Model(inputs=[seq_input, cnn_input], outputs=output)
    return model

# --- Example Usage ---
# Define example input shapes. Replace with your actual feature dimensions.
# (timesteps, features)
SEQ_SHAPE = (128, 13)  # e.g., 128 time steps of 13 MFCCs
CNN_SHAPE = (400, 1)   # e.g., a raw audio chunk of 400 samples

# Build the model
hybrid_model = build_hybrid_model(seq_input_shape=SEQ_SHAPE, cnn_input_shape=CNN_SHAPE)

# Compile the model
hybrid_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Print model summary to see the architecture
hybrid_model.summary()

# To train this model, you would need to prepare two sets of features:
# X_train_seq, X_train_cnn, y_train
# model.fit([X_train_seq, X_train_cnn], y_train, ...)
