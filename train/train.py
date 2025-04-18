import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.text_utils import download_dataset, preprocess_text, create_sequences
from config import Config

def build_model(total_words):
    """Build the LSTM model"""
    model = Sequential([
        Embedding(total_words, Config.EMBEDDING_DIM, input_length=Config.SEQUENCE_LENGTH),
        LSTM(Config.LSTM_UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(Config.LSTM_UNITS),
        Dropout(0.2),
        Dense(Config.LSTM_UNITS//2, activation='relu'),
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Setup
    download_dataset()
    text = preprocess_text()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    
    # Prepare data
    X, y, total_words = create_sequences(text, tokenizer)
    
    # Build model
    model = build_model(total_words)
    model.summary()
    
    # Callbacks
    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(Config.MODEL_NAME, save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        X, y,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_split=0.2,
        callbacks=callbacks
    )

if _name_ == "_main_":
    main()