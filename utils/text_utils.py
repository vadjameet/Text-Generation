import numpy as np
import string
import requests
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config

def download_dataset():
    """Download the Shakespeare dataset"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(Config.DATA_PATH):
        print("Downloading dataset...")
        response = requests.get(Config.DATA_URL)
        with open(Config.DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
    else:
        print("Dataset already exists. Using cached version.")

def preprocess_text():
    """Load and preprocess the text data"""
    with open(Config.DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Basic cleaning
    translator = str.maketrans('', '', string.punctuation.replace('.', '').replace('!', '').replace('?', ''))
    text = text.translate(translator)
    
    return text

def create_sequences(text, tokenizer):
    """Create training sequences and labels"""
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    input_sequences = []
    output_words = []
    
    for i in range(Config.SEQUENCE_LENGTH, len(sequences)):
        seq = sequences[i-Config.SEQUENCE_LENGTH:i]
        input_sequences.append(seq)
        output_words.append(sequences[i])
    
    X = np.array(input_sequences)
    y = np.array(output_words)
    
    return X, y, total_words

def generate_text(seed_text, model, tokenizer, num_words=Config.GENERATION_LENGTH, temperature=Config.TEMPERATURE):
    """Generate new text from seed text"""
    output_text = seed_text
    
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        if len(token_list) < Config.SEQUENCE_LENGTH:
            token_list = pad_sequences([token_list], maxlen=Config.SEQUENCE_LENGTH, padding='pre')
        else:
            token_list = [token_list[-Config.SEQUENCE_LENGTH:]]
        
        predicted_probs = model.predict(np.array(token_list), verbose=0)[0]
        
        # Apply temperature for more varied outputs
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)
        
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        seed_text += " " + output_word
        output_text += " " + output_word
    
    return output_text