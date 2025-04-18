import tensorflow as tf
from utils.text_utils import generate_text
from config import Config

def load_model_and_tokenizer():
    """Load the trained model and create tokenizer"""
    model = tf.keras.models.load_model(Config.MODEL_NAME)
    
    # We need to recreate the tokenizer (in a real app, we'd save this)
    from utils.text_utils import preprocess_text
    text = preprocess_text()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([text])
    
    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer()
    
    print("\nShakespeare Text Generator")
    print("Enter 'quit' to exit\n")
    
    while True:
        seed = input("Enter seed text: ")
        if seed.lower() == 'quit':
            break
            
        generated = generate_text(seed, model, tokenizer)
        print("\nGenerated text:")
        print(generated + "\n")

if _name_ == "_main_":
    main()
    