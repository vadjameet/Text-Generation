# Configuration constants
class Config:
    # Data
    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    DATA_PATH = "data/shakespeare.txt"
    
    # Model
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 128
    EPOCHS = 30
    EMBEDDING_DIM = 256
    LSTM_UNITS = 1024
    MODEL_NAME = "models/shakespeare_lstm.h5"
    
    # Generation
    GENERATION_LENGTH = 200
    TEMPERATURE = 0.7