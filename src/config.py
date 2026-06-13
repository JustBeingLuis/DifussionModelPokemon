import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

IMAGE_SIZE = 64       
BATCH_SIZE = 180  
EPOCHS = 2000          
LEARNING_RATE = 1e-4   

TIMESTEPS = 100     
