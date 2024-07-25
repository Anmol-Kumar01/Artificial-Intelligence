#Author : Anmol Kumar 
# Date  : 25 July 2024 

import os
import logging
import warnings

# Set environment variables to suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
from transformers import pipeline

# Load sentiment analysis pipeline
# classifier = pipeline("sentiment-analysis")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Classify text
result = classifier("I love progamming but it is so tidious task")
print(result)

