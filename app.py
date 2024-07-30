from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

print("TensorFlow version:", tf.__version__)

app = Flask(__name__)
CORS(app)

# Define max sequence length and vocab size
max_sequence_length = 300
max_vocab_size = 10000  # Adjust this according to your tokenizer settings

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and tokenizer
try:
    model_path = 'spam_detection_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    
    # Load the model with custom objects
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
    
    # Load the tokenizer
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    model = None
    tokenizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded correctly'}), 500
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text field provided'}), 400
    
    text = data['text']
    try:
        # Preprocess text
        preprocessed_text = text.lower()
        sequence = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        probability = prediction[0][0] * 100
        probability_formatted = "{:.2f}".format(probability)  # Format to 2 decimal places
        logging.info(f"Predicted probability: {probability_formatted}") 
        
        return jsonify({'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
