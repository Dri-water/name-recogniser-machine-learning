from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle


model = load_model("race_from_name")

race_mapping = {'Chinese': 0, 'Indian': 1, 'Malay': 2, 'Others': 3}

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_race(name):
    # Preprocess the input name
    preprocessed_name = name.lower().replace('[^\w\s]', '')
    name_sequence = tokenizer.texts_to_sequences([preprocessed_name])
    padded_sequence = pad_sequences(name_sequence, padding='post', maxlen=11)
    # Make a prediction
    prediction = model.predict(padded_sequence)
    predicted_race_label = np.argmax(prediction)

    # Map predicted label back to race category
    reverse_race_mapping = {v: k for k, v in race_mapping.items()}
    predicted_race = reverse_race_mapping[predicted_race_label]

    return predicted_race
# Test the function
input_name = input("Enter a name to predict the race: ")
predicted_race = predict_race(input_name)
print(f"Predicted Race: {predicted_race}")