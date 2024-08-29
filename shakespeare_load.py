import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model

# Download and preprocess the text
filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(filepath, "rb").read().decode(encoding="utf-8").lower()
text = text[300000:800000]  # Adjust text range as needed

# Define character mappings
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40

# Load the pre-trained model
model = load_model("textgenerator.model.keras")

# Define function for text generation
def generate_text(seed_text, length, temperature):
    generated_text = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(characters), p=preds)
        next_char = index_to_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text

# Example text generation with different temperature settings
seed_text = "shall i compare thee to a summer's day?\n"
print("----------Temperature: 0.2------------")
print(generate_text(seed_text, length=400, temperature=0.2))
print("----------Temperature: 0.4------------")
print(generate_text(seed_text, length=400, temperature=0.4))
print("----------Temperature: 0.6------------")
print(generate_text(seed_text, length=400, temperature=0.6))
print("----------Temperature: 0.8------------")
print(generate_text(seed_text, length=400, temperature=0.8))
print("----------Temperature: 1.0------------")
print(generate_text(seed_text, length=400, temperature=1.0))
