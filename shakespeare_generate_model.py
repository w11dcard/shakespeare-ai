import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Activation
from keras._tf_keras.keras.optimizers import RMSprop

# Download and preprocess the text
filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(filepath, "rb").read().decode(encoding="utf-8").lower()
text = text[300000:800000]

# Extract unique characters and create character mappings
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

# Generate sequences and their corresponding next characters
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])
    
# One-hot encode the sequences and next characters
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1
    
# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the trained model
model.save("textgenerator.model.keras")