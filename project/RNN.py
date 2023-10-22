import numpy as np
import pandas as pd
import keras
import json 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Reshape, Flatten
from keras.preprocessing.image import ImageDataGenerator

def save_history(history, filename='history.json'):
    history_dict = history.history
    history_dict_for_json = {k: [float(i) for i in v] for k, v in history_dict.items()}
    with open(filename, 'w') as f:
        json.dump(history_dict_for_json, f)

def load_from_csv(train_file, test_file):
    train_data = pd.read_csv(train_file).values
    test_data = pd.read_csv(test_file).values

    train_images = train_data[:, 1:].reshape(-1, 28, 28, 1).astype('float32') / 255
    train_labels = train_data[:, 0]

    test_images = test_data[:, 1:].reshape(-1, 28, 28, 1).astype('float32') / 255
    test_labels = test_data[:, 0]

    return (train_images, train_labels), (test_images, test_labels)

train_file = 'data_set/emnist-letters-train.csv'
test_file = 'data_set/emnist-letters-test.csv'
(train_images, train_labels), (test_images, test_labels) = load_from_csv(train_file, test_file)

unique_labels = np.unique(train_labels)
num_classes = len(unique_labels) + 1

train_labels = keras.utils.to_categorical(train_labels,num_classes=num_classes)
test_labels = keras.utils.to_categorical(test_labels,num_classes=num_classes)

# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(train_images)

# Model setup
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Reshape((-1, 28)))
model.add(SimpleRNN(128, activation='relu', return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for epoch in [8,32,128,512]:
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                        epochs=epoch, 
                        validation_data=(test_images, test_labels),
                        steps_per_epoch=train_images.shape[0] // 128)
    
    save_history(history, "./rnn_epochs_"+str(epoch)+"_history.json")
    model.save("./rnn_epochs_"+str(epoch)+".h5")
