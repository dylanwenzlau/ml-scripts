import json
import time
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import Sequential
import util


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = 'meme_text_gen'
MODEL_PATH = util.get_model_path(BASE_PATH, MODEL_NAME)
os.mkdir(MODEL_PATH)


SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 16
ROWS_TO_SCAN = 2000000
NUM_EPOCHS = 48
BATCH_SIZE = 256


print('loading json data...')
t = time.time()

training_data = json.load(open(BASE_PATH + '/training_data_sample.json'))

print('loading json took %ds' % round(time.time() - t))
util.print_memory()


print('scanning %d of %d json rows...' % (min(ROWS_TO_SCAN, len(training_data)), len(training_data)))
t = time.time()

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
label_id_counter = 0
for i, row in enumerate(training_data):
    template_id = str(row[0]).zfill(12)
    text = row[1].lower()
    start_index = len(template_id) + 2 + 1 + 2  # template_id, spaces, box_index, spaces
    box_index = 0
    for j in range(0, len(text)):
        char = text[j]
        # note: it is critical that the number of spaces plus len(box_index) is >= the convolution width
        texts.append(template_id + '  ' + str(box_index) + '  ' + text[0:j])
        if char in labels_index:
            label_id = labels_index[char]
        else:
            label_id = label_id_counter
            labels_index[char] = label_id
            label_id_counter += 1
        labels.append(label_id)
        if char == '|':
            box_index += 1

    if i >= ROWS_TO_SCAN:
        break


print('training text 0: %s' % texts[0])
print('training text 10: %s' % texts[10])
print('training text 1000: %s' % texts[1000])
print('scanning json took %ds' % round(time.time() - t))
util.print_memory()


print('tokenizing %d texts...' % len(texts))
del training_data  # free memory
t = time.time()

char_to_int = util.map_char_to_int(texts)
sequences = util.texts_to_sequences(texts, char_to_int)
del texts  # free memory

print('example sequence 10: ', sequences[10])
print('tokenizing took %ds' % round(time.time() - t))
util.print_memory()


print('saving tokenizer and labels to file...')

# save tokenizer, label indexes, and parameters so they can be used for predicting later
with open(MODEL_PATH + '/params.json', 'w') as handle:
    json.dump({
        'sequence_length': SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'num_rows_used': len(sequences),
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'char_to_int': char_to_int,
        'labels_index': labels_index
    }, handle)

print('found %s unique tokens.' % len(char_to_int))

print('padding sequences...')
t = time.time()
data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
del sequences  # free memory
labels = np.asarray(labels)
print('padding sequences took %ds' % round(time.time() - t))
util.print_memory()

print('data:', data)
print('labels:', labels)
print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)

# split data into training and validation
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# validation set can be much smaller if we use a lot of data (source: andrew ng on coursera video)
validation_ratio = 0.2 if data.shape[0] < 1000000 else 0.02
num_validation_samples = int(validation_ratio * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
del data, labels  # free memory

util.print_memory()

print('training model...')

model = Sequential()
model.add(Embedding(len(char_to_int) + 1, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(len(labels_index), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print('model summary: ')
model.summary()
# save model summary in model folder so we can reference it later when comparing models
with open(MODEL_PATH + '/summary.txt', 'w') as handle:
    model.summary(print_fn=lambda x: handle.write(x + '\n'))

# make sure we only keep the weights from the epoch with the best accuracy, rather than the last set of weights
checkpointer = ModelCheckpoint(filepath=MODEL_PATH + '/model.h5', verbose=1, save_best_only=True)
history_checkpointer = util.SaveHistoryCheckpoint(model_path=MODEL_PATH)

util.print_memory()

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpointer, history_checkpointer])

util.copy_model_to_latest(BASE_PATH, MODEL_PATH, MODEL_NAME)

print('total time: %ds' % round(util.total_time()))
util.print_memory()
