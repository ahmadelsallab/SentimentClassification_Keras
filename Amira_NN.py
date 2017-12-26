'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
from sklearn.tests.test_cross_validation import test_train_test_split
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D

import keras.utils.np_utils as np_utils
import random
from sklearn.preprocessing import LabelEncoder
from io import open

# Embedding
max_features = 20000

embedding_size = 50

# Convolution
filter_length = 2
nb_filter = 2
pool_length = 2
hidden_dims = 10

# LSTM
maxlen = 20
lstm_output_size = 5

# Training
batch_size = 100
nb_epoch = 100
test__train_split_ratio = 0.1# 0.1 means 10% test
nb_classes = 3
lr = 0.01

def corpus_to_indices(text):
    #words_map = build_words_map(text)
        
    #text_to_indices(text, words_map)

    # The vocabulary map
    words_map = {}

    # Index of words
    index = 0
    
    # Initialize the output list
    text_indices = []
    maxlen = 0
    # Loop line by line
    for line in text:
        # Split into words
        line_words = line.split()

        if len(line_words) > maxlen:
            maxlen = len(line_words) 
        # Initialize the line_indices
        line_indices = []
        # Loop word by word
        for word in line_words:
            # Store the word once in the wordMap
            if not word in words_map:
                words_map[word] = index
                # Increment the index for the next word
                index += 1

            # Add the index to the line_indices
            line_indices.append(words_map[word])

        # Add the line_indices to the output list
        text_indices.append(line_indices)


    return text_indices, len(words_map), maxlen

def load_data(data_file_name, annotation_file_name):

    # Load text
    f_data = open(data_file_name, 'r', encoding='UTF8')
    
    text = []
    for line in f_data:
        text.append(line)
    
    
    text_indices, voc_size, maxlen = corpus_to_indices(text)
    
    # Load labels
    f_labels = open(annotation_file_name, 'r')
    
    labels = []
    for line in f_labels:
        labels.append(int(line) - 1)
    
    '''
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels = utils.to_categorical(encoded_Y)
    '''




    X_train, y_train, X_test, y_test = split_train_test(text_indices, labels)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, y_train, X_test, y_test, voc_size, maxlen

def split_train_test(corpus, labels):
    # Randomize the dataSet
    random_data = []
    random_labels = []
    
    # Sample indices from 0..len(corpus)
    size = len(corpus)
    rand_indices = random.sample(range(size), size)
    
    
    # Insert in the final dataset N=self.datasetSize random tweets from the rawData
    for index in rand_indices:
        random_data.append(corpus[index])
        random_labels.append(labels[index])
    
    # Calculate the test set size
    test_set_size = int(test__train_split_ratio * size)
    
    # The trainSet starts from the begining until before the end by the test set size 
    train_set = random_data[0 : size - test_set_size]
    test_set  = random_data[len(train_set) : size]
    train_set_labels = random_labels[0 : size - test_set_size]
    test_set_labels  = random_labels[len(train_set) : size]    
    return train_set, train_set_labels, test_set, test_set_labels


print('Loading data...')
data_file_name = "tweets.txt"
lables_file_name = "sentiment.txt"
X_train, y_train, X_test, y_test, max_features, maxlen = load_data(data_file_name, lables_file_name)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
import tensorflow as tf
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = K.variable((self.init((input_shape[-1],1))))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/tf.expand_dims(K.sum(ai, axis=1), 1)
        
        weighted_input = x*weights
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# model = Sequential()
# model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=False))
#model.add(Dropout(0.9))

'''
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))


'''
'''
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
'''
#model.add(Dropout(0.25))
#model.add(Bidirectional(LSTM(lstm_output_size)))
# model.add(LSTM(lstm_output_size, return_sequences=True))
#model.add(LSTM(lstm_output_size))
#model.add(Dropout(0.9))
# model.add(TimeDistributed(Dense(100)))
# model.add(AttLayer())

'''
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
#model.add(Dense(3, activation='softmax'))
# model.add(Dense(nb_classes))
#model.add(Dropout(0.5))
# model.add(Activation('softmax'))


embedding_layer = Embedding(max_features,
                            embedding_size,
                            input_length=maxlen,
                            trainable=True)
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
#l_lstm = Bidirectional(LSTM(lstm_output_size, return_sequences=True))(embedded_sequences)
l_lstm = LSTM(lstm_output_size, return_sequences=True)(embedded_sequences)
l_dense = TimeDistributed(Dense(10))(l_lstm)
l_att = AttLayer()(l_dense)
preds = Dense(nb_classes, activation='softmax')(l_att)
model = Model(sentence_input, preds)
print(model.summary())

import keras.optimizers
opt = keras.optimizers.adagrad(lr)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test))


model.save('atb_model', overwrite=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test loss:', score)
print('Test accuracy:', acc)




