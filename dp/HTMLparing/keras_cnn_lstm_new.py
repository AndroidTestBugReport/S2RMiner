# LSTM and CNN for sequence classification in the IMDB dataset
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Model
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz", num_words=top_words)#len(X_train[0])!=len(X_train[1]) #len(X_train=25000)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)#after this X_train.shape=(25000,500)   #add 0 at beginning of every word(I think it should be every word 4.10.2019)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))#here omit the first stack dimension (None, 500, 32)# every word is extend to 32
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) #(None, 500, 32)
model.add(MaxPooling1D(pool_size=2)) #(None, 250, 32)

#layer_name='my_layer'#new
#intermediate_model=Model(input=model.input,output=model.get_layer(layer_name).out)#new

model.add(LSTM(100)) #(None, 100)
#model.add(LSTM(100, return_sequences=True))
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=1, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




model.predict(X_train[0:3]) # this predict needs the first dimension of the X_train