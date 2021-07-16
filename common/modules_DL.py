from libs import *

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

def lstm(dim_input, dim_output, vocabSize=500):
    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(vocabSize, embed_dim, input_length = dim_input))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(dim_output, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.summary()

    return model


def dense_net(dim_input, dim_output, vocabSize):
    model = Sequential()
    #model.add(Dense(128, input_shape=dim_input))
    model.add(Embedding(vocabSize, 128, input_length = dim_input))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(dim_output, activation='softmax'))
    model.summary()

    return model


def training(model_input, dim_input, dim_output, vocabSize, X_train, X_test, y_train, y_test):
    #vocabSize = model_input.layers[0].input_dim
    #dim_input = model_input.layers[0].input_length
    #dim_output = model_input.layers[-1].units
    #print(model_input.summary())
    model = KerasClassifier(build_fn=model_input, dim_input=dim_input, dim_output=dim_output, vocabSize=vocabSize, epochs=20, batch_size=32, verbose=0)
    params = {
            'epochs':[20],
            'batch_size':[32]
            #'optimizer' :           ['Adam', 'Nadam'],
            #'dropout_rate' :        [0.2, 0.3],
            #'activation' :          ['relu', 'elu']
        }

    grid = GridSearchCV(estimator=model, 
                                param_grid=params)
    print('Training...')
    grid.fit(X_train, y_train, verbose=1)
    print('Finished training!')
    model_best = grid.best_estimator_
    preds = model_best.predict(X_test)
    score = model_best.score(X_test, y_test)

    print("score: %.2f" % (score))

    return model_best