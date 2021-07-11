import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "\\common")

from classification import *

from sklearn.model_selection import GridSearchCV, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def lstm(dim_input, dim_output):
    vocabSize=500
    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(vocabSize, embed_dim, input_length = dim_input))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(dim_output, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    return model

df = pd.read_csv('Tweets.csv')

detector = Classification('Tweets.csv', 'text', 'airline_sentiment')
detector.run_text_preprocessing()

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X1 = tokenizer.texts_to_sequences(df['text'].values)
#y = pd.get_dummies(df['airline_sentiment']).values
X, y = detector.X_vectorized, detector.y
X = pad_sequences(X)
X1 = pad_sequences(X1)
#X_train, X_test, y_train, y_test = detector.X_train, detector.X_test, detector.y_train, detector.y_test
#X_train = pad_sequences(X_train)
#X_test = pad_sequences(X_test)
print(X.shape)
print(X1.shape)
dim_input = X.shape[1]
dim_output = len(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = KerasClassifier(build_fn=lstm, dim_input=dim_input, dim_output=dim_output)

params = {
        'epochs':[20],
        'batch_size':[128]
        #'optimizer' :           ['Adam', 'Nadam'],
        #'dropout_rate' :        [0.2, 0.3],
        #'activation' :          ['relu', 'elu']
    }

grid = GridSearchCV(estimator=model, 
                            param_grid=params)

grid.fit(X_train, y_train)
model_best = grid.best_estimator_
preds = model_best.predict(X_test)
score = model_best.score(X_test, y_test)

#rounded_preds = model.predict_classes(X_test, batch_size=128, verbose=0)
#rounded_y_test=np.argmax(y_test, axis=1)
#confuse_matrix = confusion_matrix(rounded_y_test, rounded_preds)
print("score: %.2f" % (score))
#print("confusion matrix", confuse_matrix)