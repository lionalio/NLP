from libs import *

class TextPreprocessing():
    def __init__(self, filename, text_columns, label, delimiter=',', test_split=0.2):
        self.df = pd.read_csv(filename, delimiter=delimiter)
        self.text_columns = text_columns
        self.label = label
        self.X = self.df[text_columns]
        self.y = self.df[label]
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_vectorized = None

    def create_count_vectorizer(self, max_df=0.8, min_df=2, stop_words='english'):
        cnt_vec = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
        self.X_vectorized = cnt_vec.fit_transform(self.X)

    def create_tfidf_vectorizer(self, max_df=0.8, min_df=2, stop_words='english'):
        tfidf = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
        self.X_vectorized = tfidf.fit_transform(self.X)

    def create_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            tarin_test_split(self.X_vectorized, self.y, test_split=self.test_split)