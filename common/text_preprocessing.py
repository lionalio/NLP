from libs import *

class TextPreprocessing():
    def __init__(self, filename, text_columns, label, delimiter=',', test_size=0.2):
        self.df = pd.read_csv(filename, delimiter=delimiter)
        self.text_columns = text_columns
        self.label = label
        self.X = self.df[text_columns]
        #print(self.X)
        #self.X = self.X.apply(self.cleaning)
        self.y = self.df[label]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_vectorized = None
        self.vectorizer = None
        self.tokenizer = None
        self.model_w2v = None
        self.max_features = 3000
        print('Begin text processing')
        self.X = self.X.apply(self.text_preprocessing)
        print('end text processing')

    def save_to_pkl(self):
        objs = [self.X, self.X_train, self.X_test, self.y_train, self.y_test]
        filenames = ["X.pkl", "X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
        for obj, filename in zip(objs, filenames):
            file_to_store = open(filename, "wb")
            pickle.dump(obj, file_to_store)
            file_to_store.close()
        #self.X_train.to_pickle('X_train.pkl')
        #self.X_test.to_pickle('X_test.pkl')
        #self.y_train.to_pickle('y_train.pkl')
        #self.y_test.to_pickle('y_test.pkl')

    def set_tokenizer(self, tokenizer):
        print('Set {} as tokenizer'.format(tokenizer.__name__))
        self.tokenizer = tokenizer

    def cleaning(self, text):
        # Remove all the special characters
        process_text = re.sub(r'\W', ' ', text)

        # remove all single characters
        process_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', process_text)

        # Remove single characters from the start
        process_text = re.sub(r'\^[a-zA-Z]\s+', ' ', process_text)

        # Substituting multiple spaces with single space
        process_text = re.sub(r'\s+', ' ', process_text, flags=re.I)

        # Removing prefixed 'b'
        process_text = re.sub(r'\^b\s+', '', process_text)

        # lowering case
        process_text = process_text.lower()

        return process_text

    def tokenization(self, text):
        tokens = word_tokenize(text)
        return tokens

    def spelling_correction(self, text):
        textblob =  TextBlob(text)
        text_correct = str(textblob.correct())
        return text_correct

    def lemmatizing(self, tokenized_text):
        lem = WordNetLemmatizer()
        words = []
        for word in tokenized_text:
            words.append(lem.lemmatize(word, pos='v'))

        return words

    def stemming(self, tokenized_text):
        stemmer = PorterStemmer()
        words = []
        for word in tokenized_text:
            words.append(stemmer.stem(word))

        return words

    def remove_emojis(self, text):
        """
        https://dataaspirant.com/nlp-text-preprocessing-techniques-implementation-python/
        Result :- string without any emojis in it
        Input :- String
        Output :- String
        """
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

        without_emoji = emoji_pattern.sub(r'', text)
        return without_emoji

    def remove_stopwords(self, tokenized_text):
        words = []
        for word in tokenized_text:
            if word not in all_stopwords:
                words.append(word)

        return words

    def text_preprocessing(self, text):
        processed_text = self.cleaning(text)
        processed_text = self.remove_emojis(processed_text)
        #processed_text = self.spelling_correction(processed_text)  # Astronomical slow!
        tokenized_text = self.tokenization(processed_text)
        #tokenized_text = self.stemming(tokenized_text)
        tokenized_text = self.lemmatizing(tokenized_text)
        tokenized_text = self.remove_stopwords(tokenized_text)
        
        return ' '.join(tokenized_text)

    def run_text_preprocessing(self, vectorizer='tfidf', embedding=False, run_classification=True):
        if vectorizer == 'tfidf':
            self.create_tfidf_vectorizer()
        elif vectorizer == 'count':
            self.create_count_vectorizer()
        
        if embedding:
            print(self.vectorizer)
            self.create_embedded_vectorizer()
            print(self.X_vectorized)
        if run_classification:
            self.create_train_test()
        self.save_to_pkl()

    def create_count_vectorizer(self, max_df=0.8, min_df=2, stop_words='english'):
        print('Using count as vectorizer')
        cnt_vec = CountVectorizer(max_features=self.max_features, max_df=max_df, min_df=min_df, stop_words=stop_words)
        self.X_vectorized = cnt_vec.fit_transform(self.X).toarray()
        self.vectorizer = cnt_vec

    def create_tfidf_vectorizer(self, max_df=0.8, min_df=2, stop_words='english'):
        print('Using tfidf as vectorizer')
        tfidf = TfidfVectorizer(max_features=self.max_features, max_df=max_df, min_df=min_df, stop_words=stop_words)
        self.X_vectorized = tfidf.fit_transform(self.X).toarray()
        self.vectorizer = tfidf

    def create_word_embedding(self, min_word_count = 40,  # Minimum word count
                                    num_workers = 4,     # Number of parallel threads
                                    context = 10,        # Context window size
                                    downsampling = 1e-3 # (0.001) Downsample setting for frequent words)
                            ):
        print('Create word embedding using word2vec')
        model = Word2Vec(size=self.max_features, \
                        min_count=min_word_count,\
                        window=context,
                        sample=downsampling,
                        workers=n_cores-1
                        )
        model.build_vocab(self.X, progress_per=10000)
        model.train(self.X, total_examples=len(self.X), epochs=20, report_delay=1)
        model_name = "model_w2v"
        model.save(model_name)
        print('Finished and saved in ', model_name)
        self.model_w2v = model

    def load_word_embedding(self, file_w2v):
        self.model_w2v = KeyedVectors.load(file_w2v)

    def function_embedding(self, text):
        if self.model_w2v is None:
            raise Exception('Please consider load word embedding first or create a new one.')
        size = self.max_features
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        tfidf = dict(zip(self.vectorizer.get_feature_names(), self.vectorizer.idf_))
        tokens = self.tokenizer(text)
        for word in tokens: # self.vectorizer.get_feature_names():
            try:
                vec += self.model_w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError: 
                
                continue
        if count != 0:
            vec /= count
        return vec

    def create_embedded_vectorizer(self):
        self.X_vectorized = self.X.apply(self.function_embedding)


    def create_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X_vectorized, self.y, test_size=self.test_size)