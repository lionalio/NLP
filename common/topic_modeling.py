from text_preprocessing import TextPreprocessing
from libs import *

class TopicModelling(TextPreprocessing):
    def __init__(self, filename, text_columns, label, delimiter=',', test_size=0.2):
        super().__init__(filename, text_columns, label, delimiter, test_size)
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_vectorized = None
        self.run_text_preprocessing(run_classification=False)

    def topic_modelling(self):
        LDA = LatentDirichletAllocation(n_components=5, random_state=42)
        LDA.fit(self.X_vectorized)
        for i,topic in enumerate(LDA.components_):
            print(f'Top 10 words for topic #{i}:')
            print([self.X_vectorized.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')