from libs import *
from text_preprocessing import *

class Classification(TextPreprocessing):
    def __init__(self, filename, text_columns, label, delimiter=',', test_split=0.2):
        super().__init__(filename, text_columns, label, delimiter, test_split)
        self.method_classifier = None
        self.params_classifier = None
        self.method_set = False
        self.parameter_set = False
        self.load_data = False

    def load_processed_data(self, processed_data):
        self.X, self.y = processed_data.X, processed_data.y
        self.X_vectorized = processed_data.X_vectorized
        self.X_train, self.X_test, self.y_train, self.y_test = \
            processed_data.X_train, processed_data.X_test, processed_data.y_train, processed_data.y_test
        self.load_data = True

    def timing(function):
        def wrapper(self):
            print('Running ', function.__name__)
            start = time.time()
            function(self)
            stop = time.time()
            print('Time elapsed: ', stop - start)
            
        return wrapper

    def set_methods(self, clf):
        self.method_classifier = clf
        self.method_set = True

    def set_parameters(self, params):
        self.params_classifier = params
        self.parameter_set = True

    def classifier(self,  method='GridSearch'):
        if self.parameter_set is False:
            raise Exception('Error: All parameters are not yet set!')
        if method == 'GridSearch':
            opt = GridSearchCV(
                estimator=self.method_classifier, 
                param_grid=self.params_classifier
            )
        elif method == 'BayesSearch':     
            opt = BayesSearchCV(
                estimator=self.method_classifier,
                search_spaces=self.params_classifier,
                n_iter=20,
                random_state=7
            )
        opt.fit(self.X_train, self.y_train)   
        self.method_classifier = opt.best_estimator_
        print(opt.best_params_)
        
    @timing
    def evaluate(self):
        preds = self.method_classifier.predict(self.X_test)
        print('accuracy_score: ', accuracy_score(self.y_test, preds))
        print('confusion matrix for ', self.method_classifier.__class__.__name__ , ": ", confusion_matrix(self.y_test, preds))
        if len(np.unique(self.y)) == 2:
            if type(self.y[0]) != str:
                self.plot_roc()

    def plot_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if "predict_proba" not in dir(self.method_classifier):
            print("This function doesnt have probability calculation")
            return
        probs = self.method_classifier.predict_proba(self.X_test_engineer)
        fpr, tpr, _ = roc_curve(self.y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def run(self, method='GridSearch'):
        if self.method_set is False:
            raise Exception('Methods are not yet set. Aborting!')
        if self.parameter_set is False:
            print('Warning: All parameters are taking default values. Consider tuning!')
        if self.load_data is False:
            super().run_text_preprocessing()
        self.classifier(method=method)
        self.evaluate()
