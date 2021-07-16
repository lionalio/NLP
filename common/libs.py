# Common tools
import pandas as pd
import numpy as np
import time
import itertools
import copy
import os, glob, pickle
import sys
import re
import multiprocessing

# Data Preparation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# For hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# For NLP vectorization
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import regexp_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
#import spacy
#from spacy.lang.en import English
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Resampling
from imblearn.over_sampling import SMOTE

# Machine Learning / AI
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    GradientBoostingRegressor, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Lasso, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,  roc_curve, auc, \
    silhouette_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# list of stopwords from nltk
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
#sp = spacy.load('en_core_web_sm')
stopwords_nltk = list(stopwords.words('english'))
#stopwords_spacy = list(sp.Defaults.stop_words)
stopwords_gensim = list(gensim.parsing.preprocessing.STOPWORDS)
all_stopwords = []
all_stopwords.extend(stopwords_nltk)
#all_stopwords.extend(stopwords_spacy)
all_stopwords.extend(stopwords_gensim)
# all unique stop words
all_stopwords = list(set(all_stopwords))

n_cores = multiprocessing.cpu_count()