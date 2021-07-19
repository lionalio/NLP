For NLP practicing

To install python environment, please refer to requirements.txt

All the text preprocessing steps are located in one file. They can be called separately or call in one run and get all the stages of preprocessing

## 1/ Cleaning text

Cleaning can be called within class of text processing:

    ProcessingClass.cleaning()

What does it clean:

Markup: * Remove all the special characters
    
        * Remove all single characters
        
        * Remove single characters from the start
        
        * Substituting multiple spaces with single space
        
        * Removing prefixed 'b'
        
        * Lowering case

        * Removing emojis (Perhaps useful for sentiment analysis after all...)

## 2/ Tokenization

Various tokenization like: 

Markup: * Splitting based on space

        * Word tokenize using nltk package (used as default, although performance is mediocre)

        * Regexp tokenize using nltk package

        * tokenize using spaCy (tremendously slow!)

        * tokenize using gensim

        * tokenize using keras (Not so sure about its usefulness afterward...)


## 3/ Feature extraction

Extracting features convert a collection of text documents to a matrix of token occurrences. This stage is generally carried out after cleaning stage.

### 3.1/ Bag-of-words

Based on the bag-of-words methodology, such as:

Markup: * Count vectorizer: occurence counting of words in corpus (simplest but high performance)

        * Tfidf vectorizer: term-frequency * log(inversed document frequency)

### 3.2/ Word Embedding:

word2vec: texts --> generate representation vectors.

There are many good tutorials online about word2vec, like this one and this one, 

In general, when build some model using words, simply labeling/one-hot encoding them is a plausible way to go. However, when using such encoding, the words lose their meaning.

The word2vec intends to give you just that: a numeric representation for each word, that will be able to capture such relations as above (Like: 'Paris' is closer to 'France', but not 'nuclear'). this is part of a wider concept in machine learning — the feature vectors.

Such representations, encapsulate different relations between words, like synonyms, antonyms, or analogies. Word2vec representation is created using 2 algorithms: Continuous Bag-of-Words model (CBOW) and the Skip-Gram model
Markup: * Continuous Bag-of-Words Model: which predicts the middle word based on surrounding context words. The context consists of a few words before and after the current (middle) word. This architecture is called a bag-of-words model as the order of words in the context is not important.

        * Continuous Skip-gram Model: predict words within a certain range before and after the current word in the same sentence. A worked example of this is given below.


### 3.3/ Doc2vec:

Doc2Vec model is opposite to Word2Vec, a document is transformed to a vectorised representation as a single unit, regardless of length of the document. It doesn’t only give the simple average of the words in the sentence.

### 3.4/ All feature extractions:

There are various implemented text representations. It is recommended that the text must be processed before enter the vectorized representation by any kind. The processing step, however, is not perfect. 

#### 3.4.1/ Bag-of-words only

#### 3.4.2/ Word embedding only

#### 3.4.3/ Word embedding + TFIDF weighting

#### 3.4.4/ Doc2vec + TFIDF weighting

To install python environment, please refer to requirements.txt

Depending on the type of NLP problems, they can be sentiment analysis, which corresponds to classification problem, or topic modelling, which is likely an unsupervised problem. (Will be implemented later)

NOTE: The Random Forest algorithm, which is very useful, is NOT recommended to use here, as the features after vectorizing the text can be very sparse, making the concept of random forest irrelevant. Instead, use the algorithms that can deal with high dimension problem, such as naive Bayes, logistic regression, ... 