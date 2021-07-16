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


To install python environment, please refer to requirements.txt

Depending on the type of NLP problems, they can be sentiment analysis, which corresponds to classification problem, or topic modelling, which is likely an unsupervised problem. (Will be implemented later)

NOTE: The Random Forest algorithm, which is very useful, is NOT recommended to use here, as the features after vectorizing the text can be very sparse, making the concept of random forest irrelevant. Instead, use the algorithms that can deal with high dimension problem, such as naive Bayes, logistic regression, ... 