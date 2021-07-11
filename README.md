For NLP practicing

All the text preprocessing steps are located in one file. They can be called separately or call in one run and get all the stages of preprocessing

To install python environment, please refer to requirements.txt

Depending on the type of NLP problems, they can be sentiment analysis, which corresponds to classification problem, or topic modelling, which is likely an unsupervised problem. (Will be implemented later)

NOTE: The Random Forest algorithm, which is very useful, is NOT recommended to use here, as the features after vectorizing the text can be very sparse, making the concept of random forest irrelevant. Instead, use the algorithms that can deal with high dimension problem, such as naive Bayes, logistic regression, ... 