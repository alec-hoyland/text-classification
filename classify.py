# https://stackabuse.com/text-classification-with-python-and-scikit-learn/
import numpy as np
import re, nltk, pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


"""
Preprocesses the dataset to create a cleaned corpus.
"""

def preprocess(dataset):
    corpus = []

    lemmatizer = WordNetLemmatizer()

    for document in dataset:
        document = str(document)

        # remove all the special characters
        document = re.sub(r'\W', ' ', document)

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [lemmatizer.lemmatize(word) for word in document]
        document = ' '.join(document)

        # add the document to the corpus
        corpus.append(document)
    return corpus

"""
Vectorizes the corpus using a bag-of-words term-frequency model.
"""

def vectorize(corpus):
    vectorizer = CountVectorizer(
        max_features    = 1500,
        min_df          = 5,
        max_df          = 0.7,
        stop_words      = stopwords.words('english'))
    return vectorizer.fit_transform(corpus).toarray()

if __name__ == "__main__":

    # import the dataset
    movie_data = load_files("/home/hoyland/data/datasets/sentiment-analysis/txt_sentoken")
    x, y = movie_data.data, movie_data.target

    # preprocess the corpus
    corpus = preprocess(x)

    # create document-term matrix
    dtm = vectorize(corpus)

    # split the dataset into training and testing sets
    x_training, x_testing, y_training, y_testing = train_test_split(dtm, y, test_size=0.2, random_state=0)

    # instantiate a random forest model
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(x_training, y_training)
    y_prob = classifier.predict_log_proba(x_testing)
    y_predicted = classifier.predict(x_testing)

    print(confusion_matrix(y_testing, y_predicted))
    print(classification_report(y_testing, y_predicted))
    print(accuracy_score(y_testing, y_predicted))
