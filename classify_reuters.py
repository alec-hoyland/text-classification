import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, reuters
from nltk.tokenize import punkt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

"""
Construct training and test partitions from the Reuters dataset.
"""

def get_reuters():
    documents = reuters.fileids()

    # fetch the document IDs
    train_docs_id   = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id    = list(filter(lambda doc: doc.startswith("test"), documents))
    
    # build the training and testing datasets
    train_docs      = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs       = [reuters.raw(doc_id) for doc_id in test_docs_id]

    # create training and test labels using Reuters categories
    mlb = MultiLabelBinarizer()
    train_labels    = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    test_labels     = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

    return train_docs, test_docs, train_labels, test_labels


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

if __name__ == "__main__":

    train_docs, test_docs, train_labels, test_labels = get_reuters()
    train_docs  = preprocess(train_docs)
    test_docs   = preprocess(test_docs)

    
    # create a vectorizer object
    tfidf_vectorizer = TfidfVectorizer(
        analyzer        = "word",
        stop_words      = stopwords.words('english'),
        max_df          = 0.7,
        max_features    = 10000)
    
    # create sparse matrix representation of documents
    vect_train_docs   = tfidf_vectorizer.fit_transform(train_docs)
    vect_test_docs    = tfidf_vectorizer.transform(test_docs)
    
    # classifier
    classifier = OneVsRestClassifier(ComplementNB())
    classifier.fit(vect_train_docs, train_labels)
 
    # get predictions using trained classifier
    predictions = classifier.predict(vect_test_docs)  

    # metrics    
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
    
    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
