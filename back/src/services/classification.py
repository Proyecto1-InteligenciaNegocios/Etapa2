import pandas as pd
import numpy as np
import nltk
import spacy
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('snowball_data')
nltk.download('wordnet')
nltk.download('omw')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from spacy.lang.es.examples import sentences
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

import sklearn
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from sklearn.pipeline import Pipeline
from joblib import dump, load

def definirStopwords():
    palabrasEliminar = set(stopwords.words('spanish'))
    palabrasEliminar.add('...')
    palabrasEliminar.add("''")
    palabrasEliminar.add('`')
    palabrasEliminar.add('....')
    palabrasEliminar.add('``')

    return palabrasEliminar


def removerTodos(texto):

    nlp = spacy.load("es_core_news_sm")

    #Eliminacion signos de puntuacion
    texto = texto.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    #Tokenizacion
    palabras = nltk.word_tokenize(texto)
    palabrasEliminar = definirStopwords()
    #Lematizador
    lemas = []
    for palabra in palabras:
        # Convertir a minusculas
        palabra = palabra.lower()
        # Revisar palabras a eliminar, palabras vacias, numeros y palabras en español
        if (palabra.strip()):
            if (palabra not in palabrasEliminar):
                if (not palabra.isdigit()):
                    #Lematizador
                    doc = nlp(palabra)
                    lema = doc[0].lemma_
                    lemas.append(lema)
    return lemas

def vectorizacion(texto):
    df = pd.read_csv('BagOfWords.csv', sep=',', encoding = 'utf-8')
    vectorizado = texto["Review"].map(str)
    count = CountVectorizer()
    bag_of_words = count.fit_transform(vectorizado)
    bag_of_words_array = bag_of_words.toarray()
    bag_of_words_df = pd.DataFrame(bag_of_words_array, columns=count.get_feature_names_out())
    bag_of_words_df["Class"] = float('nan')
    
    result_df = pd.DataFrame(columns=df.columns)

    for col in df.columns:
        if col in count.get_feature_names_out():
            result_df[col] = bag_of_words_df[col]
        else:
            result_df[col] = 0  # Fill with 0 for empty columns

    # Fill in the remaining columns (not in selected_columns) with 0
    for col in result_df.columns:
        if col not in df.columns:
            result_df[col] = 0

    result_df_new = result_df.fillna(int(0))
    return result_df_new


def classify_single_text(texto):
    if isinstance(texto, pd.DataFrame) != True:
        texto = pd.DataFrame([texto], columns=['Review'])

    preprocessed_text = removerTodos(texto)
    result_df = vectorizacion(preprocessed_text)
    
    pipeline = load('model.joblib')
    
    Y = result_df['Class']
    X = result_df.drop(['Class'], axis=1)

    y_pred = pipeline.predict(X)

    result_df['Class'] = y_pred
    
    return y_pred
