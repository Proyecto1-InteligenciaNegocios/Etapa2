import pandas as pd
import numpy as np
import nltk
import spacy
import string
import seaborn as sns
import matplotlib.pyplot as plt
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

from matplotlib import style
from sklearn.linear_model import LogisticRegression
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
import io

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
    vectorizado = texto.map(str)
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

    result_df = result_df.fillna(int(0))


def classify_multiple_texts(texto) -> pd.DataFrame:
    if not isinstance(texto, pd.DataFrame):
        texts_df = pd.DataFrame([texto], columns=['Review'])

    df_columns = texts_df.columns

    if len(df_columns) > 1:
        raise Exception("The dataframe must have only one column")
    elif len(df_columns) < 1:
        raise Exception("The dataframe must have at least one column")
    
    column_name = df_columns[0]
    for i in range(len(texts_df[column_name])):
        limpio = removerTodos(texts_df[column_name].iloc[i])
        texts_df[column_name].iloc[i] = limpio

    vectorizacion(text_df)

    new_X = texts_df[column_name]
    pipeline = load('model.joblib')
    class_column = pipeline.predict(new_X)
    texts_df["Class"] = class_column
    texts_df["Review"] = new_X
    return texts_df

def classify_single_text(text: str) -> int:
    if not isinstance(texto, pd.DataFrame):
        texts_df = pd.DataFrame([texto], columns=['Review'])
        
    pipeline = load('model.joblib')
    preprocessed_text = removerTodos(text)
    
    return pipeline.predict([preprocessed_text])[0], preprocessed_text
