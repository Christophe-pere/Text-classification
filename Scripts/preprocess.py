'''

    Python file containing functions and classes to
    clean text and encode it


'''
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from keras.utils import np_utils

import pandas as pd
from googletrans import Translator
import unidecode
import contractions
import re
import unicodedata
import inflect
import nltk
from num2words import num2words
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from tqdm import tqdm
from textblob import TextBlob 
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import seaborn as sns
import xgboost as xgb
# ---- Call tqdm to see progress bar with pandas
tqdm().pandas()


N = 0.5


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())

def func_remove_char_specific(text):
    table = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~•'
    table = str.maketrans('', '', table)
    words = text.split()
    stripped = [w.translate(table) for w in words]
    return ' '.join(stripped)

def func_remove_upper_case(text):
    words = text.split()
    stripped = [w.lower() if w.isupper() else w for w in words]
    return " ".join(stripped)

# ---- XGBoost classifier ---------------------------------------



# ---- Evaluation metrics ---------------------------------------

def func_plot_history(history):
    
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)


    # Plot training & validation loss values
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()

# ---- XGBoost evaluation
def func_plot_eval_xgb(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(15,10))
    plt.subplot(221)
    # Plot training & validation accuracy values
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('XGBoost Log Loss')
    plt.legend(loc='upper left')
    plt.grid(True)


    # Plot training & validation loss values
    plt.subplot(222)
    plt.plot(x_axis, results['validation_0']['merror'], label='Train')
    plt.plot(x_axis, results['validation_1']['merror'], label='Test')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.xlabel('Epochs')
    plt.title('XGBoost Classification Error')
    plt.legend( loc='upper left')
    plt.grid(True)
    plt.show()

    



def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
        #plt.savefig('confusion-matrix.png')
    

    
    

        
def func_precision_recall(y_true, y_test, N=0.5,  verbose=True):
    r = [1 if i[1]>N else 0 for i in y_test]
    conf  = confusion_matrix(y_true, r)
    tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    if verbose:
        print('''     Predicted       Predicted  
                 NO               YES
        Real   TN={}          FP={}
        NO     
        Real   FN={}           TP={}
        YES       '''.format(tn, fp, fn, tp))
        print('''
                  TP                     
    Precision = _______ = {}%    
                 TP+FP       


               TP
    Recall = ______  = {}%
              FN+TP           '''.format(round(tp/(tp+fp)*100,2), round(tp/(fn+tp)*100,2)))
    return round(tp/(tp+fp)*100,2), round(tp/(fn+tp)*100,2)

def func_detect_lang_google( x):

    translate = Translator()
    try:
        return translate.detect(x).lang
    except:
        return np.nan
    
    
def func_translate_google(x, src="en", dest="fr"):
    """
        Function to translate text with google translate API
        input:
            - x : string to translate
            - src : initial language to translate
            - dest: destination language of the translation
        output:
            - x : string translated 
    """
    
    translate = Translator()
    #try: 
    return translate.translate(x, src=src, dest=dest).text
    #except:
    #    return x
def func_stem_words(words, country='english'):
    """
        Stem words in list of tokenized words
        input:
            - words: tokenized words
        output:
            - stems : tokenized and stem words
    """
    words = word_tokenize(words)
    stemmer = SnowballStemmer(country, ignore_stopwords=True) # french 
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return " ".join(stems)

def func_lemmatize_verbs(words):
    """
        Lemmatize verbs in list of tokenized words
        input:
            - words: tokenized words
        output:
            - lemmas : tokenized and lemmatized words
    """
    words = word_tokenize(words)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return " ".join(lemmas)

def func_remove_phone_number(x):
    '''
        Function to find and remove phone number with regex
        input:
            - x : string / raw string 
        output:
            - x : without phone number
    '''
    # ---- Phone number 
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})') 
    for i in r.findall(x):
        x = x.replace(i, '')
    return x 

def is_number(s):
    '''
        Function to test if a word is a number
        input: 
            - s : word
        output:
            - boolean value 
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def func_numeric(x):
    '''
        Function to remove number 
    '''
    x_new = word_tokenize(x)
    x_ = []
    for i in x_new:
        if not is_number(i):
            x_.append(i)
    return " ".join(x_)


def func_remove_stop_words(x, stop_word):
    '''
        Function to remove a list of words
        input:
            - x : raw string 
            - stop_word_fr: stopwords to delete 
        output:
            - x_new : new string without stopwords 
    '''
    x_new = word_tokenize(x)
    x_ = []
    for i in x_new:
        if i not in stop_word:
            x_.append(i)
    return " ".join(x_)


def func_remove_chars(df, list_chars):
    
    
    for i in list_chars:
        df = df.str.replace(i, " ")
    df = df.apply(func_remove_non_ascii)
    df = df.apply(lambda x :re.sub(r"^\s+", "", x, flags=re.UNICODE))
    return df

def func_remove_hour(x):
    
    
    r_ = re.compile(r'\d{2}[a-z]\d{2}|\d{1}[a-z]\d{2}|\d{2}[a-z]\d{1}|\d{1}[a-z]\d{1}|\d{2}[a-z]|\d{1}[a-z]')
    for i in r_.findall(x):
        x = x.replace(i, " ")
    return x



def func_remove_num_str(x):
    '''
        Function to remove word like 12e or 1e or 005f 
        input:
            - x : raw text data
        output
            - x : raw text data without the kind of regex 
    '''
    r_ = re.compile(r'\d{3}[a-z]|\d{2}[a-z][a-z][a-z][a-z]|\d{2}[a-z][a-z][a-z]|\d{2}[a-z][a-z]|\d{2}[a-z]|\d{1}[a-z][a-z][a-z][a-z]|\d{1}[a-z][a-z][a-z]|\d{1}[a-z][a-z]|\d{1}[a-z]')
    for i in r_.findall(x):
        x = x.replace(i, ' ')
    return x 

def func_replace_words(x, dict_):
    
    
    x = word_tokenize(x)
    for i in itertools.product(enumerate(x), dict_.keys()):
        if i[0][1]==i[1]:
            x[i[0][0]] = dict_[i[1]]
    return " ".join(x)


