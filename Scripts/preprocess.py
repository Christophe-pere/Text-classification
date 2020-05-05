# -*- coding: utf-8 -*-
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



N = 0.5




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



# ---- XGBoost evaluation
def func_plot_eval_xgb(model, label):
    # retrieve performance metrics
    if len(label)>2:
        error_   = "merror"
        logloss_ = "mlogloss"
    else:
        error_  = "error"
        logloss_= "logloss"
        
    results = model.evals_result()
    epochs = len(results['validation_0'][error_])
    x_axis = range(0, epochs)

    plt.figure(figsize=(15,10))
    plt.subplot(221)
    # Plot training & validation accuracy values
    plt.plot(x_axis, results['validation_0'][logloss_], label='Train')
    plt.plot(x_axis, results['validation_1'][logloss_], label='Test')
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('XGBoost Log Loss')
    plt.legend(loc='upper left')
    plt.grid(True)


    # Plot training & validation loss values
    plt.subplot(222)
    plt.plot(x_axis, results['validation_0'][error_], label='Train')
    plt.plot(x_axis, results['validation_1'][error_], label='Test')
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








class PreProcessing:
    
    def __init__(self, data):
        print('''Welcome in this preprocessing classe
        ''')
        self.data = data
        
    def remove_phone_number(self):
        '''
            Function to find and remove phone number with regex
            input:
                - x : string / raw string 
            output:
                - x : without phone number
        '''
        # ---- Phone number 
        phone = []
        r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})') 
        for i in r.findall(self):
            phone.append(i)
            self = self.replace(i, '')
        return self , phone