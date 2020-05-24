# -*- coding: utf-8 -*-
# ----------------------------------------------------------
#
#
#
#         Class for preprocessing specific texts
#
#
#
# ----------------------------------------------------------


import pandas as pd
import numpy as np
import re
import itertools
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import text_to_word_sequence
from googletrans import Translator

class CustomPreProcessing(object):
    '''
    Class to preprocess specific mails and extract informations.
    '''
    
    def __init__(self):
        print('''Welcome in this custom preprocessing class
        ''')
    @classmethod
    def remove_whitespace(self, text):
        """
        Function to remove extra whitespaces from text
        @param text: (str) text
        @return: (str) text clean from extra space
        """
        text = text.strip()
        return " ".join(text.split())
    
    @classmethod
    def remove_phone_number(self, x):
        '''
        Function to find and remove phone number with regex    
        @param x: (str) text 
        @return: (str) text without phone number
        '''
        # ---- Phone number 
        phone = []
        r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})') 
        for i in r.findall(x):
            phone.append(i)
            x = x.replace(i, ' ')
        return x , phone
    
    
    @classmethod
    def remove_upper_case(self, text):
        '''
        Function to transform upper string in title words
        @param text: (str) text 
        @return: (str) text without upper words 
        '''
        sentences = text.split("\n")
        new_sentences = []
        for i in sentences:
            words = text.split()
            stripped = [w.title() if w.isupper() else w for w in words]
            new_sentences.append(" ".join(stripped))
        return "\n".join(new_sentences)
    
    @classmethod
    def strip_text(self, text, sentence):
        '''
        Function to cut the text where the sentence is find and return the first part.
        Powerful to split the text from signature.
        @param text: (str) text
        @param sentence: (str) beginning of the signature
        @return: (str) first part of the text without the signature
        '''
        if sentence in text:
            text = text.split(sentence)[0]
        return text

    @classmethod
    def find_corres(self, text, list_words):
        '''
        Function to locate a word or list of words in a string.
        @param text: (str) text
        @param list_words: (str or list) word or list of words to find in the text
        @return: (bool) indicate if the text contained the list
        '''
        if type(list_words)==str:     # change string in list if one word is passed
            list_words=[list_words]
            
        results = []
        for i in list_words:
            if i in text:
                results.append(True)
            else:
                results.append(False)
        return True if sum(results)>0 else False
    
    @classmethod
    def remove_string(self, text, list_sentences):
        '''
        Function
        @param text: (str) text
        @param list_sentences: (list) list of sentences to be delete
        @return: (str) text without a list of sentences
        '''
        text = text.lower()
        for i in list_sentences:
            if text.find(i.lower()) != -1:           # ---- Looking for the substring 
                text = text.replace(i.lower(), ' ')
        return text
    
    @classmethod
    def find_words(self, text, words_list):
        '''
        Function to locate words in text, estime if the words are closed and if so delete them.
        @param text: (str) text
        @param words_list: (str or list) word or list of words to locate and delete
        @return: (str) text with or without the list (words can't be seperated by more than 5 words)
        '''
        if type(words_list)==str:     # change string in list if one word is passed
            words_list=[words_list]
        if type(text)==tuple:
            print(text)
        if  all(x.lower() in text.lower() for x in words_list ): # ---- Looking for the words in the text 
            text_ = text.split()
            idx = []
            for i,j in itertools.product(words_list, enumerate(text_)): # loop to locate the words and their position in the text
                if i.lower() in j[1].lower(): 
                    idx.append(j[0])
            successive_index = [True if (t - s <=5) else False for s, t in zip(idx, idx[1:])] # if next word in the 5 successive words 
            if sum(successive_index)==len(words_list)-1:
                text = text.split()
                for i in sorted(set(idx))[::-1]:
                    del text[i]
                if len(text)>0:
                    return ' '.join(text)   # return text without lower char 
                else: 
                    return "empty"
            else:
                return text
        else:
            return text 
        
        
class PreProcessing(object):
    '''
    Class to preprocess text
    
    '''
    
    
    def __init__(self):
        print("Welcome in the preprocessing")
    
    
    @classmethod    
    def detect_lang_google(self, x):
        '''
        Function to detect the language of the string
        @param x: (str) sentences of text to detect language
        @return: (str or nan) language of the sentence
        '''
        translate = Translator()
        try:
            return translate.detect(x).lang
        except:
            return np.nan
    
    @classmethod
    def remove_numbers(self, text):
        '''
        Function to remove number in text.
        @param text: (str) sentence
        @return: (str) clean text
        '''
        text = ''.join([i for i in text if not i.isdigit()])         
        return text
    
    @classmethod
    def remove_URL(self, text):
        '''
        Function to remove url from text.
        @param text: (str) sentence
        @return: (str) clean text
        
        '''
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)
    
    @classmethod
    def remove_html(self, text):
        '''
        Function regex to clean text from html balises.
        @param text: (str) sentence 
        @return: (str) clean text 
        '''
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)
    
    
    @classmethod
    def remove_emoji(self, text):
        '''
        Function to remove emojis, symbols and pictograms etc from text
        @param text: (str) sentences 
        @return: (str) clean text 
        '''
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    
    @classmethod
    def preprocess(self, text):
        '''
        Function to remove special characters
        @param text: (pandas dataframe) text
        @return: (pandas dataframe) clean text 
        '''
        text = text.str.replace("(<br/>)", "")
        text = text.str.replace('(<a).*(>).*(</a>)', '')
        text = text.str.replace('(&amp)', '')
        text = text.str.replace('(&gt)', '')
        text = text.str.replace('(&lt)', '')
        text = text.str.replace('(\xa0)', ' ')  
        text = text.str.replace("\n", " ")
        text = text.str.replace("\x92", "'")
        return text
    
    @classmethod
    def remove_char_specific(self, text):
        '''
        Function to remove specific characters
        @param text: (str) text
        @return: (str) text without specific characters
        '''
        table = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~•'
        table = str.maketrans(' ', ' ', table)
        words = text.split()
        stripped = [w.translate(table) for w in words]
        return ' '.join(stripped)
    
    @classmethod
    def remove_upper_case(self, text):
        '''
        Function to transform upper string in title words
        @param text: (str) text 
        @return: (str) text without upper words 
        '''
        words = text.split()
        stripped = [w.title() if w.isupper() else w for w in words]
        return " ".join(stripped)
    
    @classmethod
    def remove_stop_words(self, x, stop_word):
        '''
        Function to remove a list of words
        @param x : (str) text 
        @param stop_word: (list) list of stopwords to delete 
        @return: (str) new string without stopwords 
        '''
        x_new = text_to_word_sequence(x)    # tokenize text 
        x_ = []
        for i in x_new:
            if i not in stop_word:
                x_.append(i)
        return " ".join(x_)
    
    @classmethod
    def get_top_n_words(self, corpus, n=None):
        '''
        Function to return a list of most frequent unigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = CountVectorizer().fit(corpus)             # bag of words
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)  
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    @classmethod
    def get_top_n_words_sw(self, corpus, stop_word=None, lang="fr", n=None):
        '''
        Function to return a list of most frequent unigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param stop_word: (list) list containing stopwords
        @param lang: (str) language of the text
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        if lang=="fr":
            corpus = corpus.apply(lambda x: self.remove_stop_words(x, stop_word))
        vec = CountVectorizer(stop_words = "english").fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    @classmethod
    def get_top_n_bigram(self, corpus, n=None):
        '''
        Function to return a list of most frequent bigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus) 
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    @classmethod
    def get_top_n_bigram_sw(self, corpus, stop_word=None, lang="fr", n=None):
        '''
        Function to return a list of most frequent bigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param stop_word: (list) list containing stopwords
        @param lang: (str) language of the text
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        if lang=="fr":
            corpus = corpus.apply(lambda x: self.remove_stop_words(x, stop_word))
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    @classmethod
    def get_top_n_trigram(self, corpus, n=None):
        '''
        Function to return a list of most frequent trigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

    @classmethod
    def get_top_n_trigram_sw(self, corpus, stop_word=None, lang="fr", n=None):
        '''
        Function to return a list of most frequent trigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param stop_word: (list) list containing stopwords
        @param lang: (str) language of the text
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        if lang=="fr":
            corpus = corpus.apply(lambda x: self.remove_stop_words(x, stop_word))
        vec = CountVectorizer(ngram_range=(3, 3), stop_words="english").fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    @classmethod
    def get_top_n_5grams_sw(self, corpus, stop_word=None, lang="fr", n=None):
        '''
        Function to return a list of most frequent trigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param stop_word: (list) list containing stopwords
        @param lang: (str) language of the text
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        if lang=="fr":
            corpus = corpus.apply(lambda x: self.remove_stop_words(x, stop_word))
        vec = CountVectorizer(ngram_range=(5, 5), stop_words="english").fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]