#!/usr/bin/python3.5

import nltk
import string
from math import log
from collections import Counter
from featureBuilder import FeatureBuilder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def hasPunct(str):
    """ Return true if any of str characters is a punctuation. """
    for c in str:
        if c in string.punctuation:
            return True
    return False

def score(wordFreq, docPresence, totalDocCount):
    """Calculation used to get TF-IDF scores"""
    return log(wordFreq)*( log(totalDocCount)-log(docPresence) )

class TFIDFSen(FeatureBuilder):
    
    #def buildFeatureVector(self, messages):
        #"""Define a feature vector for the given messages
        #BagOfWords builds feature vectors of the 30 most frequent words"""
        #wordCounts = Counter()
        #for message in messages:
            #wordCounts.update(Counter(nltk.word_tokenize(message.lower())))
            #pass
        #stopwords = set(nltk.corpus.stopwords.words('english'))
        #wordCounts = Counter({k:v for k,v in wordCounts.items() if not (hasPunct(k) or k in stopwords)})
        #self.features = {tup[0] for tup in wordCounts.most_common(50)}
        ##print(self.features)
        #return self.features
    
    def buildFeatureVector(self, messages):
        """Define a feature vector for the given messages.
        Builds feature vectors of the 30 words with highest corpus-wide TF-IDF scores"""
        
        # gather information necessary to perform TF-IDF calculation
        words = {} # {word:{'freq':frequencyOfWord, 'docs':numberOfDocsContainingWord},...}
        for message in messages:
            messageWordCount = Counter(nltk.word_tokenize(message.lower()))
            for word,count in messageWordCount.items():
                if not hasPunct(word):
                    if word not in words:
                        words[word] = {'freq':0, 'docs':0}
                    words[word]['freq'] += count
                    words[word]['docs'] += 1
        
        # perform TF-IDF calculation
        stopwords = set(nltk.corpus.stopwords.words('english'))
        TFIDFScores = Counter({word:score(values['freq'],values['docs'],len(messages)) 
                                for word,values in words.items()
                                if not (hasPunct(word) or word in stopwords)})
        self.features = {tup[0]:tup[1] for tup in TFIDFScores.most_common(500)}
        return self.features
        #self.features = TFIDFScores
        #return TFIDFScores

    def vectorize(self, message):
        """Creates and returns a vector representation of message.
        A 'roughVector' is first created from message,
        which is then standardized by the super().vectorize(...) function."""
        sid = SentimentIntensityAnalyzer()
        roughVector = Counter()
        sentokes = nltk.sent_tokenize(message)
        for sent in sentokes:
            #print(sent)
            ss = sid.polarity_scores(sent)
            compScore = ss['compound']
            for key in self.features:
                if key in sent:
                    roughVector[key] += compScore*self.features[key]
        vec = super().vectorize(roughVector)
        #temp = Counter({w:v for w,v in vec.items() if v != 0})
        #if temp:
            #print(temp)
            #print(message)
            #print()
        return vec

