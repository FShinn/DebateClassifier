#!/usr/bin/python3.5

import nltk
import string
from math import log
from collections import Counter
from featureBuilder import FeatureBuilder

def hasPunct(str):
    """ Return true if any of str characters is a punctuation. """
    for c in str:
        if c in string.punctuation:
            return True
    return False

def score(wordFreq, docPresence, totalDocCount):
    """Calculation used to get TF-IDF scores"""
    return log(wordFreq)*( log(totalDocCount)-log(docPresence) )

class TFIDF(FeatureBuilder):
    
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
        TFIDFScores = Counter({word:score(values['freq'],values['docs'],len(messages)) 
                                for word,values in words.items()})
        # most_common() returns a tuple of (key, value). We only need key.
        self.features = {tup[0] for tup in TFIDFScores.most_common(30)}
    
    def vectorize(self, message):
        """Creates and returns a vector representation of message.
        A 'roughVector' is first created from message,
        which is then standardized by the super().vectorize(...) function."""
        roughVector = Counter(nltk.word_tokenize(message.lower()))
        return super().vectorize(roughVector)

