#!/usr/bin/python3.5

import nltk
import string
from collections import Counter
from featureBuilder import FeatureBuilder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def hasPunct(str):
    """ Return true if any of str characters is a punctuation. """
    for c in str:
        if c in string.punctuation:
            return True
    return False

def getNNPs(sentence):
    NNPs = []
    tags = nltk.pos_tag(nltk.word_tokenize(sentence))
    return [t[0].lower() for t in tags if t[1] == 'NNP']

class SenAna(FeatureBuilder):
    
    def buildFeatureVector(self, messages):
        '''
        feature vector is defined by 10 most common NNPs
        '''
        nouns = []
        for message in messages:
            sentokes = nltk.sent_tokenize(message)
            for sent in sentokes:
                tags = nltk.pos_tag(nltk.word_tokenize(sent))
                for t in tags:
                    if t[1] == 'NNP':
                        nouns.append(t[0].lower())
        c = Counter(nouns)
        self.features = {tup[0] for tup in c.most_common(50)}

    def vectorize(self, message):
        '''

        '''
        sid = SentimentIntensityAnalyzer()
        roughVector = {}
        sentokes = nltk.sent_tokenize(message)
        for sent in sentokes:
            ss = sid.polarity_scores(sent)
            compScore = ss['compound']
            NNPs = getNNPs(sent)
            for PN in NNPs:
                if PN in roughVector:
                    roughVector[PN] += compScore
                else:
                    roughVector[PN] = compScore
            #print(sent)
            #print(NNPs)
            #print(compScore)
        #print(message)
        #print(roughVector)
        return super().vectorize(roughVector)
