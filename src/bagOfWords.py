#!/usr/bin/python3.5

import nltk
import string
from collections import Counter
from featureBuilder import FeatureBuilder

def hasPunct(str):
    """ Return true if any of str characters is a punctuation. """
    for c in str:
        if c in string.punctuation:
            return True
    return False

class BagOfWords(FeatureBuilder):
    
    def buildFeatureVector(self, messages):
        """Define a feature vector for the given messages
        BagOfWords builds feature vectors of the 30 most frequent words"""
        wordCounts = Counter()
        for message in messages:
            wordCounts.update(Counter(nltk.word_tokenize(message.lower())))
            pass
        wordCounts = Counter({k:v for k,v in wordCounts.items() if not hasPunct(k)})
        self.features = {tup[0] for tup in wordCounts.most_common(30)}
    
    def vectorize(self, message):
        """Creates and returns a vector representation of message.
        A 'roughVector' is first created from message,
        which is then standardized by the super().vectorize(...) function."""
        roughVector = Counter(nltk.word_tokenize(message.lower()))
        return super().vectorize(roughVector)


# ------------------ TESTING --------------------
if __name__ == "__main__":
    BOW = BagOfWords({"phone":1,"iPhone":2,"blackberry":3})
    print(BOW)
    print(BOW.getVector())

