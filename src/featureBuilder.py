#!/usr/bin/python3.5

from collections import Counter

class FeatureBuilder:
    
    def __init__(self):
        self.features = {}
        self.vectors = []
        pass
    
    def buildFeatureVector(self, messages):
        """Define a feature vector for the given messages
        By this default implementation, features are left empty."""
        pass
    
    def vectorize(self, roughVector):
        """Takes a rough feature vector and returns a standardized form.
        features in roughVector but not in self.features are not saved,
        and features in self.features but not in roughVector are saved as 0.
        
        It is convenient for classes implementing FeatureBuilder to override
        this function, preparing roughVector from raw data, and then calling 
        >>> return super().vectorize(roughVector)"""
        vector = Counter()
        for feature in self.features:
            if feature in roughVector:
                vector[feature] = roughVector[feature]
            else:
                vector[feature] = 0
        #print(self.features)
        #print(vector)
        return vector
    
    def addVector(self, vector, label=None):
        """Adds a vector and a list of vectors.
        Vector should be a counter with keys matching that of self.features,
        which is easily ensured by using the value returned by vectorize(roughVector)."""
        #print(vector)
        self.vectors.append(vector)
    
    def getVectors(self):
        """Returns list of vector values sorted by feature keys.
        This is to provide a consistent vector representation"""
        keyList = list(self.features)
        keyList.sort()
        #print(keyList)
        return [[vector[key] for key in keyList] for vector in self.vectors]


# ------------------ TESTING --------------------
if __name__ == "__main__":
    FB = FeatureBuilder({"phone":1,"iPhone":2,"blackberry":3})
    print(FB)
    print(FB.getVector())


