#!/usr/bin/python3.5

#import spacy

from sklearn.cluster import KMeans
from collections import Counter
import os
import argparse

from filereader import readFiles
from bagOfWords import BagOfWords
from TFIDF import TFIDF
from senAna import SenAna
from TFIDFSen import TFIDFSen
from featureBuilder import FeatureBuilder
from sklearn.metrics.cluster import adjusted_rand_score

normIt = False
rnf = False
numClust = 0

def collectFeatureVectors(featureBuilder, posts):
    """Use featureBuilder to construct feature vectors from posts.
    Vectors are built without knowing stance.
    Posts are vectorized without stance,
    though can optionally be stored with a vector for evaluation later."""
    featureBuilder.buildFeatureVector([post["message"] for post in posts])
    for post in posts:
        vector = featureBuilder.vectorize(post["message"])
        #print(vector)
        featureBuilder.addVector(vector)
    return featureBuilder.getVectors()

def norm(vector):
    """Makes a vector unit length.
    If original vLen is 0, leave vector as [0,...]"""
    vLen = sum([e**2 for e in vector])**(1/2)
    if vLen != 0:
        return [e/vLen for e in vector]
    else:
        return [0 for e in vector]

def normalize(vectors):
    """Make all vectors unit length"""
    return [norm(v) for v in vectors]

def removeNullFeatures(featureVectors,posts):
    """Removes any vectors with all entries 0 and associate post"""
    indicies = [] # track indicies to delete
    for i in range(len(featureVectors)):
        if not any([v != 0 for v in featureVectors[i]]):
            indicies.append(i)
    indicies.reverse()
    print("Removed {0:d} featureless posts and corresponding vectors".format(len(indicies)))
    for i in indicies:
        del(featureVectors[i])
        del(posts[i])

def clusterPosts(featureVectors,posts,numClusters=2):
    global normIt
    global rnf
    global numClust
    """Uses KMeans(...).fit(...) to cluster a list of lists.
    Returns a list of tuples [(post,clusterID),...]
    Values returned by featureVectors, klabels, and posts
    are connected to the same post by index-alignment."""
    if numClust == None:
        numClust = 2

    if normIt:
        print('normalizing')
        featureVectors = normalize(featureVectors)
    if rnf:
        print('removing feature vectors')
        removeNullFeatures(featureVectors,posts)
    klabels = KMeans(n_clusters=numClust, random_state=9).fit(featureVectors).labels_
    return [(posts[i],klabels[i]) for i in range(len(klabels))]

def classifyTopic(topic, posts, featureBuilder):
    """Returns a list of tuples containing posts and corresponding clusterID.
    Feature vectors are first selected and computed using featureBuilder,
    then used to classify posts using KMeans clustering."""
    featureVectors = collectFeatureVectors(featureBuilder,posts)
    return clusterPosts(featureVectors,posts,2)

def evaluate(postClusters):
    """ Takes in a list of (post, clusterID) tuples.
    Outputs raw counts of pairings and gives an adjusted_rand_score
    which scale is -1.0 to 1.0, with 
       -1.0 is somehow worst than random pairing (can't find docs to explain)
       0.0 is random pairing
       1.0 is perfect clustering
    """

    stanceClusterPairs = [(tup[0]["stance"],tup[1]) for tup in postClusters]
    tally = Counter(stanceClusterPairs)
    stanceSet = set()
    group = Counter()
    for (stance,clusterID),count in tally.items():
        stanceSet.add(stance)
        if len(stance) > 40:
            stance = stance[:37] + "..."
        print("stance: {0:40s} | clusterID: {1:2d} | count: {2:2d}".format(stance,clusterID,count))

    stanceMap = {stance: i for i, stance in enumerate(stanceSet)}
    labels_true = list()
    labels_pred = list()
    for tup in postClusters:
        stanceID = tup[0]["stance"]
        clusterID = tup[1]
        labels_true.append(stanceMap[stanceID])
        labels_pred.append(clusterID)

    randScore = adjusted_rand_score(labels_true, labels_pred)
    print("randscore: {0:.3f} {1:13s} (-1.0 worst, 0.0 random, 1.0 perfect)".format(randScore, ""))
    print()
    return randScore

def dump(topic,postClusters):
    """Dumps messages to an output file by clusterID"""
    postsByCluster = {tup[1]:[] for tup in postClusters}
    for post,clusterID in postClusters:
        postsByCluster[clusterID].append(post)
    
    if not os.path.exists("dumps"):
        os.makedirs("dumps")
    with open(os.path.join("dumps",topic + ".dump"),'w',encoding="UTF-8") as outFile:
        for ID in sorted(postsByCluster.keys()):
            outFile.write("ClusterID: {0:d}\n".format(ID))
            for post in postsByCluster[ID]:
                outFile.write(post['stance'] + '\n')
                outFile.write(post['message'] + '\n--------------------------------\n')
            outFile.write("\n\n")

def pickFeatureBuilder(featureBuilderName):
    featureBuilder = FeatureBuilder()

    if featureBuilderName == 'BOW':
        #print("Creating Bag Of Words Feature Builder")
        featureBuilder = BagOfWords()
    elif featureBuilderName == 'SENA':
        #print("Creating Sentiment Analysis Feature Builder")
        featureBuilder = SenAna()
    elif featureBuilderName == 'TFIDF':
        #print("Creating TFIDF Feature Builder")
        featureBuilder = TFIDF()
    elif featureBuilderName == 'TSEN':
        #print("Creating TFIDF Sentiment Feature Builder")
        featureBuilder = TFIDFSen()        
    return featureBuilder

# ------------------ RUNNING --------------------
if __name__ == "__main__":
    # (0) parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("-n", "--normalize", action="store_true")
    parser.add_argument("-rnf", "--removenullfeatures", action="store_true")
    parser.add_argument("-c", "--clusters", type=int)

    featureBuilderName = args.method
    normIt = args.normalize
    rnf = args.removenullfeatures
    numClust = args.clusters
    
    # (1) load data
    data = readFiles("stance")
    # data = [(topic, [(filename, stance, message), ... ]), ... ]
    
    # (2) run classification on each topic
    topicSum = 0.0
    topicCount = 0
    for topic, posts in data:
        print(topic)

        featureBuilder = pickFeatureBuilder(featureBuilderName)

        postClusters = classifyTopic(topic, posts, featureBuilder)
        topicSum += evaluate(postClusters)
        topicCount += 1
        dump(topic,postClusters)
    topicAve = topicSum / topicCount
    print()
    print()
    print("Total average randscore: {0:0.3f}".format(topicAve))
