import math
import numpy
import os
import pickle

from dwl import get_dev_test_dicts, get_train_dicts

#Make sure the parameters here are excetly the same as
#in the training file

FREQ_CUTOFF = 0
TRAIN_FILE = 'data/test'
DEV_FILE = 'data/test'
TEST_FILE = 'data/test'
MODELS_DIR = 'data/models/'

trainSrcDict, trainTgtDict = get_train_dicts(TRAIN_FILE)
devSrcDict = get_dev_test_dicts(DEV_FILE)
testSrcDict = get_dev_test_dicts(TEST_FILE)
    
devAndTestSrcDict = {}
for word in set(devSrcDict.keys()) | set(testSrcDict.keys()):
    if trainSrcDict[word]+devSrcDict[word]+testSrcDict[word] > FREQ_CUTOFF:
        devAndTestSrcDict[word] = len(devAndTestSrcDict)

dwl = {}
for fileName in os.listdir(MODELS_DIR):
    word = fileName.replace('word_', '').replace('.pickle', '')
    fileName = MODELS_DIR+fileName
    word_clf = pickle.load(open(fileName, 'r'))
    dwl[word] = word_clf

# Returns the probability of occurrence of a phrase 
# in the target given source words
def get_dwl_score(srcWords, tgtPhrase):
    
   X = numpy.zeros((len(devAndTestSrcDict)), dtype=int)
   for srcWord in srcWords:
        if srcWord in devAndTestSrcDict:
            X[ devAndTestSrcDict[srcWord] ] += 1
            
   phraseScore = 0.
   for word in tgtPhrase.split():
       if word in dwl:
           try:
               probs = dwl[word].predict_proba(X)
               phraseScore += math.log(probs[0][1]) - math.log(probs[0][0])
           except:
               pass
              
   return phraseScore
   
if __name__=='__main__':
    print get_dwl_score('a b c d', '1')