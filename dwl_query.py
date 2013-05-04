import math
import numpy
import os
import pickle

from dwl import get_dev_test_dicts, get_train_dicts

#Make sure the parameters here are excetly the same as
#in the training file

OOV_LOG_PROB = -10.
FREQ_CUTOFF = 1
TRAIN_FILE = 'data/news-commentary-v8.fr-en.joint.filt'
DEV_FILE = 'data/newstest2011.fr.mixed'
TEST_FILE = 'data/newstest2012.fr.mixed'
MODELS_DIR = 'data/models/'

trainSrcDict, trainTgtDict = get_train_dicts(TRAIN_FILE)
devSrcDict = get_dev_test_dicts(DEV_FILE)
testSrcDict = get_dev_test_dicts(TEST_FILE)
    
devAndTestSrcDict = {}
for word in set(devSrcDict.keys()) | set(testSrcDict.keys()):
    if trainSrcDict[word]+devSrcDict[word]+testSrcDict[word] > FREQ_CUTOFF:
        devAndTestSrcDict[word] = len(devAndTestSrcDict)

# Returns the probability of occurrence of a phrase 
# in the target given source words
def get_dwl_score(srcWords, tgtPhrase):
    
   X = numpy.zeros((len(devAndTestSrcDict)), dtype=int)
   for srcWord in srcWords:
        if srcWord in devAndTestSrcDict:
            X[ devAndTestSrcDict[srcWord] ] += 1
            
   phraseScore = 0.
   for word in tgtPhrase.split():
       if word in trainTgtDict:
            try:
               fileName = MODELS_DIR+'word_'+word+'.pickle'
               word_clf = pickle.load(open(fileName, 'r'))
               probs = word_clf.predict_proba(X)
               phraseScore += math.log(probs[0][1]) #- math.log(probs[0][0])
            except:
                pass
       else:
            #If the word is OOV return a very small probability
            phraseScore += OOV_LOG_PROB
              
   return phraseScore
   
if __name__=='__main__':
    srcPhrase = raw_input("Enter the source phrase: ")
    
    #while True:
    targetWord = raw_input("Enter the target word: ")
    print math.exp(get_dwl_score(srcPhrase.split(), targetWord))