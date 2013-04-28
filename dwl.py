import sys
import numpy
import pickle
import multiprocessing as mp
import math

from operator import itemgetter
from scipy.sparse import coo_matrix, vstack
from collections import Counter
from sklearn.linear_model import LogisticRegression

FREQ_CUTOFF = 1
TRAIN_FILE = 'data/news-commentary-v8.fr-en.joint.filt'
DEV_FILE = 'data/newstest2011.fr.mixed'
TEST_FILE = 'data/newstest2012.fr.mixed'

NUM_PROC = 5

global X, devAndTestSrcDict, sizeTrData

def get_dev_test_dicts(fileName):
    
    srcDict = Counter()
    for line in open(fileName, 'r'):
        src = line.strip()
        for word in src.strip().split():
            srcDict[word] += 1
        
    sys.stderr.write("Dict size: "+str(len(srcDict))+"\n")
    return srcDict

def get_train_dicts(fileName):
    
    srcDict = Counter()
    tgtDict = Counter()
    errorLines = 0
    
    for line in open(fileName, 'r'):
        
        try:
            src, tgt = line.strip().split('|||')
        except:
            errorLines += 1
            continue
        
        for word in src.strip().split():
            srcDict[word] += 1
            
        for word in tgt.strip().split():
            tgtDict[word] += 1
        
    sys.stderr.write("File processed with "+str(errorLines)+" erroneous lines\n")
    return srcDict, tgtDict

def get_numlines(fileName):
    
    numLines = 0
    for line in open(fileName, 'r'):
        numLines += 1
        
    return 1.*numLines

def set_src_feat_vector():
    
    global X, sizeTrData, devAndTestSrcDict, TRAIN_FILE
    
    sizeTrData = get_numlines(TRAIN_FILE)
    X1 = numpy.zeros((math.ceil(sizeTrData/2), len(devAndTestSrcDict)), dtype=int)
    
    for numLine, line in enumerate(open(TRAIN_FILE, 'r')):
        
        if numLine == math.ceil(sizeTrData/2):
            break
        
        src, tgt = line.strip().split('|||')
        for word in src.split():
            if word in devAndTestSrcDict:
                X1[numLine][devAndTestSrcDict[word]] += 1
                
    X1 = coo_matrix(X1)
    X2 = numpy.zeros((sizeTrData-math.ceil(sizeTrData/2), len(devAndTestSrcDict)), dtype=int)
    
    for numLine, line in enumerate(open(TRAIN_FILE, 'r')):
        
        if numLine >= math.ceil(sizeTrData/2):
            src, tgt = line.strip().split('|||')
            for word in src.split():
                if word in devAndTestSrcDict:
                    X2[numLine-math.ceil(sizeTrData/2)][devAndTestSrcDict[word]] += 1
                
    X2 = coo_matrix(X2)
    X = vstack([X1, X2])

def train_dwl(targetWord):
    
    global X, TRAIN_FILE, sizeTrData
    
    Y = numpy.zeros((sizeTrData), dtype=int)
    shouldTrain = False
    
    for lineNum, line in enumerate(open(TRAIN_FILE, 'r')):
        src, tgt = line.strip().split('|||')
        if targetWord in tgt.strip().split():
            shouldTrain = True
            Y[ lineNum ] = 1
            
    if not shouldTrain:
        return
    
    try:
        outModelFile = open('data/models/word_'+targetWord+'.pickle', 'w')
        clf = LogisticRegression()
        clf.fit(X,Y)
        pickle.dump(clf, outModelFile)
    except:
        pass

if __name__=='__main__':
    
    global devAndTestSrcDict
    
    trainSrcDict, trainTgtDict = get_train_dicts(TRAIN_FILE)
    devSrcDict = get_dev_test_dicts(DEV_FILE)
    testSrcDict = get_dev_test_dicts(TEST_FILE)
    
    devAndTestSrcDict = {}
    for word in set(devSrcDict.keys()) | set(testSrcDict.keys()):
        if trainSrcDict[word]+devSrcDict[word]+testSrcDict[word] > FREQ_CUTOFF:
            devAndTestSrcDict[word] = len(devAndTestSrcDict)
    
    trainTargetWords = [key for key in trainTgtDict.keys() if trainTgtDict[key] > FREQ_CUTOFF]
    sys.stderr.write("Feature size: "+str(len(devAndTestSrcDict))+"\n")
    sys.stderr.write("Num target words: "+str(len(trainTargetWords))+"\n")
    
    pool = mp.Pool(NUM_PROC, set_src_feat_vector)
    pool.map(train_dwl, trainTargetWords)
