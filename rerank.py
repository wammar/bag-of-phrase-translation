import features
from operator import itemgetter
import multiprocessing as mp

global SRC_SENTS, K_BEST_SENTS, W

def get_best_hypothesis(refNum):
    
    global SRC_SENTS, K_BEST_SENTS, W
    
    (bestScore, bestHyp) = (-1e300, '')
    for (hyp, feat) in K_BEST_SENTS[refNum].values():
        score = 0.
        newFeat = feat + features.add_features(SRC_SENTS[refNum], hyp)
        for w, f in zip(W, newFeat):
            score += w * f
            
        if score > bestScore:
            (bestScore, bestHyp) = (score, hyp)
            
    return (refNum, bestHyp)

def rerank(srcSents, tgtSents, kBestSents, weights):
    
    global SRC_SENTS, K_BEST_SENTS, W
    
    (SRC_SENTS, K_BEST_SENTS, W) = (srcSents, kBestSents, weights)
    
    pool = mp.Pool(15)
    refNums = [refNum for refNum, ref in sorted(tgtSents.items(), key=itemgetter(0))]
    for (refNum, bestHyp) in pool.imap(get_best_hypothesis, refNums):
        print bestHyp
