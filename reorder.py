import os
import sys
from fst import SimpleFst
from copy import deepcopy
import multiprocessing as mp
import math
from operator import itemgetter

import models

#Path to the language model
LM_PATH = 'data/lm'
#k-best outputs
K = 5
#number of parallel processes
NUM_PROC = 1
#number of best paths to try
NUM_BEST_TRY = 1000000000

'''
TODO:
4. Implement our semiring
'''

def print_fst(fst, phraseList, fileName):
    
    #make input symbols file
    isyms = open('isyms.txt', 'w')
    isyms.write("<s>"+' '+str(1)+'\n')
    for i, phrase in enumerate(phraseList):
        isyms.write('_'.join(phrase)+' '+str(i+2)+'\n')
    isyms.write("</s>"+' '+str(i+3)+'\n')
    isyms.close()
        
    #copy input symbols file to output symbols file
    os.system('cp isyms.txt osyms.txt')
    
    fst.write(fileName)
    os.system('fstdraw --isymbols=isyms.txt --osymbols=osyms.txt '+fileName+' '+fileName+'.dot')
    os.system('dot -Tpdf '+fileName+'.dot > '+fileName+'.pdf')
    sys.stderr.write('FST stored in '+fileName+'.pdf ...\n')

def pre_process(phraseList):
    
    wordDict = {}
    repeatedWords = {}
    for pIndex, phrase in enumerate(phraseList):
        for wIndex, word in enumerate(phrase):
            if word not in wordDict:
                wordDict[word] = 0
            elif word in repeatedWords:
                repeatedWords[word] += 1
                phraseList[pIndex][wIndex] = word+'~'+str(repeatedWords[word])
            else:
                repeatedWords[word] = 1
                phraseList[pIndex][wIndex] = word+'~1'
                
    return phraseList

def get_possible_bigrams(phraseList):
    
    bigrams = {}
    #add the initial bigram
    bigrams["<s> <s>"] = 0
    numStates = 1
    
    for i, phrase1 in enumerate(phraseList):
        for j, phrase2 in enumerate(phraseList):
            if len(phrase1) == 1:
                if phrase2[-1]+' '+phrase1[0] not in bigrams:
                    bigrams[ phrase2[-1]+' '+phrase1[0] ] = numStates
                    numStates += 1
            elif len(phrase1) >= 2:
                if phrase1[-2]+' '+phrase1[-1] not in bigrams:
                    bigrams[ phrase1[-2]+' '+phrase1[-1] ] = numStates
                    numStates += 1
    
    #make bigrams for transitions from the initial state
    startSym = '<s>'
    for phrase in phraseList:
        if len(phrase) == 1:
            if startSym+' '+phrase[0] not in bigrams:
                bigrams[ startSym+' '+phrase[0] ] = numStates
                numStates += 1
        elif len(phrase) >= 2:
            pass
    
    return bigrams
    
def get_next_state(currentState, nextPhrase, bigramList):
    
    layerNum = currentState / len(bigramList)
    bigramIndex = currentState % len(bigramList)
    
    for bigram, index in bigramList.iteritems():
        if index == bigramIndex:
            currentStateBigram = bigram
            break
    
    if len(nextPhrase) == 1:
        a, b = currentStateBigram.split()
        nextBigram = b+' '+nextPhrase[-1]
        
        #If the last word of the current state is the same as the next word
        #then it is most probably repeating the same words hence can be skipped
        #This is not always true though
        #Just comment the return and say "pass" if you do not want this filter
        if b == nextPhrase[-1]:
            return -1
        if a == nextPhrase[-1]:
            #return -1
            pass
        
    elif len(nextPhrase) >= 2:
        nextBigram = nextPhrase[-2]+' '+nextPhrase[-1]
        #If the last two words of the next phrase are same as the current state
        #then most probably it is repeating the phrase hence avoid it
        #Just comment the return and say "pass" if you do not want this filter
        if nextBigram == currentStateBigram:
            return -1
        if nextPhrase[-2] == currentStateBigram.split()[1]:
            #return -1
            pass
    
    if currentState == 0:
        nextStateNumber = (layerNum)*len(bigramList) + bigramList[nextBigram]
    else:
        nextStateNumber = (layerNum+1)*len(bigramList) + bigramList[nextBigram]
    
    return nextStateNumber
        
def get_lm_prob(currentState, nextPhrase, bigramList, lm):
    
    layerNum = currentState / len(bigramList)
    bigramIndex = currentState % len(bigramList)
    
    for bigram, index in bigramList.iteritems():
        found = False
        if index == bigramIndex:
            currentStateBigram = bigram
            found = True
            break
    
    if not found:
        print bigramIndex, len(bigramList)
        sys.stderr.write('\nError\n')
        sys.exit()
    
    (state, score) = lm.score_sequence( (currentStateBigram.split()[0], currentStateBigram.split()[1],), nextPhrase)
    return score
        
def create_fst(phraseList, bigramList, lm):
    
    fst = SimpleFst()
    
    numPhrases = len(phraseList)
    numBigrams = len(bigramList)
    
    #add start state
    fst.start = fst.add_state()
    
    #add all other states 
    #numPhrases(numLayers) x numBigrams-1 (start bigram not reqd)
    for i in range(numPhrases*(numBigrams-1)):
        fst.add_state()
    
    fst[numPhrases*(numBigrams-1)+2].final = 0.0
    
    #zero cost for starting the sentence
    fst.add_arc(1, 0, "<s>", "<s>", 0.0)
        
    #add arcs from starting state to the first layer
    prevLayerActiveStates = []
    for phrase in phraseList:
        nextState = get_next_state(0, phrase, bigramList)
        if nextState not in prevLayerActiveStates:
            prevLayerActiveStates.append(nextState)
        fst.add_arc(0, nextState, ' '.join(phrase), ' '.join(phrase), -1.*get_lm_prob(0, phrase, bigramList, lm))
    
    #add arcs from one layer to the next
    for layer in range(numPhrases-1):
        nextLayerActiveStates = []
        for currentState in prevLayerActiveStates:
            for phrase in phraseList:
                nextState = get_next_state(currentState, phrase, bigramList)
                if nextState not in nextLayerActiveStates and nextState != -1:
                    nextLayerActiveStates.append(nextState)
                if nextState != -1:
                    fst.add_arc(currentState, nextState, ' '.join(phrase), ' '.join(phrase), -1.*get_lm_prob(currentState, phrase, bigramList, lm))
        
        prevLayerActiveStates = []
        prevLayerActiveStates = deepcopy(nextLayerActiveStates)
    
    #add arcs from the last layer to the last state
    for currentState in prevLayerActiveStates:
        nextState = numPhrases*(numBigrams-1)+2
        fst.add_arc(currentState, nextState, "</s>", "</s>", -1.*get_lm_prob(currentState, ["</s>"], bigramList, lm))
    
    #remove unreachable states
    fst.connect()    
    return fst
    
def get_best_valid_path(fst, numPaths, phraseDict, lm, k):
    
    t = fst.shortest_path(numPaths)
    paths = t.paths()
    numValidPaths = 0
    outputSentences = {}
    
    for path in paths:
        arcLabels = []
        isValid = True
        for arc in path:
            #check that there is no repitition of phrases
            if arc.ilabel == 0:
                pass
            elif arc.ilabel in arcLabels:
                isValid = False
                break
            else:
                arcLabels.append(arc.ilabel)
                
        if isValid:
            sentence = ''
            numValidPaths += 1
            #Calculate the final lm_scroe for the valid path
            for arcLabel in arcLabels:
                if phraseDict[arcLabel] != '<s>' and phraseDict[arcLabel] != '</s>':
                    sentence += ' '.join(phraseDict[arcLabel].split('|||')) + ' '
            outputSentences[sentence.strip()] = -1.*lm.score_sequence(("<s>",), sentence.strip().split()+['</s>'])[1]
            
        if numValidPaths > k-1:
            break
            
    return outputSentences
            
def get_to_print_phrase_dict(phraseList):
    
    phraseDict = {}
    phraseDict[1] = '<s>'
    for phrase in phraseList:
        phraseDict[len(phraseDict)+1] = '|||'.join(phrase)
    phraseDict[len(phraseDict)+1] = '</s>'
    
    return phraseDict
    
def return_rearranged_sentences(line):
    
    global LM_PATH, K
    #read the language model
    lm = models.LM(LM_PATH)
    
    things = line.split(' ||| ')
    sentId, tm_score = things[0], float(things[1])
    
    phraseList = []
    for phrase in things[2:]:
        phraseList.append(phrase.split())
    
    if len(phraseList) > 10:
        sys.stderr.write("Too long\n")
        return
    
    phraseList = pre_process(phraseList)
    bigramList = get_possible_bigrams(phraseList)

    fst = create_fst(phraseList, bigramList, lm)
    
    #print_fst(fst, phraseList, str(sentId))
    printPhraseDict = get_to_print_phrase_dict(phraseList)
    
    outputSentences = get_best_valid_path(fst, NUM_BEST_TRY, printPhraseDict, lm, K)
    for sentence, lm_score in sorted(outputSentences.items(), key=itemgetter(1)):
        print sentId, '|||', tm_score, '|||', -1.*lm_score, '|||', sentence
        
    del lm, fst, phraseList, bigramList, printPhraseDict, outputSentences
    sys.stderr.write("done\n")
    
if __name__=='__main__':
    
    #lines = [line.strip() for line in sys.stdin]
    
    lines = ['0 ||| 0.1 ||| i ||| love ||| you ||| .']

    pool = mp.Pool(NUM_PROC)
    pool.map(return_rearranged_sentences, lines)
