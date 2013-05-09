#!/usr/bin/env python
# Simple translation model and language model data structures
import sys
import io
from collections import namedtuple, defaultdict
from math import log
from utils import dot_product

# A translation model is a dictionary where keys are tuples of French words
# and values are lists of (english, logprob) named tuples. For instance,
# the French phrase "que se est" has two translations, represented like so:
# tm[('que', 'se', 'est')] = [
#     phrase(english='what has', logprob=-0.301030009985), 
#     phrase(english='what has been', logprob=-0.301030009985)]
# k is a pruning parameter: only the top k translations are kept for each f.
phrase = namedtuple("phrase", "english, fwd, bwd, fwd_lex, bwd_lex")
def TM(filename, k, weights, testset=None):
    _weights = {'fwd':weights['fwd'],
                'bwd':weights['bwd'],
                'fwd_lex':weights['fwd_lex'],
                'bwd_lex':weights['bwd_lex']}
    # if a test set is provided, load all src tirgrams in it
    NGRAM_FILTERING_THRESHOLD = 6
    testset_ngrams = set()
    if testset != None:
        sys.stderr.write("phrase table will be filtered for test set {0}...".format(testset))
        for line in io.open(testset, encoding='utf8', mode='r'):
            tokens = line.strip().split()
            for phrase_len in range(1, NGRAM_FILTERING_THRESHOLD+1):
                for start_index in range(0, len(tokens)-phrase_len):
                    _phrase = ' '.join(tokens[start_index:start_index+phrase_len])
                    testset_ngrams.add(_phrase)
        sys.stderr.write("done. ngrams count = {0}\n".format(len(testset_ngrams)))
        
    sys.stderr.write("Reading translation model from %s...\n" % (filename,))
    tm = {}
    tm_size=0
    for line in io.open(filename, encoding='utf8'):
        (f, e, logprobs) = line.strip().split(" ||| ")
        f_tokens = f.strip().split()
        
        # filter out irrelevant phrase pairs
#        assert(testset != None)
#        sys.stderr.write(u'this thing is weird: {0}'.format(' '.join(f_tokens[0:NGRAM_FILTERING_THRESHOLD])))
        if ' '.join(f_tokens[0:NGRAM_FILTERING_THRESHOLD]) not in testset_ngrams:
            #sys.stderr.write(u'this phrase was eliminated: {0}\n'.format(f))
            continue
        
        if tm_size > 0 and tm_size % 100000 == 0:
            sys.stderr.write('tm_size={0}\n'.format(tm_size))

        logprobs = logprobs.strip().split()
        bwd = log(float(logprobs[0]))
        bwd_lex = log(float(logprobs[1]))
        fwd = log(float(logprobs[2]))
        fwd_lex = log(float(logprobs[3]))
        p = phrase(e, fwd, bwd, fwd_lex, bwd_lex)
        f_tuple = tuple(f_tokens)
        if f_tuple in tm:
            tm[f_tuple].append(p)
        else:
            tm[f_tuple] = [p]
        tm_size += 1

    for f in tm: # prune all but top k translations
        tm_size -= len(tm[f])
        tm[f].sort(key=lambda x: dot_product({'fwd':x.fwd, 
                                              'bwd':x.bwd, 
                                              'fwd_lex':x.fwd_lex, 
                                              'bwd_lex':x.bwd_lex}, _weights))
        del tm[f][k:]
        tm_size += len(tm[f])
    sys.stderr.write('final tm_size={0}\n'.format(tm_size))
    return tm

# # A language model scores sequences of English words, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = models.LM(filename)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for word in sentence.split():
#     (lm_state, word_logprob) = lm.score(lm_state, word)
#     logprob += word_logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
# unigram lm
class LM:
    def __init__(self, filename):
        sys.stderr.write("Reading language model from %s...\n" % (filename,))
        self.table = {}
        for line in io.open(filename, encoding='utf8'):
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))
                self.table[ngram] = ngram_stats(logprob, backoff)

    def begin(self):
        return ("<s>",)

    def score(self, word):
        ngram = (word,)
        if ngram in self.table:
            return self.table[ngram].logprob
        else: #backoff
            return self.table[("<unk>",)].logprob
        
    def score_sequence(self, word_list):
        assert(len(word_list) > 0)
        total_score = 0.0
        for word in word_list:
            word_score = self.score(word)
            total_score += word_score
        return total_score

