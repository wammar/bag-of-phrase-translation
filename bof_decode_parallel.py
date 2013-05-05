#!/usr/bin/env python
import argparse
import sys
import models
import heapq
import io
import os
import multiprocessing as mp
from collections import namedtuple, defaultdict
from hash_string import hash_string, hash_merge
from oracle_reordering import get_oracle_reordering, get_best_translation
from utils import dot_product
import dwl_query
from dwl_query import get_dwl_score
from datetime import datetime

global OOV_LOGPROB, weights, Hypothesis, Node, tm, lm, ref_sents, NUM_PROC

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/dev/newstest2011.fr.tok', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-r', '--ref', dest='ref', default='data/dev/newstest2011.en.tok', help='File containing reference translations')
parser.add_argument('-o', '--output', dest='output', default='data/bof/newstest2011.fr.bof', help='(output) File containing kbest bag-of-phrase translations')
parser.add_argument('-oracle', '--oracle', dest='oracle', default='data/oracle/newstest2011.fr.oracle', help='(output) File containing kbest oracle-reordered translations')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-kbest', '--kbest', dest='kbest', default=1, type=int, help='size of the kbest list')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/nc.1.lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Verbose mode (default=off)')
opts = parser.parse_args()

def merge_features(f1, f2):
  tbr = {'fwd': f1['fwd'] + f2['fwd'],
         'bwd': f1['bwd'] + f2['bwd'],
         'fwd_lex': f1['fwd_lex'] + f2['fwd_lex'],
         'bwd_lex': f1['bwd_lex'] + f2['bwd_lex'],
         'dwl':f1['dwl'] + f2['dwl'],
         'dwl_oov':f1['dwl_oov'] + f2['dwl_oov'],
         'p_count':f1['p_count'] + f2['p_count'], 
         't_count':f1['t_count'] + f2['t_count'], 
         '1lm':f1['1lm'] + f2['1lm']}
  return tbr

def draw(f, tree):
  UNIT_WIDTH = 6
  for span_len in reversed(range(1, len(f)+1)):
    sys.stderr.write(u'len={0} '.format(span_len))
    for i in range(span_len*UNIT_WIDTH/2):
      sys.stderr.write(u' ')
    for from_src_pos in range(0, len(f)-span_len+1):
      sys.stderr.write(u' /{0}\ '.format(len(tree[(from_src_pos, from_src_pos + span_len)].hyps)))
    sys.stderr.write(u'\n\n\n')
  

def extract_phrase_bag(f, hyp, pieces):
  if hyp.tgt_phrase != None:
#    pieces.append(' '.join(f[hyp.from_src_pos:hyp.to_src_pos]))
    pieces.append(hyp.tgt_phrase)
    return
  extract_phrase_bag(f, hyp.left_child, pieces)
  extract_phrase_bag(f, hyp.right_child, pieces)
  
def initialise_global_vars():
    
    global OOV_LOGPROB, weights, Hypothesis, Node, tm, lm, ref_sents
    
    OOV_LOGPROB = -10.0
    weights = {'fwd':10, 
               'bwd':0, 
               'fwd_lex':1,
               'bwd_lex':1, 
               'dwl':0,
               'dwl_oov':0,
               'p_count':0, 
               't_count':0, 
               '1lm':10}
    Hypothesis = namedtuple('Hypothesis', 'logprob, features, from_src_pos, to_src_pos, tgt_phrase, left_child, right_child, hash')
    Node = namedtuple('Node', 'from_src_pos, to_src_pos, hyps')
    
    MAX_PHRASE_OPTIONS = opts.kbest
    
    tm = models.TM(opts.tm, MAX_PHRASE_OPTIONS, weights, opts.input)
    lm = models.LM(opts.lm)
    sys.stderr.write('Decoding %s...\n' % (opts.input,))
      
    ref_sents = [tuple(line.strip().split(' ||| ')) for line in io.open(opts.ref, encoding='utf8').readlines()[:opts.num_sents]] if opts.ref != "None" else None
      
def translate(input_sent):
    
    global OOV_LOGPROB, weights, Hypothesis, Node, tm, lm, ref_sents
    
    f, sent_counter = input_sent
    
    output_file = io.open('{0}.{1}.{2}best'.format(opts.output, opts.kbest, sent_counter), encoding='utf8', mode='w')
    if ref_sents:
      oracle_file = io.open('{0}.{1}.{2}best'.format(opts.oracle, opts.kbest, sent_counter), encoding='utf8', mode='w')
    
    sys.stderr.write(u'\n===========================================================================================\n')
    sys.stderr.write(u'now processing sent # {0}: {1}\n\n'.format(sent_counter, ' '.join(f)))
    
    # create a dictionary that maps (from_src_pos,to_src_pos) pairs to the corresponding Node
    tree = defaultdict(Node)
    
    # process lower cells first
    for span_len in range(1, len(f)+1):
      # process cells left to right
      #sys.stderr.write(u'processing span length {0}\n'.format(span_len))
      for from_src_pos in range(0, len(f)-span_len+1):
        # now, create the cell
        to_src_pos = from_src_pos+span_len
        #sys.stderr.write(u'processing cell tree[({0}, {1})]...'.format(from_src_pos, to_src_pos))
        tree[(from_src_pos, to_src_pos)] = Node(from_src_pos=from_src_pos, to_src_pos=to_src_pos, hyps=[])

        # TODO: this creates empty list for all possible src spans. bad. 
        # instead, check first whether the key exists in tm.
        # also, consider making tm a dictionary instead of a defaultdict
      
        # src phrase in tm?
        if f[from_src_pos:to_src_pos] not in tm:
          # src phrase is oov?
          if span_len == 1:
            features = {'fwd':OOV_LOGPROB, 
                        'bwd':OOV_LOGPROB, 
                        'fwd_lex':OOV_LOGPROB, 
                        'bwd_lex':OOV_LOGPROB,
                        'dwl':dwl_query.OOV_LOG_PROB,
                        'dwl_oov':1.0,
                        'p_count':1.0,
                        't_count':1.0,
                        '1lm':OOV_LOGPROB}
            tree[(from_src_pos, to_src_pos)].hyps.append( Hypothesis(logprob=dot_product(weights, features), 
                                                                     features=features,
                                                                     from_src_pos=from_src_pos,
                                                                     to_src_pos=to_src_pos,
                                                                     tgt_phrase=f[from_src_pos],
                                                                     left_child=None,
                                                                     right_child=None,
                                                                     hash=hash_string(f[from_src_pos])))
            # end of oov handling 
          # not an oov, synthetic phrases will be added later
          else:
            pass
          # end of src phrase not in tm
        else:
          # add phrase pairs to the hypotheses
          for phrase in tm[f[from_src_pos:to_src_pos]]:
            _dwl = get_dwl_score(f, phrase.english) if weights['dwl'] != 0 else 0.0
            features = {'fwd':phrase.fwd, 
                        'bwd':phrase.bwd,
                        'fwd_lex':phrase.fwd_lex,
                        'bwd_lex':phrase.bwd_lex,
                        'dwl':_dwl,
                        'dwl_oov':1.0 if _dwl == dwl_query.OOV_LOG_PROB else 0.0,
                        'p_count':1.0,
                        't_count':phrase.english.count(' ')+1,
                        '1lm':lm.score_sequence(phrase.english.split())}
            tree[(from_src_pos, to_src_pos)].hyps.append( Hypothesis(logprob=dot_product(weights, features),
                                                                     features=features,
                                                                     from_src_pos=from_src_pos,
                                                                     to_src_pos=to_src_pos,
                                                                     tgt_phrase=phrase.english, 
                                                                     left_child=None, 
                                                                     right_child=None,
                                                                     hash=hash_string(phrase.english)))
        # end of src phrase in tm
        # end of tm lookup for src phrase

        # find the kbest synthetic hypotheses
        synthetic_hash_values=set()
        for mid_src_pos in range(from_src_pos + 1, to_src_pos):
          # for every way the current span can be split
          for left_hyp in tree[(from_src_pos, mid_src_pos)].hyps:
            for right_hyp in tree[(mid_src_pos, to_src_pos)].hyps:
              # consider all combinations 
              # TODO: you can stop after greedily combining kBEST hyps 
              features = merge_features(left_hyp.features, right_hyp.features)
              new_hyp = Hypothesis(logprob=dot_product(weights, features),
                                   features=features,
                                   from_src_pos=from_src_pos,
                                   to_src_pos=to_src_pos,
                                   tgt_phrase=None,
                                   left_child=left_hyp,
                                   right_child=right_hyp,
                                   hash=hash_merge(left_hyp.hash, right_hyp.hash))
              # TODO: instead of relying on the hash value only to determine duplicates, you can also rely on the logprob, after quantization
              # remove duplicate synthetic hypotheses
              if new_hyp.hash not in synthetic_hash_values:
                tree[(from_src_pos, to_src_pos)].hyps.append(new_hyp)
                synthetic_hash_values.add(new_hyp.hash)
              
        # now sort and prune
        tree[(from_src_pos, to_src_pos)].hyps.sort(reverse=True)
        #sys.stderr.write(u'created {0} hyps before pruning...'.format( len(tree[(from_src_pos, to_src_pos)].hyps) ))
        del tree[(from_src_pos, to_src_pos)].hyps[opts.kbest:]
        #sys.stderr.write(u'{0} hyps left after pruning.\n'.format( len(tree[(from_src_pos, to_src_pos)].hyps) ))
      # finished processing all cells with span_len
    # finished processing all spans
  
    # print tree
    #draw(f, tree)

    # now, spit out the kbest bag-of-phrase translations
    kbest_oracles = []
    sys.stderr.write(u'{0}-best translations:\n'.format(opts.kbest))
    for hyp in tree[(0, len(f))].hyps:
      pieces = []
      extract_phrase_bag(f, hyp, pieces)
      bof_translation = u'{0} ||| {1} ||| {2}\n'.format(sent_counter, hyp.logprob, ' ||| '.join(pieces))
      output_file.write(bof_translation)
      sys.stderr.write(bof_translation)  
      if ref_sents:
        best_reordering = get_oracle_reordering(pieces, ref_sents[sent_counter])
        kbest_oracles.append(best_reordering)
        sys.stderr.write(best_reordering+u'\n')
  
    # now, out of the kbest-oracle-reorderings, select the one that would give the highest best bleu score
    if ref_sents:
      oracle_one_best = get_best_translation(kbest_oracles, ref_sents[sent_counter])
      oracle_file.write(u'{0}\n'.format(oracle_one_best))
      sys.stderr.write(u'the oracle_one_best translation is: {0}\n'.format(oracle_one_best))

    #close the output file
    output_file.close()
    if ref_sents:
      oracle_file.close()
    
if __name__=='__main__':
    
    os.system('mkdir data/bof/')
    os.system('mkdir data/oracle/')
    
    NUM_PROC = 10
    #input sentence now contains (input_sent, sent_num)
    input_sents = [(tuple(line.strip().split()), lineNum) for lineNum, line in enumerate(io.open(opts.input, encoding='utf8').readlines()[:opts.num_sents])]
    
    pool = mp.Pool(NUM_PROC, initialise_global_vars)
    pool.map(translate, input_sents)
    
    for sent_num in range(len(input_sents)):
        if sent_num == 0:
            os.system('cat data/bof/newstest2011.fr.bof.'+str(opts.kbest)+'.'+str(sent_num)+'best > data/newstest2011.fr.bof')
            os.system('cat data/oracle/newstest2011.fr.oracle.'+str(opts.kbest)+'.'+str(sent_num)+'best > data/newstest2011.fr.oracle')
        else:
            os.system('cat data/bof/newstest2011.fr.bof.'+str(opts.kbest)+'.'+str(sent_num)+'best >> data/newstest2011.fr.bof')
            os.system('cat data/oracle/newstest2011.fr.oracle.'+str(opts.kbest)+'.'+str(sent_num)+'best >> data/newstest2011.fr.oracle')