import models
import argparse
import sys
import io


parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-tmin')
parser.add_argument('-srcin')
parser.add_argument('-kbest')
parser.add_argument('-tmout')
opts = parser.parse_args()

# if a test set is provided, load all src tirgrams in it
NGRAM_FILTERING_THRESHOLD = 6
testset_ngrams = set()
sys.stderr.write("phrase table will be filtered for test set {0}...".format(opts.srcin))
for line in io.open(opts.srcin, encoding='utf8', mode='r'):
  tokens = line.strip().split()
  for phrase_len in range(1, NGRAM_FILTERING_THRESHOLD+1):
    for start_index in range(0, len(tokens)-phrase_len):
      _phrase = ' '.join(tokens[start_index:start_index+phrase_len])
      testset_ngrams.add(_phrase)
sys.stderr.write("done. ngrams count = {0}\n".format(len(testset_ngrams)))
        
sys.stderr.write("Reading translation model from %s...\n" % (opts.tmin,))
out_tm = io.open(opts.tmout, encoding='utf8', mode='w')
tm_size=0
for line in io.open(opts.tmin, encoding='utf8'):
  (f, e, logprobs) = line.strip().split(" ||| ")
  f_tokens = f.strip().split()

  # filter out irrelevant phrase pairs
  if ' '.join(f_tokens[0:NGRAM_FILTERING_THRESHOLD]) not in testset_ngrams:
    continue
  else:
    out_tm.write(line)
out_tm.close()

