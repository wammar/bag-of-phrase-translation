import cdec.score
import io

def bleu(hyp_filename, ref_filename):
  with io.open(hyp_filename, encoding='utf8') as hyp, io.open(ref_filename, encoding='utf8') as ref:
    stats = sum(cdec.score.BLEU(r).evaluate(h) for h, r in zip(hyp, ref))
    print 'stats={0}'.format(stats)
    return stats.score

def dot_product(weights, features):
  assert(len(weights) == len(features))
  tbr = 0.0
  for k,w in weights.iteritems():
    tbr += w * features[k]
  return tbr

