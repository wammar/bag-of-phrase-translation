bag-of-phrase-translation
=========================

Main ideas
----------
-
-

Secondary ideas
---------------
- use unigram lm in bof_decode
- use dwl in bof_decode
- use a commutative hash function to avoid duplicate hypotheses
- 

Future work
-----------

1. The reordering model is not exact, it compares paths which have covered
same number of phrases, but doesn't consider which phrase was at which position
in the source sentence.
2. Add future costs to the weights in the arcs.

1.21613955125
BEST WEIGHTS:
weights = {'fwd':10, 
           'bwd':0, 
           'fwd_lex':1, 
           'bwd_lex':1, 
           'dwl':0,
           'dwl_oov':0,
           'p_count':-10, 
           't_count':0, 
           '1lm':10}
