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

Solution: Add future costs to the weights in the arcs.
