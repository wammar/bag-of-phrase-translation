bag-of-phrase-translation
=========================



Future work
-----------

1. The reordering model is not exact, it compares paths which have covered
same number of phrases, but doesn't consider which phrase was at which position
in the source sentence.

Solution: Add future costs to the weights in the arcs.
