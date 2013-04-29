
BestMerge = namedtuple('BetMerge', 'left_index, right_index, unigrams, bigrams, trigrams, fourgrams')

# phrases is a list of unicode strings, refs is also a list of unicode strings
def get_oracle_reordering(_phrases, refs):
  phrases = list(_phrases)
  for step in range(0, len(phrases)):
    best = (left_index=0, right_index=1, unigrams=0, bigrams=0, trigrams=0, fourgrams=0)
    for left_index in range(0, len(phrases)):
      for right_index in range(0, len(phrases)):
        if left_index == right_index:
          continue
        best = update_best_merge(best, left, right, refs)
    # finished considering all merges in this step
    left_phrase, right_phrase = phrases[best.left_index], phrases[best.right_index]
    del phrases[ min(best.left_index, best.right_index) ]
    del phrases[ max(best.left_index, best.right_index) ]
    phrases.append(' '.join([left_phrase, right_phrase]))
  assert(len(phrases) == 1)
  return phrases[0]
