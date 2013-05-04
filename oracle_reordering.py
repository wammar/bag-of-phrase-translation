from collections import namedtuple, defaultdict
import sys

Merge = namedtuple('Merge', 'left_index, right_index, unigrams, bigrams, trigrams, fourgrams')
Translation = namedtuple('Translation', 'translation, unigrams, bigrams, trigrams, fourgrams')

def count_matches(query, text):
  count = 0
  start_index = 0
  while True:
    start_index = text.find(query, start_index)
    if start_index == -1:
      break
    start_index += 1
    count += 1
  return count

def update_best_merge(prev_best, phrases, left_index, right_index, refs):
  ngram_matches = [0, 0, 0, 0]
  combined_phrase = phrases[left_index] + u' ' + phrases[right_index]
  combined_phrase_words = combined_phrase.split()
  for ref in refs:
    for size in range(1, min(5, len(combined_phrase_words)+1)):
      for from_index in range(0, len(combined_phrase_words)-size+1):
        partial_phrase = ' '.join(combined_phrase_words[from_index:from_index+size])
        count = count_matches(partial_phrase, ref)
        ngram_matches[size-1] += count
  new_merge = Merge(left_index, right_index, ngram_matches[0], ngram_matches[1], ngram_matches[2], ngram_matches[3])
  if ngram_matches[3] > prev_best.fourgrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] > prev_best.trigrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] == prev_best.trigrams and ngram_matches[1] > prev_best.bigrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] == prev_best.trigrams and ngram_matches[1] == prev_best.bigrams and ngram_matches[0] > prev_best.unigrams:
    return new_merge
  else:
    return prev_best    

def update_best_translation(prev_best, translation, refs):
  ngram_matches = [0, 0, 0, 0]
  translation_words = translation.split()
  for ref in refs:
    for size in range(1, min(5, len(translation_words)+1)):
      for from_index in range(0, len(translation_words)-size+1):
        partial_phrase = ' '.join(translation_words[from_index:from_index+size])
        count = count_matches(partial_phrase, ref)
        ngram_matches[size-1] += count
  new_translation = Translation(translation, ngram_matches[0], ngram_matches[1], ngram_matches[2], ngram_matches[3])
  if ngram_matches[3] > prev_best.fourgrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] > prev_best.trigrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] == prev_best.trigrams and ngram_matches[1] > prev_best.bigrams or \
        ngram_matches[3] == prev_best.fourgrams and ngram_matches[2] == prev_best.trigrams and ngram_matches[1] == prev_best.bigrams and ngram_matches[0] > prev_best.unigrams:
    return new_translation
  else:
    return prev_best    

# phrases is a list of unicode strings, refs is also a list of unicode strings
def get_oracle_reordering(_phrases, refs):
  phrases = list(_phrases)
  for step in range(0, len(phrases)-1):
    assert( len(phrases) == len(_phrases)-step )
    best = Merge(left_index=0, right_index=1, unigrams=0, bigrams=0, trigrams=0, fourgrams=-1)
    for left_index in range(0, len(phrases)):
      for right_index in range(0, len(phrases)):
        if left_index == right_index:
          continue
        best = update_best_merge(best, phrases, left_index, right_index, refs)
    # finished considering all merges in this step
    left_phrase, right_phrase = phrases[best.left_index], phrases[best.right_index]
    del phrases[ max(best.left_index, best.right_index) ]
    del phrases[ min(best.left_index, best.right_index) ]
    phrases.append(' '.join([left_phrase, right_phrase]))
  assert(len(phrases) == 1)
  return phrases[0]

def get_best_translation(kbest, refs):
  assert(len(kbest) > 0)
  best = Translation(translation=kbest[0], unigrams=0, bigrams=0, trigrams=0, fourgrams=-1)
  for i in range(0, len(kbest)):
    best = update_best_translation(best, kbest[i], refs)
  return best.translation
