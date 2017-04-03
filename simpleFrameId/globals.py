# pretrained embeddings
EMBEDDINGS_LEVY_DEPS_300 = 'deps.words.txt' # 174.015 words, 300 dim

# lexicons
LEXICON_FULL_BRACKETS_FIX = "fn1.5_full_lexicon_expanded"

# corpora
# full training sets
CORPUS_DAS_TRAIN = "train-das"
CORPORA_TRAIN = [CORPUS_DAS_TRAIN]

#test sets
CORPUS_DAS_TEST = "test-das"
CORPUS_YAGS_TEST = "test-yags"
CORPUS_YAGS_POSFIX_SPELL_TEST = "test-yags-posfix-spell"
CORPUS_YAGS_POSFIX_TEST = "test-yags-posfix"
CORPUS_MASC_TEST = "test-masc"
CORPUS_TW_G_TEST = "test-tw-g"
COPRUS_TW_M_TEST = "test-tw-m"
CORPUS_TW_S_TEST = "test-tw-s"
CORPORA_TEST = [CORPUS_DAS_TEST, CORPUS_YAGS_POSFIX_SPELL_TEST, CORPUS_YAGS_POSFIX_TEST, CORPUS_YAGS_TEST,
                CORPUS_MASC_TEST, CORPUS_TW_G_TEST, COPRUS_TW_M_TEST, CORPUS_TW_S_TEST]

CORPORA_ALL = CORPORA_TRAIN + CORPORA_TEST
