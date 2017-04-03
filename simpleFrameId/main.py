from globals import *
from data import get_graphs
from extras import Lexicon, VSM
from representation import DependentsBowMapper, SentenceBowMapper, DummyMapper
from classifier import SharingDNNClassifier, DataMajorityBaseline, LexiconMajorityBaseline, WsabieClassifier
from evaluation import Score
from reporting import ReportManager
from config import Config
from resources import ResourceManager
import time
from numpy import random

HOME = "/home/local/UKP/martin/repos/frameID/"  # adjust accordingly

if __name__ == "__main__":

    random.seed(4)  # fix the random seed

    vsms = [EMBEDDINGS_LEVY_DEPS_300]  # vector space model to use
    lexicons = [LEXICON_FULL_BRACKETS_FIX]  # lexicon to use (mind the all_unknown setting!)
    multiword_averaging = [False]  # treatment of multiword predicates, false - use head embedding, true - use avg
    all_unknown = [False, True]  # makes the lexicon treat all LU as unknown, corresponds to the no-lex setting

    # WSABIE params
    num_components = [1500]
    max_sampled = [10]  # maximum number of negative samples used during WARP fitting 'warp'
    num_epochs = [500]

    configs = []
    for lexicon in lexicons:
        for all_unk in all_unknown:
            # DummyMapper doesn't do anything
            configs += [Config(DataMajorityBaseline, DummyMapper, lexicon, None, False, all_unk, None, None, None)]
            configs += [Config(LexiconMajorityBaseline, DummyMapper, lexicon, None, False, all_unk, None, None, None)]

    # Add configurations for NN classifiers
    for lexicon in lexicons:
        for vsm in vsms:
            for mwa in multiword_averaging:
                for all_unk in all_unknown:
                   configs += [Config(SharingDNNClassifier, SentenceBowMapper, lexicon, vsm, mwa, all_unk, None, None, None)]
                   configs += [Config(SharingDNNClassifier, DependentsBowMapper, lexicon, vsm, mwa, all_unk, None, None, None)]

    # Add configurations for WSABIE classifiers
    for lexicon in lexicons:
        for vsm in vsms:
            for mwa in multiword_averaging:
                for all_unk in all_unknown:
                    for num_comp in num_components:
                       for max_sampl in max_sampled:
                            for num_ep in num_epochs:
                                configs += [Config(WsabieClassifier, SentenceBowMapper, lexicon, vsm, mwa, all_unk, num_comp, max_sampl, num_ep)]
                                configs += [Config(WsabieClassifier, DependentsBowMapper, lexicon, vsm, mwa, all_unk, num_comp, max_sampl, num_ep)]

    print "Starting resource manager"
    sources = ResourceManager(HOME)

    print "Initializing reporters"
    reports = ReportManager(sources.out)

    print "Running the experiments!"
    runs = len(configs)*len(CORPORA_TRAIN)*len(CORPORA_TEST)
    print len(configs), "configurations, ", len(CORPORA_TRAIN)*len(CORPORA_TEST), " train-test pairs -> ", \
        runs, " runs"

    current_train = 0
    current_config = 0
    current_test = 0
    for corpus_train in CORPORA_TRAIN:
        current_train += 1
        current_config = 0

        g_train = get_graphs(*sources.get_corpus(corpus_train))
        reports.conll_reporter_train.report(g_train)

        for conf in configs:
            current_config += 1
            start_time = time.time()

            lexicon = Lexicon()
            # go to configuration, check which lexicon is needed, locate the lexicon in FS, load the lexicon
            lexicon.load_from_list(sources.get_lexicon(conf.get_lexicon()))
            reports.lexicon_reporter.report(lexicon)

            # same for VSM
            vsm = VSM(sources.get_vsm(conf.get_vsm()))
            mapper = conf.get_feat_extractor()(vsm, lexicon)

            # prepare the data
            X_train, y_train, lemmapos_train, gid_train = mapper.get_matrix(g_train)

            # train the model
            clf = conf.get_clf()(lexicon, conf.get_all_unknown(), conf.get_num_components(), conf.get_max_sampled(),
                                 conf.get_num_epochs())
            clf.train(X_train, y_train, lemmapos_train)

            current_test = 0
            for corpus_test in CORPORA_TEST:
                score = Score()  # storage for scores
                score_v = Score()  # storage for verb-only scores
                score_known = Score()  # storage for known lemma-only scores

                start_time = time.time()

                reports.set_config(conf, corpus_train, corpus_test)

                current_test += 1

                # prepare test data
                g_test = get_graphs(*sources.get_corpus(corpus_test))
                reports.conll_reporter_test.report(g_test)
                X_test, y_test, lemmapos_test, gid_test = mapper.get_matrix(g_test)

                # predict and compare
                for x, y_true, lemmapos, gid, g in zip(X_test, y_test, lemmapos_test, gid_test, g_test):
                    y_predicted = clf.predict(x, lemmapos)
                    correct = y_true == y_predicted

                    score.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
                    if lemmapos.endswith(".v"):
                        score_v.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
                    if not lexicon.is_unknown(lemmapos):
                        score_known.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)

                    reports.result_reporter.report(gid, g, lemmapos, y_predicted, y_true, lexicon)
                reports.summary_reporter.report(corpus_train, corpus_test, conf, score, time.time() - start_time)
                reports.summary_reporter_v.report(corpus_train, corpus_test, conf, score_v, time.time() - start_time)
                reports.summary_reporter_known.report(corpus_train, corpus_test, conf, score_known, time.time() - start_time)

                print "============ STATUS: - train", current_train, "/", len(CORPORA_TRAIN), \
                    "conf", current_config, "/", len(configs),\
                    "test", current_test, "/", len(CORPORA_TEST)









