from globals import *
from data import get_graphs
from resources import ResourceManager
from reporting import ConllReporter

def check_corpora_read_ok(sources, out):
    print "Checking datasets"

    # set corpora to test gere
    for corpus in [CORPUS_YAGS_TEST, CORPUS_DAS_TRAIN, CORPUS_DAS_TEST,
                   CORPUS_YAGS_TEST, CORPUS_MASC_TEST, CORPUS_TW_G_TEST, COPRUS_TW_M_TEST, CORPUS_TW_S_TEST]:
        g = get_graphs(*sources.get_corpus(corpus), verbose=False)
        reporter = ConllReporter(out+corpus+".conll")
        reporter.report(g)


if __name__ == "__main__":
    src = "your/path/here"
    root = ResourceManager(src)
    check_corpora_read_ok(root, "your/path/here/tmp")