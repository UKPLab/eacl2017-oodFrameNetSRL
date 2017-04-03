import codecs, os, shutil
from evaluation import acc

# Reporting classes

class ReportManager:
    def __init__(self, report_folder):
        if os.path.exists(report_folder):
            shutil.rmtree(report_folder)
        os.makedirs(report_folder)
        self.report_folder = report_folder
        self.result_reporter = ResultReporter(os.path.join(self.report_folder, "results"))
        self.lexicon_reporter = LexiconReporter(os.path.join(self.report_folder, "lexicon"))
        self.conll_reporter_train = ConllReporter(os.path.join(self.report_folder, "train.conll"))
        self.conll_reporter_test = ConllReporter(os.path.join(self.report_folder, "test.conll"))
        self.summary_reporter = ResultSummaryReporter(os.path.join(self.report_folder, "summary"))
        self.summary_reporter_v = ResultSummaryReporter(os.path.join(self.report_folder, "summary_v"))
        self.summary_reporter_known = ResultSummaryReporter(os.path.join(self.report_folder, "summary_known"))

    def set_config(self, config, train, test):
        self.result_reporter = ResultReporter(os.path.join(self.report_folder, "results_"+train+"_"+test+"_"+str(config)))
        self.lexicon_reporter = LexiconReporter(os.path.join(self.report_folder, "lexicon_"+config.lexicon if config.lexicon is not None else "NA"))


class Reporter(object):
    def __init__(self, out_path):
        self.out = codecs.open(out_path, "w", "utf-8")
        if hasattr(self, 'columns'):
            self.write_header()

    def write_header(self):
        self.out.write("\t".join(self.columns)+"\n")
    def close(self):
        self.out.close()


class ResultReporter(Reporter):
    def __init__(self, out_path):
        self.columns = ["gid", "sent", "lemmapos", "pos", "predicted_id", "true_id", "predicted_frame", "true_frame", "ambig", "unknown"]
        super(self.__class__, self).__init__(out_path)

    def report(self, instance_id, g, lemmapos, predicted, true, lexicon):
        self.out.write("\t".join([str(instance_id), g.sent,
                                  lemmapos, lemmapos.split(".")[1],
                                  str(predicted), str(true), lexicon.get_frame(predicted), lexicon.get_frame(true),
                                  str(lexicon.is_ambiguous(lemmapos)), str(lexicon.is_unknown(lemmapos))])+"\n")


class ResultSummaryReporter(Reporter):
    def __init__(self, out_path):
        self.columns = ["train", "test", "clf", "feats", "lex", "vsm", "MWE_avg", "all_unk", "num_components", "max_sampled", "num_epochs", "total", "correct", "ambig", "ambig_correct", "unambig", "unambig_correct", "unk", "unk_correct",
                        "total_acc", "ambig_acc", "unambig_acc", "unk_acc", "time"]
        super(self.__class__, self).__init__(out_path)

    def report(self, train, test, config, score, time_delta):
        self.out.write(
            "\t".join([train, test, config.clf.__name__, config.feat_extractor.__name__, config.lexicon if config.lexicon is not None else "NA",
                       config.vsm if config.vsm is not None else "NA", str(config.multiword_averaging), str(config.all_unknown), 
                       str(config.num_components) if config.num_components is not None else "NA", 
                       str(config.max_sampled) if config.max_sampled is not None else "NA", 
                       str(config.num_epochs) if config.num_epochs is not None else "NA", 
                       str(score.total), str(score.correct), str(score.total_ambig), str(score.correct_ambig), str(score.total_unambig),
                       str(score.correct_unambig), str(score.total_unknown), str(score.correct_unknown),
                       str(acc(score.correct, score.total)), str(acc(score.correct_ambig, score.total_ambig)),
                       str(acc(score.correct_unambig, score.total_unambig)), str(acc(score.correct_unknown, score.total_unknown)),
                       str(time_delta)])+"\n"
        )


class LexiconReporter(Reporter):
    def __init__(self, out_path):
        self.columns = ["lemma", "frames"]
        super(self.__class__, self).__init__(out_path)

    def report(self, lexicon):
        for lemma in lexicon.frameLexicon:
            self.out.write("\t".join([lemma, ", ".join([str(lexicon.get_id(frame))+": "+frame for frame in lexicon.frameLexicon[lemma]])]) + "\n")


class ConllReporter(Reporter):
    def report(self, graphs):
        for g in graphs:
            self.out.write(g.pretty() + "\n")