# Evaluation routines


def acc(correct, total):
    return 1.0 * correct / total if total != 0 else 0


class Score:
    def __init__(self, skip_unknown_frames=True):
        self.total = 0
        self.correct = 0
        self.total_ambig = 0
        self.correct_ambig = 0
        self.total_unambig = 0
        self.correct_unambig = 0

        self.total_unknown = 0
        self.correct_unknown = 0

        # if the frame is missing in the lexicon AND in the training data, there is no system that will predict it.
        self.skip_unknown_frames = skip_unknown_frames

    def consume(self, correct, ambig, unknown, gold_frame):
        if self.skip_unknown_frames and gold_frame == -1:
            pass
        else:
            self.total += 1
            self.correct += int(correct)

            self.total_ambig += int(ambig)
            self.correct_ambig += int(ambig & correct)

            self.total_unambig += int(not ambig)
            self.correct_unambig += int(correct & (not ambig))

            self.total_unknown += int(unknown)
            self.correct_unknown += int(unknown & correct)

    def report_accuracies(self):
        print "Acc", acc(self.correct, self.total)
        print "Ambig", acc(self.correct_ambig, self.total_ambig)
        print "Unambig", acc(self.correct_unambig, self.total_unambig)
        print "Unknown", acc(self.correct_unknown, self.total_unknown)

    def report_counts(self):
        print "Total", self.total
        print "Correct", self.correct
        print "Total_ambig", self.total_ambig
        print "Correct_ambig", self.correct_ambig
        print "Total_unambig", self.total_unambig
        print "Correct_unambig", self.correct_unambig
        print "Total_unknown", self.total_unknown
        print "Correct_unknown", self.correct_unknown

    def report(self):
        print "=========================="
        self.report_accuracies()
        self.report_counts()
        print "=========================="

