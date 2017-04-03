import numpy as np
import codecs

# Extra classes for managing external resources


class Lexicon:  # Lexicon manager. Stores information about lemma.pos -> frame mappings
    def __init__(self):
        self.frameLexicon = {}
        self.frameToId = {}
        self.idToFrame = {}
        self.source = "NA"

    def get_id(self, frame):
        if frame not in self.frameToId:
            print "Unknown frame", frame, "assigning id=-1"
        return self.frameToId.get(frame, -1)

    def get_available_frame_ids(self, lemmapos):
        return [self.frameToId[x] for x in self.frameLexicon.get(lemmapos, [])]

    def get_all_frame_ids(self):
        return list(self.idToFrame.keys())

    def get_frame(self, id):
        return self.idToFrame.get(id, "UNKNOWN_FRAME")

    # Load from pre-defined lexicon in format [frame \t lemmapos]
    def load_from_list(self, src):
        with codecs.open(src, "r", "utf-8") as f:
            frames = []
            for line in f:
                frame, lemmapos = line.strip().rstrip().split("\t")
                self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
                frames += [frame]
        frames = list(set(frames))
        self.frameToId = {frames[i]:i for i in range(len(frames))}
        self.idToFrame = {y:x for (x,y) in self.frameToId.items()}
        self.source = src.split("/")[-1]

    def is_unknown(self, lemmapos):
        return lemmapos not in self.frameLexicon

    def is_ambiguous(self, lemmapos):
        return len(self.frameLexicon.get(lemmapos, []))>1

    # Load from training data
    def load_from_graphs(self, g_train):
        frames = []
        for g in g_train:
            predicate = g.get_predicate_head()
            lemmapos = predicate["lemmapos"]
            frame = predicate["frame"]
            self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
            frames += [frame]
        frames = list(set(frames))
        self.frameToId = {frames[i]: i for i in range(len(frames))}
        self.idToFrame = {y: x for (x, y) in self.frameToId.items()}
        self.source = "training_data"


class VSM:
    def __init__(self, src):
        self.map = {}
        self.dim = None
        self.source = src.split("/")[-1] if src is not None else "NA"
        # create dictionary for mapping from word to its embedding
        if src is not None:
            with open(src) as f:
                i = 0
                for line in f:
                    word = line.split()[0]
                    embedding = line.split()[1:]
                    self.map[word] = np.array(embedding, dtype=np.float32)
                    i += 1
                self.dim = len(embedding)
        else:
            self.dim = 1

    def get(self, word):
        word = word.lower()
        if word in self.map:
            return self.map[word]
        else:
            return np.zeros(self.dim, dtype=np.float32)