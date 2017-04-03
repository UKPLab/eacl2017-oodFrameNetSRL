import numpy as np

# Feature mappers convert graphs into matrices given lexicon and vsm


class FeatureMapper:
    def __init__(self, vsm, lexicon, multiword_averaging=False):
        self.vsm = vsm
        self.lexicon = lexicon
        self.multiword_averaging = multiword_averaging

    def get_repr(self, graph):
        raise NotImplementedError("Not implemented")

    def get_repr_sent(self, words, predicate_id):
        raise NotImplementedError("Not implemented")

    def get_matrix(self, graph_list):
        X = []
        y = []
        lemmapos = []
        gid = []
        for g in graph_list:
            X += [self.get_repr(g)]
            frame = g.get_predicate_head()["frame"]
            y += [self.lexicon.get_id(frame)]
            lemmapos += [g.get_predicate_head()["lemmapos"]]
            gid += [g.gid]
        X = np.vstack(X)
        y = np.array(y, dtype=np.int)
        return X, y, lemmapos, gid


class DummyMapper(FeatureMapper): # Dummy mapper for cases where no features are needed, e.g. for majority baselines
    def get_repr(self, graph):
        return np.zeros(self.vsm.dim)


def avg_embedding(wordlist, emb):
    res = []
    for word in wordlist:
        word = word.lower()
        res += [emb.get(word)]
    return np.mean(res, axis=0)


class SentenceBowMapper(FeatureMapper):
    def get_repr(self, graph):
        words = graph.sent.split(" ")
        if not self.multiword_averaging:
            predicate_head = graph.get_predicate_head()
            tgt_w = [predicate_head["word"].lower(), ]
        else:
            tgt_w = graph.get_predicate_node_words()
        return self.get_repr_sent(words, tgt_w)

    def get_repr_sent(self, words, tgt_w):
        return np.concatenate((avg_embedding(words, self.vsm), avg_embedding(tgt_w, self.vsm)), axis=0)


class DependentsBowMapper(FeatureMapper):
    def get_repr(self, graph):
        predicate_head = graph.get_predicate_head()
        deps = graph.get_direct_dependents(graph.predicate_head)
        parent = graph.G.predecessors(graph.predicate_head)
        if parent is not None and len(parent)>0:
            deps += [parent[0]]
        words = [graph.G.node[n]["word"].lower() for n in deps]
        if not self.multiword_averaging:
            tgt_w = [predicate_head["word"].lower(), ]
        else:
            tgt_w = graph.get_predicate_node_words()
        return np.concatenate((avg_embedding(words, self.vsm), avg_embedding(tgt_w, self.vsm)), axis=0)