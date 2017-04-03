import codecs, sys
from graph import DependencyGraph

# Data management routines


def fix_tid(src_tid, sep):  # fixes and unrolls the offsets
    if sep not in src_tid:
        fixed_span = [str(int(src_tid)+1)]
    else:
        vals = src_tid.split(sep)
        fixed_span = [str(int(val)+1) for val in vals]

    unrolled_span = []  # unroll spans, e.g. 2:5 -> [2,3,4,5]; 6_7_9 -> [6,7,8,9]
    if len(fixed_span) <= 1:
        return tuple([int(i) for i in fixed_span])
    else:
        for x in range(len(fixed_span)-1):
            for y in range(int(fixed_span[x]), int(fixed_span[x+1])+1):
                unrolled_span += [y]
        return tuple(set(sorted([int(i) for i in unrolled_span])))


def collect_srl_data(in_fes):  # load SRL data (~frame.elements). All the offsets are shifted by 1!
    srl_data = {}  # {sentence_id: {fe_id: [[fee_frame, fee_lemmapos, {role: role_span}], [fee_frame2, {role: role_span}], ...]}
    for line in in_fes:
        line = line.strip().split("\t")
        fee_tid = fix_tid(line[5], "_")  # predicate offsets are given as tid_tid_tid_tid
        fee_frame = line[3]
        fee_lemmapos = line[4].lower()
        sid = int(line[7])
        role_info = line[8:]
        srl_data[sid] = srl_data.get(sid, {})
        srl_data[sid][fee_tid] = srl_data[sid].get(fee_tid, [])
        fee_info = []  # ugly but so is the data! Multiple fee possible on single span
        fee_info += [fee_frame]
        fee_info += [fee_lemmapos]

        role_dict = {}
        for x in range(0, len(role_info), 2):
            role_dict[role_info[0]] = fix_tid(role_info[1], ":")  # role offsets are given as start:end
        fee_info += [role_dict]
        srl_data[sid][fee_tid] += [fee_info]
    return srl_data


def collect_sentence_data(in_sentences):  # load parse data (~all.lemma.tags)
    sid = 0
    sentences = {}
    for line in in_sentences:
        line = line.strip()
        if line:
            line = line.split("\t")
            num_tok = int(line[0])
            line = line[1:]
            data = [line[x*num_tok:x*num_tok+num_tok] for x in range(0, len(line)/num_tok)]  # TODO list comprehension ninja required here
            sentences[sid] = {}
            try:
                tid = 1
                for form, pos, dep, head, _, lemma in zip(*data):
                    sentences[sid][tid] = {}
                    sentences[sid][tid]["form"] = form
                    sentences[sid][tid]["pos"] = pos
                    sentences[sid][tid]["dep"] = dep
                    sentences[sid][tid]["head"] = int(head)
                    sentences[sid][tid]["lemma"] = lemma
                    tid += 1
            except Exception:
                print "Malformed parse data in sentence", sid
                sentences[sid] = None
            finally:
                sid += 1
    return sentences


def merge_to_graph(srl_data, sentences, verbose=False):  # zip sentence and SRL data together and turn them into a graph
    for sid in sentences:
        if sid in srl_data:
            sentence = sentences[sid]
            if sentence is not None:
                nodes = {tid: sentence[tid]["form"] for tid in sentence}
                edges = [(sentence[tid]["head"], tid, sentence[tid]["dep"]) for tid in sentence]
                srl = srl_data[sid]
                for pred_tid in srl:
                    for pred_info in srl[pred_tid]:
                        g = DependencyGraph(nodes, edges)
                        frame, lemmapos, roles = pred_info
                        roles_by_tid = {}
                        for (x, y) in roles.items():
                            for role_tid in y:
                                roles_by_tid[int(role_tid)] = x
                        try:
                            g.add_srl((pred_tid, frame, lemmapos), roles_by_tid)
                            yield g
                        except Exception:
                            print "SRL data error in sentence", sid, sys.exc_info()[0]
                            if verbose:
                                print "pred:", pred_tid, frame, lemmapos
                                print roles_by_tid
                                print g.pretty()


# This is the method you are looking for
def get_graphs(src_sentences, src_fes, verbose=False):  # files in, graphs out
    i = 0
    with codecs.open(src_sentences, "r", "utf-8") as in_sentences:
        with codecs.open(src_fes, "r", "utf-8") as in_fes:
            srl_data = collect_srl_data(in_fes)
            sentences = collect_sentence_data(in_sentences)
            graphs = [x for x in merge_to_graph(srl_data, sentences, verbose)]
            print src_sentences.split("/")[-1], src_fes.split("/")[-1], "labeled:", len(srl_data), "parsed:", len(sentences), "graphs:", len(graphs)
            for graph in graphs:
                graph.gid = i
                i += 1
            return graphs


