import networkx as nx

class DependencyGraph:
    def __init__(self, nodes, edges):
        """ Initialize a dependency graph from a list of nodes and a list of edges
			Nodes are represented as a dictionary {node_id:word, ...}
			Edges are a list of triples [(src_id, tgt_id, label), ...] """
        self.G = nx.DiGraph()
        for node_id in nodes:
            self.G.add_node(node_id, word=nodes[node_id])
        self.G.add_node(0, word="ROOT")
        for edge in edges:
            label = edge[2]
            # add prepositions to labels
            if label == 'prep':
                label += "_" + self.G.node[edge[1]]["word"].lower()
            self.G.add_edge(edge[0], edge[1], label=label)

        self.predicate_head = None
        self.predicate_nodes = None
        self.roles = None
        self.sent = " ".join(nodes[nid] for nid in sorted(list(nodes.keys())))
        self.gid = None

    def add_srl(self, predicate_node, role_nodes):
        """ Add SRL information to the graph
			Predicate is specified as a tuple (node_ids, sense, lemmapos)
			Roles are specified as a dictionary {node_id:role, ...}
			This can be done only once, since only one predicate-argument structure at a time is considered """
        self.predicate_nodes = []
        if (self.predicate_head is not None) or (self.roles is not None):
            raise Exception("Each graph must contain only one predicate-argument structure")
        for x in predicate_node[0]:
            self.G.node[int(x)]["frame"] = predicate_node[1]
            self.G.node[int(x)]["lemmapos"] = predicate_node[2]
            self.predicate_nodes += [int(x)]
        self.predicate_head = predicate_node[0][0]
        self.roles = []
        node_groups = {}  #group nodes by role
        for node_id in role_nodes:
            node_groups[role_nodes[node_id]] = node_groups.get(role_nodes[node_id], []) + [node_id]
        for role in node_groups:
            head = self.get_head(node_groups[role])
            self.G.node[head]["role"] = role
            self.roles += [head]

    def pretty(self):
        """ Pretty-print the graph """
        s = ""
        for n in self.G.nodes():
            if self.G.node[n] != {}:
                gid = str(self.gid) if self.gid!=None else "NOID"
                word = self.G.node[n]["word"]
                head = self.G.predecessors(n)[0] if len(self.G.predecessors(n)) > 0 else "_"
                dep_label = self.G[head][n]["label"] if len(self.G.predecessors(n)) > 0 else "_"
                role = self.G.node[n].get("role", "_")
                pred = self.G.node[n].get("frame", "_")
                s += "\t".join([x for x in [str(gid), str(n), word, str(head), dep_label, role, pred]])+"\n"
        return s

    def get_predicate_head(self):
        return self.G.node[self.predicate_head]

    def get_predicate_node_words(self):
        return [self.G.node[x]["word"].lower() for x in self.predicate_nodes]

    def get_direct_dependents(self, node):
        """ Get direct dependents of a node """
        return self.G.successors(node)

    def get_path(self, src, tgt):
        """ Get path from the source node (id) to the target node (id)
			Path is represented as a list of dependency relations concatenated by "->" """
        edges = None
        if tgt in self.G.predecessors(src) and tgt!=0:  # don't want the ROOT
            return "-1"  # the parent relation
        try:
            edges = nx.shortest_path(self.G, src, tgt)
        except nx.exception.NetworkXNoPath:
            edges = None
        finally:
            if edges is not None:
                dep_labels = [self.G[edges[n]][edges[n + 1]]["label"] for n in range(len(edges) - 1)]
                return "->".join(dep_labels)
            else:
                return None

    def create_pathmap(self):
        """ Internal function that calculates paths between all possible node pairs in the graph """
        self.pathmap = {}
        self.all_paths = []
        for n1 in self.G.nodes():
            self.pathmap[n1] = {}
            for n2 in self.G.nodes():
                if n1 != n2:
                    path = self.get_path(n1, n2)
                    if path is not None:
                        p = self.get_path(n1, n2)
                        self.pathmap[n1][n2] = p
                        self.all_paths += [p]
        self.all_paths = set(self.all_paths)

    def find_node(self, src, path):
        """ Find node in a graph given the source and the path """
        res = []
        if path == '':
            return [src]
        if path not in self.all_paths:
            return None
        for tgt in self.G.nodes():
            if tgt != src:
                if self.pathmap[src] is not None:
                    if tgt in self.pathmap[src]:
                        if self.pathmap[src][tgt] == path:
                            res += [tgt]
        return res if len(res) > 0 else None

    def get_node_label(self,
                       node_id):
        """ Get node label given the node id
			If it's a preposition, take the noun it points to! """
        in_rel = self.G.in_edges(node_id)
        if in_rel is not None and len(in_rel)>0:
            label = self.G[in_rel[0][0]][in_rel[0][1]]["label"]  # check the label
            if label.startswith("prep"):
                succ = self.G.successors(in_rel[0][1])
                if succ is None or len(succ) == 0:
                    return "#ERR"  # no successor? That's weird!
                else:
                    pobj = self.G.successors(in_rel[0][1])[0]  # here we assume that a preposition has only one successor, the pobj
                    return self.G.node[pobj]["word"]
        return self.G.node[node_id]["word"]

    def get_head(self, nodes):
        """ Get the head node for a role span.
		First, try to find a node with outgoing arc.
		If none found, pick the node with most dependents inside the span """
        head = None # leftmost node is default
        if len(nodes) == 1:
            head = nodes[0]
        else:
            for node_id in nodes:
                parent = self.G.predecessors(node_id)[0]
                if parent not in nodes:
                    head = node_id
                    break
        return head
