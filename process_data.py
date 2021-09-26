import networkx as nx
from glob import glob
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed, dump
from stellargraph import StellarGraph


graph_paths = glob('data/psi-graph/*/*.txt')

# Generate dictionary
nodes = {}

for graph_path in tqdm(graph_paths):
    with open(graph_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
    for line in lines[2:]:
        e = line.strip().split()
        if len(e) == 2:
            nodes[e[0]] = nodes.get(e[0], 0) + 1
            nodes[e[1]] = nodes.get(e[1], 0) + 1

nodes = dict(sorted(nodes.items(), key=lambda k: k[1], reverse=True)[:900])


def load_graph(graph_path):
    G = nx.DiGraph()
    with open(graph_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
    for line in lines[2:]:
        e = line.strip().split()
        if len(e) == 2:
            if e[0] in nodes and e[1] in nodes:
                G.add_edge(e[0], e[1])
    if not len(G):
        return None
    G = nx.relabel_nodes(G, dict(zip(G, range(len(G)))))
    for node_id, node_data in G.nodes(data=True):
        node_data["feature"] = [G.degree(node_id)]
    square = StellarGraph.from_networkx(G, node_features="feature")
    if 'benign' in graph_path:
        label = 0
    else:
        label = 1
    return square, label


output = Parallel(cpu_count())(delayed(load_graph)(graph_path)
                               for graph_path in tqdm(graph_paths))
output = [v for v in output if v]
X, y = zip(*output)
dump([X, y], 'data/data.sav', compress=1)
