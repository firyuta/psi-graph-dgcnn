import pandas as pd
import tensorflow as tf
import networkx as nx
from tqdm import tqdm
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from sklearn import metrics
from multiprocessing import cpu_count
from joblib import Parallel, delayed

N_JOBS = cpu_count()
EPOCHS = 100
DICT_SIZE = 900


class DGCNN:
    def __init__(self, train_paths, test_paths, report_path) -> None:
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.report_path = report_path

        self.paths = self.train_paths + self.test_paths
        self.nodes = self.load_dict()

    def create_model(self, X):
        generator = PaddedGraphGenerator(graphs=X)
        k = 35
        layer_sizes = [32, 1]

        dgcnn_model = DeepGraphCNN(
            layer_sizes=layer_sizes,
            activations=["tanh", "tanh"],
            k=k,
            bias=False,
            generator=generator,
        )
        x_inp, x_out = dgcnn_model.in_out_tensors()
        x_out = Conv1D(filters=16, kernel_size=sum(
            layer_sizes), strides=sum(layer_sizes))(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=0.5)(x_out)
        predictions = Dense(units=1, activation="sigmoid")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"],
        )
        print(model.summary())
        return model

    def load_dict(self):
        nodes = {}
        for path in tqdm(self.train_paths):
            with open(path, 'r') as f:
                lines = f.read().split('\n')[:-1]
            for line in lines[2:]:
                e = line.strip().split()
                if len(e) == 2:
                    nodes[e[0]] = nodes.get(e[0], 0) + 1
                    nodes[e[1]] = nodes.get(e[1], 0) + 1
        nodes = dict(
            sorted(nodes.items(), key=lambda k: k[1], reverse=True)[:DICT_SIZE])
        return nodes

    def load_graphs(self):
        X, y_train, y_test = [], [], []
        for i, path in tqdm(enumerate(self.paths)):
            G = nx.DiGraph()
            with open(path, 'r') as f:
                lines = f.read().split('\n')[:-1]
            for line in lines[2:]:
                e = line.strip().split()
                if len(e) == 2:
                    if e[0] in self.nodes and e[1] in self.nodes:
                        G.add_edge(e[0], e[1])
            if not len(G):
                continue
            G = nx.relabel_nodes(G, dict(zip(G, range(len(G)))))
            for node_id, node_data in G.nodes(data=True):
                node_data["feature"] = [G.degree(node_id)]
            square = StellarGraph.from_networkx(G, node_features="feature")
            label = 0 if 'benign' in path else 1
            X.append(square)
            if i < len(self.train_paths):
                y_train.append(label)
            else:
                y_test.append(label)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
        return X, y_train, y_test

    def run(self):
        # load data
        X, y_train, y_test = self.load_graphs()

        gen = PaddedGraphGenerator(graphs=X)
        train_gen = gen.flow(
            list(y_train.index),
            targets=y_train.values,
            batch_size=32,
            symmetric_normalization=False,
        )
        test_gen = gen.flow(
            list(y_test.index),
            targets=y_test.values,
            batch_size=1,
            symmetric_normalization=False,
        )

        # train
        model = self.create_model(X)
        model.fit(
            train_gen, epochs=EPOCHS, verbose=2, shuffle=True,
        )

        # test
        y_prob = model.predict(test_gen)
        y_pred = tf.greater(y_prob, .5)

        clf_report = metrics.classification_report(y_test, y_pred, digits=4)
        cfn_matrix = metrics.confusion_matrix(y_test, y_pred)
        with open(self.report_path, 'w') as f:
            f.write(clf_report + '\n')
            f.write(str(cfn_matrix) + '\n')
