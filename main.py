import pandas as pd
import tensorflow as tf
import stellargraph as sg
from joblib import load
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt


# Load data
X, y = load('data/data.sav')

# Define DGCNN model
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

# Train the model
y = pd.Series(y)
y_train, y_test = train_test_split(
    y, test_size=.3, random_state=42, stratify=y)

gen = PaddedGraphGenerator(graphs=X)

train_gen = gen.flow(
    list(y_train.index - 1),
    targets=y_train.values,
    batch_size=32,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(y_test.index - 1),
    targets=y_test.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 100
history = model.fit(
    train_gen, epochs=epochs, verbose=2, shuffle=True,
)
model.save('model/dgcnn.h5')
sg.utils.plot_history(history)
plt.savefig('result/history.png')

# Test
y_prob = model.predict(test_gen)
y_pred = tf.greater(y_prob, .5)

clf_report = metrics.classification_report(y_test, y_pred, digits=4)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cnf_matrix.ravel()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
auc = metrics.roc_auc_score(y_test, y_prob)
other_metrics = pd.DataFrame({'TPR': '%.4f' % TPR,
                              'FPR': '%.4f' % FPR,
                              'FNR': '%.4f' % FNR,
                              'ROC AUC': '%.4f' % auc}, index=[0]).to_string(col_space=9, index=False)
print(clf_report)
print(cnf_matrix)
print(other_metrics)
