from bert_serving.client import BertClient
from inspect import signature
from keras.callbacks import Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from numpy import loadtxt
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from train import rmse, fp_rate, fetch_embeddings, MIN_PRED, CLASS_LABELS
import datetime
import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import time
import tqdm

data_path = None

if len(sys.argv) >= 2:
    data_path = sys.argv[1]

if data_path is None or not os.path.exists(data_path):
    print("Please provide a valid data path. Exiting now ...")
    sys.exit(-1)

test_csv_file = os.path.join(data_path, "./jigsaw-toxic-comment-classification-challenge/test.csv")
test_csv_labels_file = os.path.join(data_path, "./jigsaw-toxic-comment-classification-challenge/test_labels.csv")
df_test = pd.read_csv(test_csv_file, index_col="id")
df_test_labels = pd.read_csv(test_csv_labels_file, index_col="id")
df = df_test.join(df_test_labels)

# remove all the rows which were not scored, so we have only 0 or 1 scores
df = df[df["toxic"] != -1]
print(df.shape)

models = {}
for class_label in CLASS_LABELS:
    model_path = os.path.join(data_path, "models/%s.h5" % class_label)
    model = load_model(model_path, custom_objects={'rmse': rmse, 'fp_rate': fp_rate})
    models[class_label] = model


test_embeddings_cache_file = os.path.join(data_path, "test_bert_embeddings_array.npy")
X = fetch_embeddings(df, test_embeddings_cache_file)

perform_eval = True

if perform_eval:
    for class_label in CLASS_LABELS:
        model = models[class_label]
        Y = df[class_label].values
        score = model.evaluate(X, Y, verbose=1)
        print("Class Label:", class_label)
        for i in range(len(model.metrics_names)):
            print("%s: %.6f" % (model.metrics_names[i], score[i]))

bc = BertClient(check_length=False)

# perform actual prediction
perform_predict = False
if perform_predict:
    N = 5
    for i, r in df[df.identity_hate==1].sample(N).iterrows():
        txt = r.comment_text
        encodings = bc.encode([txt])

        predicted = {}
        actual = {}
        for class_label in CLASS_LABELS:
            model = models[class_label]
            p = model.predict(encodings)[0][0]
            predicted[class_label] = p
            actual[class_label] = r[class_label]

        print("-" * 40)
        print("Input Text:", txt)
        print("Actual:", actual)
        print("Predict:", predicted)
        print()


def run_prediction(bc, txt, models, CLASS_LABELS):
    bc_enc = bc.encode([txt], show_tokens=True)
    encodings = bc_enc[0]
    tokens = bc_enc[1][0]
    predicted = {}
    for class_label in CLASS_LABELS:
        model = models[class_label]
        p = model.predict(encodings)[0][0]
        if p >= MIN_PRED:
            predicted[class_label] = p
    return predicted, tokens


try:
    while True:
        txt = input("Input text please(Ctrl+D to exit): ")
        if txt.strip() == "":
            continue
        else:
            predicted, tokens = run_prediction(bc, txt, models, CLASS_LABELS)
            print("Tokens:", tokens)
            print("Predict:", predicted)
            print()
except EOFError as e:
    print("Terminating the input loop!")


movie_script_path = os.path.join(data_path, "valkaama-script.txt")

from spacy.lang.en import English
with open(movie_script_path) as f:
    movie_script = f.read()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(movie_script)
    predict_times = []
    for s in doc.sents:
        txt = str(s)
        start_time = time.time()
        predicted, tokens = run_prediction(bc, txt, models, CLASS_LABELS)
        time_diff = time.time() - start_time
        predict_times.append(time_diff)
        if any([x >= 0.5 for x in predicted.values()]):
            print("ToxicityLabels: %s" % (predicted))
            print("Text: %s"% (txt.replace("\n", " ").strip()))
            print()
    
    # convert seconds to milli-seconds
    predict_times = np.array(predict_times) * 1000
    print(stats.describe(predict_times))
    fig = plt.figure()
    fig.suptitle('Prediction time over different predict() calls', fontsize=20)
    plt.plot(predict_times)
    plt.xlabel('sequence_number', fontsize=18)
    plt.ylabel('prediction_times', fontsize=16)
    fig.savefig(os.path.join(data_path, 'prediction_times.png'))
    plt.show()
