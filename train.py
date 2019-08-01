import datetime
import tensorflow as tf
import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

from bert_serving.client import BertClient
from inspect import signature
from keras.callbacks import Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from keras.callbacks import CSVLogger
from keras.callbacks import Callback

ENC_SIZE = 768  # size of bert encoding
MIN_PRED = 0.5  # this value means if score >= 0.5 then label it
NUM_EPOCHS = 20
BATCH_SIZE = 64
CLASS_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def fetch_embeddings(df, embeddings_cache_file):
    if os.path.exists(embeddings_cache_file):
        print("Load from cache")
        X = np.load(embeddings_cache_file)
    else:
        bc = BertClient(check_length=False)
        print("Fetch from bert server")
        bert_encodings = []
        for i, r in df.iterrows():
            txt = r.comment_text
            e = bc.encode([txt])
            bert_encodings.append(e)

        print(len(bert_encodings))
        X = np.vstack(bert_encodings)
        print("Save to cache file %s" % embeddings_cache_file)
        np.save(embeddings_cache_file, X)

    return X


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def fp_rate(y_true, y_pred, threshold=MIN_PRED):
    c1 = K.less(y_true, threshold)
    c2 = K.greater_equal(y_pred, threshold)
    f1 = K.cast(c1, dtype='float32')
    f2 = K.cast(c2, dtype='float32')
    return K.mean(f1 * f2)


def create_keras_model(epochs, input_dim):
    opt = SGD(lr=0.01, decay=.01/epochs, momentum=0.3)
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activity_regularizer=l2(0.0001), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activity_regularizer=l2(0.0001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', rmse, fp_rate])
    return model


def predict(model, X):
    return np.ravel(model.predict(X))


def plot_learning_curves(metrics_path, class_label, csv_logs):
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 1)

    metrics_df = pd.read_csv(csv_logs, index_col="epoch")
    plots_png_file = os.path.join(metrics_path, class_label + "-metrics.png")
    print("Write plots to: %s" % plots_png_file)
    
    metrics_df[["rmse", "val_rmse"]].plot(title=class_label, ax=plt.subplot(gs[0, 0]))
    metrics_df[["loss", "val_loss"]].plot(ax=plt.subplot(gs[1, 0]))
    metrics_df[["fp_rate", "val_fp_rate"]].plot(ax=plt.subplot(gs[2, 0]))
    fig.savefig(plots_png_file)
    plt.close()


def main(data_path):
    train_csv_path = os.path.join(data_path,
                                  "jigsaw-toxic-comment-classification-challenge/train.csv")

    df_train = pd.read_csv(train_csv_path)

    bc = BertClient(check_length=False)

    embeddings_cache_file = os.path.join(data_path, "bert_embeddings_array.npy")
    if os.path.exists(embeddings_cache_file):
        X = np.load(embeddings_cache_file)
    else:
        df = df_train
        bert_encodings = []
        for i, r in df.iterrows():
            txt = r.comment_text
            e = bc.encode([txt])
            bert_encodings.append(e)

        print(len(bert_encodings))
        X = np.vstack(bert_encodings)
        np.save(embeddings_cache_file, X)

    if not os.path.exists("metrics"):
        os.mkdir("metrics")

    metrics_path = os.path.join(data_path, "metrics/")
    models_path = os.path.join(data_path, "models/")

    df = df_train.copy()

    for class_label in CLASS_LABELS:
        print("\nTraining model for %s" % class_label)
        Y = df[class_label].values
        (X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=.3, stratify=Y, random_state=0)
        model = create_keras_model(NUM_EPOCHS, ENC_SIZE)
        csv_logs = os.path.join(metrics_path, class_label + "-training-log.csv")
        if os.path.isfile(csv_logs):
            os.unlink(csv_logs)
        csv_logger = CSVLogger(csv_logs, append=True, separator=',')

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[csv_logger], verbose=1)
        model_file = os.path.join(models_path, class_label + ".h5")
        model.save(model_file)
        plot_learning_curves(metrics_path, class_label, csv_logs)

    print("Training complete")


if __name__ == "__main__":
    data_path = None

    if len(sys.argv) >= 2:
        data_path = sys.argv[1]

    if data_path is None or not os.path.exists(data_path):
        print("Please provide a valid data path. Exiting now ...")
        sys.exit(-1)

    main(data_path)
