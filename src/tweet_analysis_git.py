# standard library imports
import sys
import string
import argparse
import pickle as pk
import multiprocessing
import numpy as np
import pandas as pd
import gensim
import keras
from sklearn.model_selection import train_test_split
# this local packages
import conv
import prep_and_encode_data as prep_enc_data
import eda_plot

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 500)

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
EMBEDDING_DIM = 150
EXITCODE_SUCCESS, EXITCODE_FAILED, EXITCODE_ABORTED = 0, 1, 10


def checkout_data(src_df):
    try:
        # check what the distribution of text/values in the dataframe looks like
        eda_plot.exploration(src_df)
        # check what the most common words in the normalized tweets for each class/category are
        eda_plot.plot_topn_words(src_df, n=10)
        # plot some more curves to see the kind of data we're dealing with length of normalized tweets
        eda_plot.plot_hist_length(src_df['tokenized'].tolist(), bins=40, level='word') # length in number of words
        eda_plot.plot_hist_length(src_df['text'].tolist(), bins=40, level='char') # length in number of characters
    except Exception as ex:
        print(f"Failed to generate desired plot due to: {ex}")


def train_model(src_df):
    # trying word2vec in gensim for more advanced text encoding
    MAX_SEQUENCE_LENGTH = src_df['tokenized'].apply(len).max()
    print(f"MAX_SEQUENCE_LENGTH (length of longest normalized tweet): {MAX_SEQUENCE_LENGTH} words.")

    # taking cue from: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
    w2v_model = gensim.models.Word2Vec(size=EMBEDDING_DIM, window=10, min_count=2, alpha=0.1, 
    	min_alpha=0.0001, negative=10, sample=6e-5, workers=cores-1)
    w2v_model.build_vocab(src_df['tokenized'], progress_per=10000)
    w2v_model.train(src_df['tokenized'], total_examples=w2v_model.corpus_count, epochs=30)
    w2v_model.init_sims(replace=True)

    embeddings, cnn_data, labels = prep_enc_data.get_word2vec(src_df, w2v_model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    uniq_cnt = embeddings.shape[0]

    x_train, x_test, y_train, y_test = train_test_split(cnn_data, labels, test_size=0.2, random_state=13)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=7)
    
    model = conv.ConvNet(embeddings, MAX_SEQUENCE_LENGTH, uniq_cnt, EMBEDDING_DIM, labels.shape[-1]) # load compiled model from conv.py
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)] # restore_best_weights=True
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=128, 
    	callbacks=callbacks, workers=cores-1, use_multiprocessing=True)
    test_results = model.evaluate(x_test, y_test, batch_size=128, workers=cores-1, use_multiprocessing=True)
    print(f"\nConvolutional neural network's Test accuracy: {test_results[1]}, test loss: {test_results[0]}")
    model.save('w2v_cnn.h5') # save the model so it can be used to plot confusion matrix later or for any other use
    eda_plot.plot_confusion_matrix(x_test, y_test, saved_model='w2v_cnn.h5')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-from_pkl", default=False, help="use pickled data or csv file")
    cmd_args = parser.parse_args()
    from_pkl = cmd_args.from_pkl
    try:
        if from_pkl:
            src_df = pd.read_pickle(f"data_tokenized.pkl")
            print(f"Loaded file: data_tokenized.pkl into a dataframe")
        else:
            text = pd.read_csv("tweets_10k.csv", usecols = ['text', 'class_label'])
            src_df = text.reindex(np.random.permutation(text.index))
            src_df = prep_enc_data.prep_and_pkl(src_df)
    except Exception as ex:
        print(f"Failed to load data due to: {ex}")
        return EXITCODE_FAILED

    try:
        checkout_data(src_df)
        train_model(src_df)
        return EXITCODE_SUCCESS
    except Exception as ex:
        print(f"Ran into an exception: {ex}")
        return EXITCODE_FAILED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Ctrl + C was hit. Exiting.")
        sys.exit(EXITCODE_ABORTED)