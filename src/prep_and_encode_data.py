from nltk.tag import StanfordPOSTagger
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet as wn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from datetime import datetime
import pandas as pd
import re, string
import numpy as np


stop_words = set(stopwords.words('english'))
path_to_jar = "../../../stanford-postagger-full-2020-08-06/stanford-postagger.jar"
path_to_model = "../../../stanford-postagger-full-2020-08-06/models/english-left3words-distsim.tagger"
st = StanfordPOSTagger(path_to_model, path_to_jar, encoding='utf8', java_options='-Xmx10G')
lm = WordNetLemmatizer()


def wordnet_pos_code(tag):
    if tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def prep_and_pkl(data):
    #print(data.head())
    data.loc[:, 'text'] = data['text'].str.replace(r'http\S+', '').str.replace('@', 'at ').str.replace('#', '').apply(lambda x: re.sub(r'(\S)\1{2,}', r'\1', x))
    data['tokenized'] = st.tag_sents([x.split() for x in data.text.tolist()])
    start_time = datetime.now()
    data.loc[:, 'tokenized'] = data['tokenized'].apply(lambda ls: [x for x in ls if x[0] not in list(set(stop_words)) + list(string.punctuation)]) \
            .apply(lambda ls: [lm.lemmatize(x, wordnet_pos_code(y)) if y else lm.lemmatize(x) for x, y in ls])\
            .apply(clean_text)

    data['tokenized_text'] = data['tokenized'].apply(lambda ls: ' '.join(ls)).astype("string")
    print(data.head())
    data.rename(columns={"text": "original_text", "tokenized_text": "text"}) # renaming columns 
    print(f"Dataframe has {len(data)} rows, processed in {(datetime.now() - start_time).seconds / 60: 0.1f} minutes.")
    pd.to_pickle(data, f"data_tokenized.pkl")
    return data


def clean_text(ls):
    return [re.sub(r"(\W)", "", x).lower() for x in ls] # handle any embedded non-alphanumeric chars in the token


def get_word2vec(src_df, w2v_model, max_seq_len, dim_size):
    df, y_df = src_df[['text', 'tokenized']], src_df[["class_label"]]
    MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = max_seq_len, dim_size
    VOCAB_SIZE = len(set([x for ls in df['tokenized'].tolist() for x in ls]))
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df['text'].tolist())
    sequences = tokenizer.texts_to_sequences(df["text"].tolist())
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')
    cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(y_df))
    indices = np.arange(cnn_data.shape[0])
    np.random.shuffle(indices)
    cnn_data = cnn_data[indices]
    labels = labels[indices]

    embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
    for word,index in word_index.items():
        embedding_weights[index,:] = w2v_model[word] if w2v_model.wv.__contains__(word) else np.random.rand(EMBEDDING_DIM)
    #print(embedding_weights.shape)
    return embedding_weights, cnn_data, labels