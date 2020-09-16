import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from nltk.corpus import stopwords
from collections import Counter
import itertools
from statistics import mean, median
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import numpy as np


def exploration(src_df):
    print(src_df.info())
    print(src_df.describe(include='all'))
    for col in src_df.columns:
        print(src_df[col].value_counts(dropna=False))


def plot_hist_length(data, bins, level):
    print(f"Plotting histogram for length of normalized tweet in {level}s.")
    x = [len(x) for x in data]
    print(f"Normalized tweet length (in {level}s) - shortest: {min(x)}, longest: {max(x)}, median: {median(x)}, avg: {mean(x)}.")
    x_rounded = [round(x,-1) for x in x] # round length to nearest 10: 77 -> 80, 173: 170
    #plt.tight_layout(pad=0.5)
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(x, bins=bins, range=(min(x_rounded), max(x_rounded)), edgecolor='red', linewidth=2)
    ax.set_ylabel('number of tweets')
    ax.set_xlabel(f'length of normalized tweets - number of {level}s')
    ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.set_xticklabels(labels=bins, rotation=70)
    #plt.show()
    plt.savefig(f'plot_hist_tweet_{level}_count.png', bbox_inches='tight')


def plot_topn_words(data, n):
    data['tokenized'] = data['tokenized'].apply(lambda ls: [x for x in ls if x not in stopwords.words('english') and x.strip()])
    list2d_relevant = data[data['class_label']==1]['tokenized'].tolist()
    list1d_relevant = [x for x in list(itertools.chain(*list2d_relevant))]
    relevant_tup = Counter(list1d_relevant).most_common(2*n)
    #pd.Series(list1d).value_counts().head(n).plot(kind='bar')
    #plt.show()
    not_relevant_tup = {}
    list2d_not_relevant = data[data['class_label']==0]['tokenized'].tolist()
    list1d_not_relevant = list(itertools.chain(*list2d_not_relevant))
    not_relevant_tup[0] = Counter(list1d_not_relevant).most_common(n)

    list2d_not_sure = data[data['class_label']==2]['tokenized'].tolist()
    list1d_not_sure = list(itertools.chain(*list2d_not_sure))
    not_relevant_tup[1] = Counter(list1d_not_sure).most_common(n)

    fig = plt.figure(tight_layout=True)
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, :]) # row 0, col 0
    ax.bar([x[0] for x in relevant_tup], [x[1] for x in relevant_tup], align='center')
    ax.set_xticklabels(labels=[x[0] for x in relevant_tup], rotation=70)
    ax.set_ylabel('relevant tweets')
    ax.set_xlabel(f'top {n*2} words')

    for i in range(2):
        ax = fig.add_subplot(gs[1, i])
        ax.bar([x[0] for x in not_relevant_tup[i]], [x[1] for x in not_relevant_tup[i]], align='center')
        ax.set_xticklabels(labels = [x[0] for x in not_relevant_tup[i]], rotation=70)
        ax.set_ylabel(f"{'non-relevant' if i==0 else 'not-sure'} tweets")
        ax.set_xlabel(f'top {n} words')
    fig.align_labels()
    #plt.show()
    plt.savefig('plot_topn_words_by_class.png', bbox_inches='tight')


# taking cue from Emmanuel Ameisen's notebook
def plot_confusion_matrix(x_test, y_test, saved_model, normalize=False):
    classes = ['Irrelevant', 'Disaster', 'Unsure']
    model = load_model(saved_model)
    y_pred = model.predict(x_test, batch_size=128, workers=-1, use_multiprocessing=True)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.winter)
    plt.title('confusion Matrix', fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=20)
    
    #plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    #plt.show()
    plt.savefig('plot_confusion_matrix.png', bbox_inches='tight')
