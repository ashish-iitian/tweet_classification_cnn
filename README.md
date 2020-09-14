# tweet_classification_cnn
NLP is one of the hottest fields of interest in today's world. With more and more data being collected in different ways, there is a wealth of information to learn from and understand what communication/conversation means and make informed inferences therefrom. Businesses today collect reviews from customers and more often than not, people take to social media platforms to share their thoughts/experiences about various products/services. So, it is becoming increasingly important to glean some actionable insights from data, learn what users like/dislike about something. Aside from the commercial appeal/application of NLP, we can also learn about some event(s) occurring elsewhere, right from the digital devices/apps that connect us all online. This project shows one such application of NLP algorithms, trained over Twitter data where I try to detect whether a tweet talks about some "disastrous" event or not. One can think about such an application being useful in informing the community/authorities of any adverse/dangerous event taking place somewhere, using a machine-learning model trained to tag tweets.

## The Data
In this project, I use a dataset likely made available by [Crowdflower](https://www.crowdflower.com/data-for-everyone/) but I got it from [here](https://data.world/crowdflower/disasters-on-social-media). To quote the source, _contributors looked at over 10,000 tweets culled with a variety of searches like "ablaze", "quarantine", and "pandemonium", then noted whether the tweet referred to a disaster event (as opposed to a joke with the word or a movie review or something non-disastrous)_. There seem to be 3 different categories/labels for the tweets: *"Not Relevant(0), Relevant(1), Can't Decide(2)"*, so the problem was treated as a multiclass classification problem (though I could've dropped the far less prominent class - label 2).

## The Steps

### Reading Input File
For my project, once I figured out how I would like to preprocess and clean the data (details in next section), I decided to pickle and save the data to faciliate a faster model development/training process. Currently, the main script ([tweet_analysis_git.py](https://github.com/ashish-iitian/tweet_classification_cnn/blob/master/src/tweet_analysis_git.py)) accepts a boolean argument ("from_pkl") to indicate whether we want to use the pickled data or we prefer to read the csv input file that'd then undergo preprocessing.

### Exploratory Data Analysis
Once the data is read, I conducted some preliminary checks to see what its composition is and what can be learned about the features at hand. In this case, it meant checking out some tweets and their labels in dataset, getting a sense of what/how text normalization should be applied. I looked for any NULL columns (there were none unsurprisingly) etc. before trying some visualization using **Matplotlib**.

I plotted some graphs to see how long raw tweets are (in terms of word count and character count), what are some popular words that appear in tweets classified as "disaster"-related or other categories etc. I used a histogram to see the former and bar plots for the latter to see the "important" words juxtaposed.

### Cleaning and Tagging The Data
Cleaning and normalizing the tweets took the longest time (as expected). Given that people these days in their online posts use a lot on acronyms and new urban slangs, text clean-up/normalization can be a challenging task. I tried a few approaches (a **naive Python** approach, then an advanced Pythonic way using **regex**, trying out **nltk** and its methods) before settling in on the one used in this project. 

#### Basic clean-up
For example, looking at sample tweets, I found frequent use of urls and hashtags. I cleaned the text to strip any urls (since a hyperlink by itself is not meaningful), replaced "@" by "at" to make any geo-tagging more meaningful as a feature and stripped "#" from hashtags.

Susequently, I also applied Python regex to replace multiple occurrences (more than 2 times) of a character to handle texts like "Noooo" or "lolll" etc. where the repetition is unnecessary and just an informal expression of sentiment behind the tweet. The idea was that this would help the *"text encoder"* used to parse as many meaningful words as possible. 

#### Part-Of-Speech (POS) Tagging
Instead of using nltk's very own NLTKTagger or MIT's spaCY for this, I used **StanfordPOSTagger** from nltk. I didn't do any comparative study between them (could be a to-do) but decided to go in favor of StanfordPOSTagger for its better performance (F1 score as shown [here](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy)). 

With StanfordPOSTagger, I tried POS-Tagging for over tokenized text and passing entire text to StanfordPOSTagger for tagging. The first approach took a long time but tagging sentences was much faster. After POS-Tagging, I got rid of **stopwords** and **punctuation marks**. 

Finally, I applied **lemmatization** to the tokens using the POS tag so I have the normalized tokens from the raw tweets. This processed data was next saved into a pickle file for faster model development/training. 

After text normalization and right before model training in the main script, I plotted the curves from the EDA stage again to see if things make sense and to verify important information is not being lost.

### Feature Engineering & Model Training

#### Feature Engineering
At this point of time, the data we have is normalized but still exists in text/"string" data type. We need to encode it into a numeric format of fixed size so it can be passed to model training. There are multiple options for that, like **Bag-of-Words** model and **Tf-Idf-Weighting** through **sklearn.feature_extraction.text** package, using **CountVectorizer** and **TfidfVectorizer** respectively. I found that **TfidfVectorizer** vectorizer gives better performance compared to a naive **CountVectorizer** (subject of a different project which I will share soon). 

However for this project, I chose to use **Word2Vec** class from **gensim** package for text embedding since **Word2Vec** is smart enough to learn *context* for words and capture rich relationships within *window* specified. Word2Vec performs much better on larger corpus (> 10M words) but we will test it anyway. I instantiate a word2vec model using parameters as suggested [here](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec). Next I *build the vocabulary* for the model by passing on the entire corpus over which it gets trained subsequently. 

After the word2vec model is trained, I extract *token/word vectors* from it for the words in normalized and tokenized tweets and then construct our matrix of **embedding_weights** which is also **padded** so that each tweet can be represented by a vector/matrix of the same dimensionality. 

#### Model Training
After taking care of **feature engineering**, I define our **Convolutional Neural Network** model. We try 3 different filter sizes and use **max-pooling** to reduce dimensionality of feature map. We also use **dropout** techniques to make our model resilient to overfitting. We specify the **loss function** to minimize and the **optimizer** to use, alongside the metric to measure the model's performance.

I also leverage **keras.callbacks** that provides us additional control over the model training. Finally, I use sklearn's **train_test_split** method to split our text-embedded matrix into a train/test dataset, and further splitting the training dataset to obtain validation data. I provide the validation dataset to the keras model being trained for periodic evaluation. 

Next, I evaluated the model over the test dataset. It gave me an accuracy of around 78%. I also saved the trained model so it could be loaded again to generate the **confusion matrix** (since Keras doesn't allow model object to be passed to a function, I saved the model as h5 object).

## Next Steps
- Trying different parameters for word2vec model training (*alpha, min_alpha, negative*)
- Testing LSA word embedding technique which reportedly has better performance than word2vec for smaller data size (< 10M words in corpus)
- Trying hyperparameter-tuning for CNN model (trying different *filter size, dropout* etc.)
