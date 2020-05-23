import pandas as pd
from pathlib import Path
from utils.text_filters import remove_noise
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_Excel(filename):
    data_Frame = pd.read_excel(Path(__file__).parents[1] / 'resources' / 'corpora' / 'custom' / filename)
    return data_Frame

def get_Column_values(data_Frame,col):
    return [sent for sent in data_Frame.get(col)]

def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    
    inspired  by function from this wonderful article: 
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
    
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.title.split()))

def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try:
        index = sentiment_dict.loc[sentiment_dict['words'] == word].index[0]
        out = sentiment_dict['sentiment_coeff'][index]
    except:
        out = 0
    return out

if __name__ == "__main__":
    file_original = read_Excel('ratingReviews.xlsx')
    file_cleaned = file_original.dropna().drop_duplicates().reset_index(drop=True).rename(columns={'Comments':'title','Ratings':'rate'})
    comments = get_Column_values(file_original,'Comments')
    # stop_words = stopwords.words('english')
    stop_words = ()
    sentences = []
    for comment in comments:
        sentences.append(remove_noise(word_tokenize(comment),stop_words))
    file_cleaned.title = file_cleaned.title.apply(lambda x: ' '.join(remove_noise(word_tokenize(x),stop_words)),comments)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(file_cleaned)
    w2v_model = Word2Vec(min_count=3,
                     window=4,
                     size=300,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)
    w2v_model.build_vocab(sentences, progress_per=50000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    words = list(w2v_model.wv.vocab)
    w2v_model.save("/home/vajapu/Documents/VS-workspace/Python/output/models/w2v/model.bin")
    word_vectors = Word2Vec.load('/home/vajapu/Documents/VS-workspace/Python/output/models/w2v/model.bin').wv
    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
    negative_cluster_center = model.cluster_centers_[0]
    positive_cluster_center = model.cluster_centers_[1]
    #print(word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None))

    #assign clusters
    words = pd.DataFrame(word_vectors.vocab.keys())
    words.columns = ['words']
    words['vectors'] = words.words.apply(lambda x: word_vectors.wv[f'{x}'])
    words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
    words.cluster = words.cluster.apply(lambda x: x[0])
    words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]
    words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
    words['sentiment_coeff'] = words.closeness_score * words.cluster_value
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(words)
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(file_cleaned.title)
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(file_cleaned.title)
    replaced_tfidf_scores = file_cleaned.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
    replaced_closeness_scores = file_cleaned.title.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, words), x.split())))
    replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_cleaned.title, file_cleaned.rate]).T
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
    replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(replacement_df)
    replacement_df.to_excel(Path(__file__).parents[1] / 'resources' / 'corpora' / 'custom' / 'kmeansPrediction.xlsx', index = False)