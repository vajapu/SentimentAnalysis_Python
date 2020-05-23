import pandas as pd
from pathlib import Path
from unsupervised_sentiment_analysis_vader import getSentiment
from utils.text_filters import remove_noise
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import random

def read_Excel(filename):
    data_Frame = pd.read_excel(Path(__file__).parents[1] / 'resources' / 'corpora' / 'custom' / filename)
    return data_Frame

def get_Column_values(data_Frame,col):
    return [sent for sent in data_Frame.get(col)]

def get_dataset_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield (dict([token, True] for token in tweet_tokens[0]),tweet_tokens[1])

def get_pseudo_sentiment(comments):
    sentiment = []
    cleaned_tokens_list = []

    for comment in comments:
        data = pd.read_json(getSentiment(comment))
        #print(data)
        sentiment.append(data.get('overall').get('sentiment'))
        i=0
        for colName in data.iteritems():
            if colName != 'overall':
                #print( data.values[0][i])
                compound = data.values[3][i]
                if compound > 0.05:
                    pos = 'positive'
                elif compound < 0.05:
                    pos = 'negative'
                else:
                    pos = 'neutral'
                if str(data.values[4][i]) != 'nan' and pos != 'neutral':
                    cleaned_tokens_list.append([data.values[4][i],pos])
                i+=1
    # freq_dist_pos = FreqDist(cleaned_tokens_list)
    # print(freq_dist_pos.most_common(10))
    tokens_for_model = get_dataset_for_model(cleaned_tokens_list)
    dataset = [com_dict for com_dict in tokens_for_model]
    return [sentiment,dataset]

def getData(comments,ratings):
    sentiment = []
    cleaned_tokens_list = []
    for comment in comments:
        rating = ratings[comments.index(comment)]
        if rating > 3:
            pos = 'positive'
        elif rating < 3:
            pos = 'negative'
        cleaned_tokens_list.append([word_tokenize(comment),pos])
        sentiment.append(pos)
    tokens_for_model = get_dataset_for_model(cleaned_tokens_list)
    dataset = [com_dict for com_dict in tokens_for_model]
    return [sentiment,dataset]

if __name__ == "__main__":
    comments = get_Column_values(read_Excel('ratingReviews.xlsx'),'Comments')
    ratings = get_Column_values(read_Excel('ratingReviews.xlsx'),'Ratings')
    #pseudo_sentiment = get_pseudo_sentiment(comments)
    pseudo_sentiment = getData(comments,ratings)
    sentiment = pseudo_sentiment[0]
    dataset_modul = pseudo_sentiment[1]
    data = {'Comments':comments,
    'Ratings':ratings,
    'sentiment':sentiment}
    # Convert the dictionary into DataFrame  
    df = pd.DataFrame(data)
    df.to_excel(Path(__file__).parents[1] / 'resources' / 'corpora' / 'custom' / 'ratingReviews.xlsx', index = False)
    
    random.shuffle(dataset_modul)
    total = int(len(dataset_modul)*0.7)

    train_data = dataset_modul[:total]
    test_data = dataset_modul[total:]

    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(20))

#custom text analysis
    custom_tweet = "With the latest update, all seems to be working well now. Thank you..."
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))