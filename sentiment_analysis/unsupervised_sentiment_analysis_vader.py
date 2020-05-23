from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.text_filters import remove_noise
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import json

def getSentiment(stmnt):
    sid = SentimentIntensityAnalyzer()
    sent = 0.0
    count = 0
    sentList = sent_tokenize(stmnt)
    data = {}

    for sentence in sentList:
        #print(sentence)
        #base_words = remove_noise(word_tokenize(sentence))
        base_words = word_tokenize(sentence)
        ss = sid.polarity_scores(' '.join([str(elem) for elem in base_words]))
        count += 1
        sent += ss['compound']  # Tally up the overall sentiment
        val = {}
        for k in ss:
           #print('{0}: {1}, '.format(k, ss[k]), end='')
           val[''.join(k)]=ss[k]
        if len(base_words) > 0:
            val['base_words'] = base_words
        #print()
        data[sentence] = val

    if count != 0:
        if sent > 0.05:
            pol = 'positive'
        elif sent < -0.05:
            pol = 'negative'
        else:
            pol = 'neutral'
        data['overall'] = {'sentiment':pol,'intensity':float(sent / count)}

    return json.dumps(data,allow_nan=False) 

print(getSentiment("While it's convenient to access content on your phone, awful"))