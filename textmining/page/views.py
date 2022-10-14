from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from django.shortcuts import render

from datetime import datetime, timedelta

import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools

import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


# Create your views here.


def index(request):
    search_word = "iphone"

# https://jsikim1.tistory.com/143
    now_day = datetime.now()
    end_day = now_day.strftime("%Y-%m-%d")
    start_day = (now_day - timedelta(weeks=7)).strftime("%Y-%m-%d")

    search_query = search_word + ' since:' + start_day + ' until:' + end_day

# https://github.com/JustAnotherArchivist/snscrape/issues/164
    scraped_tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        if i >= 100:
            break
        scraped_tweets.append([tweet.url, tweet.date, tweet.content, tweet.id, tweet.username,
                              tweet.outlinks, tweet.outlinksss, tweet.tcooutlinks, tweet.tcooutlinksss])

    df = pd.DataFrame(scraped_tweets)
    df.columns = ["url", "date", "content", "id", "username",
                  "outlinks", "outlinksss", "tcooutlinks", "tcooutlinksss"]

    df = df[df['content'].str.contains(
        '|'.join(case_combination(search_word)))]

    stop_words = stopwords.words('english')
    stop_words.extend(["rt", "iphone"])

    positive = []
    negative = []

    for tweet in df.content:
        cleaned_tweet = []
        cleaned_tweet_string = CleanText(tweet, Num=True, Eng=False)
        tweet_tokens = word_tokenize(cleaned_tweet_string)
        for token in tweet_tokens:
            if token.lower() not in stop_words:
                cleaned_tweet.append(token)

        cleaned_tweet_str = ' '.join(cleaned_tweet)

        compound_point = sia.polarity_scores(cleaned_tweet_str)['compound']
        if compound_point > 0:
            positive.append(cleaned_tweet_str)
        else:
            negative.append(cleaned_tweet_str)

    data = {"positive": positive, "negative": negative}

    return render(request, 'page/index.html', data)

# ref: https://windybay.net/post/41/


def case_combination(word):
    sequence = ((c.lower(), c.upper()) for c in word)
    return [''.join(x) for x in itertools.product(*sequence)]


def CleanText(readData, Num=True, Eng=True):
    # Remove Retweets
    text = re.sub('RT @[\w_]+: ', '', readData)
    # Remove Mentions
    text = re.sub('@[\w_]+', '', text)
    # Remove or Replace URL
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ',
                  text)  # http로 시작되는 url
    text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ',
                  text)  # http로 시작되지 않는 url
    # Remove only hashtag simbol "#" because hashtag contains huge information
    text = re.sub(r'#', ' ', text)
    # Remove Garbage Words (ex. &lt, &gt, etc)
    text = re.sub('[&]+[a-z]+', ' ', text)
    # Remove Special Characters
    text = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', text)
    # Remove 출처 by yamada
    text = re.sub(r"(출처.*)", ' ', text)
    # Remove newline
    text = text.replace('\n', ' ')

    if Num is True:
        # Remove Numbers
        text = re.sub(r'\d+', ' ', text)

    if Eng is True:
        # Remove English
        text = re.sub('[a-zA-Z]', ' ', text)

    # Remove multi spacing & Reform sentence
    text = ' '.join(text.split())

    return text
