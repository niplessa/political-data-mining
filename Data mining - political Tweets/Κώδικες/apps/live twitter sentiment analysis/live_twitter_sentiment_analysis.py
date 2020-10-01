# -*- coding: utf-8 -*-
"""live twitter sentiment analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qu4VDSShIG2Q1TBmuTFAzHT4x1SdwDj1
"""

import tweepy

#Variables that contains the user credentials to access Twitter API
access_token = "1237766600159367168-LlkBsrjgaOkgs2R7iZd5TBe3ul3oFc"
access_token_secret = "WVGUEGrFxbTo1QicE5TiSMPY9xZdxrr0mKqDUEin1Ti1g"
consumer_key = "lZuIHazx7PtCN2BmdLDj4BC8q"
consumer_secret = "59Gperq9TNXC9WphXpRhSrROXvNEct282bG2LCpimXBZ7kFoLk"

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

"""###get the tweets streaming"""

from tweepy import Stream
from tweepy import StreamListener
import json
from  textblob import TextBlob
import re
import csv
import nltk


p1=re.compile('(www|http)\S+')
p2=re.compile('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)')

import datetime

trump = 0
sanders = 0

header_name = ['Trump','Sanders']

with open('sentiment.csv','w') as file :
  writer = csv.DictWriter(file,fieldnames = header_name)
  writer.writeheader()

class MyListener(StreamListener):
  def on_data (self, data):
    raw_tweets = json.loads(data)
    try:
      tweets = raw_tweets['text']
      #tweets = ' '.join(re.sub(p2,'',tweets).split()) #remove junk
      tweets = re.sub (p1,'', tweets) #remove links
      tweets = ' '.join(re.sub('RT','',tweets).split()) #remove RTs
      tweets = tweets.lower()

      global trump
      global sanders

      trump_sentiment = 0
      sanders_sentiment = 0
      blob = TextBlob(tweets.strip())
      
      if "trump" in blob and 'sanders' not in blob: 
        print("Trump")
        print(blob.sentiment.polarity)
        trump_sentiment += blob.sentiment.polarity
      elif 'sanders' or 'bernie' in blob and 'trump' not in blob:
        print("Sanders")
        print(blob.sentiment.polarity)
        sanders_sentiment += blob.sentiment.polarity

      trump += trump_sentiment
      sanders += sanders_sentiment

      with open('sentiment.csv','a') as file :
         writer = csv.DictWriter(file,fieldnames = header_name)
         info = {'Trump': trump,'Sanders': sanders}
         writer.writerow(info)



      print(tweets)
      print('\n')

    except:
      print("Error")



  def on_error(self,status):
    print(status)

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

candidates = ['Sanders','Bernie','Trump']

twitter_stream = Stream(auth,MyListener())
twitter_stream.filter(track = candidates)
