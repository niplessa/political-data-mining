import GetOldTweets3 as got
import csv

#column names in csv
columns = ['text','date']

#setting no of tweets:
a=5000

        
###--------Tweets about Sanders---------------###
'''
with open('about_sanders.csv','a',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    
    tweetCriteria = got.manager.TweetCriteria().setMaxTweets(a)\
                                                .setQuerySearch('sanders')\
                                                .setLang('eng')\
       
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    for i in range(len(tweet)):
        date = str(tweet[i].date)
        date = date.split()
        text = tweet[i].text

        print(i,text,date[0])
        writer.writerow({'text' : text,'date' :str(date[0])}) #write entry to .csv file
'''
###--------Tweets about Trump---------------###

with open('about_trump.csv','a',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    
    tweetCriteria = got.manager.TweetCriteria().setMaxTweets(a)\
                                                .setQuerySearch('trump')\
                                                .setLang('eng')\
       
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    for i in range(len(tweet)):
        date = str(tweet[i].date)
        date = date.split()
        text = tweet[i].text

        print(i,text,date[0])
        writer.writerow({'text' : text,'date' :str(date[0])}) #write entry to .csv file

