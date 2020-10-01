import GetOldTweets3 as got
import csv

#column names in csv
columns = ['username','text','date']

#setting no of tweets:
a=2000


###--------Tweets from CNN politics---------------###

with open('fox.csv','a',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    
    tweetCriteria = got.manager.TweetCriteria().setMaxTweets(a)\
                                                .setUsername('@foxnewspolitics')\
       
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    for i in range(len(tweet)):
        date = str(tweet[i].date)
        date = date.split()
        uname = str(tweet[i].username)
        text = str(tweet[i].text)

        print(i,text,date[0])
        writer.writerow({'username' : uname ,'text' : text,'date' :str(date[0])}) #write entry to .csv file
