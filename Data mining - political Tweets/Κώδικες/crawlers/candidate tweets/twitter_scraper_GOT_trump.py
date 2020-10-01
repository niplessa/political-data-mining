import GetOldTweets3 as got
import csv

#column names in csv
columns = ['username','text','date']

#setting no of tweets:
a=10000


###--------Bernie's tweets---------------###

with open('trump.csv','a',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    
    tweetCriteria = got.manager.TweetCriteria().setUsername("@realDonaldTrump")\
                                                .setMaxTweets(a)\
                                               
            
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    for i in range(len(tweet)):
        uname = str(tweet[i].username)
        date = str(tweet[i].date)
        date = date.split()
        text = tweet[i].text
        print(i,uname,text,date[0])
        writer.writerow({'username' : uname, 'text' : text,'date' :str(date[0])}) #write entry to .csv file
