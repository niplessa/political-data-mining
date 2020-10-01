from tkinter import *

import pandas as pd


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize 
from string import punctuation
from nltk.stem import WordNetLemmatizer 

#from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

import re

from string import punctuation


stopwords = stopwords.words('english')

#load bernie's tweets

df_bernie = pd.read_csv('datasets/bernie.csv')
#drop duplicate articles on original df_bernieset
df_bernie.drop_duplicates(subset="text",keep = "first", inplace = True)
print("Total retrieved articles: ",df_bernie.shape[0])
print(df_bernie.shape)
#df_bernie.head()

#load trump's tweets

df_trump = pd.read_csv("datasets/trump.csv")
#drop duplicate articles on original df_trumpset
df_trump.drop_duplicates(subset="text",keep = "first", inplace = True)
print("Total retrieved articles: ",df_trump.shape[0])
print(df_trump.shape)
#df_trump.head()

"""**combine the two dataframes into a single one**"""

tweets = pd.concat([df_trump, df_bernie], axis=0, ignore_index=True)

"""**Create label 1 for Trump, 0 for Sanders**"""

tweets['label'] = tweets.username.apply(lambda uname: 1 if uname == 'realDonaldTrump' else 0 )

tweets.head().append(tweets.tail())


#convert tweets.text to string
tweets['text']=tweets['text'].astype('str')

#lowercase
tweets['clean_text'] = tweets['text'].str.lower()

#remove words len<2
tweets['clean_text'] = tweets['clean_text'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>2] ))

#remove links with regex
p=re.compile('(www|http)\S+')
tweets['clean_text'] = tweets['clean_text'].apply(lambda x: re.sub(p,' ',x))

#remove @
tweets['clean_text'] = tweets['clean_text'].apply(lambda x: re.sub('@','',x))

#remove stopwords function
stops =  list(stopwords) + list(punctuation)

def remove_stops(text):
    text_no_stops = []
    for i in text:
        if i not in stops:
            if len(i) == 1:
                pass
            else:
                text_no_stops.append(i)
        else:
            pass
    return text_no_stops

#tokenize text
tweets['tokenized_text'] = tweets['clean_text'].apply(lambda x: regexp_tokenize(x,"[\w']+"))

#actually remove stopwords
tweets['tokenized_text'] = tweets['tokenized_text'].apply(lambda x: remove_stops(x))

#lematization function
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    lemmatized = []
    for word in text:
        lemmatized.append(lemmatizer.lemmatize(word))
    return lemmatized

#lemmatize
tweets['lemmatized_text'] = tweets['tokenized_text'].apply(lemmatize_text)

#detokenize (create the final string of words)
tweets['lemmatized_string'] = tweets['lemmatized_text'].apply(lambda x: ' '.join(x))

"""**sentiment analysis**"""

"""**tweets are now clean and sentiment analysis has been conducted. Time for predictions!**"""

#create the feature dataset
features = ['lemmatized_string']
X = tweets[features]

#create the target dataset
Y = tweets['label']

"""**split the dataset into training and test set**"""

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,test_size=0.3)

"""**Baseline accuracy**"""

print(Y_train.value_counts())
print()
most_freq_class = Y_train.value_counts().index[0]
print('Most frequent class in training dataset:', most_freq_class)
print()

print('Baseline accuracy:', round(Y_test.value_counts()[most_freq_class] / Y_test.count(), 3))

"""**TfIDf Vectorizer**"""

vect = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), min_df=2)

"""**training set**"""

#document term matrix
X_train_dtm = vect.fit_transform(X_train.lemmatized_string)

# Compressed Sparse Row matrix
import scipy as sp
extra = sp.sparse.csr_matrix(X_train.drop('lemmatized_string', axis=1).astype(float))

#combine matrices
X_train_dtm_extra = sp.sparse.hstack((X_train_dtm, extra))

"""**test set**"""

X_test_dtm = vect.transform(X_test.lemmatized_string)
extra1 = sp.sparse.csr_matrix(X_test.drop('lemmatized_string', axis=1).astype(float))

#combine matrices
X_test_dtm_extra = sp.sparse.hstack((X_test_dtm, extra1))

"""**Multinomial Naive Bayes classifier**"""

mnb = MultinomialNB()
mnb.fit(X_train_dtm, Y_train)

# Predict the class using the Multinomial Naive Bayes classifier
Y_pred_class = mnb.predict(X_test_dtm)

# Testing accuracy classification score
print('Testing accuracy score:', round(metrics.accuracy_score(Y_test, Y_pred_class), 3))



def classify() :
    tweet = tweet1.get()
    source_text = []
    source_text.append(tweet)

    source =vect.transform(source_text)

    result = mnb.predict_proba(source)


    for i in range(len(result)):
      #print('Probability Sanders: {} \nProbability Trump: {}'.format(result[i][0],result[i][1]))
      if result[i][0] > result[i][1] :
        tweet2['text'] = tweet
        answer['text'] = f'Sanders tweeted this! - Sanders probability: {result[i][0]}'
        
      else :
        tweet2['text'] = tweet
        answer['text'] = f'Trump tweeted this! - Trump probability: {result[i][1]}'
        



#graphical interface

root  = Tk()
root.title("Bernie or Trump: Tweet classifier")
root.geometry('600x300')

label = Label(root, text='Enter a tweet: ')
label.pack()

label2 = Label(root, text ='Baseline accuracy: 0.503 - Model accuracy score: 0.951',bg='yellow')
label2.place(x=0, y=300, anchor='sw')

tweet1 = Entry(root,width=200)
tweet1.pack()

btn = Button(root,text='Classify this tweet!',command=classify)
btn.pack()

tweet2 = Label(root,text='',bg="white")
tweet2.pack()
answer = Label(root,text='',bg="red")
answer.pack()
        


root.mainloop()


'''
'''
