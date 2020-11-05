# -*- coding: utf-8 -*-
"""
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Data Gathered:      https://www.kaggle.com/cfpb/us-consumer-finance-complaints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from io import StringIO # This is used for fast string concatination
import nltk # Use nltk for valid words
import collections as co # Need to make hash 'dictionaries' from nltk for fast processing
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer #Bag of Words



# The dataset used was collected from the following website:
# https://www.kaggle.com/cfpb/us-consumer-finance-complaints

# read the CSV file
dataset = pd.read_csv('consumer_complaints.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(dataset.head())

# checking the dataset shape
print(dataset.shape)
# (555957, 18)


# prints out names of columns
print(dataset.columns)

# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(dataset.info())


# checking for missing data
dataset.isnull().sum() 

# making object into categorical variables
dataset['product'] = dataset['product'].astype('category')
dataset['sub_product'] = dataset['sub_product'].astype('category')
dataset['company'] = dataset['company'].astype('category')
dataset['state'] = dataset['state'].astype('category')
dataset['submitted_via'] = dataset['submitted_via'].astype('category')
dataset['company_response_to_consumer'] = dataset['company_response_to_consumer'].astype('category')
dataset['timely_response'] = dataset['timely_response'].astype('category')
dataset['consumer_disputed?'] = dataset['consumer_disputed?'].astype('category')

# checking data to check that all objects have been changed to categorical variables.
dataset.info()

# =============================================================================
# Converting 'date_received' and 'date_sent_to_company' to datetime
# =============================================================================

dataset['date_received'] = dataset['date_received'].astype('datetime64')
dataset['date_sent_to_company'] = dataset['date_sent_to_company'].astype('datetime64')

# Complete the call to convert the date column
dataset['date_received'] =  pd.to_datetime(dataset['date_received'],
                              format='%m-%d-%Y')

# Complete the call to convert the date column
dataset['date_sent_to_company'] =  pd.to_datetime(dataset['date_sent_to_company'],
                              format='%m-%d-%Y')

print(dataset.info())

#creating column to asses the number of days to process claim
dataset['days_to_process'] = dataset['date_sent_to_company'] - dataset['date_received']

#plt.figure(figsize = (12, 8))
#sns.countplot(x = 'days_to_process', data = dataset, palette = 'viridis', order = dataset['days_to_process'].value_counts().index)
#plt.xticks(rotation = 90)
#plt.title('Days Needed for the Company to Respond', fontsize = 16)
#plt.ylabel('count', fontsize = 14)
#plt.xlabel('Number of days to Process', fontsize = 14)
#plt.show()

print(dataset['days_to_process'])

# =============================================================================
# Examining States
# =============================================================================
dataset['state'].unique()

dataset.groupby(['state']).size().sort_values(ascending=False)

# Examines the top 25 States that have complaints
plt.figure(figsize = (12, 8))
sns.countplot(x = 'state', data = dataset, palette = 'viridis', order = dataset['state'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of TOP 25 States', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('States', fontsize = 14)
plt.show()

# =============================================================================
# Examining Products
# =============================================================================
dataset['product'].unique()

dataset.groupby(['product']).size().sort_values(ascending=False)
#Mortgage                   186475
#Debt collection            101052
#Credit reporting            91854
#Credit card                 66468
#Bank account or service     62563
#Consumer Loan               20990
#Student loan                15839
#Payday loan                  3877
#Money transfers              3812
#Prepaid card                 2470
#Other financial service       557
#dtype: int64

# Examines the top 25 countries that make a reservation
plt.figure(figsize = (12, 8))
sns.countplot(x = 'product', data = dataset, palette = 'viridis', order = dataset['product'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Products', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Products', fontsize = 14)
plt.show()



# =============================================================================
# Examining Companies
# =============================================================================
dataset['company'].unique()

dataset.groupby(['company']).size().sort_values(ascending=False)
#No     443823
#Yes    112134

# Examines the top 25 companies which had the most complaints
plt.figure(figsize = (12, 8))
sns.countplot(x = 'company', data = dataset, palette = 'viridis', order = dataset['company'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Companies Involved', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Companies', fontsize = 14)
plt.show()


# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['company'].value_counts().head(25).index.tolist()
sizes = dataset['company'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of Top 25 Companies', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Examining Methods of Submission
# =============================================================================
dataset['submitted_via'].unique()

dataset.groupby(['submitted_via']).size().sort_values(ascending=False)
#Web            361338
#Referral       109379
#Phone           40026
#Postal mail     36752
#Fax              8118
#Email             344

# Examines method of submission
plt.figure(figsize = (12, 8))
sns.countplot(x = 'submitted_via', data = dataset, palette = 'viridis', order = dataset['submitted_via'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Methods of Submission', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Methods of Submission', fontsize = 14)
plt.show()


# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['submitted_via'].value_counts().head(25).index.tolist()
sizes = dataset['submitted_via'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of Submission Methods', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Examining Company Response to Consumer
# =============================================================================
dataset['company_response_to_consumer'].unique()

dataset.groupby(['company_response_to_consumer']).size().sort_values(ascending=False)
#Closed with explanation            404293
#Closed with non-monetary relief     70237
#Closed with monetary relief         38262
#Closed without relief               17909
#Closed                              13399
#Closed with relief                   5305
#In progress                          3763
#Untimely response                    2789

# Examines company responses to the company
plt.figure(figsize = (12, 8))
sns.countplot(x = 'company_response_to_consumer', data = dataset, palette = 'viridis', order = dataset['company_response_to_consumer'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Company Response to Consumer', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Company Response', fontsize = 14)
plt.show()


# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['company_response_to_consumer'].value_counts().index.tolist()
sizes = dataset['company_response_to_consumer'].value_counts().tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of Breakdown of Company Response to Consumer', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()

# =============================================================================
# Examining Reponses
# =============================================================================
dataset['timely_response'].unique()

dataset.groupby(['timely_response']).size().sort_values(ascending=False)
#Yes    541909
#No      14048

# Examines the response
plt.figure(figsize = (12, 8))
sns.countplot(x = 'timely_response', data = dataset, palette = 'viridis', order = dataset['timely_response'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Responses', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Timely Response', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['timely_response'].value_counts().index.tolist()
sizes = dataset['timely_response'].value_counts().tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Was there a timely response to the complaint?', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()

# =============================================================================
# 
# =============================================================================
# Table I want to create:
#                       Yes                 NO
# Mortgage
# Student Debt

# Stacked barchart!

dataset.groupby('product')['timely_response'].value_counts()

fig = plt.figure(figsize = (20,10))
labels = dataset['product'].value_counts().index.tolist()
sizes = dataset['product'].value_counts().tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle =30)
plt.title('Matching the Reserved Room and Assigned Room', fontdict = None, position= [0.48,1], size = 'xx-large')
plt.show()










# =============================================================================
# Splitting the data into timely or late responses
# =============================================================================
# splitting data on 'Yes' or 'No' Timely Responses
# retrieving everyone who has NOT been timely responded
timely_response_data = dataset[dataset['timely_response'].isin(['Yes'])]

late_response_data  = dataset[dataset['timely_response'].isin(['No'])]




# =============================================================================
# Preparing data for WordCloud
# =============================================================================

# Only interested in data with consumer complaints
d = dataset[dataset['issue'].notnull()]


sns.set(style="white", color_codes=True)

# We want a very fast way to concat strings.
s=StringIO()
d['issue'].apply(lambda x: s.write(x))

k=s.getvalue()
s.close()
k=k.lower()
k=k.split()

# Next only want valid strings
words = co.Counter(nltk.corpus.words.words())
stopWords =co.Counter( nltk.corpus.stopwords.words() )
k=[i for i in k if i in words and i not in stopWords]
s=" ".join(k)
c = co.Counter(k)

# At this point we have k,s and c
# k Array of words, with stop words removed
#
# s Concatinated string of all comments
#
# c Collection of words

# Take a look at the 14 most common words
c.most_common(14)

s[0:100]

print(k[0:10],"\n\nLength of k %s" % len(k))


# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",max_words=len(k),max_font_size=40, relative_scaling=.8).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# =============================================================================
# TO DO -> 
# =============================================================================
# draw a time graph for date_received!
# draw a graph the company by product type
# draw a graph for days_process
# find the  most common words in Narrative. 

# creating a wordcloud for issue was not informative 
# find better methods to analyse text.


# =============================================================================
# Analysing Consumer Complaint Narrative
# =============================================================================

# Only interested in data with consumer complaint narrative
narrative_dataset = dataset[dataset['consumer_complaint_narrative'].notnull()]


"""
Remove XXXX from the list!
"""

import re

review_list=[]
for review in narrative_dataset.consumer_complaint_narrative:
    review = re.sub("[^a-zA-z]"," ",review) # if expression in the sentence is not a word then this code change them to space
    review = review.lower() # turns all word in the sentence into lowercase.
    review = nltk.word_tokenize(review) # splits the words that are in the sentence from each other.
    lemma = nltk.WordNetLemmatizer()
    review = [lemma.lemmatize(word) for word in review] # this code finds the root of the word for a word in the sentence and change them to their root form.
    review = " ".join(review)
    review_list.append(review) # store sentences in list


from sklearn.feature_extraction.text import CountVectorizer #Bag of Words

max_features=500 # "number" most common(used) words in reviews

count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english") # stop words will be dropped by stopwords command

sparce_matrix=count_vectorizer.fit_transform(review_list).toarray()# this code will create matrix that consist of 0 and 1.
 
sparce_matrix.shape 


print("Top {} the most used word by reviewers: {}".format(max_features,count_vectorizer.get_feature_names()))

# =============================================================================
# Trying new things
# =============================================================================

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist



tokenized_word=word_tokenize(review)
print(tokenized_word)


tokenized_word.remove('xxxx')
tokenized_word.remove('bring')
tokenized_word.remove('total')
tokenized_word.remove('said')
tokenized_word.remove('talk')
tokenized_word.remove('hotel')
tokenized_word.remove('see')


fdist = FreqDist(tokenized_word)
print(fdist)

fdist.most_common(30)



# removing StopWords
stop_words=set(stopwords.words("english"))
print(stop_words)

# identifying StopWords
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
print("Tokenized Words:",tokenized_word)
print("Filtered Words:",filtered_word)

# Stemming
ps = PorterStemmer()

stemmed_words=[]
for w in filtered_word:
    stemmed_words.append(ps.stem(w))

print("Filtered Words:",filtered_word)
print("Stemmed Words:",stemmed_words)

stemmed_words.remove('xxxx')
stemmed_words.remove('bring')
stemmed_words.remove('total')
stemmed_words.remove('said')
stemmed_words.remove('talk')
stemmed_words.remove('hotel')
stemmed_words.remove('see')
print(stemmed_words)

##Lexicon Normalization
##performing stemming and Lemmatization
#lem = WordNetLemmatizer()
#
#stem = PorterStemmer()
#
#word = "verifying"
#print("Lemmatized Word:",lem.lemmatize(word,"v"))
#print("Stemmed Word:",stem.stem(word))



#tokens = nltk.word_tokenize(review)
#print(tokens)
#
#nltk.pos_tag(tokens)


# =============================================================================
# 
# =============================================================================


plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(tokenized_word))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Display of Tokenized Words')
plt.show()


plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(stemmed_words))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Display of Stemmed Words')
plt.show()



plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(filtered_word))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Display of Filtered')
plt.show()