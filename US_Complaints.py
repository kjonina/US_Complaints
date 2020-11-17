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



# The df used was collected from the following website:
# https://www.kaggle.com/cfpb/us-consumer-finance-complaints

# read the CSV file
df = pd.read_csv('consumer_complaints.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking shape
print("The df has {} rows and {} columns.".format(*df.shape))

# ... and duplicates
print("It contains {} duplicates.".format(df.duplicated().sum()))


# prints out names of columns
print(df.columns)

# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(df.info())


# checking for missing data
df.isnull().sum() 

# making object into categorical variables
df['product'] = df['product'].astype('category')
df['sub_product'] = df['sub_product'].astype('category')
df['company'] = df['company'].astype('category')
df['state'] = df['state'].astype('category')
df['submitted_via'] = df['submitted_via'].astype('category')
df['company_response_to_consumer'] = df['company_response_to_consumer'].astype('category')
df['timely_response'] = df['timely_response'].astype('category')
df['consumer_disputed?'] = df['consumer_disputed?'].astype('category')

# checking data to check that all objects have been changed to categorical variables.
df.info()

# =============================================================================
# Converting 'date_received' and 'date_sent_to_company' to datetime
# =============================================================================

df['date_received'] = df['date_received'].astype('datetime64')
df['date_sent_to_company'] = df['date_sent_to_company'].astype('datetime64')

#creating column to asses the number of days to process claim
df['days_to_process'] = df['date_sent_to_company'] - df['date_received']

print(df['days_to_process'])


# =============================================================================
# Examining States
# =============================================================================

df['state'].unique()

# Examines the States that have complaints
df.groupby(['state']).size().sort_values(ascending=False)

# Examines the top 25 States that have complaints
sns.set()
plt.figure(figsize = (12, 8))
sns.countplot(x = 'state', data = df, palette = 'magma', order = df['state'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Top 25 States', fontsize = 16)
plt.ylabel('Number of Complaints', fontsize = 14)
plt.xlabel('States', fontsize = 14)
plt.show()

# =============================================================================
# Examining Products
# =============================================================================
df['product'].unique()

# Examines the products
df.groupby(['product']).size().sort_values(ascending=False)

# Examines the products
plt.figure(figsize = (12, 8))
sns.countplot(y = 'product', data = df, palette = 'viridis', order = df['product'].value_counts().index)
plt.title('Breakdown of Products', fontsize = 16)
plt.ylabel('Type of Products', fontsize = 14)
plt.xlabel('Number of Complaints', fontsize = 14)
plt.show()



# =============================================================================
# Examining Companies
# =============================================================================
df['company'].unique()

df.groupby(['company']).size().sort_values(ascending=False)

# Examines the top 25 companies which had the most complaints
plt.figure(figsize = (12, 8))
sns.countplot(y = 'company', data = df, palette = 'magma', order = df['company'].value_counts().head(25).index)
plt.title('Breakdown of Companies Involved', fontsize = 16)
plt.ylabel('Companies', fontsize = 14)
plt.xlabel('Number of Complaints', fontsize = 14)
plt.show()


# =============================================================================
# Examining Methods of Submission
# =============================================================================
df['submitted_via'].unique()

# Examines method of submission
df.groupby(['submitted_via']).size().sort_values(ascending=False)

# Examines method of submission
plt.figure(figsize = (12, 8))
sns.countplot(y = 'submitted_via', data = df, palette = 'viridis', order = df['submitted_via'].value_counts().head(25).index)
plt.title('Breakdown of Methods of Submission', fontsize = 16)
plt.ylabel('Methods of Submission', fontsize = 14)
plt.xlabel('Number of Complaints', fontsize = 14)
plt.show()

# =============================================================================
# Examining Company Response to Consumer
# =============================================================================
df['company_response_to_consumer'].unique()


# Examines company responses to the company
df.groupby(['company_response_to_consumer']).size().sort_values(ascending=False)

# Examines company responses to the company
plt.figure(figsize = (12, 8))
sns.countplot(y = 'company_response_to_consumer', data = df, palette = 'viridis', order = df['company_response_to_consumer'].value_counts().index)
plt.title('Breakdown of Company Response to Consumer', fontsize = 16)
plt.ylabel('Company Response', fontsize = 14)
plt.xlabel('count', fontsize = 14)
plt.show()


# =============================================================================
# Examining days_to_process
# =============================================================================

time_df = df['days_to_process'].value_counts().head(20)

fig = plt.figure(figsize = (20,10))
sns.lineplot(data = time_df)
plt.title('The Response in Days between ', fontsize = 20)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Duration in Days', fontsize = 14)
plt.show()


# =============================================================================
# Examining days_to_process by Product
# ============================================================================= 
product_days = df.groupby(['product']
        )['days_to_process'].sum().sort_values(ascending = False)


#Creating a graph for TV Shows
plt.figure(figsize = (12, 8))
sns.barplot(y = "product", x = "days_to_process", data = df,
                 palette = 'Reds_d')
plt.set_title('Top 15 TV Shows Mid-Covid', fontsize = 20)
plt.set_ylabel('TV Shows', fontsize = 14)
plt.set_xlabel('Duration', fontsize = 14)
plt.set_xticklabels(plt.get_xticklabels(), rotation = 90)


# =============================================================================
# Examining Reponses
# =============================================================================
df['timely_response'].unique()

## Examines the response
df.groupby(['timely_response']).size().sort_values(ascending=False)

## Examines the response
#plt.figure(figsize = (12, 8))
#sns.countplot(x = 'timely_response', data = df, palette = 'viridis', order = df['timely_response'].value_counts().index)
#plt.title('Breakdown of Responses', fontsize = 16)
#plt.ylabel('count', fontsize = 14)
#plt.xlabel('Timely Response', fontsize = 14)
#plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = df['timely_response'].value_counts().index.tolist()
sizes = df['timely_response'].value_counts().tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Was there a timely response to the complaint?', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()

# =============================================================================
# Exploring Timely Response by Product
# =============================================================================

df.groupby('product')['timely_response'].value_counts()


# isolating variables in a table
product_timely = pd.DataFrame({'product': df['product'],
                   'timely_response': df['timely_response']})

#pivoting the table
product_timely_pivot = pd.pivot_table(
        product_timely, index = ['timely_response'], columns = ['product'], aggfunc=len)

# print pitot table to insure it works
print(product_timely_pivot)

# creating a graph
plt.figure(figsize = (12, 8))


sns.catplot(x = 'product', hue = 'timely_response', kind = 'count', palette = 'pastel', edgecolor = '.6', data = product_timely_pivot)



plt.title('Breakdown of Ratings by Content Type', fontsize = 16)
plt.ylabel('Rating', fontsize = 14)
plt.xlabel('count', fontsize = 14)




# =============================================================================
# Preparing data for WordCloud
# =============================================================================

# Only interested in data with consumer complaints
d = df[df['issue'].notnull()]


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
# draw a time graph for date_received + date_sent_to_company!
# draw a graph the company by product type
# draw a graph for days_process
# find the  most common words in Narrative. 
#
# creating a wordcloud for issue was not informative 
# find better methods to analyse text.


# =============================================================================
# Analysing Consumer Complaint Narrative
# =============================================================================

# Only interested in data with consumer complaint narrative
narrative_df = df[df['consumer_complaint_narrative'].notnull()]


"""
Remove XXXX from the list!
"""

import re

review_list=[]
for review in narrative_df.consumer_complaint_narrative:
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