# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 10:44:21 2022

@author: Pratik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#from sklearn.linear_model import LogisticRegression

twitter_data = pd.read_csv(r'C:\Users\Pratik\Desktop\IIT KGP\Seminar-II (IM69002)\Datasets\Twitter_Data.csv')
reddit_data = pd.read_csv(r'C:\Users\Pratik\Desktop\IIT KGP\Seminar-II (IM69002)\Datasets\Reddit_Data.csv')

twitter_data = twitter_data.dropna(how='any')
reddit_data = reddit_data.dropna(how='any')

twitter_data.rename(columns={'clean_text':'comment'}, inplace=True)
reddit_data.rename(columns={'clean_comment':'comment'}, inplace=True)

twitter_data.category = twitter_data.category.apply(lambda x: int(x))
reddit_data.category = reddit_data.category.apply(lambda x: int(x))

def cleanText(string):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    cleanedText = ' '.join(''.join([i for i in string if not i.isdigit()]).split())
    
    for i in punc:
        cleanedText = cleanedText.replace(i, '')
        
    a = [i for i in cleanedText if i.isalpha() or i == ' ']
            
    final_text = ' '.join(''.join(a).split())
    return final_text

twitter_data.comment = twitter_data.comment.apply(cleanText)
reddit_data.comment = reddit_data.comment.apply(cleanText)

twitter_data.drop_duplicates(subset='comment', keep=False, inplace=True)
reddit_data.drop_duplicates(subset='comment', keep=False, inplace=True)

# Concat reddit data and twiter data
concat_df = pd.concat([reddit_data, twitter_data], ignore_index=True)
concat_df.drop_duplicates(subset='comment', keep=False, inplace=True)



#.....................................Total No. of comments available - counting...........................................
counts = [len(reddit_data), len(twitter_data), len(concat_df)]
labels = ['Reddit', 'Twitter', 'Total']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10,7))
plt.bar(labels, counts,color=colors, edgecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for i in range(len(counts)):
    plt.annotate(str(counts[i]),xy = (labels[i],counts[i]), 
                 ha='center', va='center',size=15, xytext=(0, 10),textcoords='offset points')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()



#.....................................Counting The Negative, Neutral & Positive Comments.....................
negative_count = [len(reddit_data[reddit_data.category == -1]),
                 len(twitter_data[twitter_data.category == -1])]

neutral_count = [len(reddit_data[reddit_data.category == 0]),
                 len(twitter_data[twitter_data.category == 0])]

positive_count = [len(reddit_data[reddit_data.category == 1]),
                 len(twitter_data[twitter_data.category == 1])]

x = ['Reddit', 'Twitter']
x_indexes = np.arange(len(x))

plt.figure(figsize=(14,8))

plt.bar(x_indexes - 0.25, negative_count, width=0.25, label='Negative', edgecolor='white', color='red')
plt.bar(x_indexes, neutral_count, width=0.25, label='Neutral', edgecolor='white', color='blue')
plt.bar(x_indexes + 0.25, positive_count, width=0.25, label='Positive', edgecolor='white', color='green')

for i in range(2):
    plt.annotate(negative_count[i], xy=(i-0.25,negative_count[i]), ha='center', va='center',size=15, xytext=(0, 10),textcoords='offset points')
    plt.annotate(neutral_count[i], xy=(i,neutral_count[i]), ha='center', va='center',size=15, xytext=(0, 10),textcoords='offset points')
    plt.annotate(positive_count[i], xy=(i+0.25,positive_count[i]), ha='center', va='center',size=15, xytext=(0, 10),textcoords='offset points')

plt.yticks(fontsize=14)
plt.xticks(ticks=x_indexes, labels=x, fontsize=14)

plt.legend(prop={'size':15})
plt.grid(axis='y', alpha=0.65)
plt.tight_layout()



#...........................................Count the Average Length of All Nature of Comment.........................
negative_avg = int(np.mean([len(i) for i in concat_df[concat_df.category == -1].comment]))
neutral_avg = int(np.mean([len(i) for i in concat_df[concat_df.category == 0].comment]))
positive_avg = int(np.mean([len(i) for i in concat_df[concat_df.category == 1].comment]))

labels = ['Negative', 'Neutral', 'Positive']
avg = [negative_avg, neutral_avg, positive_avg]

plt.figure(figsize=(14,8))
plt.bar(labels, avg, color=colors)

for i in range(3):
    plt.annotate(avg[i], xy=(labels[i],avg[i]), ha='center', va='center',size=15, xytext=(0, 10),textcoords='offset points')
    
plt.title('Comment Length Average', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(axis='y', alpha=0.5)
plt.tight_layout()

#.........................................Create Word Clouds to See Which Words Appear Frequently...............................
negative_words = ''
neutral_words = ''
positive_words = ''

stopwords = set(STOPWORDS)

for comment, category in zip(concat_df.comment, concat_df.category):
    tokens = comment.split()
    
    for word in tokens:  
        if category == -1:
            negative_words += word + ' '
        elif category == 0:
            neutral_words += word + ' '
        else:
            positive_words += word + ' '

negative_cloud = WordCloud(width = 800, height = 800,background_color ='white',
                stopwords = stopwords,min_font_size = 10).generate(negative_words)

neutral_cloud = WordCloud(width = 800, height = 800,background_color ='white',
                stopwords = stopwords,min_font_size = 10).generate(neutral_words)

positive_cloud = WordCloud(width = 800, height = 800,background_color ='white',
                stopwords = stopwords,min_font_size = 10).generate(positive_words)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), facecolor=None)

ax1.imshow(negative_cloud)
ax1.set_title('Negative', fontsize=18, color='red')

ax2.imshow(neutral_cloud)
ax2.set_title('Neutral', fontsize=18, color='blue')

ax3.imshow(positive_cloud)
ax3.set_title('Positive', fontsize=18, color='green')

plt.tight_layout()

#........................................................
from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('wordnet')

lem = WordNetLemmatizer()

def tokenize_lem(sentence):
    outlist= []
    token = sentence.split()
    for tok in token:
        outlist.append(lem.lemmatize(tok))
    print(outlist)
    return " ".join(outlist)

concat_df["comment"] = concat_df["comment"].apply(tokenize_lem)

#...................................................
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, df):
        self.df = df
        
    def fixData(self):
        dataframe = self.df
        lowest_len = min([i for i in dataframe.category.value_counts()])
        
        # Create the final data frame
        final_df = pd.concat([dataframe[dataframe.category == -1][:lowest_len],
                             dataframe[dataframe.category == 0][:lowest_len],
                             dataframe[dataframe.category == 1][:lowest_len]])
        
        # To shuffle the rows in the data frame
        final_df = final_df.sample(frac=1).reset_index(drop=True)
        return final_df

a = Data(concat_df)
fixed_df = a.fixData()
print(f'Before: \n{concat_df.category.value_counts()}\n')
print(f'After: \n{fixed_df.category.value_counts()}')

train_X, test_X, train_y, test_y = train_test_split([i for i in fixed_df.comment], [i for i in fixed_df.category], test_size=0.25, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_X_vectors = vectorizer.fit_transform(train_X)
test_X_vectors = vectorizer.transform(test_X)

#.....................................1) KNN..........................................
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(train_X_vectors, train_y)
knn_prediction=knn.predict(test_X_vectors)

print(f'Accuracy: {knn.score(test_X_vectors, test_y)}')

#..............................2) Naive Bayes.................................................
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(train_X_vectors, train_y)
nb_prediction = nb.predict(test_X_vectors)

print(f'Accuracy: {nb.score(test_X_vectors, test_y)}')

#........................................3) Decision Tree ..........................
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(train_X_vectors, train_y)
dtc_prediction = dtc.predict(test_X_vectors)

print(f'Accuracy: {dtc.score(test_X_vectors, test_y)}')

#........................................4) XG-Boost.................................
from sklearn.ensemble import GradientBoostingClassifier

xgboost= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
                                    max_depth=1, random_state=0)
xgboost.fit(train_X_vectors, train_y)
xgboost_predicted= xgboost.predict(test_X_vectors)

print(f'Accuracy: {xgboost.score(test_X_vectors, test_y)}')

#..............................5) SVM.................................................
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(train_X_vectors, train_y)
svm_prediction = svm.predict(test_X_vectors)

print(f'Accuracy: {svm.score(test_X_vectors, test_y)}')



