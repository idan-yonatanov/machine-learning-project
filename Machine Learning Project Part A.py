# Imports
import random
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector



# Reading the data from pkl file 
data = pd.read_pickle('C:/Users/dimay/Downloads/XY_train.pkl')

# Fill some columns with None or nan with the string 'Nan' to indicate the null values
data.email = data.email.fillna("Nan")
data.gender = data.gender.fillna("Nan")
data.email_verified = data.email_verified.fillna("Nan")
data.blue_tick = data.blue_tick.fillna("Nan")
data.embedded_content = data.embedded_content.fillna("Nan")
data.platform = data.platform.fillna("Nan")

############################### EDA ###############################

############## sentiment feature ############## 
# Building a bar chart of sentiment feature
total_samples = len(data)
positive_count = data['sentiment'].value_counts().get('positive', 0)# Count positive sentiments
negative_count = data['sentiment'].value_counts().get('negative', 0)# Count negative sentiments
percentage_data = [positive_count / total_samples * 100,# Calculate the percentage of positive and negative sentiments in the data
                   negative_count / total_samples * 100]
labels = ['positive', 'negative']
bar_chart = plt.bar(labels, percentage_data, color=['blue', 'orange'])# Variable that stores a bar chart
for b in bar_chart:# Loop that it's purpose is to add a number value per column in the bar chart 
    height = round(b.get_height(),2)# Using round function to view the number value as 23.15 for example 
    plt.annotate( "{}%".format(height),(b.get_x() + b.get_width()/2,
                 height+.05),ha="center",va="bottom",fontsize=8)
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.title('Bar Chart Of Sentiment Feature')
plt.show()
# Building a heat map of correlations between sentiment feature and boolean features
# Conversion of boolean features to binary values as 0 or 1
df = pd.DataFrame(data)
boolean_features = df[['gender', 'email_verified', 'blue_tick', 'sentiment']]
boolean_features['sentiment'] = boolean_features['sentiment'].replace('positive', 1)
boolean_features['sentiment'] = boolean_features['sentiment'].replace('negative', 0)
boolean_features['gender'] = boolean_features['gender'].replace('M', 1)
boolean_features['gender'] = boolean_features['gender'].replace('F', 0)
boolean_features['email_verified'] = boolean_features['email_verified'].replace(True, 1)
boolean_features['email_verified'] = boolean_features['email_verified'].replace(False, 0)
boolean_features['blue_tick'] = boolean_features['blue_tick'].replace(True, 1)
boolean_features['blue_tick'] = boolean_features['blue_tick'].replace(False, 0)
# Heat map correlations
corr_matrix = boolean_features.corr()# Building a matrix of correlations between boolean features
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")#Building a heat map with seaborn module on the basis of the matrix
plt.title("Heat Map - Correlations Between Boolean Features")
plt.show()
############## End sentiment feature ############## 

############## text feature ##############  
# Building a dictionary of common words of the text feature 
data = data.astype({'text': str})# Convert the text features values to string
all_text = ' '.join(data['text'].tolist())# Create a list from all the values of the text feature
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase 
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
N = 15# Top 15 words in the text feature column 
top_words = dict(word_count.most_common(N))# Create a dictionary of keys and values of top words from word_count
# Plot of common words of text feature - bar chart
fig, ax = plt.subplots(figsize=(15, 5))# It creates a figure and axis object using the subplots function 
plt.bar(top_words.keys(), top_words.values(), width=0.5,)# Create bar chart from top_words
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top {} Most Common Words'.format(N))
plt.show()
# Word cloud of text feature
df_text_cloud = pd.DataFrame(data)
text_cloud = ' '.join(df_text_cloud['text'].dropna().astype(str).tolist())# Create a list of words from text feature
wordcloud_text = WordCloud(width=800, height=400, background_color='white').generate(text_cloud)# Create variable of word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud_text, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Of The Text Feature")
plt.show()
# Word cloud of text feature with respect to lable "negative" in sentiment feature
data_negative = data[data['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset 
text_negative = ' '.join(data_negative['text'].dropna().astype(str).tolist())# Create a list of words from text feature
wordcloud_text = WordCloud(width=800, height=400, background_color='white').generate(text_negative)# Create variable of word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud_text, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Of The Text Feature With Respect To Negative Sentiment")
plt.show()
# Plot of common words of text feature with respect to lable "negative" in sentiment feature - bar chart
data_negative = data_negative.astype({'text': str})# Convert the negative text lable values to string
all_text = ' '.join(data_negative['text'].tolist())# Create a list from all the values of the negative text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words = dict(word_count.most_common(N))# Create a dictionary of keys and values of top words from word_count
fig, ax = plt.subplots(figsize=(15, 5))# It creates a figure and axis object using the subplots function
plt.bar(top_words.keys(), top_words.values(), width=0.5)# Create bar chart from top_words
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top {} Most Common Words With Respect To Negative Sentiment'.format(N))
plt.show()
# Word cloud of text feature with respect to lable "positive" in sentiment feature
data_positive = data[data['sentiment'] == "positive"]# Extract the positive sentiment samples from the entire dataset 
text_positive = ' '.join(data_positive['text'].dropna().astype(str).tolist())# Create a list of words from text feature
wordcloud_text = WordCloud(width=800, height=400, background_color='white').generate(text_positive)# Create variable of word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud_text, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Of The Text Feature With Respect To Positive Sentiment")
plt.show()
# Plot of common words of text feature with respect to lable "positive" in sentiment feature - bar chart
data_positive = data_positive.astype({'text': str})# Convert the positive text lable values to string
all_text = ' '.join(data_positive['text'].tolist())# Create a list from all the values of the positive text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words = dict(word_count.most_common(N))# Create a dictionary of keys and values of top words from word_count
fig, ax = plt.subplots(figsize=(15, 5))# It creates a figure and axis object using the subplots function
plt.bar(top_words.keys(), top_words.values(), width=0.5)# Create bar chart from top_words
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top {} Most Common Words With Respect To positive Sentiment'.format(N))
plt.show()
############## End text feature ############## 

############## gender feature ############## 
# Bar chart of gender feature
colors = ['green','hotpink', 'steelblue', 'brown', 'orange', 'purple', 'yellow']# List of colors
class_counts = data['gender'].value_counts()# Count the number of instances of each lable of gender feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
bar_plot = ax.bar(class_counts.index, class_counts.values, color = colors)# Variable that stores a bar chart
for rect in bar_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of Gender Feature")
ax.set_xlabel("Gender")
ax.set_ylabel("Frequency")
plt.show()
# Pie chart of gender feature
data['gender'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of gender feature
plt.axis('off')
plt.title("Pie Chart Of Gender Feature")
plt.show()
# Bar chart of genders compare by sentiment feature
table_gender = pd.crosstab(data["gender"], data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_gender.plot(kind="bar", stacked=True)# Create a bar chart from table_gender
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Sentiment VS Gender")
plt.legend(title="sentiment", loc="upper right")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of genders compare by sentiment feature
gender_percentages = data.groupby('gender')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'gender' feature and then counts the occurrences of 'sentiment' values within each group.
gender_percentages = gender_percentages.unstack()
ax = gender_percentages.plot(kind='bar', stacked=True)# Create a bar chart from gender_percentages
plt.xlabel('Gender')
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Gender')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End gender feature ############## 

############## email_verified feature ##############
# Bar chart of email_verified feature
df_email_verified = pd.DataFrame(data)
df_email_verified['email_verified'] = df_email_verified['email_verified'].replace(True, 'True')# Replace the old values of the lable of email_verified feature with new one
df_email_verified['email_verified'] = df_email_verified['email_verified'].replace(False, 'False')# Replace the old values of the lable of email_verified feature with new one
verified_counts = df_email_verified['email_verified'].value_counts()# Count the number of instances of each lable of email_verified feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
email_verified_plot = ax.bar(verified_counts.index, verified_counts.values, color = colors)# Variable that stores a bar chart
for v in email_verified_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = v.get_height()
    ax.text(v.get_x() + v.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of Email_verified Feature")
ax.set_xlabel("Email_verified")
ax.set_ylabel("Frequency")
plt.show()
# Pie chart of email_verified feature
data['email_verified'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of email_verified feature
plt.axis('off')
plt.title("Pie Chart Of Email_verified Feature")
plt.show()
# Bar chart of email_verified compare by sentiment feature
table_email_verified = pd.crosstab(data["email_verified"], data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_email_verified.plot(kind="bar", stacked=True)# Create a bar chart from table_email_verified
plt.xlabel("Email_verified")
plt.ylabel("Frequency")
plt.title("Sentiment VS Email_verified")
plt.legend(title="sentiment", loc="upper right")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of email_verified compare by sentiment feature
email_verified_percentages = data.groupby('email_verified')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'email_verified' feature and then counts the occurrences of 'sentiment' values within each group.
email_verified_percentages = email_verified_percentages.unstack()
ax = email_verified_percentages.plot(kind='bar', stacked=True)# Create a bar chart from email_verified_percentages
plt.xlabel('Email_verified')
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Email_verified')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End email_verified feature ##############

############## email feature ##############
df_email = pd.DataFrame(data)
df_email['replace_email_values'] = ["email" if email != 'Nan' else "Nan" for email in df_email['email']]# Creating a new feature with a condition
email_counts = df_email['replace_email_values'].value_counts()# Count the number of instances of each lable of email feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
email_plot = ax.bar(email_counts.index, email_counts.values, color = colors)# Variable that stores a bar chart
for e in email_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = e.get_height()
    ax.text(e.get_x() + e.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of Email Feature")
ax.set_xlabel("Email")
ax.set_ylabel("Frequency")
plt.show()
# Pie chart of email feature
df_email['replace_email_values'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of email feature
plt.axis('off')
plt.title("Pie Chart Of Email Feature")
plt.show()
# Bar chart of email compare by sentiment feature
table_email = pd.crosstab(df_email['replace_email_values'], df_email["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_email.plot(kind="bar", stacked=True)# Create a bar chart from table_email
plt.xlabel("Email")
plt.ylabel("Frequency")
plt.title("Sentiment VS Email")
plt.legend(title="sentiment", loc="upper left")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of email compare by sentiment feature
email_percentages = df_email.groupby('replace_email_values')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'email' feature and then counts the occurrences of 'sentiment' values within each group.
email_percentages = email_percentages.unstack()
ax = email_percentages.plot(kind='bar', stacked=True)# Create a bar chart from email_percentages
plt.xlabel("Email")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Email')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End email feature ##############

############## blue_tick feature ##############
# Bar chart of blue_tick feature
df_blue_tick = pd.DataFrame(data)
df_blue_tick['blue_tick'] = df_blue_tick['blue_tick'].replace(True, 'True')# Replace the old values of the lable of blue_tick feature with new one
df_blue_tick['blue_tick'] = df_blue_tick['blue_tick'].replace(False, 'False')# Replace the old values of the lable of blue_tick feature with new one
blue_tick_counts = df_blue_tick['blue_tick'].value_counts()# Count the number of instances of each lable of blue_tick feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
blue_tick_plot = ax.bar(blue_tick_counts.index, blue_tick_counts.values, color = colors)# Variable that stores a bar chart
for t in blue_tick_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = t.get_height()
    ax.text(t.get_x() + t.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of Blue_tick Feature")
ax.set_xlabel("Blue_tick")
ax.set_ylabel("Frequency")
plt.show()
# Pie chart of blue_tick feature
data['blue_tick'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of blue_tick feature
plt.axis('off')
plt.title("Pie Chart Of Blue_tick Feature")
plt.show()
# Bar chart of blue_tick compare by sentiment feature
table_blue_tick = pd.crosstab(data["blue_tick"], data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_blue_tick.plot(kind="bar", stacked=True)# Create a bar chart from blue_tick
plt.xlabel("Blue_tick")
plt.ylabel("Frequency")
plt.title("Sentiment VS Blue_tick")
plt.legend(title="sentiment", loc="upper right")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of blue_tick compare by sentiment feature
blue_tick_percentages = df_blue_tick.groupby('blue_tick')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'blue_tick' feature and then counts the occurrences of 'sentiment' values within each group.
blue_tick_percentages = blue_tick_percentages.unstack()
ax = blue_tick_percentages.plot(kind='bar', stacked=True)# Create a bar chart from blue_tick_percentages
plt.xlabel("Blue_tick")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Blue_tick')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End blue_tick feature ##############

############## embedded_content feature ##############
# Bar chart of embedded_content feature
df_embedded_content = pd.DataFrame(data)
df_embedded_content['embedded_content'] = df_embedded_content['embedded_content'].replace(False, 'False')# Replace the old values of the lable of embedded_content feature with new one
embedded_content_counts = df_embedded_content['embedded_content'].value_counts()# Count the number of instances of each lable of embedded_content feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
embedded_content_plot = ax.bar(embedded_content_counts.index, embedded_content_counts.values, color = colors)# Variable that stores a bar chart
for e in embedded_content_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = e.get_height()
    ax.text(e.get_x() + e.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of Embedded_content Feature")
ax.set_xlabel("Embedded_content")
ax.set_ylabel("Frequency")
plt.show()
# Pie chart of embedded_content feature
data['embedded_content'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of embedded_content feature
plt.axis('off')
plt.title("Pie Chart Of Embedded_content Feature")
plt.show()
# Bar chart of embedded_content compare by sentiment feature
table_embedded_content = pd.crosstab(data["sentiment"], data["embedded_content"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_embedded_content.plot(kind="bar", stacked=False)# Create a bar chart from embedded_content
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
plt.title("Sentiment VS Embedded_content")
plt.legend(title="embedded_content", loc="upper right", fontsize = "7")
plt.xticks(rotation=0)
plt.show()
# Pie chart of embedded_content with respect to lable "negative" in sentiment feature
data_negative_content = data[data['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset  
data_negative_content['embedded_content'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of embedded_content feature
plt.axis('off')
plt.title("Pie Chart Of Embedded_content Feature With Respect To Negative Sentiment")
plt.show()
# Relative Stacked bar chart of embedded_content compare by sentiment feature
embedded_content_percentages = df_embedded_content.groupby('embedded_content')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'embedded_content' feature and then counts the occurrences of 'sentiment' values within each group.
embedded_content_percentages = embedded_content_percentages.unstack()
ax = embedded_content_percentages.plot(kind='bar', stacked=True)# Create a bar chart from embedded_content_percentages
plt.xlabel("Embedded_content")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Embedded_content')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End embedded_content feature ##############

############## platform feature ##############
# Bar chart of platform feature
platform_counts = data['platform'].value_counts()# Count the number of instances of each lable of platform feature
fig, ax = plt.subplots()# It creates a figure and axis object using the subplots function
platform_plot = ax.bar(platform_counts.index, platform_counts.values, color = colors)# Variable that stores a bar chart
for p in platform_plot:# Loop that it's purpose is to add a number value per column in the bar chart
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.0, height, str(height), ha='center', va='bottom')
ax.set_title("Bar Chart Of platform Feature")
ax.set_xlabel("Platform")
ax.set_ylabel("Frequency")
fig.set_size_inches(10, 5)
plt.show()
# Pie chart of platform feature
data['platform'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of platform feature
plt.axis('off')
plt.title("Pie Chart Of platform Feature")
plt.show()
# Bar chart of platform compare by sentiment feature
table_platform = pd.crosstab(data["sentiment"], data["platform"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_platform.plot(kind="bar", stacked=False)# Create a bar chart from platform
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
plt.title("Sentiment VS Platform")
plt.legend(title="platform", loc="upper right", fontsize = "7")
plt.xticks(rotation=0)
plt.show()
# Pie chart of platform with respect to lable "negative" in sentiment feature
data_negative_platform = data[data['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset  
data_negative_platform['platform'].value_counts().plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of platform feature
plt.axis('off')
plt.title("Pie Chart Of platform Feature With Respect To Negative Sentiment")
plt.show()
# Relative Stacked bar chart of platform compare by sentiment feature
platform_percentages = data.groupby('platform')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'platform' feature and then counts the occurrences of 'sentiment' values within each group.
platform_percentages = platform_percentages.unstack()
ax = platform_percentages.plot(kind='bar', stacked=True)# Create a bar chart from platform_percentages
plt.xlabel("Platform")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Platform')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
############## End platform feature ##############

##############  date_of_new_follower feature ##############
new_follower_data = pd.DataFrame(data)
new_follower_data['number_of_followers'] = new_follower_data['date_of_new_follower'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
# Bar chart of date_of_new_follower compare by sentiment feature
new_follower_vs_sentiment = new_follower_data.groupby('sentiment')['number_of_followers'].sum()# Calculating the sum of 'number_of_followers' and grouping by 'sentiment'
ax = new_follower_vs_sentiment.plot(kind='bar', stacked=False, color = colors)# Create a bar chart
for i, v in enumerate(new_follower_vs_sentiment):# Adding numeric values on top of the bars
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
plt.xlabel("Sentiment")
plt.ylabel("Number Of Followers")
plt.title("Sentiment VS Date_of_new_follower")
plt.xticks(rotation=0)
plt.show()
# Pie chart of date_of_new_follower feature
new_follower_vs_sentiment.plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of date_of_new_follower feature
plt.axis('off')
plt.title("Pie Chart Of Number Of  Followers Per Sentiment")
plt.show()
# Relative Stacked bar chart of date_of_new_follower compare by sentiment feature
new_follower_percentages = new_follower_data.groupby('number_of_followers')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'number_of_followers' feature and then counts the occurrences of 'sentiment' values within each group.
new_follower_percentages = new_follower_percentages.unstack()
ax = new_follower_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Number Of Followers")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Number Of Followers')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.13, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(15, 7)
plt.show()
##############  End date_of_new_follower feature ##############

##############  date_of_new_follow feature ##############
new_follow_data = pd.DataFrame(data)
new_follow_data['number_of_follows'] = new_follow_data['date_of_new_follow'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
# Bar chart of date_of_new_follow compare by sentiment feature
new_follow_vs_sentiment = new_follow_data.groupby('sentiment')['number_of_follows'].sum()# Calculating the sum of 'number_of_follows' and grouping by 'sentiment'
ax = new_follow_vs_sentiment.plot(kind='bar', stacked=False, color = colors)# Create a bar chart
for i, v in enumerate(new_follow_vs_sentiment):# Adding numeric values on top of the bars
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
plt.xlabel("Sentiment")
plt.ylabel("Number Of Follows")
plt.title("Sentiment VS Date_of_new_follow")
plt.xticks(rotation=0)
plt.show()
# Pie chart of date_of_new_follow feature
new_follow_vs_sentiment.plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of date_of_new_follow feature
plt.axis('off')
plt.title("Pie Chart Of Number Of  Follows Per Sentiment")
plt.show()
# Relative Stacked bar chart of date_of_new_follow compare by sentiment feature
new_follow_percentages = new_follow_data.groupby('number_of_follows')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'number_of_follows' feature and then counts the occurrences of 'sentiment' values within each group.
new_follow_percentages = new_follow_percentages.unstack()
ax = new_follow_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Number Of Follows")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Number Of Follows')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.13, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(15, 7)
plt.show()
##############  End date_of_new_follow feature ##############

##############  previous_messages_dates feature ##############
previous_messages_data = pd.DataFrame(data)
previous_messages_data['number_of_previous_messages'] = previous_messages_data['previous_messages_dates'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
# Bar chart of previous_messages_dates compare by sentiment feature
previous_messages_vs_sentiment = previous_messages_data.groupby('sentiment')['number_of_previous_messages'].sum()# Calculating the sum of 'number_of_previous_messages' and grouping by 'sentiment'
ax = previous_messages_vs_sentiment.plot(kind='bar', stacked=False, color = colors)# Create a bar chart
for i, v in enumerate(previous_messages_vs_sentiment):# Adding numeric values on top of the bars
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
plt.xlabel("Sentiment")
plt.ylabel("Number Of Previous Messages")
plt.title("Sentiment VS Previous_messages_dates")
plt.xticks(rotation=0)
plt.show()
# Pie chart of previous_messages_dates feature
previous_messages_vs_sentiment.plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of previous_messages_dates feature
plt.axis('off')
plt.title("Pie Chart Of Number Of Previous Messages Per Sentiment")
plt.show()
# Relative Stacked bar chart of previous_messages_dates compare by sentiment feature
previous_messages_percentages = previous_messages_data.groupby('number_of_previous_messages')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'number_of_previous_messages' feature and then counts the occurrences of 'sentiment' values within each group.
previous_messages_percentages = previous_messages_percentages.unstack()
ax = previous_messages_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Number Of Previous Messages")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Number Of Previous Messages')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.13, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(15, 7)
plt.show()
##############  End previous_messages_dates feature ##############

##############  account_creation_date feature ##############
account_creation_data = pd.DataFrame(data)
account_creation_data['account_creation_date'] = pd.to_datetime(account_creation_data['account_creation_date'])# Convert 'account_creation_date' column to datetime type
account_creation_data['year'] = account_creation_data['account_creation_date'].dt.year# Extract year into separate column
account_creation_data['month'] = account_creation_data['account_creation_date'].dt.month# Extract month into separate column
# Bar chart of 'year' of account_creation_date compare by sentiment feature
table_account_creation_year = pd.crosstab(account_creation_data["year"], account_creation_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_account_creation_year.plot(kind="bar", stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel("Frequency")
plt.title("Sentiment VS Account_creation_date - Year")
plt.legend(title="sentiment", loc="upper right")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of 'year' of account_creation_date compare by sentiment feature
account_creation_year_percentages = account_creation_data.groupby('year')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'year' feature and then counts the occurrences of 'sentiment' values within each group.
account_creation_year_percentages = account_creation_year_percentages.unstack()
ax = account_creation_year_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Account_creation Year')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Bar chart of 'month' of account_creation_date compare by sentiment feature
table_account_creation_month = pd.crosstab(account_creation_data["month"], account_creation_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_account_creation_month.plot(kind="bar", stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel("Frequency")
plt.title("Sentiment VS Account_creation_date - Month")
plt.legend(title="sentiment", loc="upper left", fontsize = "9")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of 'month' of account_creation_date compare by sentiment feature
account_creation_month_percentages = account_creation_data.groupby('month')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'month' feature and then counts the occurrences of 'sentiment' values within each group.
account_creation_month_percentages = account_creation_month_percentages.unstack()
ax = account_creation_month_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Account_creation Month')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of account_creation_date (hours) compare by sentiment feature
account_creation_data['hour'] = account_creation_data['account_creation_date'].dt.hour# Extract hour into separate column
account_creation_data['hour_for_label'] = account_creation_data['hour'].apply(lambda x: '0 - 7' if x >= 0 and x < 8# Creating a label of hours for each hour in his right domain
                                                                      else ('8 - 15' if x >= 8 and x < 16 else '16 - 23'))
account_creation_hour_percentages = account_creation_data.groupby('hour_for_label')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'hour_for_label' feature and then counts the occurrences of 'sentiment' values within each group.
account_creation_hour_percentages = account_creation_hour_percentages.unstack()
ax = account_creation_hour_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Hours Range")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Account Creation Hours')
plt.legend(labels=['negative', 'positive'], loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# Bar chart of account_creation_date (hours) compare by sentiment feature
table_account_creation = pd.crosstab(account_creation_data["hour_for_label"], account_creation_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_account_creation.plot(kind="bar", stacked=False)# Create a bar chart from account_creation_date
plt.xlabel("Hours Range")
plt.ylabel("Frequency")
plt.title("Sentiment VS Account_creation_date")
plt.legend(title="sentiment", loc="upper left", fontsize = "5")
plt.xticks(rotation=0)
plt.show()
##############  End account_creation_date feature ##############

##############  message_date feature ##############
message_date_data = pd.DataFrame(data)
message_date_data['message_date'] = pd.to_datetime(message_date_data['message_date'])# Convert 'message_date' column to datetime type
message_date_data['year'] = message_date_data['message_date'].dt.year# Extract year into separate column
message_date_data['month'] = message_date_data['message_date'].dt.month# Extract month into separate column
# Bar chart of 'year' of message_date compare by sentiment feature
table_message_date_year = pd.crosstab(message_date_data["year"], message_date_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_message_date_year.plot(kind="bar", stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel("Frequency")
plt.title("Sentiment VS Message_date - Year")
plt.legend(title="sentiment", loc="upper left")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of 'year' of message_date compare by sentiment feature
message_date_year_percentages = message_date_data.groupby('year')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'year' feature and then counts the occurrences of 'sentiment' values within each group.
message_date_year_percentages = message_date_year_percentages.unstack()
ax = message_date_year_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Message_date Year')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Bar chart of 'month' of message_date compare by sentiment feature
table_message_date_month = pd.crosstab(message_date_data["month"], message_date_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_message_date_month.plot(kind="bar", stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel("Frequency")
plt.title("Sentiment VS Message_date - Month")
plt.legend(title="sentiment", loc="best", fontsize = "6.5")
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of 'month' of message_date compare by sentiment feature
message_date_month_percentages = message_date_data.groupby('month')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'month' feature and then counts the occurrences of 'sentiment' values within each group.
message_date_month_percentages = message_date_month_percentages.unstack()
ax = message_date_month_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Message_date Month')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of message_date (hours) compare by sentiment feature
message_date_data['hour'] = message_date_data['message_date'].dt.hour# Extract hour into separate column
message_date_data['hour_for_label'] = message_date_data['hour'].apply(lambda x: '0 - 7' if x >= 0 and x < 8# Creating a label of hours for each hour in his right domain
                                                                      else ('8 - 15' if x >= 8 and x < 16 else '16 - 23'))
message_date_hour_percentages = message_date_data.groupby('hour_for_label')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'hour_for_label' feature and then counts the occurrences of 'sentiment' values within each group.
message_date_hour_percentages = message_date_hour_percentages.unstack()
ax = message_date_hour_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Hours Range")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Message_date Hours')
plt.legend(labels=['negative', 'positive'], loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# Bar chart of message_date (hours) compare by sentiment feature
table_message_date = pd.crosstab(message_date_data["hour_for_label"], message_date_data["sentiment"])# Create a contingency table, also known as a cross-tabulation or a crosstab
table_message_date.plot(kind="bar", stacked=False)# Create a bar chart from message_date
plt.xlabel("Hours Range")
plt.ylabel("Frequency")
plt.title("Sentiment VS Message_date")
plt.legend(title="sentiment", loc="upper right", fontsize = "9")
plt.xticks(rotation=0)
plt.show()
##############  End message_date feature ##############

############## Interesting relationships between explanatory variables ##############
# gender vs embedded_content
relationships_data = pd.DataFrame(data)
ge_percentages = relationships_data.groupby('embedded_content')['gender'].value_counts(normalize=True) * 100# Groups the data by the 'embedded_content' feature and then counts the occurrences of 'gender' values within each group
ge_percentages = ge_percentages.unstack()
ax = ge_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Embedded_content")
plt.ylabel('Frequency %')
plt.title('Gender vs Embedded_content')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# gender vs platform
gp_percentages = relationships_data.groupby('platform')['gender'].value_counts(normalize=True) * 100# Groups the data by the 'platform' feature and then counts the occurrences of 'gender' values within each group
gp_percentages = gp_percentages.unstack()
ax = gp_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Platform")
plt.ylabel('Frequency %')
plt.title('Gender vs Platform')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(7, 5)
plt.show()
# embedded_content vs platform
ep_percentages = relationships_data.groupby('platform')['embedded_content'].value_counts(normalize=True) * 100# Groups the data by the 'platform' feature and then counts the occurrences of 'embedded_content' values within each group
ep_percentages = ep_percentages.unstack()
ax = ep_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Platform")
plt.ylabel('Frequency %')
plt.title('Embedded_content vs Platform')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(7, 5)
plt.show()
# previous_messages_dates vs gender 
# Bar chart of previous_messages_dates compare by gender feature
previous_messages_vs_gender = previous_messages_data.groupby('gender')['number_of_previous_messages'].sum()# Calculating the sum of 'number_of_previous_messages' and grouping by 'gender'
ax = previous_messages_vs_gender.plot(kind='bar', stacked=True, color = colors)# Create a bar chart
for i, v in enumerate(previous_messages_vs_gender):# Adding numeric values on top of the bars
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
plt.xlabel("Gender")
plt.ylabel("Number Of Previous Messages")
plt.title("Gender VS Previous_messages_dates")
plt.xticks(rotation=0)
plt.show()
# Pie chart of previous_messages_dates vs gender feature
previous_messages_vs_gender.plot(kind='pie', y='', autopct='%1.0f%%')# Creating a pie chart from the properties of previous_messages_dates feature
plt.axis('off')
plt.title("Pie Chart Of Gender VS Previous_messages_dates")
plt.show()
# message_date vs platform
# Relative Stacked bar chart of 'year' of message_date compare by platform feature
message_date_year_percentages = message_date_data.groupby('year')['platform'].value_counts(normalize=True) * 100# Groups the data by the 'year' feature and then counts the occurrences of 'platform' values within each group.
message_date_year_percentages = message_date_year_percentages.unstack()
ax = message_date_year_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel('Frequency %')
plt.title('Platform VS Message_date - Year')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of 'month' of message_date compare by platform feature
message_date_month_percentages = message_date_data.groupby('month')['platform'].value_counts(normalize=True) * 100# Groups the data by the 'month' feature and then counts the occurrences of 'platform' values within each group.
message_date_month_percentages = message_date_month_percentages.unstack()
ax = message_date_month_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel('Frequency %')
plt.title('Platform VS Message_date - Month')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of message_date (hours) compare by platform feature
message_date_hour_percentages = message_date_data.groupby('hour_for_label')['platform'].value_counts(normalize=True) * 100# Groups the data by the 'hour_for_label' feature and then counts the occurrences of 'platform' values within each group.
message_date_hour_percentages = message_date_hour_percentages.unstack()
ax = message_date_hour_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Hours Range")
plt.ylabel('Frequency %')
plt.title('Platform VS Message_date - Hours')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# message_date vs email_verified
# Relative Stacked bar chart of 'year' of message_date compare by email_verified feature
message_date_year_percentages = message_date_data.groupby('year')['email_verified'].value_counts(normalize=True) * 100# Groups the data by the 'year' feature and then counts the occurrences of 'email_verified' values within each group.
message_date_year_percentages = message_date_year_percentages.unstack()
ax = message_date_year_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel('Frequency %')
plt.title('Email_verified VS Message_date - Year')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of 'month' of message_date compare by email_verified feature
message_date_month_percentages = message_date_data.groupby('month')['email_verified'].value_counts(normalize=True) * 100# Groups the data by the 'month' feature and then counts the occurrences of 'email_verified' values within each group.
message_date_month_percentages = message_date_month_percentages.unstack()
ax = message_date_month_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel('Frequency %')
plt.title('Email_verified VS Message_date - Month')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of message_date (hours) compare by email_verified feature
message_date_hour_percentages = message_date_data.groupby('hour_for_label')['email_verified'].value_counts(normalize=True) * 100# Groups the data by the 'hour_for_label' feature and then counts the occurrences of 'email_verified' values within each group.
message_date_hour_percentages = message_date_hour_percentages.unstack()
ax = message_date_hour_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Hours Range")
plt.ylabel('Frequency %')
plt.title('Email_verified VS Message_date - Hours')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# message_date vs blue_tick
# Relative Stacked bar chart of 'year' of message_date compare by blue_tick feature
message_date_year_percentages = message_date_data.groupby('year')['blue_tick'].value_counts(normalize=True) * 100# Groups the data by the 'year' feature and then counts the occurrences of 'blue_tick' values within each group.
message_date_year_percentages = message_date_year_percentages.unstack()
ax = message_date_year_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Years")
plt.ylabel('Frequency %')
plt.title('Blue_tick VS Message_date - Year')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of 'month' of message_date compare by blue_tick feature
message_date_month_percentages = message_date_data.groupby('month')['blue_tick'].value_counts(normalize=True) * 100# Groups the data by the 'month' feature and then counts the occurrences of 'blue_tick' values within each group.
message_date_month_percentages = message_date_month_percentages.unstack()
ax = message_date_month_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Months")
plt.ylabel('Frequency %')
plt.title('Blue_tick VS Message_date - Month')
plt.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
# Relative Stacked bar chart of message_date (hours) compare by blue_tick feature
message_date_hour_percentages = message_date_data.groupby('hour_for_label')['blue_tick'].value_counts(normalize=True) * 100# Groups the data by the 'hour_for_label' feature and then counts the occurrences of 'blue_tick' values within each group.
message_date_hour_percentages = message_date_hour_percentages.unstack()
ax = message_date_hour_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Hours Range")
plt.ylabel('Frequency %')
plt.title('Blue_tick VS Message_date - Hours')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
############## End interesting relationships between explanatory variables ##############

############################### End EDA ###############################


############################### Dataset Creation ###############################

############## Pre - Processing ##############
data_creation = pd.read_pickle('C:/Users/dimay/Downloads/XY_train.pkl')

#### Handling missing values ####
print(data_creation.isnull().sum())
# Missing values - gender
print(data_creation['gender'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['gender'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['gender'].isnull().sum()}")
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'gender')
print(data_creation['gender'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['gender'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['gender'].isnull().sum()}")
# Missing values - email_verified
print(data_creation['email_verified'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['email_verified'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['email_verified'].isnull().sum()}")
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'email_verified')
print(data_creation['email_verified'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['email_verified'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['email_verified'].isnull().sum()}")
# Missing values - blue_tick
print(data_creation['blue_tick'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['blue_tick'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['blue_tick'].isnull().sum()}")
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'blue_tick')
print(data_creation['blue_tick'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['blue_tick'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['blue_tick'].isnull().sum()}")
# Missing values - embedded_content
print(data_creation['embedded_content'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['embedded_content'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['embedded_content'].isnull().sum()}")
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'embedded_content')
print(data_creation['embedded_content'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['embedded_content'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['embedded_content'].isnull().sum()}")
# Missing values - platform
print(data_creation['platform'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['platform'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['platform'].isnull().sum()}")
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'platform')
print(data_creation['platform'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['platform'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['platform'].isnull().sum()}")
# Missing values - email
data_creation.email = data_creation.email.fillna("email")
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
data_creation['email'] = data_creation['email'].replace('email', None)
data_creation['email_domain_suffix'] = data_creation['email_domain_suffix'].replace('email', None)
print(data_creation['email_domain_suffix'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['email_domain_suffix'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['email'].isnull().sum()}")
def fill_null_with_category(df1, column_target, column_temp):# Fill nulls for categorical variables
    value_counts = df1[column_temp].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column_target].isnull()].tolist()# Generate a list with missing values in the specified column
    username_length = random.randint(5, 10)# Random length for the username
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    df1.loc[null_indices, column_target] = username + '.' + np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category 
fill_null_with_category(data_creation, 'email', 'email_domain_suffix')
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
print(data_creation['email_domain_suffix'].value_counts(normalize=True))
print(data_creation.groupby('sentiment')['email_domain_suffix'].value_counts(normalize=True) * 100)
print(f"Number of nulls: {data_creation['email'].isnull().sum()}")
#### End handling missing values ####

#### Chacking relationships between categorical feature to sentiment feature ####
# Relative Stacked bar chart of embedded_content compare by sentiment feature
embedded_content_percentages = data_creation.groupby('embedded_content')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'embedded_content' feature and then counts the occurrences of 'sentiment' values within each group.
embedded_content_percentages = embedded_content_percentages.unstack()
ax = embedded_content_percentages.plot(kind='bar', stacked=True)# Create a bar chart from embedded_content_percentages
plt.xlabel("Embedded_content")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Embedded_content')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of blue_tick compare by sentiment feature
blue_tick_percentages = data_creation.groupby('blue_tick')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'blue_tick' feature and then counts the occurrences of 'sentiment' values within each group.
blue_tick_percentages = blue_tick_percentages.unstack()
ax = blue_tick_percentages.plot(kind='bar', stacked=True)# Create a bar chart from blue_tick_percentages
plt.xlabel("Blue_tick")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Blue_tick')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
plt.show()
# Relative Stacked bar chart of platform compare by sentiment feature
platform_percentages = data_creation.groupby('platform')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'platform' feature and then counts the occurrences of 'sentiment' values within each group.
platform_percentages = platform_percentages.unstack()
ax = platform_percentages.plot(kind='bar', stacked=True)# Create a bar chart from platform_percentages
plt.xlabel("Platform")
plt.ylabel('Frequency %')
plt.title('Percentage Of Types Of Sentiment With Respect To Platform')
plt.legend(labels=['negative', 'positive'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
#### End chacking relationships between categorical feature to sentiment feature ####

############## End Pre - Processing ##############

############## Feature extraction ##############
### email_domain_suffix feature ###
# Relative Stacked bar chart of email_domain_suffix compare by sentiment feature
email_domain_suffix_percentages = data_creation.groupby('email_domain_suffix')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'email_domain_suffix' feature and then counts the occurrences of 'sentiment' values within each group.
email_domain_suffix_percentages = email_domain_suffix_percentages.unstack()
ax = email_domain_suffix_percentages.plot(kind='bar', stacked=True)# Create a bar chart sentiment
plt.xlabel("Email_domain_suffix")
plt.ylabel('Frequency %')
plt.title('Email_domain_suffix VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
plt.show()
### number_of_previous_messages feature ### 
data_creation['number_of_previous_messages'] = data_creation['previous_messages_dates'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
### number_of_followers feature ###
data_creation['number_of_followers'] = data_creation['date_of_new_follower'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
### number_of_follows feature ###
data_creation['number_of_follows'] = data_creation['date_of_new_follow'].apply(lambda x: len(x))# Count the number of datetime in a list for each row
### Creating features from account_creation_date feature - account_creation_year, account_creation_month and hour_ranges_of_account_creation ###
data_creation['account_creation_date'] = pd.to_datetime(data_creation['account_creation_date'])# Convert 'account_creation_date' column to datetime type
data_creation['account_creation_year'] = data_creation['account_creation_date'].dt.year# Extract year into separate column
data_creation['account_creation_month'] = data_creation['account_creation_date'].dt.month# Extract month into separate column
data_creation['hour'] = data_creation['account_creation_date'].dt.hour# Extract hour into separate column
data_creation['hour_ranges_of_account_creation'] = data_creation['hour'].apply(lambda x: '0 - 7' if x >= 0 and x < 8# Creating a label of hours for each hour in his right domain
                                                                      else ('8 - 15' if x >= 8 and x < 16 else '16 - 23'))
data_creation = data_creation.drop('hour', axis = 1)
### Creating features from message_date feature - message_date_year, message_date_month and hour_ranges_of_message_date ###
data_creation['message_date'] = pd.to_datetime(data_creation['message_date'])# Convert 'message_date' column to datetime type
data_creation['message_date_year'] = data_creation['message_date'].dt.year# Extract year into separate column
data_creation['message_date_month'] = data_creation['message_date'].dt.month# Extract month into separate column
data_creation['hour'] = data_creation['message_date'].dt.hour# Extract hour into separate column
data_creation['hour_ranges_of_message_date'] = data_creation['hour'].apply(lambda x: '0 - 7' if x >= 0 and x < 8# Creating a label of hours for each hour in his right domain
                                                                      else ('8 - 15' if x >= 8 and x < 16 else '16 - 23'))
data_creation = data_creation.drop('hour', axis = 1)
### seniority feature ###
message_date_year_list = data_creation['message_date_year'].tolist()
account_creation_year_list = data_creation['account_creation_year'].tolist()
year_gap_list = []  
for index, val in enumerate(message_date_year_list):# Calculate the gap of years and stores it in year_gap_list
    year_gap = val - account_creation_year_list[index]
    year_gap_list.append(year_gap)
year_gap = pd.Series(year_gap_list, name='seniority')# Convert the list to a pandas Series
data_creation['seniority'] = year_gap# Add the Series to the DataFrame
# Relative Stacked bar chart of seniority compare by sentiment feature
seniority_percentages = data_creation.groupby('seniority')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'seniority' feature and then counts the occurrences of 'sentiment' values within each group.
seniority_percentages = seniority_percentages.unstack()
ax = seniority_percentages.plot(kind='bar', stacked=True)# Create a bar chart sentiment
plt.xlabel("Seniority")
plt.ylabel('Frequency %')
plt.title('Seniority VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
plt.show() 
### text_word_count feature ###
data_creation['text_word_count'] = data_creation['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)# count the number of words in the text column for each row
# Relative Stacked bar chart of text_word_count compare by sentiment feature
text_word_count_percentages = data_creation.groupby('text_word_count')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'text_word_count' feature and then counts the occurrences of 'sentiment' values within each group.
text_word_count_percentages = text_word_count_percentages.unstack()
ax = text_word_count_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Text_word_count")
plt.ylabel('Frequency %')
plt.title('Text_word_count VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
### sum_top_common_negative_words feature ###
data_negative = data_creation[data_creation['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset
data_negative = data_negative.astype({'text': str})# Convert the negative text lable values to string
all_text = ' '.join(data_negative['text'].tolist())# Create a list from all the values of the negative text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words = dict(word_count.most_common(N))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top negative words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(N))
    return sum(row_top_words.get(word, 0) for word in top_words.keys())
data_creation['sum_top_common_negative_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature
# Relative Stacked bar chart of sum_top_common_negative_words compare by sentiment feature
sum_top_common_negative_words_percentages = data_creation.groupby('sum_top_common_negative_words')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'sum_top_common_negative_words' feature and then counts the occurrences of 'sentiment' values within each group.
sum_top_common_negative_words_percentages = sum_top_common_negative_words_percentages.unstack()
ax = sum_top_common_negative_words_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Sum_top_common_negative_words")
plt.ylabel('Frequency %')
plt.title('Sum_top_common_negative_words VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
### top_common_negative_words_percentage feature ###
data_creation['top_common_negative_words_percentage'] = data_creation['sum_top_common_negative_words'] / data_creation['text_word_count']
data_creation['top_common_negative_words_percentage'] = data_creation['top_common_negative_words_percentage'].fillna(0)
# Relative Stacked bar chart of top_common_negative_words_percentage compare by sentiment feature
top_common_negative_words_percentage_percentages = data_creation.groupby('top_common_negative_words_percentage')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'top_common_negative_words_percentage' feature and then counts the occurrences of 'sentiment' values within each group.
top_common_negative_words_percentage_percentages = top_common_negative_words_percentage_percentages.unstack()
ax = top_common_negative_words_percentage_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Top_common_negative_words_percentage")
plt.ylabel('Frequency %')
plt.title('Top_common_negative_words_percentage VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
### sum_top_common_positive_words feature ###
data_positive = data_creation[data_creation['sentiment'] == "positive"]# Extract the positive sentiment samples from the entire dataset
data_positive = data_positive.astype({'text': str})# Convert the positive text lable values to string
all_text = ' '.join(data_positive['text'].tolist())# Create a list from all the values of the positive text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words = dict(word_count.most_common(N))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top positive words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(N))
    return sum(row_top_words.get(word, 0) for word in top_words.keys())
data_creation['sum_top_common_positive_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature
# Relative Stacked bar chart of sum_top_common_positive_words compare by sentiment feature
sum_top_common_positive_words_percentages = data_creation.groupby('sum_top_common_positive_words')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'sum_top_common_positive_words' feature and then counts the occurrences of 'sentiment' values within each group.
sum_top_common_positive_words_percentages = sum_top_common_positive_words_percentages.unstack()
ax = sum_top_common_positive_words_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Sum_top_common_positive_words")
plt.ylabel('Frequency %')
plt.title('Sum_top_common_positive_words VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
### top_common_positive_words_percentage feature ###
data_creation['top_common_positive_words_percentage'] = data_creation['sum_top_common_positive_words'] / data_creation['text_word_count']
data_creation['top_common_positive_words_percentage'] = data_creation['top_common_positive_words_percentage'].fillna(0)
# Relative Stacked bar chart of top_common_positive_words_percentage compare by sentiment feature
top_common_positive_words_percentage_percentages = data_creation.groupby('top_common_positive_words_percentage')['sentiment'].value_counts(normalize=True) * 100# Groups the data by the 'top_common_positive_words_percentage' feature and then counts the occurrences of 'sentiment' values within each group.
top_common_positive_words_percentage_percentages = top_common_positive_words_percentage_percentages.unstack()
ax = top_common_positive_words_percentage_percentages.plot(kind='bar', stacked=True)# Create a bar chart 
plt.xlabel("Top_common_positive_words_percentage")
plt.ylabel('Frequency %')
plt.title('Top_common_positive_words_percentage VS Sentiment')
plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.3, 1))
plt.xticks(rotation=0)
fig = ax.get_figure()
fig.set_size_inches(10, 5)
plt.show()
### Features drop after operating feature extraction ###
data_creation.drop('text', axis=1, inplace=True)
data_creation.drop('message_date', axis=1, inplace=True)
data_creation.drop('account_creation_date', axis=1, inplace=True)
data_creation.drop('previous_messages_dates', axis=1, inplace=True)
data_creation.drop('date_of_new_follower', axis=1, inplace=True)
data_creation.drop('date_of_new_follow', axis=1, inplace=True)
data_creation.drop('email', axis=1, inplace=True)
############## End feature extraction ##############

############## Feature representation ##############
# Droping textID because we will not representation it
data_creation.drop('textID', axis=1, inplace=True)
# Numeric features normalizing
def minmax_scale_column(df1, column_to_scale):# MinMaxScaler - normalizing
    scaler = MinMaxScaler()# Initialize the scaler
    scaler.fit(df1[[column_to_scale]])# Fit the scaler to the column
    df1[column_to_scale] = scaler.transform(df1[[column_to_scale]])# Transform the column using the scaler
    return df1
data_creation = minmax_scale_column(data_creation, 'number_of_previous_messages')
data_creation = minmax_scale_column(data_creation, 'text_word_count')
data_creation = minmax_scale_column(data_creation, 'sum_top_common_negative_words')
data_creation = minmax_scale_column(data_creation, 'sum_top_common_positive_words')
data_creation = minmax_scale_column(data_creation, 'number_of_followers')
data_creation = minmax_scale_column(data_creation, 'number_of_follows')
### One hot encoding - category encoders ###
# gender representation
gender = pd.get_dummies(data_creation['gender'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, gender], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation = data_creation.astype({'F': int})# Convert the feature values to int
data_creation = data_creation.astype({'M': int})# Convert the feature values to int
data_creation.drop('gender', axis=1, inplace=True)
# email_verified representation
data_creation['email_verified'] = data_creation['email_verified'].replace(True , 1)# Conversion of boolean features to binary values as 0 or 1
data_creation['email_verified'] = data_creation['email_verified'].replace(False , 0)# Conversion of boolean features to binary values as 0 or 1
# blue_tick representation
data_creation['blue_tick'] = data_creation['blue_tick'].replace(True , 1)# Conversion of boolean features to binary values as 0 or 1
data_creation['blue_tick'] = data_creation['blue_tick'].replace(False , 0)# Conversion of boolean features to binary values as 0 or 1
# sentiment representation
data_creation['sentiment'] = data_creation['sentiment'].replace('positive' , 1)# Conversion of boolean features to binary values as 0 or 1
data_creation['sentiment'] = data_creation['sentiment'].replace('negative' , 0)# Conversion of boolean features to binary values as 0 or 1
# platform representation
platform = pd.get_dummies(data_creation['platform'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, platform], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation['facebook'] = data_creation['facebook'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation['instagram'] = data_creation['instagram'].apply(lambda x: 1 if x else 0)
data_creation['telegram'] = data_creation['telegram'].apply(lambda x: 1 if x else 0)
data_creation['tiktok'] = data_creation['tiktok'].apply(lambda x: 1 if x else 0)
data_creation['whatsapp'] = data_creation['whatsapp'].apply(lambda x: 1 if x else 0)
data_creation['x'] = data_creation['x'].apply(lambda x: 1 if x else 0)
data_creation.drop('platform', axis=1, inplace=True)
# embedded_content representation
embedded_content = pd.get_dummies(data_creation['embedded_content'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, embedded_content], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={False: 'none_embedded_content'}, inplace=True)# Rename 
data_creation['none_embedded_content'] = data_creation['none_embedded_content'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation['jpeg'] = data_creation['jpeg'].apply(lambda x: 1 if x else 0)
data_creation['link'] = data_creation['link'].apply(lambda x: 1 if x else 0)
data_creation['mp4'] = data_creation['mp4'].apply(lambda x: 1 if x else 0)
data_creation.drop('embedded_content', axis=1, inplace=True)
# message_date_year representation
message_date_year = pd.get_dummies(data_creation['message_date_year'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, message_date_year], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={2022: 'message_date_year_2022'}, inplace=True)# Rename
data_creation['message_date_year_2022'] = data_creation['message_date_year_2022'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={2023: 'message_date_year_2023'}, inplace=True)# Rename
data_creation['message_date_year_2023'] = data_creation['message_date_year_2023'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('message_date_year', axis=1, inplace=True)
# message_date_month representation
message_date_month = pd.get_dummies(data_creation['message_date_month'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, message_date_month], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={1: 'message_date_month_1'}, inplace=True)# Rename
data_creation['message_date_month_1'] = data_creation['message_date_month_1'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={2: 'message_date_month_2'}, inplace=True)# Rename
data_creation['message_date_month_2'] = data_creation['message_date_month_2'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={3: 'message_date_month_3'}, inplace=True)# Rename
data_creation['message_date_month_3'] = data_creation['message_date_month_3'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={4: 'message_date_month_4'}, inplace=True)# Rename
data_creation['message_date_month_4'] = data_creation['message_date_month_4'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={5: 'message_date_month_5'}, inplace=True)# Rename
data_creation['message_date_month_5'] = data_creation['message_date_month_5'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={6: 'message_date_month_6'}, inplace=True)# Rename
data_creation['message_date_month_6'] = data_creation['message_date_month_6'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={7: 'message_date_month_7'}, inplace=True)# Rename
data_creation['message_date_month_7'] = data_creation['message_date_month_7'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={8: 'message_date_month_8'}, inplace=True)# Rename
data_creation['message_date_month_8'] = data_creation['message_date_month_8'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={9: 'message_date_month_9'}, inplace=True)# Rename
data_creation['message_date_month_9'] = data_creation['message_date_month_9'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={10: 'message_date_month_10'}, inplace=True)# Rename
data_creation['message_date_month_10'] = data_creation['message_date_month_10'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={11: 'message_date_month_11'}, inplace=True)# Rename
data_creation['message_date_month_11'] = data_creation['message_date_month_11'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={12: 'message_date_month_12'}, inplace=True)# Rename
data_creation['message_date_month_12'] = data_creation['message_date_month_12'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('message_date_month', axis=1, inplace=True)
# seniority representation
seniority = pd.get_dummies(data_creation['seniority'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, seniority], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={7: '7_years_seniority'}, inplace=True)# Rename
data_creation['7_years_seniority'] = data_creation['7_years_seniority'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={8: '8_years_seniority'}, inplace=True)# Rename
data_creation['8_years_seniority'] = data_creation['8_years_seniority'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={9: '9_years_seniority'}, inplace=True)# Rename
data_creation['9_years_seniority'] = data_creation['9_years_seniority'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={10: '10_years_seniority'}, inplace=True)# Rename
data_creation['10_years_seniority'] = data_creation['10_years_seniority'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('seniority', axis=1, inplace=True)
# email_domain_suffix representation
email_domain_suffix = pd.get_dummies(data_creation['email_domain_suffix'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, email_domain_suffix], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={'com': 'email_suffix_com'}, inplace=True)# Rename
data_creation['email_suffix_com'] = data_creation['email_suffix_com'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'de': 'email_suffix_de'}, inplace=True)# Rename
data_creation['email_suffix_de'] = data_creation['email_suffix_de'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'edu': 'email_suffix_edu'}, inplace=True)# Rename
data_creation['email_suffix_edu'] = data_creation['email_suffix_edu'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'gov': 'email_suffix_gov'}, inplace=True)# Rename
data_creation['email_suffix_gov'] = data_creation['email_suffix_gov'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'il': 'email_suffix_il'}, inplace=True)# Rename
data_creation['email_suffix_il'] = data_creation['email_suffix_il'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'jp': 'email_suffix_jp'}, inplace=True)# Rename
data_creation['email_suffix_jp'] = data_creation['email_suffix_jp'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'ke': 'email_suffix_ke'}, inplace=True)# Rename
data_creation['email_suffix_ke'] = data_creation['email_suffix_ke'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'org': 'email_suffix_org'}, inplace=True)# Rename
data_creation['email_suffix_org'] = data_creation['email_suffix_org'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'ru': 'email_suffix_ru'}, inplace=True)# Rename
data_creation['email_suffix_ru'] = data_creation['email_suffix_ru'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('email_domain_suffix', axis=1, inplace=True)
# account_creation_year representation
account_creation_year = pd.get_dummies(data_creation['account_creation_year'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, account_creation_year], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={2013: 'account_creation_year_2013'}, inplace=True)# Rename
data_creation['account_creation_year_2013'] = data_creation['account_creation_year_2013'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={2014: 'account_creation_year_2014'}, inplace=True)# Rename
data_creation['account_creation_year_2014'] = data_creation['account_creation_year_2014'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={2015: 'account_creation_year_2015'}, inplace=True)# Rename
data_creation['account_creation_year_2015'] = data_creation['account_creation_year_2015'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('account_creation_year', axis=1, inplace=True)
# account_creation_month representation
account_creation_month = pd.get_dummies(data_creation['account_creation_month'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, account_creation_month], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={1: 'account_creation_month_1'}, inplace=True)# Rename
data_creation['account_creation_month_1'] = data_creation['account_creation_month_1'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={2: 'account_creation_month_2'}, inplace=True)# Rename
data_creation['account_creation_month_2'] = data_creation['account_creation_month_2'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={3: 'account_creation_month_3'}, inplace=True)# Rename
data_creation['account_creation_month_3'] = data_creation['account_creation_month_3'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={4: 'account_creation_month_4'}, inplace=True)# Rename
data_creation['account_creation_month_4'] = data_creation['account_creation_month_4'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={5: 'account_creation_month_5'}, inplace=True)# Rename
data_creation['account_creation_month_5'] = data_creation['account_creation_month_5'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={6: 'account_creation_month_6'}, inplace=True)# Rename
data_creation['account_creation_month_6'] = data_creation['account_creation_month_6'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={7: 'account_creation_month_7'}, inplace=True)# Rename
data_creation['account_creation_month_7'] = data_creation['account_creation_month_7'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={8: 'account_creation_month_8'}, inplace=True)# Rename
data_creation['account_creation_month_8'] = data_creation['account_creation_month_8'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={9: 'account_creation_month_9'}, inplace=True)# Rename
data_creation['account_creation_month_9'] = data_creation['account_creation_month_9'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={10: 'account_creation_month_10'}, inplace=True)# Rename
data_creation['account_creation_month_10'] = data_creation['account_creation_month_10'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={11: 'account_creation_month_11'}, inplace=True)# Rename
data_creation['account_creation_month_11'] = data_creation['account_creation_month_11'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={12: 'account_creation_month_12'}, inplace=True)# Rename
data_creation['account_creation_month_12'] = data_creation['account_creation_month_12'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('account_creation_month', axis=1, inplace=True)
# hour_ranges_of_message_date representation
hour_ranges_of_message_date = pd.get_dummies(data_creation['hour_ranges_of_message_date'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, hour_ranges_of_message_date], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={'0 - 7': 'message_date_0-7'}, inplace=True)# Rename
data_creation['message_date_0-7'] = data_creation['message_date_0-7'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'8 - 15': 'message_date_8-15'}, inplace=True)# Rename
data_creation['message_date_8-15'] = data_creation['message_date_8-15'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'16 - 23': 'message_date_16-23'}, inplace=True)# Rename
data_creation['message_date_16-23'] = data_creation['message_date_16-23'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('hour_ranges_of_message_date', axis=1, inplace=True)
# hour_ranges_of_account_creation representation
hour_ranges_of_account_creation = pd.get_dummies(data_creation['hour_ranges_of_account_creation'])# Perform one-hot encoding
data_creation = pd.concat([data_creation, hour_ranges_of_account_creation], axis=1)# Concatenate the one-hot encoded columns with the original DataFrame
data_creation.rename(columns={'0 - 7': 'account_creation_0-7'}, inplace=True)# Rename
data_creation['account_creation_0-7'] = data_creation['account_creation_0-7'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'8 - 15': 'account_creation_8-15'}, inplace=True)# Rename
data_creation['account_creation_8-15'] = data_creation['account_creation_8-15'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.rename(columns={'16 - 23': 'account_creation_16-23'}, inplace=True)# Rename
data_creation['account_creation_16-23'] = data_creation['account_creation_16-23'].apply(lambda x: 1 if x else 0)# Conversion of boolean features to binary values as 0 or 1
data_creation.drop('hour_ranges_of_account_creation', axis=1, inplace=True)
############## End feature representation ##############

############## Feature selection ############## 
model = LogisticRegression(random_state=0, max_iter=1000)# Create a LogisticRegression object
X = data_creation.drop(['sentiment'], axis=1)
y = data_creation['sentiment']
sfs = SequentialFeatureSelector(model,#Using forward selection, feature selection will stop when improvement of scoring metric is less than tol
                                n_features_to_select="auto",
                                tol=None,
                                direction='forward',
                                scoring='roc_auc',
                                cv=5)
selected_features = sfs.fit(X, y)
feature_names = X.columns[sfs.get_support()]# Get the selected feature names  
print(feature_names)
data_after_selection = pd.DataFrame()
for i in feature_names:# Adding the features after feature selection to a new data frame
    data_after_selection[i] = data_creation[i]
############## End feature selection ##############

############## Dimensionality reduction ##############
pca = PCA()# Initialize an object of PCA
pca.fit(data_after_selection)
pca_data = pca.transform(data_after_selection)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the bar plot of the PC's
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.xticks(rotation=0, fontsize = 7)
fig = ax.get_figure()
fig.set_size_inches(15, 10)
plt.show()
# Creating the final dataset
data_after_selection['sentiment'] = y
final_dataset = pd.DataFrame()
for i in data_after_selection:# Adding the features after feature selection to a new data frame
    final_dataset[i] = data_after_selection[i]
############## End dimensionality reduction ##############
