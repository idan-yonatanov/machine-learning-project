# Imports
import random
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree # for training and ploting DT
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn_extra.cluster import KMedoids
from gower import gower_matrix
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.ensemble import BaggingClassifier


# Reading the data from pkl file 
data = pd.read_pickle('C:/Users/dimay/Downloads/XY_train.pkl')
data_x_test = pd.read_pickle('C:/Users/dimay/Downloads/X_test.pkl')
data_y_test = pd.read_pickle('C:/Users/dimay/Downloads/sample_y_test.pkl')
data_XY_train_pkl = pd.read_pickle('C:/Users/dimay/Downloads/XY_train.pkl')

# Data holdout split: 20% validation, 80% training
data_senntimet = pd.Series(data['sentiment'])# Contains only the target feature - sentiment
data.drop('sentiment', axis=1, inplace=True)# Contains only the features without the target feature - sentiment
x_train, x_test, y_train, y_test = train_test_split(data, data_senntimet, stratify = data_senntimet, test_size=0.2, random_state=123)# Split the data in a stratified fashion
print("Train label percentages\n-----------\n", (pd.value_counts(y_train)/y_train.shape[0])*100)
print("\nTest label percentages\n-----------\n", pd.value_counts(y_test)/y_test.shape[0]*100)

# Merge between x_train and y_train before dataset creation 
data_creation = pd.DataFrame()
for i in x_train:
    data_creation[i] = x_train[i]
data_creation = pd.concat([data_creation, y_train], axis=1)

############################### Dataset Creation x_train ###############################

############## Pre - Processing ##############
#### Handling missing values ####
# Missing values - gender
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'gender')
# Missing values - email_verified
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'email_verified')
# Missing values - blue_tick
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'blue_tick')
# Missing values - embedded_content
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'embedded_content')
# Missing values - platform
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'platform')
# Missing values - email
data_creation.email = data_creation.email.fillna("email")
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
data_creation['email'] = data_creation['email'].replace('email', None)
data_creation['email_domain_suffix'] = data_creation['email_domain_suffix'].replace('email', None)
def fill_null_with_category(df1, column_target, column_temp):# Fill nulls for categorical variables
    value_counts = df1[column_temp].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column_target].isnull()].tolist()# Generate a list with missing values in the specified column
    username_length = random.randint(5, 10)# Random length for the username
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    df1.loc[null_indices, column_target] = username + '.' + np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category 
fill_null_with_category(data_creation, 'email', 'email_domain_suffix')
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
#### End handling missing values ####
############## End Pre - Processing ##############

############## Feature extraction ##############
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
### text_word_count feature ###
data_creation['text_word_count'] = data_creation['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)# count the number of words in the text column for each row  
### sum_top_common_negative_words feature ###
data_negative = data_creation[data_creation['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset
data_negative = data_negative.astype({'text': str})# Convert the negative text lable values to string
all_text = ' '.join(data_negative['text'].tolist())# Create a list from all the values of the negative text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words_negative = dict(word_count.most_common(15))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top negative words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_negative.keys())
data_creation['sum_top_common_negative_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_negative_words_percentage feature ###
data_creation['top_common_negative_words_percentage'] = data_creation['sum_top_common_negative_words'] / data_creation['text_word_count']
data_creation['top_common_negative_words_percentage'] = data_creation['top_common_negative_words_percentage'].fillna(0)   
### sum_top_common_positive_words feature ###
data_positive = data_creation[data_creation['sentiment'] == "positive"]# Extract the positive sentiment samples from the entire dataset
data_positive = data_positive.astype({'text': str})# Convert the positive text lable values to string
all_text = ' '.join(data_positive['text'].tolist())# Create a list from all the values of the positive text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words_positive = dict(word_count.most_common(15))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top positive words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_positive.keys())
data_creation['sum_top_common_positive_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_positive_words_percentage feature ###
data_creation['top_common_positive_words_percentage'] = data_creation['sum_top_common_positive_words'] / data_creation['text_word_count']
data_creation['top_common_positive_words_percentage'] = data_creation['top_common_positive_words_percentage'].fillna(0)    
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
# pca = PCA()# Initialize an object of PCA
# pca.fit(data_after_selection)
# pca_data = pca.transform(data_after_selection)
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
# labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the final dataset
X_train = pd.DataFrame()
for i in data_after_selection:# Adding the features after feature selection to a new data frame
    X_train[i] = data_after_selection[i]
y_train = y_train.replace('positive' , 1)# Conversion of boolean features to binary values as 0 or 1
y_train = y_train.replace('negative' , 0)# Conversion of boolean features to binary values as 0 or 1
############## End dimensionality reduction ##############

############################### End dataset Creation x_train ###############################

# Creating new dataset - data_creation, that includes all the features from x_test  
data_creation = pd.DataFrame()
for i in x_test:
    data_creation[i] = x_test[i]
  
############################### Dataset Creation x_test ###############################

############## Pre - Processing ##############
#### Handling missing values ####
# Missing values - gender
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'gender')
# Missing values - email_verified
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'email_verified')
# Missing values - blue_tick
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'blue_tick')
# Missing values - embedded_content
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'embedded_content')
# Missing values - platform
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'platform')
# Missing values - email
data_creation.email = data_creation.email.fillna("email")
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
data_creation['email'] = data_creation['email'].replace('email', None)
data_creation['email_domain_suffix'] = data_creation['email_domain_suffix'].replace('email', None)
def fill_null_with_category(df1, column_target, column_temp):# Fill nulls for categorical variables
    value_counts = df1[column_temp].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column_target].isnull()].tolist()# Generate a list with missing values in the specified column
    username_length = random.randint(5, 10)# Random length for the username
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    df1.loc[null_indices, column_target] = username + '.' + np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category 
fill_null_with_category(data_creation, 'email', 'email_domain_suffix')
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
#### End handling missing values ####
############## End Pre - Processing ##############

############## Feature extraction ##############
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
### text_word_count feature ###
data_creation['text_word_count'] = data_creation['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)# count the number of words in the text column for each row  
### sum_top_common_negative_words feature ###
def sum_top_words(row):# Calculate for each row the sum of top negative words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_negative.keys())
data_creation['sum_top_common_negative_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_negative_words_percentage feature ###
data_creation['top_common_negative_words_percentage'] = data_creation['sum_top_common_negative_words'] / data_creation['text_word_count']
data_creation['top_common_negative_words_percentage'] = data_creation['top_common_negative_words_percentage'].fillna(0)   
### sum_top_common_positive_words feature ###
def sum_top_words(row):# Calculate for each row the sum of top positive words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_positive.keys())
data_creation['sum_top_common_positive_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_positive_words_percentage feature ###
data_creation['top_common_positive_words_percentage'] = data_creation['sum_top_common_positive_words'] / data_creation['text_word_count']
data_creation['top_common_positive_words_percentage'] = data_creation['top_common_positive_words_percentage'].fillna(0)    
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

############## Feature dropping by the features that were selected in feature selection in X_train dataset ############## 
data_after_selection = data_creation[data_creation.columns.intersection(feature_names)]
############## End feature dropping ##############

############## Dimensionality reduction ##############
# pca = PCA()# Initialize an object of PCA
# pca.fit(data_after_selection)
# pca_data = pca.transform(data_after_selection)
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
# labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the final dataset
X_test = pd.DataFrame()
for i in data_after_selection:# Adding the features after feature selection to a new data frame
    X_test[i] = data_after_selection[i]
y_test = y_test.replace('positive' , 1)# Conversion of boolean features to binary values as 0 or 1
y_test = y_test.replace('negative' , 0)# Conversion of boolean features to binary values as 0 or 1
############## End dimensionality reduction ##############

############################### End dataset Creation x_test ###############################

# Dataset creation - data_creation from data_XY_train_pkl
data_creation = pd.DataFrame()
for i in data_XY_train_pkl:
    data_creation[i] = data_XY_train_pkl[i]

############################### Dataset Creation XY_train.pkl - all dataset ###############################

############## Pre - Processing ##############
#### Handling missing values ####
# Missing values - gender
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'gender')
# Missing values - email_verified
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'email_verified')
# Missing values - blue_tick
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'blue_tick')
# Missing values - embedded_content
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'embedded_content')
# Missing values - platform
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'platform')
# Missing values - email
data_creation.email = data_creation.email.fillna("email")
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
data_creation['email'] = data_creation['email'].replace('email', None)
data_creation['email_domain_suffix'] = data_creation['email_domain_suffix'].replace('email', None)
def fill_null_with_category(df1, column_target, column_temp):# Fill nulls for categorical variables
    value_counts = df1[column_temp].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column_target].isnull()].tolist()# Generate a list with missing values in the specified column
    username_length = random.randint(5, 10)# Random length for the username
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    df1.loc[null_indices, column_target] = username + '.' + np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category 
fill_null_with_category(data_creation, 'email', 'email_domain_suffix')
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
#### End handling missing values ####
############## End Pre - Processing ##############

############## Feature extraction ##############
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
### text_word_count feature ###
data_creation['text_word_count'] = data_creation['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)# count the number of words in the text column for each row  
### sum_top_common_negative_words feature ###
data_negative = data_creation[data_creation['sentiment'] == "negative"]# Extract the negative sentiment samples from the entire dataset
data_negative = data_negative.astype({'text': str})# Convert the negative text lable values to string
all_text = ' '.join(data_negative['text'].tolist())# Create a list from all the values of the negative text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words_negative = dict(word_count.most_common(15))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top negative words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_negative.keys())
data_creation['sum_top_common_negative_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_negative_words_percentage feature ###
data_creation['top_common_negative_words_percentage'] = data_creation['sum_top_common_negative_words'] / data_creation['text_word_count']
data_creation['top_common_negative_words_percentage'] = data_creation['top_common_negative_words_percentage'].fillna(0)   
### sum_top_common_positive_words feature ###
data_positive = data_creation[data_creation['sentiment'] == "positive"]# Extract the positive sentiment samples from the entire dataset
data_positive = data_positive.astype({'text': str})# Convert the positive text lable values to string
all_text = ' '.join(data_positive['text'].tolist())# Create a list from all the values of the positive text lable
clean_text = re.sub(r'[^\w\s]','',all_text).lower()# Remove non-alphanumeric characters and underscores and converts the entire string to lowercase
word_list = clean_text.split()# It splits the clean_text string list into a list of words
stop_words = set(stopwords.words('english'))# Stores set of words that are commonly in use such as - "the," "and," "is")
word_list = [word for word in word_list if not word in stop_words]# Remove stop words from the word list
word_count = Counter(word_list)# Count the number of instances that a word has in the list of words
top_words_positive = dict(word_count.most_common(15))# Create a dictionary of keys and values of top words from word_count
def sum_top_words(row):# Calculate for each row the sum of top positive words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_positive.keys())
data_creation['sum_top_common_positive_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_positive_words_percentage feature ###
data_creation['top_common_positive_words_percentage'] = data_creation['sum_top_common_positive_words'] / data_creation['text_word_count']
data_creation['top_common_positive_words_percentage'] = data_creation['top_common_positive_words_percentage'].fillna(0)    
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
# pca = PCA()# Initialize an object of PCA
# pca.fit(data_after_selection)
# pca_data = pca.transform(data_after_selection)
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
# labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the final dataset
X_train_XY_pkl = pd.DataFrame()
for i in data_after_selection:# Adding the features after feature selection to a new data frame
    X_train_XY_pkl[i] = data_after_selection[i]
y_train_XY_pkl = pd.Series(y)# Contains only the target feature - sentiment
############## End dimensionality reduction ##############

############################### End Dataset Creation XY_train.pkl - all dataset ###############################

# Dataset creation - data_creation from data_x_test
data_creation = pd.DataFrame()
for i in data_x_test:
    data_creation[i] = data_x_test[i]

############################### Dataset Creation data_x_test - alex dataset ###############################

############## Pre - Processing ##############
#### Handling missing values ####
# Missing values - gender
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'gender')
# Missing values - email_verified
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'email_verified')
# Missing values - blue_tick
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'blue_tick')
# Missing values - embedded_content
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'embedded_content')
# Missing values - platform
def fill_null_with_category(df1, column):# Fill nulls for categorical variables
    value_counts = df1[column].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column].isnull()].tolist()# Generate a list with missing values in the specified column
    df1.loc[null_indices, column] = np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category
fill_null_with_category(data_creation, 'platform')
# Missing values - email
data_creation.email = data_creation.email.fillna("email")
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
data_creation['email'] = data_creation['email'].replace('email', None)
data_creation['email_domain_suffix'] = data_creation['email_domain_suffix'].replace('email', None)
def fill_null_with_category(df1, column_target, column_temp):# Fill nulls for categorical variables
    value_counts = df1[column_temp].value_counts(normalize=True)# Get the proportion of each category in the column
    null_indices = df1.index[df1[column_target].isnull()].tolist()# Generate a list with missing values in the specified column
    username_length = random.randint(5, 10)# Random length for the username
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    df1.loc[null_indices, column_target] = username + '.' + np.random.choice(value_counts.index, size=len(null_indices), p=value_counts.values)# Fill in missing values with the proportion of each category 
fill_null_with_category(data_creation, 'email', 'email_domain_suffix')
data_creation['email_domain_suffix'] = data_creation['email'].apply(lambda x: x.split('.')[-1])
#### End handling missing values ####
############## End Pre - Processing ##############

############## Feature extraction ##############
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
### text_word_count feature ###
data_creation['text_word_count'] = data_creation['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)# count the number of words in the text column for each row  
### sum_top_common_negative_words feature ###
def sum_top_words(row):# Calculate for each row the sum of top negative words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_negative.keys())
data_creation['sum_top_common_negative_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_negative_words_percentage feature ###
data_creation['top_common_negative_words_percentage'] = data_creation['sum_top_common_negative_words'] / data_creation['text_word_count']
data_creation['top_common_negative_words_percentage'] = data_creation['top_common_negative_words_percentage'].fillna(0)   
### sum_top_common_positive_words feature ###
def sum_top_words(row):# Calculate for each row the sum of top positive words
    row_text = re.sub(r'[^\w\s]','',row['text']).lower()
    row_word_list = row_text.split()
    row_word_list = [word for word in row_word_list if not word in stop_words]
    row_word_count = Counter(row_word_list)
    row_top_words = dict(row_word_count.most_common(15))
    return sum(row_top_words.get(word, 0) for word in top_words_positive.keys())
data_creation['sum_top_common_positive_words'] = data_creation.apply(sum_top_words, axis=1)# Craete the new feature   
### top_common_positive_words_percentage feature ###
data_creation['top_common_positive_words_percentage'] = data_creation['sum_top_common_positive_words'] / data_creation['text_word_count']
data_creation['top_common_positive_words_percentage'] = data_creation['top_common_positive_words_percentage'].fillna(0)    
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

############## Feature dropping by the features that were selected in feature selection in XY_train.pkl dataset ############## 
data_after_selection = data_creation[data_creation.columns.intersection(feature_names)]
############## End feature dropping ##############

############## Dimensionality reduction ##############
# pca = PCA()# Initialize an object of PCA
# pca.fit(data_after_selection)
# pca_data = pca.transform(data_after_selection)
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
# labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the final dataset
X_test_alex = pd.DataFrame()
for i in data_after_selection:# Adding the features after feature selection to a new data frame
    X_test_alex[i] = data_after_selection[i]
############## End dimensionality reduction ##############

############################### End dataset Creation data_x_test - alex dataset ###############################


################################################## Part B ##################################################


############## Decision trees ##############

### Default decision tree ###
tree_model = DecisionTreeClassifier(random_state=123)
tree_model.fit(X_train, y_train)
# Plot of default Decision tree
plt.figure(figsize=(20, 13))
plot_tree(tree_model, filled=True, class_names = ['0', '1'], feature_names = X_train.columns.to_list())
plt.show()
# AUC score for default decision tree for X_train and X_test 
y_train_prediction = tree_model.predict_proba(X_train)[:, 1]
train_score = roc_auc_score(y_train, y_train_prediction)
print("train set AUC ROC score: {:.10f}".format(train_score))
y_test_prediction = tree_model.predict_proba(X_test)[:, 1]
test_score = roc_auc_score(y_test, y_test_prediction)
print("test set AUC ROC score: {:.10f}".format(test_score))
default_tree_depth = tree_model.tree_.max_depth
### End default decision tree ###

### Hyperparameter Tuning and showing the results on the training set and the test set by AUC score ###
# Defining the grid to tune the model - we will tune: (max_depth, criterion, max_features and class_weight)
param_grid = {'max_depth': np.arange(1, default_tree_depth + 1, 1),
              'criterion': ['entropy', 'gini'],
              'max_features': ['sqrt', 'log2', None],
              'class_weight': ['balanced', None]
             }
# Grid search 
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
best_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=123),
                           param_grid=param_grid, 
                           refit=True,# This means the best model will be available to us at the end, refitted on the WHOLE data
                           verbose=1,# If 2, print for each iteration
                           cv=stratified_cv,# number of folds
                           scoring='roc_auc',
                           return_train_score=True)
# Train the best model by the best hyperparameters
best_model.fit(X_train, y_train)
# Print the best hyperparameters
best_hyperparameters = best_model.best_params_
print(best_hyperparameters)
# best_tree_model stores the best performing model found during the grid search
best_tree_model = best_model.best_estimator_
# Printing the train result of AUC ROC of the best model after hyperparameter Tuning
y_prediction_train = best_tree_model.predict_proba(X_train)[:, 1]
train_roc_auc_best_model = roc_auc_score(y_train, y_prediction_train)
print("train set AUC ROC score: {:.10f}".format(train_roc_auc_best_model))
# Printing the test result of AUC ROC of the best model after hyperparameter Tuning
y_prediction_test = best_tree_model.predict_proba(X_test)[:, 1]
test_roc_auc_best_model = roc_auc_score(y_test, y_prediction_test)
print("Test set AUC ROC score: {:.10f}".format(test_roc_auc_best_model))
# Bulinding a dataFrame of the best hyperparameters and the best results of training set and the test set by AUC score
grid_results = pd.DataFrame(best_model.cv_results_)
selected_columns = ['mean_test_score', 'mean_train_score', 'param_max_depth', 'param_criterion', 'param_class_weight', 'param_max_features']
df_selected = grid_results[selected_columns]
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected['mean_test_score'] = df_selected['mean_test_score'].round(10)
df_selected['mean_train_score'] = df_selected['mean_train_score'].round(10)
column_names = {
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score',
    'param_max_depth': 'Max depth',
    'param_criterion': 'Criterion',
    'param_class_weight': 'Class weight',
    'param_max_features': 'Max Features'
}
df_selected = df_selected.rename(columns=column_names)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('The Best Hyperparameters For The Decision Tree Model', loc='center', fontsize = 14)
table = ax.table(cellText=df_selected.values, colLabels=df_selected.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
### End hyperparameter Tuning ###

### Plot the best decision tree by the best hyperparameters ###
best_model_decision_tree_final = DecisionTreeClassifier(max_features = best_hyperparameters['max_features'],
                                                        max_depth = best_hyperparameters['max_depth'],
                                                        criterion = best_hyperparameters['criterion'],
                                                        class_weight = best_hyperparameters['class_weight'], random_state=123)
best_model_decision_tree_final.fit(X_train_XY_pkl, y_train_XY_pkl)
plt.figure(figsize=(22, 16))
plot_tree(best_model_decision_tree_final, filled=True, max_depth=3 ,class_names = ['0', '1'], feature_names = X_train_XY_pkl.columns.to_list())
plt.rcParams.update({'font.size': 30})
plt.show()
### End plot the best decision tree by the best hyperparameters ###

# Printing the best AUC ROC of the best model after hyperparameter Tuning on all dataset
y_prediction_best_model_decision_tree = best_model_decision_tree_final.predict_proba(X_train_XY_pkl)[:, 1]
train_roc_auc_best_model = roc_auc_score(y_train_XY_pkl, y_prediction_best_model_decision_tree)
print("best decision tree AUC ROC score: {:.10f}".format(train_roc_auc_best_model))

### Plot feature importances of the best model ###
importance = best_model_decision_tree_final.feature_importances_
importance = pd.DataFrame({'Feature_name': X_train_XY_pkl.columns, 'Importance': best_model_decision_tree_final.feature_importances_.round(4)})
importance_sorted = importance.sort_values(by='Importance', ascending=False)# Sort by Importance column in descending order
fig, ax = plt.subplots()
ax.axis('off')
table = ax.table(cellText=importance_sorted.values, colLabels=importance_sorted.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(2,2)
plt.show()
### End plot feature importances of the best model ###

### Keeping the best result of decision tree to model comparison ###
best_result_decision_tree_model = train_roc_auc_best_model
max_features = best_hyperparameters['max_features']
max_depth = best_hyperparameters['max_depth']
criterion = best_hyperparameters['criterion']
class_weight = best_hyperparameters['class_weight']
df_result = pd.DataFrame([best_result_decision_tree_model], columns=['best tree model AUC score'])
grid_results_tree = pd.DataFrame(best_model.cv_results_)
selected_columns = ['mean_test_score', 'param_max_depth', 'param_criterion', 'param_class_weight', 'param_max_features']
df_selected = grid_results_tree[selected_columns]
column_names = {
    'param_max_depth': 'Max depth',
    'param_criterion': 'Criterion',
    'param_class_weight': 'Class weight',
    'param_max_features': 'Max Features'
}
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected = df_selected.rename(columns=column_names)
df_result_tree = pd.concat([df_result, df_selected])
df_result_tree.drop('mean_test_score', axis=1, inplace=True)
df_result_tree.reset_index(drop=True, inplace=True)
df_result_tree.iloc[1, 0] = best_result_decision_tree_model
df_result_tree.drop(index=0, inplace=True)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('The Best Decision Tree Model', loc='center', fontsize = 25)
table = ax.table(cellText=df_result_tree.values, colLabels=df_result_tree.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(22)
table.scale(5,5)
plt.show()
### End keeping the best result of decision tree to model comparison ###

############## End decision trees ##############

############## ANN ##############

### Default ANN ###
model_default_ann = MLPClassifier(random_state=123)
model_default_ann.fit(X_train, y_train)
# AUC score for default ANN for X_train and X_test
y_pred_train_ann = model_default_ann.predict_proba(X_train)[:, 1]
roc_auc_train = roc_auc_score(y_train, y_pred_train_ann)
print("Train set ROC AUC score: {:.10f}".format(roc_auc_train))
y_pred_test_ann = model_default_ann.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_test, y_pred_test_ann)
print("Test set ROC AUC score: {:.10f}".format(roc_auc_test))
# Plot of loss curve of the default ANN model
plt.plot(model_default_ann.loss_curve_)
plt.title("Loss Curve", fontsize=17)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(np.arange(0, 201, 50), fontsize=14)
plt.yticks(np.arange(0, 0.61, 0.1), fontsize=14)
plt.show()
print(model_default_ann.n_features_in_)
print(model_default_ann.hidden_layer_sizes)
### End default ANN ###

### Plots for each hyperparameter ###
# Plot of 1 hidden layers and checking 1 to 100 neurons and there scores
train_roc_aucs = []
test_roc_aucs = []
best_train_roc_auc = 0.0
best_test_roc_auc = 0.0
best_size = 0
for size_ in range(1, 101, 1):# Run on 1 to 100 neurons in single hidden layer and store the results in variables and lists
    model_1_layer = MLPClassifier(hidden_layer_sizes = (size_), random_state=123)                                                                        
    model_1_layer.fit(X_train, y_train)
    y_pred_train_1_layer = model_1_layer.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_pred_train_1_layer)
    train_roc_aucs.append(roc_auc_train)
    y_pred_test_1_layer = model_1_layer.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_pred_test_1_layer)
    test_roc_aucs.append(roc_auc_test)
    if roc_auc_train > best_train_roc_auc:
        best_train_roc_auc = roc_auc_train
    if roc_auc_test > best_test_roc_auc:
        best_test_roc_auc = roc_auc_test
        best_size = size_
# plot AUC ROC Score vs 1 Hidden Layer
plt.plot(train_roc_aucs, label='Train')
plt.plot(test_roc_aucs, label='Test')
plt.xlabel('Number Of Neurons In 1 Layer', fontsize=14)
plt.ylabel('AUC-ROC Score', fontsize=14)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(np.arange(0.996, 1, 0.001), fontsize=14)
plt.title('AUC-ROC Scores For Train And Test Sets - Number Of Neurons In 1 hidden Layer', fontsize=15)
plt.legend(loc="upper left", fontsize = "9")
plt.show()
# Plot a table with the best results for 1 layer
fig, ax = plt.subplots()
ax.axis('off')
max_train = max(train_roc_aucs)
max_test = max(test_roc_aucs)
best_results_dict_1 = {'train AUC score': max_train, 'best test AUC score':max_test, 'best size of neurons': best_size}
table = ax.table(cellText=[list(best_results_dict_1.values())], colLabels=list(best_results_dict_1.keys()))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
# Plot of 2 hidden layers and checking 1 to 100 neurons and there scores
train_roc_aucs = []
test_roc_aucs = []
best_train_roc_auc = 0.0
best_test_roc_auc = 0.0
best_size = 0
for size_ in range(1, 101, 1):# Run on 1 to 100 neurons in 2 hidden layer and store the results in variables and lists
    model_2_layer = MLPClassifier(hidden_layer_sizes = (size_, size_), random_state=123)                                                                        
    model_2_layer.fit(X_train, y_train)
    y_pred_train_2_layer = model_1_layer.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_pred_train_2_layer)
    train_roc_aucs.append(roc_auc_train)
    y_pred_test_2_layer = model_2_layer.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_pred_test_2_layer)
    test_roc_aucs.append(roc_auc_test)
    if roc_auc_train > best_train_roc_auc:
        best_train_roc_auc = roc_auc_train
    if roc_auc_test > best_test_roc_auc:
        best_test_roc_auc = roc_auc_test
        best_size = size_
# plot AUC ROC Score vs 2 Hidden Layer
plt.plot(train_roc_aucs, label='Train')
plt.plot(test_roc_aucs, label='Test')
plt.xlabel('Number Of Neurons In 2 Layers', fontsize=14)
plt.ylabel('AUC-ROC Score', fontsize=14)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(np.arange(0.993, 1, 0.001), fontsize=14)
plt.title('AUC-ROC Scores For Train And Test Sets - Number Of Neurons In 2 hidden Layer', fontsize=15)
plt.legend(loc="lower left", fontsize = "9")
plt.show()
# Plot a table with the best results for 2 layer
fig, ax = plt.subplots()
ax.axis('off')
max_train = max(train_roc_aucs)
max_test = max(test_roc_aucs)
best_results_dict_2 = {'train AUC score': max_train, 'best test AUC score':max_test, 'best size of neurons': best_size}
table = ax.table(cellText=[list(best_results_dict_2.values())], colLabels=list(best_results_dict_2.keys()))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
# Plot of 3 hidden layers and checking 1 to 100 neurons and there scores
train_roc_aucs = []
test_roc_aucs = []
best_train_roc_auc = 0.0
best_test_roc_auc = 0.0
best_size = 0
for size_ in range(1, 101, 1):# Run on 1 to 100 neurons in 3 hidden layer and store the results in variables and lists
    model_3_layer = MLPClassifier(hidden_layer_sizes = (size_, size_, size_), random_state=123)                                                                        
    model_3_layer.fit(X_train, y_train)
    y_pred_train_3_layer = model_3_layer.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_pred_train_3_layer)
    train_roc_aucs.append(roc_auc_train)
    y_pred_test_3_layer = model_3_layer.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_pred_test_3_layer)
    test_roc_aucs.append(roc_auc_test)
    if roc_auc_train > best_train_roc_auc:
        best_train_roc_auc = roc_auc_train
    if roc_auc_test > best_test_roc_auc:
        best_test_roc_auc = roc_auc_test
        best_size = size_
# plot AUC ROC Score vs 3 Hidden Layer
plt.plot(train_roc_aucs, label='Train')
plt.plot(test_roc_aucs, label='Test')
plt.xlabel('Number Of Neurons In 3 Layers', fontsize=14)
plt.ylabel('AUC-ROC Score', fontsize=14)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(np.arange(0.993, 1, 0.001), fontsize=14)
plt.title('AUC-ROC Scores For Train And Test Sets - Number Of Neurons In 3 hidden Layer', fontsize=15)
plt.legend(loc="lower right", fontsize = "9")
plt.show()
# Plot a table with the best results for 3 layer
fig, ax = plt.subplots()
ax.axis('off')
max_train = max(train_roc_aucs)
max_test = max(test_roc_aucs)
best_results_dict_3 = {'train AUC score': max_train, 'best test AUC score':max_test, 'best size of neurons': best_size}
table = ax.table(cellText=[list(best_results_dict_3.values())], colLabels=list(best_results_dict_3.keys()))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
# Plot of learning_rate_init 
learning_rate_init_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
train_roc_aucs = []
test_roc_aucs = []
best_train_roc_auc = 0.0
best_test_roc_auc = 0.0
best_learning_rate = 0.0
for rate in learning_rate_init_list:# Run on learning_rate_init list values and store the results in variables and lists
   model_learning_rate = MLPClassifier(learning_rate_init = rate, random_state=123)
   model_learning_rate.fit(X_train, y_train)   
   y_pred_train_learning_rate = model_learning_rate.predict_proba(X_train)[:, 1]
   roc_auc_train = roc_auc_score(y_train, y_pred_train_learning_rate)
   train_roc_aucs.append(roc_auc_train)
   y_pred_test_learning_rate = model_learning_rate.predict_proba(X_test)[:, 1]
   roc_auc_test = roc_auc_score(y_test, y_pred_test_learning_rate)
   test_roc_aucs.append(roc_auc_test)
   if roc_auc_train > best_train_roc_auc:
       best_train_roc_auc = roc_auc_train
   if roc_auc_test > best_test_roc_auc:
       best_test_roc_auc = roc_auc_test
       best_learning_rate = rate
# plot AUC ROC Score vs learning_rate_init
plt.plot(learning_rate_init_list, train_roc_aucs, marker='o', label='Train')
plt.plot(learning_rate_init_list, test_roc_aucs, marker='o', label='Test')
plt.xlabel('Learning Rate', fontsize=14)
plt.ylabel('AUC-ROC Score', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('AUC-ROC Scores For Train And Test Sets - Learning Rate', fontsize=15)
plt.legend(loc="center right", fontsize = "11")
plt.show()
# Plot a table with the best results for learning_rate
fig, ax = plt.subplots()
ax.axis('off')
max_train = max(train_roc_aucs)
max_test = max(test_roc_aucs)
best_results_learning_rate = {'train AUC score': max_train, 'best test AUC score':max_test, 'best Learning Rate': best_learning_rate}
table = ax.table(cellText=[list(best_results_learning_rate.values())], colLabels=list(best_results_learning_rate.keys()))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
# Plot of max_iter 
max_iterations = [10, 50, 100, 200, 300, 400, 500]
train_roc_aucs = []
test_roc_aucs = []
best_train_roc_auc = 0.0
best_test_roc_auc = 0.0
best_max_iterations = 0
for iteration in max_iterations:# Run on max_iterations list values and store the results in variables and lists
   model_max_iter = MLPClassifier(max_iter = iteration, random_state=123)
   model_max_iter.fit(X_train, y_train)   
   y_pred_train_max_iter = model_max_iter.predict_proba(X_train)[:, 1]
   roc_auc_train = roc_auc_score(y_train, y_pred_train_max_iter)
   train_roc_aucs.append(roc_auc_train)
   y_pred_test_max_iter = model_max_iter.predict_proba(X_test)[:, 1]
   roc_auc_test = roc_auc_score(y_test, y_pred_test_max_iter)
   test_roc_aucs.append(roc_auc_test)
   if roc_auc_train > best_train_roc_auc:
       best_train_roc_auc = roc_auc_train
   if roc_auc_test > best_test_roc_auc:
       best_test_roc_auc = roc_auc_test
       best_max_iterations = iteration
# plot AUC ROC Score vs max_iter
plt.plot(max_iterations, train_roc_aucs, marker='o', label='Train')
plt.plot(max_iterations, test_roc_aucs, marker='o', label='Test')
plt.xlabel('Number Of Iterations', fontsize=14)
plt.ylabel('AUC-ROC Score', fontsize=14)
plt.yticks(np.arange(0.996, 1.0, 0.001), fontsize=14)
plt.xticks(np.arange(0, 501, 50), fontsize=14)
plt.title('AUC-ROC Scores For Train And Test Sets - Number Of Iterations', fontsize=15)
plt.legend(loc="center right", fontsize = "11")
plt.show()

# Plot a table with the best results for max_iter
fig, ax = plt.subplots()
ax.axis('off')
max_train = max(train_roc_aucs)
max_test = max(test_roc_aucs)
best_results_max_iter = {'train AUC score': max_train, 'best test AUC score':max_test, 'best Number Of Iterations': best_max_iterations}
table = ax.table(cellText=[list(best_results_max_iter.values())], colLabels=list(best_results_max_iter.keys()))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
### End plots for each hyperparameter ###

### Defining the parameters to tune the model - we will tune: (hidden_layer_sizes, max_iter, learning_rate_init and numer of hidden layers)
model_tune_ann = MLPClassifier(random_state=123)
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
layer_size = range(1, 4)
neuron_count = range(1, 101, 1)
hidden_layer_combinations = []
for i in layer_size:# Store in a list all the combinations of number of layers + number of neurons in each layer
    combination = list(itertools.permutations(neuron_count, i))
    hidden_layer_combinations.extend(combination)
params = {'hidden_layer_sizes': hidden_layer_combinations,
          'max_iter': max_iterations,
          'learning_rate_init': learning_rate_init_list
         }
# Random search
mlp_random_search = RandomizedSearchCV(estimator = model_tune_ann, param_distributions = params, scoring='roc_auc',
                                       cv = stratified_cv, n_jobs=-1, refit=True, return_train_score=True, random_state=123,
                                       n_iter=100)
# Train the best model by the best hyperparameters
mlp_random_search.fit(X_train, y_train)
# best_model_ann stores the best performing model found during the random search
best_model_ann = mlp_random_search.best_estimator_
# Print the best hyperparameters
best_hyperparams = mlp_random_search.best_params_
print(best_hyperparams)
# Printing the train result of AUC ROC of the best model after hyperparameter Tuning
y_train_pred = best_model.predict_proba(X_train)[:, 1]
train_roc_auc = roc_auc_score(y_train, y_train_pred)
print("Train set ROC AUC score: {:.10f}".format(train_roc_auc))
# Printing the test result of AUC ROC of the best model after hyperparameter Tuning
y_test_pred = best_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_test_pred)
print("Test set ROC AUC score: {:.10f}".format(test_roc_auc))
# Bulinding a dataFrame of the best hyperparameters and the best results of training set and the test set by AUC score
cv_results = pd.DataFrame(mlp_random_search.cv_results_)
selected_columns = ['mean_train_score', 'mean_test_score', 'param_hidden_layer_sizes', 'param_max_iter', 'param_learning_rate_init']
df_selected = cv_results[selected_columns]
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected['mean_test_score'] = df_selected['mean_test_score'].round(9)
df_selected['mean_train_score'] = df_selected['mean_train_score'].round(9)
column_names = {
    'mean_train_score': 'Mean train score',
    'mean_test_score': 'Mean test score',
    'param_hidden_layer_sizes': 'Hidden Layer Size', 
    'param_learning_rate_init': 'Learning Rate Init',
    'param_max_iter': 'Max iterations'
}
df_selected = df_selected.rename(columns=column_names)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('The Best Hyperparameters By Random Search', loc='center')
table = ax.table(cellText=df_selected.values, colLabels=df_selected.columns)
table.auto_set_font_size(False)
table.set_fontsize(30)
table.scale(4.5, 7)
plt.show()
# Loss curve after random search for hyperparameters
plt.plot(best_model_ann.loss_curve_)
plt.title("Loss Curve", fontsize=17)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(np.arange(0, 0.81, 0.1), fontsize=14)
plt.show()
### End hyperparameter Tuning ###

### Train and test for all dataset with the best hyperparameters by random search
optimal_mlp = MLPClassifier(hidden_layer_sizes = best_hyperparams['hidden_layer_sizes'],
                            max_iter = best_hyperparams['max_iter'],
                            learning_rate_init = best_hyperparams['learning_rate_init'], random_state=123)
optimal_mlp.fit(X_train_XY_pkl, y_train_XY_pkl)
# Printing the best AUC ROC of the best model after hyperparameter Tuning on all dataset
y_prediction_optimal_mlp = optimal_mlp.predict_proba(X_train_XY_pkl)[:, 1]
train_roc_auc_best_model = roc_auc_score(y_train_XY_pkl, y_prediction_optimal_mlp)
print("best ANN modal mlp AUC ROC score: {:.10f}".format(train_roc_auc_best_model))
### End Train and test for all dataset with the best hyperparameters ###

### Confusion matrix- calculations - TPR, FPR etc and heatmap plot 
cm = confusion_matrix(y_true = y_train_XY_pkl, y_pred = optimal_mlp.predict(X_train_XY_pkl))
print(cm)
# Heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Of Best ANN Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
### End confusion matrix ###

### Keeping the best result of ANN to model comparison ###
best_result_ann_model = train_roc_auc_best_model
hidden_layer_sizes = best_hyperparams['hidden_layer_sizes']
max_iter = best_hyperparams['max_iter']
learning_rate_init = best_hyperparams['learning_rate_init']
df_result = pd.DataFrame([best_result_ann_model], columns=['best ann model AUC score'])
random_results_ann = pd.DataFrame(mlp_random_search.cv_results_)
selected_columns = ['mean_test_score', 'param_hidden_layer_sizes', 'param_learning_rate_init', 'param_max_iter']
df_selected = random_results_ann[selected_columns]
column_names = {
    'mean_test_score': 'Mean test score',
    'param_hidden_layer_sizes': 'Hidden Layer Size', 
    'param_learning_rate_init': 'Learning Rate Init',
    'param_max_iter': 'Max iterations'
}
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected = df_selected.rename(columns=column_names)
df_result_ann = pd.concat([df_result, df_selected])
df_result_ann.drop('Mean test score', axis=1, inplace=True)
df_result_ann.reset_index(drop=True, inplace=True)
df_result_ann.iloc[1, 0] = best_result_ann_model
df_result_ann.drop(index=0, inplace=True)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('The Best ANN Model', loc='center', fontsize = 25)
table = ax.table(cellText=df_result_ann.values, colLabels=df_result_ann.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(22)
table.scale(5,5)
plt.show()
### End keeping the best result of ANN to model comparison ###

############## End ANN ##############

############## SVM ##############

### Perform PCA's with 2 components for X_train, X_test and X_train_XY_pkl ###
# PCA for X_train
original_indices = X_train.index
pca = PCA(n_components=2)# Initialize an object of PCA
pca.fit(X_train)
model_pca_X_train = pca.transform(X_train)
# Scatter plot of X_train - PCA with its y_train lables to see the distribution by PC1 and PC2
X_train_transformed_df = pd.DataFrame(model_pca_X_train, index=original_indices, columns=['PC1', 'PC2'])
model_pca_X_train_plot = pd.concat([X_train_transformed_df, y_train], axis=1)
plt.figure(figsize=(10, 9))
sns.scatterplot(x='PC1', y='PC2', data = model_pca_X_train_plot, hue = 'sentiment')
plt.title("Scatter Plot Of Sentiment With Respect To PC1 And PC2", fontsize = 16)
plt.legend(loc="upper right", fontsize = "20")
plt.ylim(-2, 2)
plt.xlim(-2, 2)
plt.show()
# PCA for X_test
pca = PCA(n_components=2)# Initialize an object of PCA
pca.fit(X_test)
model_pca_X_test = pca.transform(X_test)
# PCA for X_train_XY_pkl
original_indices = X_train_XY_pkl.index
pca = PCA(n_components=2)# Initialize an object of PCA
pca.fit(X_train_XY_pkl)
model_pca_X_train_XY_pkl = pca.transform(X_train_XY_pkl)
### End perform PCA's with 2 components for X_train, X_test and X_train_XY_pkl ###

### Hyperparameter Tuning and showing the results on the training set and the test set by AUC score ###
# Defining the grid to tune the model - we will tune: C Hyperparameter
param_grid = {'C': [0.1, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10, 50, 100, 1000]}
# Create the LinearSVC classifier
svm_model = SVC(kernel='linear', probability=True, random_state=123)
# Grid search 
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
best_model = GridSearchCV(estimator=svm_model,
                           param_grid=param_grid, 
                           refit=True,# This means the best model will be available to us at the end, refitted on the WHOLE data
                           verbose=1,# If 2, print for each iteration
                           cv=stratified_cv,# number of folds
                           scoring='roc_auc',
                           return_train_score=True)
# Train the best model by the best hyperparameters
best_model.fit(X_train, y_train)
# Print the best hyperparameters
best_hyperparameters = best_model.best_params_
print(best_hyperparameters)
# best_svm_model stores the best performing model found during the grid search
best_svm_model = best_model.best_estimator_
# Printing the train result of AUC ROC of the best model after hyperparameter Tuning
y_prediction_train = best_svm_model.predict_proba(X_train)[:, 1]
train_roc_auc_best_model = roc_auc_score(y_train, y_prediction_train)
print("train set AUC ROC score: {:.10f}".format(train_roc_auc_best_model))
# Printing the test result of AUC ROC of the best model after hyperparameter Tuning
y_prediction_test = best_svm_model.predict_proba(X_test)[:, 1]
test_roc_auc_best_model = roc_auc_score(y_test, y_prediction_test)
print("Test set AUC ROC score: {:.10f}".format(test_roc_auc_best_model))
# Bulinding a dataFrame of the best hyperparameters and the best results of training set and the test set by AUC score
grid_results = pd.DataFrame(best_model.cv_results_)
selected_columns = ['param_C', 'mean_test_score', 'mean_train_score']
df_selected = grid_results[selected_columns]
df_selected = df_selected.sort_values('mean_test_score', ascending=False)
df_selected['mean_test_score'] = df_selected['mean_test_score'].round(10)
df_selected['mean_train_score'] = df_selected['mean_train_score'].round(10)
column_names = {
    'param_C': 'C Hyperparameter',
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score'
}
df_selected = df_selected.rename(columns=column_names)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('Sorted AUC Scores (Descending Order) By C Hyperparameters', loc='center')
table = ax.table(cellText=df_selected.values, colLabels=df_selected.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.7,1.9)
plt.show()
### End hyperparameter Tuning ###

### Train the best SVM model by the best hyperparameters ###
best_model_SVM_final = SVC(kernel='linear', C = best_hyperparameters['C'], probability=True, random_state=123)
best_model_SVM_final.fit(X_train_XY_pkl, y_train_XY_pkl)
plt.figure(figsize=(22, 16))
### End training the best SVM model ###

# Printing the best AUC ROC of the best model after hyperparameter Tuning on all dataset
y_prediction_best_svm_model = best_model_SVM_final.predict_proba(X_train_XY_pkl)[:, 1]
train_roc_auc_best_model = roc_auc_score(y_train_XY_pkl, y_prediction_best_svm_model)
print("best SVM AUC ROC score: {:.10f}".format(train_roc_auc_best_model))
# Scatter plot of X_train_XY_pkl - PCA with its preducted sentiment lables to see the distribution by PC1 and PC2
y_prediction_best_svm_model_plot = pd.Series(y_prediction_best_svm_model)
X_train_XY_pkl_transformed_df = pd.DataFrame(model_pca_X_train_XY_pkl, index=original_indices, columns=['PC1', 'PC2'])
model_pca_X_train_XY_pkl_plot = pd.concat([X_train_XY_pkl_transformed_df, y_prediction_best_svm_model_plot], axis=1)
model_pca_X_train_XY_pkl_plot.rename(columns={0: 'y'}, inplace=True)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', data = model_pca_X_train_XY_pkl_plot, hue = 'y')
plt.title("Scatter Plot Of Sentiment With Respect To PC1 And PC2 After SVM", fontsize = 16)
plt.legend(loc="upper right", fontsize = "20")
plt.ylim(-2, 2)
plt.xlim(-2, 2)
plt.show()

### Printing the equation of the dividing line of the best SVM model ###
print("Equation of the dividing line:", best_model_SVM_final.coef_[0])
coefficients = best_model_SVM_final.coef_[0]
max_coef = max(coefficients)
min_coef = min(coefficients)
print("Maximum coefficient:", max_coef)
print("Minimum coefficient:", min_coef)
col_names = X_train_XY_pkl.columns
max_coef_index = np.argmax(coefficients)
min_coef_index = np.argmin(coefficients)
max_coef_col = col_names[max_coef_index]
min_coef_col = col_names[min_coef_index]
# Print the results
print("Feature with maximum coefficient:", max_coef_col)
print("Feature with minimum coefficient:", min_coef_col)
### End printing the equation of the dividing line of the best SVM model ###

### Keeping the best result of ANN to model comparison ###
best_result_svm_model = train_roc_auc_best_model
C = best_hyperparameters['C']
df_result = pd.DataFrame([best_result_svm_model], columns=['best svm model AUC score'])
random_results_svm = pd.DataFrame(best_model.cv_results_)
selected_columns = ['mean_test_score', 'param_C']
df_selected = random_results_svm[selected_columns]
column_names = {
    'mean_test_score': 'Mean test score',
    'param_C': 'C Hyperparameter'
}
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected = df_selected.rename(columns=column_names)
df_result_svm = pd.concat([df_result, df_selected])
df_result_svm.drop('Mean test score', axis=1, inplace=True)
df_result_svm.reset_index(drop=True, inplace=True)
df_result_svm.iloc[1, 0] = best_result_svm_model
df_result_svm.drop(index=0, inplace=True)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('The Best SVM Model', loc='center', fontsize = 25)
table = ax.table(cellText=df_result_svm.values, colLabels=df_result_svm.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(22)
table.scale(5,5)
plt.show()

############## End SVM ##############

############## Clustering ##############

# Initial gaprh to chose the metric type
original_indices = X_train_XY_pkl.index
pca = PCA(n_components=2)
model_pca_XY_train = pca.fit_transform(X_train_XY_pkl)
model_pca_XY_train = pd.DataFrame(model_pca_XY_train, index=original_indices, columns=['PC1', 'PC2'])
model_pca_XY_train['sentiment'] = y_train_XY_pkl
model_pca_XY_train['sentiment'] = model_pca_XY_train['sentiment'].replace('negative' , 0)# Conversion of boolean features to binary values as 0 or 1
model_pca_XY_train['sentiment'] = model_pca_XY_train['sentiment'].replace('positive' , 1)# Conversion of boolean features to binary values as 0 or 1
k_medoids = KMedoids(n_clusters=2, random_state=123)
k_medoids.fit(model_pca_XY_train[['PC1', 'PC2']])
labels = k_medoids.labels_
model_pca_XY_train['cluster'] = labels
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=model_pca_XY_train)
plt.xticks(np.arange(-1, 2.1, 0.5), fontsize=14)
plt.yticks(np.arange(-1, 1.51, 0.5), fontsize=14)
plt.legend(loc="best", fontsize = 10)
plt.title("Scatter Plot Of Sentiment With Respect To PC1 And PC2 By Clustering", fontsize = 16)
plt.show()

# Clustering with gower metric type
# Select best k
inertia_scores = []
silhouette_scores = []
davies_bouldin_scores = []
# Calculte gower distance matrix
gower_mat = gower_matrix(X_train_XY_pkl)
np.fill_diagonal(gower_mat, 0)
for i in range(2,11):
    k_medoids = KMedoids(n_clusters=i, metric='precomputed', random_state=123, method='pam')
    k_medoids.fit(gower_mat)
    labels = k_medoids.labels_
    # Calculate scores
    inertia = k_medoids.inertia_
    inertia_scores.append(inertia)
    silhouette = silhouette_score(gower_mat, labels, metric='precomputed')
    silhouette_scores.append(silhouette)
    davies_bouldin = davies_bouldin_score(X_train_XY_pkl, labels)
    davies_bouldin_scores.append(davies_bouldin)
# Plots of scores for X_train_XY_pkl
# Inertia
plt.plot(range(2, 11), inertia_scores, marker='o')
plt.title("Inertia - Gower Metric")
plt.xlabel("Number Of Clusters")
plt.ylabel("Inertia Score")
plt.show()
# Silhouette
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Silhouette Score - Gower Metric")
plt.xlabel("Number Of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
# Davies-Bouldin
plt.plot(range(2, 11), davies_bouldin_scores, marker='o')
plt.title("Davies-Bouldin Score - Gower Metric")
plt.xlabel("Number Of Clusters")
plt.ylabel("Davies-Bouldin Score")
plt.show()
# Summary of the results of inertia, silhouette and davies-bouldin
k_range = range(2, 11)
results_k_df = pd.DataFrame({'k': k_range, 'Inertia': inertia_scores, 'Silhouette Scores': silhouette_scores, 'Davies-Bouldin Scores': davies_bouldin_scores})
print(results_k_df)
# Inertia: k=6, Silhouette: k=3, Davies-Bouldin: k=8
k = 6
k_medoids = KMedoids(n_clusters=k, metric='precomputed', random_state=123)
k_medoids.fit(gower_mat)
labels = k_medoids.labels_
# Plot of the division to optimal k clusters
plt.scatter(model_pca_XY_train['PC1'], model_pca_XY_train['PC2'], c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data Clustering')
plt.show()

############## End clustering ##############

############## Model comparison - model selection ##############

### Choosing the best model from the TD, ANN and SVM ###
def best_model_of_all(model_tree, model_ann, model_svm):
    if (model_tree.iloc[0, 0] > model_ann.iloc[0, 0]) and (model_tree.iloc[0, 0] > model_svm.iloc[0, 0]):
       return model_tree
    elif (model_ann.iloc[0, 0] > model_tree.iloc[0, 0]) and (model_ann.iloc[0, 0] > model_svm.iloc[0, 0]):
       return model_ann
    else:
       return model_svm
best_algorithm_of_the_project = pd.DataFrame(best_model_of_all(df_result_tree, df_result_ann, df_result_svm))
fig, ax = plt.subplots()
ax.axis('off')
table = ax.table(cellText=best_algorithm_of_the_project.values, colLabels=best_algorithm_of_the_project.columns, loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(22)
table.scale(5,5)
plt.show()
### END Choosing the best model from the TD, ANN and SVM ###

### Printing AUC curve plot ###
fpr1, tpr1, thresh1 = roc_curve(y_train_XY_pkl, y_prediction_best_model_decision_tree)
fpr2, tpr2, thresh2 = roc_curve(y_train_XY_pkl, y_prediction_optimal_mlp)
fpr3, tpr3, thresh3 = roc_curve(y_train_XY_pkl, y_prediction_best_svm_model)
# Plot the ROC curve for each model
plt.plot(fpr1, tpr1, label='Decision Tree')
plt.plot(fpr2, tpr2, label='ANN')
plt.plot(fpr3, tpr3, label='SVM')
# Set labels and title
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0.985, 1.1, 0.005), fontsize=14)
plt.ylim(0.985, 1)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = 15)
plt.legend(loc="best", fontsize = 10)
plt.show()
### End printing AUC curve plot ###

############## End model comparison - model selection ##############

############## Model improvement ##############

######  Improve of the model - Bagging technique ######
### Building the bagging model, train it and printing the AUC scores ### 
bagging_model = BaggingClassifier(estimator=optimal_mlp, random_state=123)
n_estimators = range(1,11)# The hyperparameter of bagging classifier
param_grid = {
    'n_estimators': n_estimators
}
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
# Hyperparameter tune - grid search
bagging_model_grid_search = GridSearchCV(estimator=bagging_model, param_grid=param_grid,  scoring='roc_auc', cv=stratified_cv, n_jobs=-1, refit=True, return_train_score=True)
# Train the model on train set
bagging_model_grid_search.fit(X_train, y_train)
print(bagging_model_grid_search.best_params_)
# Printing the train result of AUC ROC of the model after hyperparameter Tuning
y_train_pred = bagging_model_grid_search.predict_proba(X_train)[:, 1]
train_roc_auc = roc_auc_score(y_train, y_train_pred)
print("Train set ROC AUC score: {:.10f}".format(train_roc_auc))
# Printing the test result of AUC ROC of the model after hyperparameter Tuning
y_test_pred = bagging_model_grid_search.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_test_pred)
print("Test set ROC AUC score: {:.10f}".format(test_roc_auc))
### End building the bagging model, train it and printing the AUC scores ### 

### Bulinding a dataFrame of the best hyperparameters and the best results of training set and the test set by AUC score ###
cv_results = pd.DataFrame(bagging_model_grid_search.cv_results_)
selected_columns = ['mean_test_score', 'mean_train_score', 'param_n_estimators']
df_selected = cv_results[selected_columns]
df_selected = df_selected.sort_values('mean_test_score', ascending=False).head(1)
df_selected['mean_test_score'] = df_selected['mean_test_score'].round(10)
df_selected['mean_train_score'] = df_selected['mean_train_score'].round(10)
column_names = {
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score',
    'param_n_estimators': 'n_estimators'
}
df_selected = df_selected.rename(columns=column_names)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('Model Improvement - Bagging', loc='center', fontsize = 25)
table = ax.table(cellText=df_selected.values, colLabels=df_selected.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(22)
table.scale(5,5)
plt.show()
### End bulinding a dataFrame of the best hyperparameters and the best results of training set and the test set by AUC score ###

######  End improve of the model - Bagging technique ######

######  Improve of the data - PCA technique ######
### Create new data frame for X_train and X_test to use in this section ###
X_train_improve_the_data = pd.DataFrame()
for i in X_train:# Adding the features after feature selection to a new data frame
    X_train_improve_the_data[i] = X_train[i]
X_test_improve_the_data = pd.DataFrame()
for i in X_test:# Adding the features after feature selection to a new data frame
    X_test_improve_the_data[i] = X_test[i]
### End create new data frame for X_train and X_test to use in this section ###

### Plot PCA for X_train and X_test ###
# Plot PCA for X_train_improve_the_data
pca = PCA()# Initialize an object of PCA
pca.fit(X_train_improve_the_data)
pca_data = pca.transform(X_train_improve_the_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the bar plot of the PC's
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance', fontsize = 12)
plt.xlabel('Principal Component', fontsize = 12)
plt.title('Scree Plot - X_train', fontsize = 16)
plt.xticks(rotation=0, fontsize = 7.5)
plt.yticks(fontsize = 13)
fig = ax.get_figure()
fig.set_size_inches(15, 15)
plt.show()
# Plot PCA for X_test_improve_the_data
pca = PCA()# Initialize an object of PCA
pca.fit(X_test_improve_the_data)
pca_data = pca.transform(X_test_improve_the_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)# Calculating the percentage of variation that each feature accounts for
labels = [str(x) for x in range(1, len(per_var) + 1)]
# Creating the bar plot of the PC's
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance', fontsize = 12)
plt.xlabel('Principal Component', fontsize = 12)
plt.title('Scree Plot- X_test', fontsize = 16)
plt.xticks(rotation=0, fontsize = 7.5)
plt.yticks(fontsize = 13)
fig = ax.get_figure()
fig.set_size_inches(15, 15)
plt.show()
### End plot PCA for X_train and X_test ###

### PCA - dimensionality reduction from 35 features to 20 features with approximately 80% explanation of the variance ###
# PCA - dimensionality reduction for X_train_improve_the_data
original_indices = X_train_improve_the_data.index
#Fit the PCA model
pca = PCA()
pca.fit(X_train_improve_the_data)
# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Sort components based on explained variance ratio
sorted_components = sorted(enumerate(explained_variance_ratio), key=lambda x: x[1], reverse=True)
# Select the top components you want to keep
top_components_indices = [index for index, _ in sorted_components[:20]]
# Transform original data using selected components
transformed_data = pca.transform(X_train_improve_the_data)[:, top_components_indices]
# Store transformed data in a new DataFrame
columns = ['PC{}'.format(i+1) for i in range(len(top_components_indices))]
X_Train_improve_data = pd.DataFrame(transformed_data, columns=columns, index=original_indices)
# PCA - dimensionality reduction for X_test_improve_the_data
original_indices = X_test_improve_the_data.index
#Fit the PCA model
pca = PCA()
pca.fit(X_test_improve_the_data)
# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Sort components based on explained variance ratio
sorted_components = sorted(enumerate(explained_variance_ratio), key=lambda x: x[1], reverse=True)
# Select the top components you want to keep
top_components_indices = [index for index, _ in sorted_components[:20]]
# Transform original data using selected components
transformed_data = pca.transform(X_test_improve_the_data)[:, top_components_indices]
# Store transformed data in a new DataFrame
columns = ['PC{}'.format(i+1) for i in range(len(top_components_indices))]
X_test_improve_data = pd.DataFrame(transformed_data, columns=columns, index=original_indices)
### End PCA - dimensionality reduction from 35 features to 20 features with approximately 80% explanation of the variance ###

### Run on the optimal model ###
optimal_mlp.fit(X_Train_improve_data, y_train)
# Printing the train result of AUC ROC of the best model after hyperparameter Tuning
y_train_pred = optimal_mlp.predict_proba(X_Train_improve_data)[:, 1]
train_roc_auc = roc_auc_score(y_train, y_train_pred)
print("Train set ROC AUC score: {:.10f}".format(train_roc_auc))
# Printing the test result of AUC ROC of the best model after hyperparameter Tuning
y_test_pred = optimal_mlp.predict_proba(X_test_improve_data)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_test_pred)
print("Test set ROC AUC score: {:.10f}".format(test_roc_auc))

# Loss curve 
plt.plot(optimal_mlp.loss_curve_)
plt.title("Loss Curve", fontsize=17)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(np.arange(0, 0.81, 0.1), fontsize=14)
plt.show()
### End run on the optimal model ###

######  End improve of the data - PCA technique ######

### Predict alex test set ###
optimal_mlp.fit(X_train_XY_pkl, data_senntimet)
predicted_sentiment_final = optimal_mlp.predict(X_test_alex) 
results_y = pd.DataFrame({'sentiment': predicted_sentiment_final})
G16_ytest = results_y.to_pickle("G16_ytest.pkl")

################################################## End of part B ##################################################