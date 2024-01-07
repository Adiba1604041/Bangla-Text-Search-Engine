'''''
# Import Module
from bs4 import BeautifulSoup
import requests
# Website URL
URL = 'https://www.fake-plants.co.uk/'

# Page content from Website URL
page = requests.get(URL)

# Function to remove tags
def remove_tags(html):

	# parse html content
	soup = BeautifulSoup(html, "html.parser")

	for data in soup(['style', 'script']):
		# Remove tags
		data.decompose()

	# return data by retrieving the tag content
	return ' '.join(soup.stripped_strings)

# Print the extracted data
print(remove_tags(page.content))
'''''
import numpy as np
import pandas as pd
data = pd.read_csv( 'F:\mydemo.csv')
print(data.head())


import string
print("\n\n ####### Show punctuation #########")
print(string.punctuation)

pnct='!"#$%&\'()*+,-./:;<=>?‘’@[\\]^_`{|}~।'

def remove_punc(txt):
  txt_nopunctuation= "".join([c for c in txt if c not in pnct])
  return txt_nopunctuation
print("\n\n ######### Punctuation Removal ##########")
data['Description_clean']= data['Description'].apply(lambda x: remove_punc(x))
print(data.head())

####TOKENIZATION

from bnlp import NLTKTokenizer
def tokenize(txt):
  bnltk = NLTKTokenizer()
  tokens=bnltk.word_tokenize(txt)
  return tokens

print("\n\n ######### Tokenization ########")
data['Description_clean_tokenized']= data['Description_clean'].apply(lambda x:tokenize(x.lower()))
print(data.head())


#####STOPWORD REMOVAL
stopwords=['অতএব','অথচ','অথবা','অনুযায়ী','অনেক','অনেকে','অনেকেই','অন্তত','অন্য','অবধি','অবশ্য','অর্থাত','আই','আগামী','আগে','আগেই','আছে','আজ','আদ্যভাগে','আপনার','আপনি','আবার',
 'আমরা',
 'আমাকে',
 'আমাদের',
 'আমার',
 'আমি',
 'আর',
 'আরও',
 'ই',
 'ইত্যাদি',
 'ইহা',
 'উচিত',
 'উত্তর',
 'উনি',
 'উপর',
 'উপরে',
 'এ',
 'এঁদের',
 'এঁরা',
 'এই',
 'একই',
 'একটি',
 'একবার',
 'একে',
 'এক্',
 'এখন',
 'এখনও',
 'এখানে',
 'এখানেই',
 'এটা',
 'এটাই',
 'এটি',
 'এত',
 'এতটাই',
 'এতে',
 'এদের',
 'এব',
 'এবং',
 'এবার',
 'এমন',
 'এমনকী',
 'এমনি',
 'এর',
 'এরা',
 'এল',
 'এস',
 'এসে',
 'ঐ',
 'ও',
 'ওঁদের',
 'ওঁর',
 'ওঁরা',
 'ওই',
 'ওকে',
 'ওখানে',
 'ওদের',
 'ওর',
 'ওরা',
 'কখনও',
 'কত',
 'কবে',
 'কমনে',
 'কয়েক',
 'কয়েকটি',
 'করছে',
 'করছেন',
 'করতে',
 'করবে',
 'করবেন',
 'করলে',
 'করলেন',
 'করা',
 'করাই',
 'করায়',
 'করার',
 'করি',
 'করিতে',
 'করিয়া',
 'করিয়ে',
 'করে',
 'করেই',
 'করেছিলেন',
 'করেছে',
 'করেছেন',
 'করেন',
 'কাউকে',
 'কাছ',
 'কাছে',
 'কাজ',
 'কাজে',
 'কারও',
 'কারণ',
 'কি',
 'কিংবা',
 'কিছু',
 'কিছুই',
 'কিন্তু',
 'কী',
 'কে',
 'কেউ',
 'কেউই',
 'কেখা',
 'কেন',
 'কোটি',
 'কোন',
 'কোনও',
 'কোনো',
 'ক্ষেত্রে',
 'কয়েক',
 'খুব',
 'গিয়ে',
 'গিয়েছে',
 'গিয়ে',
 'গুলি',
 'গেছে',
 'গেল',
 'গেলে',
 'গোটা',
 'চলে',
 'চান',
 'চায়',
 'চার',
 'চালু',
 'চেয়ে',
 'চেষ্টা',
 'ছাড়া',
 'ছাড়াও',
 'ছিল',
 'ছিলেন',
 'জন',
 'জনকে',
 'জনের',
 'জন্য',
 'জন্যওজে',
 'জানতে',
 'জানা',
 'জানানো',
 'জানায়',
 'জানিয়ে',
 'জানিয়েছে',
 'জে',
 'জ্নজন',
 'টি',
 'ঠিক',
 'তখন',
 'তত',
 'তথা',
 'তবু',
 'তবে',
 'তা',
 'তাঁকে',
 'তাঁদের',
 'তাঁর',
 'তাঁরা',
 'তাঁাহারা',
 'তাই',
 'তাও',
 'তাকে',
 'তাতে',
 'তাদের',
 'তার',
 'তারপর',
 'তারা',
 'তারৈ',
 'তাহলে',
 'তাহা',
 'তাহাতে',
 'তাহার',
 'তিনঐ',
 'তিনি',
 'তিনিও',
 'তুমি',
 'তুলে',
 'তেমন',
 'তো',
 'তোমার',
 'থাকবে',
 'থাকবেন',
 'থাকা',
 'থাকায়',
 'থাকে',
 'থাকেন',
 'থেকে',
 'থেকেই',
 'থেকেও',
 'দিকে',
 'দিতে',
 'দিন',
 'দিয়ে',
 'দিয়েছে',
 'দিয়েছেন',
 'দিলেন',
 'দু',
 'দুই',
 'দুটি',
 'দুটো',
 'দেওয়া',
 'দেওয়ার',
 'দেওয়া',
 'দেখতে',
 'দেখা',
 'দেখে',
 'দেন',
 'দেয়',
 'দ্বারা',
 'ধরা',
 'ধরে',
 'ধামার',
 'নতুন',
 'নয়',
 'না',
 'নাই',
 'নাকি',
 'নাগাদ',
 'নানা',
 'নিজে',
 'নিজেই',
 'নিজেদের',
 'নিজের',
 'নিতে',
 'নিয়ে',
 'নিয়ে',
 'নেই',
 'নেওয়া',
 'নেওয়ার',
 'নেওয়া',
 'নয়',
 'পক্ষে',
 'পর',
 'পরে',
 'পরেই',
 'পরেও',
 'পর্যন্ত',
 'পাওয়া',
 'পাচ',
 'পারি',
 'পারে',
 'পারেন',
 'পি',
 'পেয়ে',
 'পেয়্র্',
 'প্রতি',
 'প্রভৃতি',
 'প্রযন্ত',
 'প্রায়',
 'প্রায়',
 'ফলে',
 'ফিরে',
 'ফের',
 'বক্তব্য',
 'বদলে',
 'বন',
 'বরং',
 'বলতে',
 'বলল',
 'বললেন',
 'বলা',
 'বলে',
 'বলেছেন',
 'বলেন',
 'বসে',
 'বহু',
 'বা',
 'বাদে',
 'বার',
 'বি',
 'বিনা',
 'বিভিন্ন',
 'বিশেষ',
 'বিষয়টি',
 'বেশ',
 'বেশি',
 'ব্যবহার',
 'ব্যাপারে',
 'ভাবে',
 'ভাবেই',
 'মতো',
 'মতোই',
 'মধ্যভাগে',
 'মধ্যে',
 'মধ্যেই',
 'মধ্যেও',
 'মনে',
 'মাত্র',
 'মাধ্যমে',
 'মোট',
 'মোটেই',
 'যখন',
 'যত',
 'যতটা',
 'যথেষ্ট',
 'যদি',
 'যদিও',
 'যা',
 'যাঁর',
 'যাঁরা',
 'যাওয়া',
 'যাওয়ার',
 'যাওয়া',
 'যাকে',
 'যাচ্ছে',
 'যাতে',
 'যাদের',
 'যান',
 'যাবে',
 'যায়',
 'যার',
 'যারা',
 'যিনি',
 'যে',
 'যেখানে',
 'যেতে',
 'যেন',
 'যেমন',
 'র',
 'রকম',
 'রয়েছে',
 'রাখা',
 'রেখে',
 'শুধু',
 'শুরু',
 'সঙ্গে',
 'সঙ্গেও',
 'সব',
 'সবার',
 'সমস্ত',
 'সহ',
 'সহিত',
 'সাধারণ',
 'সামনে',
 'সি',
 'সুতরাং',
 'সে',
 'সেই',
 'সেখান',
 'সেখানে',
 'সেটা',
 'সেটাই',
 'সেটাও',
 'সেটি',
 'স্পষ্ট',
 'স্বয়ং',
 'হইতে',
 'হইবে',
 'হইয়া',
 'হওয়া',
 'হওয়ায়',
 'হওয়ার',
 'হচ্ছে',
 'হত',
 'হতে',
 'হতেই',
 'হন',
 'হবে',
 'হবেন',
 'হয়',
 'হয়তো',
 'হয়নি',
 'হয়ে',
 'হয়েই',
 'হয়েছিল',
 'হয়েছে',
 'হয়েছেন',
 'হল',
 'হলে',
 'হলেই',
 'হলেও',
 'হলো',
 'হাজার',
 'হিসাবে',
 'হৈলে',
 'হোক',
 'হয়']

def remove_stopwords(txt_tokenized):
  txt_clean=[word for word in txt_tokenized if word not in stopwords]
  return txt_clean
print("\n\n ########## Stopword Removal ###########")
data['Des_no_stopword']=data['Description_clean_tokenized'].apply(lambda x: remove_stopwords(x))
print(data.head())


######TFIDF VECTORIZER

print("\n\n ##########TFIDF VECTORIZER #############")
from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)
tfidf.fit(data['Description_clean_tokenized'])

from pandas import DataFrame

def create_document_term_matrix(msg_list,vectorizer):
  doc_term_matrix= vectorizer.fit_transform(msg_list)
  return DataFrame(doc_term_matrix.toarray(), columns=vectorizer.get_feature_names_out())
data_s=data['Description_clean_tokenized']
Tscore_frame=create_document_term_matrix(data_s,tfidf)
Tscore_frame=Tscore_frame.transpose()

print(Tscore_frame)

Tscore_frame=create_document_term_matrix(data_s,tfidf)

from sqlalchemy import create_engine

# create sqlalchemy engine

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="",
                               db="tfidf_vector"))
Tscore_frame=Tscore_frame.transpose()
Tscore_frame.to_sql('third_table', con = engine, if_exists = 'append', chunksize = 1000,)
#Tscore_frame.to_sql('second_table', con = engine, if_exists = 'append', chunksize = 1000,)
#Tscore_frame.to_sql('second_table', con = engine, if_exists = 'append', chunksize = 1000,)

print(Tscore_frame)






''''
####Connect Database
import mysql.connector
mydb= mysql.connector.connect(
      host="localhost",
      user='root',
      password="",
      database="tfidf_vector"
)
data.to_sql('tfidf_vector', mydb, if_exists='append',index=False)

if mydb.is_connected():
 print("\n\n ####### Database Connection #########\n")
 print("Successfully Connected")

import pandas as pd

# Create dataframe
data = pd.DataFrame({
    'book_id':[12345, 12346, 12347],
    'title':['Python Programming', 'Learn MySQL', 'Data Science Cookbook'],
    'price':[29, 23, 27]
})

print(data)
'''
