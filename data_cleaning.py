import pandas as pd
from nltk.corpus import stopwords

stops = stopwords.words('english')

# import dataset
cols = ['target','id','date','query','user','tweets']
df =pd.read_csv('dataset.csv', encoding='latin',names = cols)
# Remove unnecesssary columns
df_1 = df.drop(['id','date','user','query'],axis = 1)


# Remove username
df_without_mentions = df_1['tweets'].str.replace('@[a-zA-Z0-9_]+','',regex =True)
# Remove Url
df_without_url = df_without_mentions.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',regex =True)
# Remove special characters
df_without_fullstops = df_without_url.str.replace('[.%+*/0-9?&#]+','',regex = True)
# Remove words with 2 or fewer letters
df_new = df_without_fullstops.str.replace(r'\b\w{1,2}\b', '', regex =True)
# Remove special characters
df = df_new.str.replace('([-,]+)|((\')+)|([;:()!@#=$]+)','',regex =True)
# convert strings into lower case
df = df.apply(lambda x:" ".join(x.lower() for x in x.split()))
# Removing stop words
df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stops))
df = df.str.findall(r'[\[\]]+')
print(df)
# df_1['tweets'] = df
# df = df_1[['tweets','target']]
# print(df.iloc[20000:].head(5))
#
#
# df.to_csv('dataset_cleaned.csv')