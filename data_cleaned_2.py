import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



st = PorterStemmer()
lemmatizer = WordNetLemmatizer()
df =pd.read_csv('dataset_cleaned.csv')
df['text'] = df['tweets'].fillna('').apply(nltk.word_tokenize)

df['stemmed']= df['text'].apply(lambda x:[st.stem(word) for word in x])
df['lemmatized'] = df['stemmed'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])

df = df[['lemmatized','target']]

# df.to_csv('data_finalized.csv')
print(df.iloc[20000:].head(5))