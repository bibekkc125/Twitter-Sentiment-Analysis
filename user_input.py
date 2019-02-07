from sklearn.externals import joblib
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from main import tt

loaded_model =joblib.load("trainedmodel.sav")




def predict_tweet(tt,input_str):
    stops = stopwords.words('english')
    st = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    a = re.sub(r'@[a-zA-Z0-9_]+','',input_str)
    b = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',a)
    c = re.sub(r'[.%+*/0-9?&#]+','',b)
    d = re.sub(r'([-,]+)|((\')+)|([;:()!@#=$]+)','',c)
    e = lambda x: " ".join(x for x in x.split() if x not in stops)
    f = e(d)
    g = lambda x: " ".join(x.lower() for x in x.split())
    h = g(f)
    i =nltk.word_tokenize(h)
    # print(i)
    j = lambda x: (st.stem(word) for word in x)
    k= j(i)
    l = lambda x: (lemmatizer.lemmatize(word) for word in x)
    m = l(k)
    # print(m)
    ques = np.array([m])
    testing = tt.transform(m)
    return loaded_model.predict(testing)


for i in range(5):
    input_str = input('\nEnter a string: ')
    output = predict_tweet(tt,input_str)
    if output[0] == 0:
        print ("Sentiment : Negative")
    elif output[0] == 4:
        print ("Sentiment :Positive")