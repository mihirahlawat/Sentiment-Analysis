import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data=pd.read_csv('SemEval2017-task4-dev.subtask-A.english.INPUT.txt',header=None,sep='\t',error_bad_lines=False)
X=data.iloc[:,2].values
y=data.iloc[:,1].values

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 20632):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cvec = CountVectorizer(lowercase=False,ngram_range = (1,2),max_df = .85, max_features = 1500)

X = cvec.fit_transform(corpus).toarray()
