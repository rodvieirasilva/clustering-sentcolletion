# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:28:10 2017

@author: Wash
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

#Lendo arquivos em formato .csv
iphone6_dataset = pd.read_csv('./Databases/SentCollection - IPHONE6.csv')
archeage_dataset = pd.read_csv('./Databases/SentCollection - ARCHEAGE.csv')
hobbit3_dataset = pd.read_csv('./Databases/SentCollection - HOBBIT.csv')

#Renomeando as colunas e removendo o atributo alvo (Y)
iphone6_dataset.columns = ['X', 'Y']
iphone6_dataset.drop('Y', axis=1, inplace=True)

archeage_dataset.columns = ['X', 'Y']
archeage_dataset.drop('Y', axis=1, inplace=True)

hobbit3_dataset.columns = ['X', 'Y']
hobbit3_dataset.drop('Y', axis=1, inplace=True)

#Concatenando os conjuntos de dados
sentDatabase = pd.concat([iphone6_dataset, archeage_dataset, hobbit3_dataset], ignore_index=True)

print ('Exemplos de objetos da base de dados:\n', sentDatabase.head(3))
print ('\nDimensoes da base de dados concatenada:\n', sentDatabase.shape)


#Utilizando o CountVectorizer para criar a bag-of-words
vectorizer = CountVectorizer(stop_words='english')

X_trainning_set = vectorizer.fit_transform(sentDatabase).todense()
print (X_trainning_set.shape)


"""# Import `fake_or_real_news.csv` 
df = pd.read_csv('./Databases/fake_or_real_news.csv')
    
# Inspect shape of `df` 
print(df.shape)

df = df.set_index("Unnamed: 0")


# Print first lines of `df` 
print (df.head())

# Set `y` 
y = df.label 

# Drop the `label` column
df.drop("label", axis=1)

# Make training and test sets 
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize the `count_vectorizer` 
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data 
count_train = count_vectorizer.fit_transform(X_train) 

# Transform the test set 
count_test = count_vectorizer.transform(X_test)

print(count_train)"""

"""corpus = [
 'UNC played Duke in basketball',
 'Duke lost the basketball game']

vectorizer = CountVectorizer()
print (vectorizer.fit_transform(corpus).todense())
print (vectorizer.vocabulary_)"""



