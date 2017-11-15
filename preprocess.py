"""
-- Sent Collection v.1 para análise de agrupamento --
--                  Grupo 1                        --
--Marciele de Menezes Bittencourt                  --
--Rodrigo Vieira da Silva                          --
--Washington Rodrigo Dias da Silva                 --
-----------------------------------------------------
"""
import csv
import json
from textdict import TextDict
from simplewordcloud import SimpleWordCloud
import numpy as np
import re

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer #--> Classe utilizada para converter o array de strings em um array de binários
from util import save, savecsv

def readCSV(filename, theme, product):
    result = []
    with open(filename, 'r', encoding='utf-8') as reader:
        reader.readline()
        reader = csv.reader(reader, delimiter = ',', quotechar = '"')        
        for row in reader:
            result.append({"theme":theme, "product": product, "tweet": row[0], "class":row[1]})
    return result

def stats(name, data):
    print('-- {} --'.format(name))
    print('size: {}'.format(len(data)))
    print('--------')

def process(data):
    matrix = [item.replace('\t', ' \t').split(' ') for item in data]
    dictEnglish = TextDict('dicts/englishDict.txt')
    dictEmoticons = TextDict('dicts/emoticons.txt')
    dictStopWords = TextDict('dicts/stopwords.txt')
    dictLingo = TextDict('dicts/lingo.txt')
    separateLetter = "$_-();,.:{}?[]#!@\'\"/"
    replaced = True

    # Identificação de Termos
    while(replaced):
        replaced = False
        for row in matrix:
            for i, word in enumerate(row):
                word = word.strip().lower()
                word = word.strip().lower()
                word = re.sub('http://[^\\s]*', '', word)
                word = re.sub('https://[^\\s]*', '', word)
                row[i] = word                

                # Verificando se há emoticons
                if dictEmoticons.contains(word):
                    pass

                # Verificando se há Girias e Abreviaturas
                elif dictLingo.contains(word):
                    row[i] = dictLingo.translate(word)
                else:
                    for c in separateLetter:
                        word = word.replace(c, ' ')
                    if row[i] != word:
                        replaced = True
                        row[i] = ''
                        row.extend(word.split(' '))

    TermosIdentificados = [' '.join([item for item in row if item]).strip().replace(" ", " ") for row in matrix]

    # Refazendo a matrix de palavras para conseguir remover StopWords e realizar Normalização Morfológica nos termos transformados
    matrix = [item.split(' ') for item in TermosIdentificados]
                
    # Remoção de StopWords        
    for row in matrix:
        for i, word in enumerate(row):
            if dictStopWords.contains(word):
                row[i] = ''
                        
    # Normalização Morfológica
    for row in matrix:
        for i, word in enumerate(row):
            if dictEnglish.contains(word):
                row[i] = dictEnglish.translate(word)

    return [' '.join([item for item in row if item]).strip().replace(" ", " ") for row in matrix]

def main():
    print('Started')
    archeage = readCSV("basescsv/SentCollection - ARCHEAGE.csv", "game", "archeage")
    stats("archeage", archeage)
    save('basesjson/archeage.json', archeage)
    hobbit = readCSV("basescsv/SentCollection - HOBBIT.csv", "movie", "hobbit")
    stats("hobbit", hobbit)
    save('basesjson/hobbit.json', hobbit)
    iphone6 = readCSV("basescsv/SentCollection - IPHONE6.csv", "smartphone", "iphone6")
    stats("iphone6", iphone6)
    save('basesjson/iphone6.json', iphone6)

    complete = archeage + hobbit + iphone6
    stats("complete", complete)
    save('basesjson/complete.json', complete)
    
    tweets = [item['tweet'] for item in complete]  
    save('basesjson/tweets.json', tweets)

    processed = process(tweets)  
    save('basesjson/processed.json', processed)
    wordCloud = SimpleWordCloud()
    wordCloud.plot(tweets, 'base-original-wordcloud.png') #--> Juntar as palavras no gráfico wordcloud
    wordCloud.plot(processed, 'base-processed-wordcloud.png')
    # Contabilizando as palavras unicas das mensagens sem processamento
    uniquewords = set()
    allwords = []
    for tweet in tweets:
        tweet = tweet.split()
        for word in tweet:
            allwords.append(word)  
            uniquewords.add(word)     
    uniquewords = list(uniquewords)
    stats("allwords", allwords)
    save('basesjson/allwords.json', allwords)
    stats("uniquewords", uniquewords)
    save('basesjson/uniquewords.json', uniquewords)

    # Contabilizando as palavras unicas das mensagens com processamento
    uniquewords = set()
    allwords = []
    for tweet in processed:
        tweet = tweet.split()
        for word in tweet:
            allwords.append(word)
            uniquewords.add(word)
    uniquewords = list(uniquewords)
    stats("allwordsProcessed", allwords)
    save('basesjson/allwordsProcessed.json', allwords)
    stats("uniquewordsProcessed", uniquewords)
    save('basesjson/uniquewordsProcessed.json', uniquewords)
        
    #Utilizando o CountVectorizer para criar a bag-of-words
    vectorizer = CountVectorizer(stop_words=[], binary=True, vocabulary=uniquewords) #--> Inicializa a função CountVectorizer passando o parâmetro de extração de stop_words (Em inglês)

    sklearn_bagOfWords = vectorizer.fit_transform(processed).todense()
    save('basesjson/sklearn_bagofwords.json', sklearn_bagOfWords.tolist())
    
    featuresNames = list(vectorizer.vocabulary_)
    save('basesjson/sklearn_featuresnames.json', featuresNames)
    
    print('\nsklearn_featuresnames, size: {}'.format(len(featuresNames)))
    #save('sklearn_vocabulary.json', vectorizer.vocabulary_) #--> Exibe o vocabulario extraido pela função fit_transform()
    
    savecsv('basescsv/sklearn_bagofwords.csv', featuresNames, sklearn_bagOfWords.tolist())
    

    print('\nFinished')

if __name__ == '__main__':
    main()          