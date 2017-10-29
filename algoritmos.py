import csv
import json
from textdict import TextDict

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import re

#from wordcloud import WordCloud #--> Módulo wordcloud
import matplotlib.pyplot as plt


def save(filename, data):
    with open(filename, 'w') as outfile:
            json.dump(data, outfile)

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


def Kmedia(data, k, tweets):
    model = KMeans(n_clusters=k, max_iter=1000)
    model.fit(data)
   
    savecsvParticao("KMeans\KMeans_k{}.csv".format(k), tweets, model.labels_)

    return 0

def savecsvParticao(filename, tweet, labels):
    with open(filename, 'w', encoding='utf-8') as file:
           for idx, item in enumerate(labels):
                file.write(str(tweet[idx]).replace(";", ""))
                file.write(';')
                file.write(str(item))
                file.write(';')
                file.write('\n')


def savecsv(filename, header, data):
    with open(filename, 'w', encoding='utf-8') as file:
        for cell in header:
            file.write('"{}"'.format(cell))
            file.write(';')
        file.write('\n')
        for row in data:
            for cell in row:
                file.write(str(cell))
                file.write(';')
            file.write('\n')

def main():
    print('Started')
    print('Carregando as Bases de Dados')

    # Base de Dados Completa
    with open('complete.json') as json_data:
        complete = json.load(json_data)
    tweets = [item['tweet'] for item in complete]  

    # Base de Dados Pré-Processada
    with open('sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)
    
    print('Escolha uma das opções: ')
    print('1 - Aplicar o k-medias')
    print('2 - Aplicar Single-link')
    print('3 - Aplicar ...')
    print('4 - Aplicar ...')
    opcao = input('Opção: ')

    # Aplicando o k-medias    
    if opcao == '1':
        k = input('Informe o NUmero de K ou informe 0 para Executar com varios valores para K: ')
        if k == '0':
            for x in range(1, 10):
                Kmedia(bagofwords, x, tweets)
        else:
            Kmedia(bagofwords, int(k), tweets)

    else:
        print('outro')

    print('\nFinished')

if __name__ == '__main__':
    main()          