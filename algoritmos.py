import csv
import json
from textdict import TextDict
from pca import PlotPCA

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import re

#from wordcloud import WordCloud #--> Módulo wordcloud
import matplotlib.pyplot as plt


def plotpca(title, data, Y):
    pca = PlotPCA(data=data)
    pca.plotpca(title, data, Y, set(Y))
    pca.show()

def Kmedia(data, k):
    print("Gerando a partição com Kmedia e k "+str(k))
    model = KMeans(n_clusters=k, max_iter=1000)
    model.fit(data)
    title = "KMeans_k{}".format(k)
    plotpca(title, data, model.labels_)
    return model.labels_

def AvaliaSalvaResultado(data, particao, algoritmo, k, processed, complete):
    print("Avaliando a partição encontrada")
    tweets = [item['tweet'] for item in complete]  
    theme = [item['theme'] for item in complete]  
    classe = [item['class'] for item in complete]  

    savecsvParticao("{0}\{0}_k{1}.csv".format(algoritmo, k), tweets, processed, particao)
    print("Indice Rand ajustado com relação a Classe "+str(metrics.adjusted_rand_score(classe, particao)))
    print("Indice Rand ajustado com relação ao Tema "+str(metrics.adjusted_rand_score(theme, particao)))
    print("Indice Silhueta com relação a base inicial "+str(metrics.silhouette_score(data, particao)))
    

    return 0

def savecsvParticao(filename, tweet, processed, labels):
    with open(filename, 'w', encoding='utf-8') as file:
           for idx, item in enumerate(labels):
                file.write(str(tweet[idx]).replace(";", ""))
                file.write(';')
                file.write(str(processed[idx]))
                file.write(';')
                file.write(str(item))
                file.write(';')
                file.write('\n')


def main():
    print('Started')
    print('Carregando as Bases de Dados')

    # Base de Dados Completa
    with open('basesjson/complete.json') as json_data:
        complete = json.load(json_data)
    
    # Tweets Pré-Processados
    with open('basesjson/processed.json') as json_data:
        processed = json.load(json_data)

    # Base de Dados Pré-Processada
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)
    
    print('Escolha uma das opções: ')
    print('1 - Aplicar o k-medias')
    print('2 - Aplicar Single-link')
    print('3 - Aplicar ...')
    print('4 - Aplicar ...')
    opcao = input('Opção: ')

    # Aplicando o k-medias    
    if opcao == '1':
        algoritmo = 'KMeans';
        k = input('Informe o Número de K ou informe 0 para Executar com varios valores para K: ')
        if k == '0':
            for x in range(1, 10):
                particao = Kmedia(bagofwords, x)
                AvaliaSalvaResultado(bagofwords, particao, algoritmo, k, processed, complete)
        else:
            particao = Kmedia(bagofwords, int(k))
            AvaliaSalvaResultado(bagofwords, particao, algoritmo, k, processed, complete)

    else:
        print('outro')

    print('\nFinished')

if __name__ == '__main__':
    main()          