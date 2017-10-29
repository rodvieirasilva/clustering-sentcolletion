import csv
import json
from textdict import TextDict
from pca import PlotPCA
import hdbscan
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from util import mkdir
from sklearn.metrics.pairwise import pairwise_distances
from singlelink import SingleLink

#from wordcloud import WordCloud #--> Módulo wordcloud
import matplotlib.pyplot as plt

def plotpca(pca, algoritmo, title, data, Y):
    pca.plotpca(title, data, Y, set(Y))
    pca.savefig("{0}/{1}.png".format(algoritmo, title))

def Kmedia(k):
    print("Criando Modelo com Kmedia e k="+str(k))
    model = KMeans(n_clusters=k, max_iter=1000)    
    model.title = "KMeans_k{}".format(k)
    model.name = "KMeans"
    return model

def singleLink(k):
    print("Criando Modelo com SingleLink e k="+str(k))
    model = SingleLink(k=k) 
    model.title = "SingleLink_k{}".format(k)
    model.name = "SingleLink"
    return model

def AvaliaSalvaResultado(data, model, processed, complete, pca=None):
    print("Gerando a partição para: " + model.title)
    model.fit(data)
    print("Avaliando a partição encontrada: " + model.title)
    tweets = [item['tweet'] for item in complete]  
    theme = [item['theme'] for item in complete]  
    classe = [item['class'] for item in complete]  
    algoritmo = model.name
    particao = model.labels_
    savecsvParticao("{0}/{1}.csv".format(model.name, model.title), tweets, processed, particao)
    print("Indice Rand ajustado com relação a Classe "+str(metrics.adjusted_rand_score(classe, particao)))
    print("Indice Rand ajustado com relação ao Tema "+str(metrics.adjusted_rand_score(theme, particao)))
    
    print("Indice Silhueta com relação a base inicial "+str(metrics.silhouette_score(data, particao)))

    if(pca != None):
        plotpca(pca, algoritmo, model.title, data, particao)

def savecsvParticao(filename, tweet, processed, labels):
    mkdir(filename)
    with open(filename, 'w', encoding='utf-8') as file:
           for idx, item in enumerate(labels):
                file.write(str(tweet[idx]).replace(";", ""))
                file.write(';')
                file.write(str(processed[idx]))
                file.write(';')
                file.write(str(item))
                file.write(';')
                file.write('\n')

def menu():
    print('Escolha uma das opções: ')
    print('1 - Aplicar o k-medias')
    print('2 - Aplicar Single-link')
    print('3 - Aplicar ...')
    print('4 - Aplicar ...')
    print('5 - Mostrar PCAs')
    print('-1 - Sair')
    opcao = input('Opção: ')
    return int(opcao)                

def inputK():
    k = input('Informe o Número de K ou informe 0 para Executar com varios valores para K: ')
    return int(k)

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
    
    # Plot PCA
    pca = PlotPCA(data=bagofwords)
    opcao = menu()
    while(opcao != -1):
        models = []
        interval = 0
        # Aplicando o k-medias    
        if opcao == 1:
            k = inputK()
            if k == 0:
                for x in range(1, 10):
                    models.append(Kmedia(x))
            else:
                models.append(Kmedia(k))
        elif opcao == 2:
            k = inputK()
            if k == 0:
                for x in range(1, 10):
                    models.append(singleLink(x))
            else:
                models.append(singleLink(k))
        elif opcao == 5:
            plt.show()
        else:
            print('outro')
        
        for model in models:
            
            AvaliaSalvaResultado(bagofwords, model, processed, complete, pca)
            
        opcao = menu()

    print('\nFinished')

if __name__ == '__main__':
    main()          