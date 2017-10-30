import time
import csv
import json
from textdict import TextDict
from pca import PlotPCA
import matplotlib.pyplot as plt
from sklearn import metrics, cluster, datasets, mixture
from sklearn.cluster import KMeans
import numpy as np
from util import mkdir
from sklearn.metrics.pairwise import pairwise_distances
from singlelink import SingleLink
from sklearn.neighbors import kneighbors_graph
from stats import StatList
import matplotlib.pyplot as plt

def plotpca(pca, algoritmo, title, data, Y, stat):
    pca.plotpca(title, data, Y, set(Y))
    plt.figtext(.02, .02, stat.toStringChart())
    pca.savefig("{0}/{1}.png".format(algoritmo, title))    

def GaussianMixture(k):    
    print("Criando Modelo com GaussianMixture e k="+str(k))
    t0 = time.time()
    model = mixture.GaussianMixture(n_components=k, covariance_type='full')
    model.title = "GaussianMixture_k{}".format(k)
    model.name = "GaussianMixture"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def Kmedia(k):
    print("Criando Modelo com Kmedia e k="+str(k))
    t0 = time.time()
    model = KMeans(n_clusters=k, max_iter=1000)    
    model.title = "KMeans_k{}".format(k)
    model.name = "KMeans"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def singleLink(distance, k):
    print("Criando Modelo com SingleLink e k="+str(k))
    t0 = time.time()
    model = SingleLink(distance, k) 
    model.title = "SingleLink_k{}".format(k)
    model.name = "SingleLink"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def WardLink(data, k):
    print("Criando Modelo com WardLink e k="+str(k))
    t0 = time.time()
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        data, n_neighbors=k, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)  
    model = cluster.AgglomerativeClustering(
        n_clusters=k, linkage='ward',
        connectivity=connectivity)
    model.title = "WardLink_k{}".format(k)
    model.name = "WardLink"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def SpectralClustering(k):
    print("Criando Modelo com SpectralClustering e k="+str(k))
    t0 = time.time()
    model = cluster.SpectralClustering(
        n_clusters=k, eigen_solver='arpack',
        affinity="nearest_neighbors", n_neighbors=5)
    model.title = "SpectralClustering_k{}".format(k)
    model.name = "SpectralClustering"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def DBSCAN():
    print("Criando Modelo com SpectralClustering")  
    t0 = time.time()  
    model = cluster.DBSCAN(eps=.3)
    model.title = "DBSCAN"
    model.name = "DBSCAN"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = 0
    return model

def AffinityPropagation():
    print("Criando Modelo com AffinityPropagation")
    t0 = time.time() 
    model = cluster.AffinityPropagation(
        damping=.9, preference=-200)
    model.title = "AffinityPropagation"
    model.name = "AffinityPropagation"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = 0
    return model

def AgglomerativeClustering(data, k):
    print("Criando Modelo com AgglomerativeClustering e k="+str(k)) 
    t0 = time.time()  
     # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        data, n_neighbors=3, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T) 
    model = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=k, connectivity=connectivity)
    model.title = "AgglomerativeClustering_k{}".format(k)
    model.name = "AgglomerativeClustering"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k = k
    return model

def Birch(k):
    print("Criando Modelo com Birch e k="+str(k))  
    t0 = time.time() 
    model = cluster.Birch(n_clusters=k)
    model.title = "Birch_k{}".format(k)
    model.name = "Birch"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k=k
    return model

def MeanShift():
    print("Criando Modelo com MeanShift") 
    t0 = time.time()  
    model = cluster.MeanShift(bandwidth=None, bin_seeding=True)
    model.title = "MeanShift"
    model.name = "MeanShift"
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    model.k=0
    return model

def MiniBatchKMeans(k):
    print("Criando Modelo com MiniBatchKMeans e k="+str(k))  
    t0 = time.time() 
    model = cluster.MiniBatchKMeans(n_clusters=k)
    model.title = "MiniBatchKMeans_k{}".format(k)
    model.name = "MiniBatchKMeans"
    model.k = k
    model.beginCreationTime = t0
    model.endCreationTime = time.time()
    return model

def AvaliaSalvaResultado(data, model, processed, complete, stats, pca=None):
    print("Gerando a partição para: " + model.title)
    stat = stats.add(model.k)
    stat.beginCreationTime = model.beginCreationTime
    stat.endCreationTime = model.endCreationTime    
    stat.beginExecutionTime = time.time()    
    model.fit(data)
    if hasattr(model, 'labels_'):
        particao = model.labels_
    else:
        particao = model.predict(data)
    stat.endExecutionTime = time.time()
    stat.calc(data, particao)
    print(stat.toString())
    
    print("Avaliando a partição encontrada: " + model.title)
    tweets = [item['tweet'] for item in complete]    
    savecsvParticao("{0}/{1}.csv".format(model.name, model.title), tweets, processed, particao, stat.toString())

    if(pca != None):
        plotpca(pca, model.name , model.title, data, particao, stat)

def savecsvParticao(filename, tweet, processed, labels, strstats):
    mkdir(filename)
    with open(filename, 'w', encoding='utf-8') as file:
           file.write(strstats)
           file.write('\n\n')

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
    print('4 - Aplicar GaussianMixture')
    print('5 - Aplicar MiniBatchKMeans')
    print('6 - Aplicar AffinityPropagation')
    print('7 - Aplicar MeanShift')
    print('8 - Aplicar SpectralClustering')
    print('9 - Aplicar WardLink')
    print('10 - Aplicar AgglomerativeClustering')
    print('11 - Aplicar DBSCAN')
    print('12 - Aplicar Birch')
    print('13 - Mostrar Gráficos Gerados')
    print('-1 - Sair')
    opcao = input('Opção: ')
    return int(opcao)                

def inputK():
    k = input('Informe o Número de K ou informe 0 para Executar com varios valores para K: ')
    k = int(k)
    if k == 0:
        return range(2, 10)
    return [k]

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

    statList = StatList(complete)

    opcao = menu()
    while(opcao != -1):
        models = []
        interval = 0
        # Aplicando o k-medias    
        if opcao == 1:
            ks = inputK()
            for k in ks:
                models.append(Kmedia(k))

        elif opcao == 2:
            ks = inputK()
            distance = pairwise_distances(bagofwords, Y=None, metric='euclidean', n_jobs=1)
            for k in ks:
                models.append(singleLink(distance, k))

        elif opcao == 4:
            ks = inputK()
            for k in ks:
                models.append(GaussianMixture(k))          
        elif opcao==5:
            ks = inputK()
            for k in ks:
                models.append(MiniBatchKMeans(k))
        elif opcao==6:
            models.append(AffinityPropagation())
        elif opcao==7:
            models.append(MeanShift())                
        elif opcao==8:
            ks = inputK()
            for k in ks:
                models.append(SpectralClustering(k))            
        elif opcao==9:
            ks = inputK()
            for k in ks:
                models.append(WardLink(bagofwords, k))
        elif opcao==10:
            ks = inputK()
            for k in ks:
                models.append(AgglomerativeClustering(bagofwords, k))
        elif opcao==11:
            models.append(DBSCAN()) 
        elif opcao==12:
            ks = inputK()
            for k in ks:
                models.append(Birch(k))
        elif opcao == 13:
            plt.show()      
        else:
            print('outro')
        
        for model in models:            
            AvaliaSalvaResultado(bagofwords, model, processed, complete, statList, pca)
            
        statList.plot()
        opcao = menu()

    print('\nFinished')

if __name__ == '__main__':
    main()          