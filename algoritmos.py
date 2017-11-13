import time
import csv
import json
from textdict import TextDict
from pca import PlotPCA
import matplotlib.pyplot as plt
from sklearn import metrics, cluster, datasets, mixture
import numpy as np
from util import mkdir, inputInt, printGreen, printRed, save
from scipy.spatial.distance  import pdist
from simplelinkage import SimpleLinkage
from sklearn.neighbors import kneighbors_graph
from stats import StatList
from simplewordcloud import SimpleWordCloud
import matplotlib.pyplot as plt

class Algoritmos:
    data=None
    complete=None
    processed=None
    distance=None
    algs=None
    algsK=None
    wordCloud = SimpleWordCloud()

    def __init__(self, data=None, complete=None, processed=None):

        if complete==None:
            print('Carregando Base de Dados Completa')
            # Base de Dados Completa
            with open('basesjson/complete.json') as json_data:
                complete = json.load(json_data)
        
        if processed==None:
            print('Carregando Tweets Pré-Processados')
            # Tweets Pré-Processados
            with open('basesjson/processed.json') as json_data:
                processed = json.load(json_data)

        if data==None:
            print('Carregando Base de Dados Pré-Processada')
            # Base de Dados Pré-Processada
            with open('basesjson/sklearn_bagofwords.json') as json_data:
                data = json.load(json_data)

        self.complete = complete
        self.data = data
        self.processed = processed  
        self.distance = pdist(data, metric='euclidean')
        self.pca = PlotPCA(data=data)
        self.algsP = [self.KMeans, self.SingleLink, 
                     self.WardLink, self.DBSCAN]
        self.algs = [self.KMeans, self.SingleLink, 
                     self.WardLink, self.DBSCAN, self.GaussianMixture, self.SpectralClustering, 
                     self.AgglomerativeClustering, 
                     self.Birch, self.MiniBatchKMeans, self.AffinityPropagation, self.MeanShift]
        self.algsK = [self.KMeans, self.SingleLink, 
                     self.GaussianMixture,
                     self.WardLink, self.SpectralClustering,
                     self.AgglomerativeClustering, 
                     self.Birch, self.MiniBatchKMeans ]
        self.algsDend = [self.SingleLink, 
                     self.WardLink]

    def ElbowMethod(self):
        distortions = []
        print('\nGerando gráfico de distorção para o algoritmo K-Means')
        
        for i in range (1, 51):
            km_model = cluster.KMeans(n_clusters=i)
            km_model.fit(self.data)
            distortions.append(km_model.inertia_)
            
        plt.plot(range(1, 51), distortions, marker='o')
        plt.xlabel('Número de clusters')
        plt.ylabel('Distorção')
        plt.title('Método do cotovelo')
        plt.show()
        plt.savefig('KMeans/ElbowMethod.png')
        
        print()
    
    def KMeans(self, k):
        print("Criando Modelo com KMeans e k="+str(k))
        t0 = time.time()
        model = cluster.KMeans(n_clusters=k, max_iter=1000)    
        model.title = "KMeans_k{}".format(k)
        model.name = "KMeans"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k = k
        return model

    def SingleLink(self, k):
        print("Criando Modelo com SingleLink e k="+str(k))
        t0 = time.time()
        model = SimpleLinkage(self.distance, k, 'single') 
        model.title = "SingleLink_k{}".format(k)
        model.name = "SingleLink"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k = k
        return model

    def WardLink(self, k):
        print("Criando Modelo com WardLink e k="+str(k))
        # t0 = time.time()
        # # connectivity matrix for structured Ward
        # connectivity = kneighbors_graph(
        #     self.data, n_neighbors=k, include_self=False)
        # # make connectivity symmetric
        # connectivity = 0.5 * (connectivity + connectivity.T)  
        # model = cluster.AgglomerativeClustering(
        #     n_clusters=k, linkage='ward',
        #     connectivity=connectivity)
        # model.title = "WardLink_k{}".format(k)
        # model.name = "WardLink"
        # model.beginCreationTime = t0
        # model.endCreationTime = time.time()
        # model.k = k
        t0 = time.time()
        model = SimpleLinkage(self.distance, k, 'ward') 
        model.title = "WardLink_k{}".format(k)
        model.name = "WardLink"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k = k
        return model

    def GaussianMixture(self, k):    
        print("Criando Modelo com GaussianMixture e k="+str(k))
        t0 = time.time()
        model = mixture.GaussianMixture(n_components=k, covariance_type='full')
        model.title = "GaussianMixture_k{}".format(k)
        model.name = "GaussianMixture"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k = k
        return model

    def SpectralClustering(self, k):
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

    def DBSCAN(self):
        print("Criando Modelo com SpectralClustering")  
        t0 = time.time()  
        model = cluster.DBSCAN(eps=4, min_samples = 100)
        model.title = "DBSCAN"
        model.name = "DBSCAN"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k = 0
        return model

    def AffinityPropagation(self):
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

    def AgglomerativeClustering(self, k):
        print("Criando Modelo com AgglomerativeClustering e k="+str(k)) 
        t0 = time.time()  
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            self.data, n_neighbors=3, include_self=False)
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

    def Birch(self, k):
        print("Criando Modelo com Birch e k="+str(k))  
        t0 = time.time() 
        model = cluster.Birch(n_clusters=k)
        model.title = "Birch_k{}".format(k)
        model.name = "Birch"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k=k
        return model

    def MeanShift(self):
        print("Criando Modelo com MeanShift") 
        t0 = time.time()  
        model = cluster.MeanShift(bandwidth=None, bin_seeding=True)
        model.title = "MeanShift"
        model.name = "MeanShift"
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        model.k=0
        return model

    def MiniBatchKMeans(self, k):
        print("Criando Modelo com MiniBatchKMeans e k="+str(k))  
        t0 = time.time() 
        model = cluster.MiniBatchKMeans(n_clusters=k)
        model.title = "MiniBatchKMeans_k{}".format(k)
        model.name = "MiniBatchKMeans"
        model.k = k
        model.beginCreationTime = t0
        model.endCreationTime = time.time()
        return model

    def avaliaSalvaResultado(self, model, stats):
        print("Gerando a partição para: " + model.title)
        stat = stats.add(model.k)
        stat.beginCreationTime = model.beginCreationTime
        stat.endCreationTime = model.endCreationTime    
        stat.beginExecutionTime = time.time()    
        model.fit(self.data)
        if hasattr(model, 'labels_'):
            particao = model.labels_
        else:
            particao = model.predict(self.data)
        
        if hasattr(model, 'dendrogram'):
            model.dendrogram()
            
        stat.endExecutionTime = time.time()
        print("Avaliando a partição encontrada: " + model.title)
        stat.calc(particao)
        print(stat.toStringChart())              
        print("Salvando partição encontrada: " + model.title)
        self.savecsvParticao("{0}/{1}.csv".format(model.name, model.title), particao, stat.toString())
        print("Gerando PCA: " + model.title)
        self.pca.plotpca(model.title, self.data, particao, set(particao))
        self.pca.savefig("{0}/{1}.png".format(model.name, model.title))
        self.wordCloud.plotLabels(self.processed, particao, model.name, model.title)
        save("{0}/{1}.json".format(model.name, model.title), particao.tolist())

    def savecsvParticao(self, filename, labels, strstats):
        tweets = [item['tweet'] for item in self.complete]  
        mkdir(filename)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(strstats)
            file.write('\n\n')
            file.write('"tweet_original";"tweet_processed";"cluster"\n')
            for idx, item in enumerate(labels):
                file.write(str(tweets[idx]).replace(";", ""))
                file.write(';')
                file.write(str(self.processed[idx]))
                file.write(';')
                file.write(str(item))
                file.write(';')
                file.write('\n')

def inputK():
    k = inputInt('Informe o Número de K ou informe 0 para Executar K = [2...50]: ')
    if k == 0:
        return range(2, 51)
    return [k]

def menu(listAlgs, adicional):    
    print('Escolha uma das opções: ')
    i=0
    for alg in listAlgs:
        i+=1
        print('{0} - {1}'.format(i, alg.__name__))
    i+=1
    print('{0} - {1}'.format(i, adicional))
    print('0 - Sair')
    return inputInt('Opção: ', i)

def run(algoritmos, alg, ks=None):   
    try:
        statList = StatList(algoritmos.complete, algoritmos.data)
        statList.name = alg.__name__
        statList.prefix = ''
        if alg in algoritmos.algsK:        
            if ks is None:
                ks = inputK()
            statList.prefix = 'K{0}-{1}'.format(min(ks), max(ks))
            for k in ks:
                model = alg(k)
                algoritmos.avaliaSalvaResultado(model, statList)
        else:
            model = alg()
            algoritmos.avaliaSalvaResultado(model, statList)
        statList.plot()
    except Exception as e:
        print('Erro ao rodar alg.: ' + str(e))

def mainAll(algoritmos):
    opcao = menu(algoritmos.algs, 'Rodar todos com K=[2..50]')
    while(opcao!=0):
        if opcao>len(algoritmos.algs):
            for alg in algoritmos.algs:
                run(algoritmos, alg, range(2,51))
        elif opcao!=0:
            alg = algoritmos.algs[opcao-1]
            run(algoritmos, alg)
        opcao = menu(algoritmos.algs, 'Rodar todos com K=[2..50]') 

def main():
    algoritmos = Algoritmos()
    opcao = menu(algoritmos.algsP, 'Outros')
    while(opcao!=0):
        if opcao>len(algoritmos.algsP):
            mainAll(algoritmos)
        elif opcao!=0:
            if opcao == 1:
                algoritmos.ElbowMethod()
            alg = algoritmos.algsP[opcao-1]
            run(algoritmos, alg)
        opcao = menu(algoritmos.algsP, 'Outros')    

if __name__ == '__main__':
    main()                