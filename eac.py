from simplelinkage import SimpleLinkage
from pca import PlotPCA, plotPcaTweets
from scipy.spatial.distance  import pdist
import json
import math
from util import save, inputInt, saveclu
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, datasets, metrics
from random import randrange
import glob
from pvis import PVis

class EvidenceAccumulationCLustering:
    
    np=None
    P = None
    distanceNP = None
    C  = None    
    def __init__(self, P, X=None, distanceNP=None):
        if not (X is None):
            distanceNP = kneighbors_graph(X, len(X), mode='distance', include_self=True)
            distanceNP = distanceNP.toarray()
        self.distanceNP = distanceNP
        self.P = P

    def step1(self, kneighbors=None):
        # Calculo da Co-Associação sem verificação do vizinho C(i,j) = nij/N
        C = [[0 for i in range(len(self.P[0]))] for j in range(len(self.P[0]))]
        for p in self.P:            
            for i,x in enumerate(p):
                cK = 1
                for j, l in enumerate(p):
                    if j != i:
                        if p[i] == p[j]:                
                            C[i][cK-1] += 1/len(self.P)

                        cK += 1

                    if cK > kneighbors:
                        break      
        self.C = C
        return self.C

    def step2(self, kneighbors, k, alg='single', title=None):
        distance = pdist(self.C, metric='euclidean')
        singleLink = SimpleLinkage(distance=distance, k=k, alg=alg)
        singleLink.name = "EvidenceAccumulationClustering"
        singleLink.title = title
        singleLink.fit(self.C)
        singleLink.dendrogram()
        return singleLink.labels_

    def binomial(self, x, y):        
        if y == x:
            return 1
        elif y == 1:         
            return x
        elif y > x:          
            return 0
        else:                
            a = math.factorial(x)
            b = math.factorial(y)
            c = math.factorial(x-y)
            div = a // (b * c)
            return div

    def calcM(self, k, n):
        m = 0
        for l in range(1, k+1):
            m += self.binomial(k, l) * ((-1) ** (k-l)) * (l**n)
        return (1.0/math.factorial(k)) * m

def runEac(P, X, prefix, Y, ks=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):    
    print('Rodando EvidenceAccumulationClustering - {0}'.format(prefix))
    name = "EvidenceAccumulationClustering"
    eca = EvidenceAccumulationCLustering(P=P, X=X, distanceNP=None)    
    kneighbors = len(X)
    alg='single'    
    
    for k in ks:
        titleParams = "eac-{1}Link-neighbors{2}-k{3}".format(prefix, alg, kneighbors, k)
        title = "{0}/eac-{1}Link-neighbors{2}-k{3}".format(prefix, alg, kneighbors, k)
        eca.step1(kneighbors=kneighbors)  
        y_pred = eca.step2(kneighbors=kneighbors, k=k, title=title, alg=alg)
        # Indice Rand Ajustado
        ind_rand = metrics.adjusted_rand_score(Y, y_pred)
        # Indice NMI
        ind_nmi = metrics.normalized_mutual_info_score(Y, y_pred)
        pca = PlotPCA(filename=None, data=X)
        pca.plotpca(title=title, words=X, y_pred=y_pred, classnames=set(y_pred), text='Indice Rand Ajustado: {0} NMI: {1}'.format(ind_rand, ind_nmi))    
        pca.savefig("{0}/{1}_pca.png".format(name, title))
        save("{0}/{1}.json".format(name, title), y_pred.tolist()) 
        saveclu("{0}/{1}/pvis/partitions/{3}/{2}.clu".format(name, prefix, titleParams, alg), y_pred)

def runPVis(prefix, alg):
    baseref = "EvidenceAccumulationClustering\\{0}\\pvis\\".format(prefix)
    basepartitions = "EvidenceAccumulationClustering\\{0}\\pvis\\partitions\\{1}\\".format(prefix, alg)
    filenameRefs = glob.glob("{0}*.clu".format(baseref))
    filenamesPartitions = glob.glob("{0}*.clu".format(basepartitions))
    pvis = PVis()
    ordModes =  range(1,5)
    for filenameRef in filenameRefs:        
        for ordmode in ordModes:
            filenameOut = filenameRef +'-pvis'+str(alg)+'-ordmode'+ str(ordmode) + '.pdf'
            pvis.post(ordmode=ordmode, nrgrp=0, filenameRef=filenameRef, filenamesPartitions=filenamesPartitions, filenameOut=filenameOut)

def runKMeans(X, prefix):
    print('Executando 30 KMeans, k =[2,20] - {0}'.format(prefix))
    P = []
    folder = "EvidenceAccumulationClustering"
    for i in range(0,30):        
        k = randrange(2, 21)
        print('Executando {0} - KMeans, k={1}'.format(i, k))
        model = cluster.KMeans(n_clusters=k, max_iter=1000)
        titleParams = "{0:02.0f}-KMeans-k{1:02.0f}".format(i, k)
        model.title = "{0}/KMeans/{1}".format(prefix, titleParams)
        model.name = "KMeans"
        model.fit(X)
        p = model.labels_
        P.append(p)
        pca = PlotPCA(filename=None, data=X)
        titleFig = "{0} - {1} - KMeans k={2}".format(prefix, i, k)
        pca.plotpca(title=titleFig, words=X, y_pred=p, classnames=set(p))    
        pca.savefig("{0}/{1}_pca.png".format(folder, model.title))
        save("{0}/{1}.json".format(folder, model.title), p.tolist())
        saveclu("{0}/{1}/pvis/partitions/{3}/{2}.clu".format(folder, prefix, titleParams, model.name), p)

    return P

def loadP(prefix):
    files = glob.glob("EvidenceAccumulationClustering/{0}/KMeans/*.json".format(prefix))
    P = []
    for f in files:
        with open(f) as json_data:
            P.append(json.load(json_data)) 
    return P

def menu():
    print('Escolha uma das opções: ')
    print('1 - Executar 30 K-Means com K=[2,20] Dados da base IRIS')
    print('2 - Executar 30 K-Means com K=[2,20] Dados da base de Tweets')
    print('3 - Aplicar EAC Dados da base IRIS')
    print('4 - Aplicar EAC Partições KMeans da base de Tweets')
    print('5 - Aplicar K-Means e Executar EAC para todas as bases')
    print('6 - Rodar PVis para todas as particoes IRIS encontradas')
    print('7 - Rodar PVis para todas as particoes Tweets encontradas')
    print('0 - Sair')
    return inputInt('Opção: ', 7)

def loadTweets():
    X = []
    title = 'tweets'
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        X = json.load(json_data)
    Yclazz, Ytheme, Yclasstheme = plotPcaTweets('EvidenceAccumulationClustering/tweets')
    saveclu("EvidenceAccumulationClustering/{0}/pvis/realClass.clu".format(title), Yclazz)
    saveclu("EvidenceAccumulationClustering/{0}/pvis/realTheme.clu".format(title), Ytheme)
    saveclu("EvidenceAccumulationClustering/{0}/pvis/realClassTheme.clu".format(title), Yclasstheme)
    return X, title, Ytheme

def loadIris():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # Mostrando os dados da base IRIS com o PCA
    pca = PlotPCA(filename=None, data=X)
    title = "iris"
    titleFig = "PCA Dataset IRIS"
    #pca.plotpcaColors(title=titleFig, words=X, y_pred=Y, classnames=iris.target_names) 
    pca.plotpca(title=titleFig, words=X, y_pred=Y, classnames=iris.target_names) 
    pca.savefig("EvidenceAccumulationClustering/{0}/{1}_pca.png".format(title, titleFig))
    save("EvidenceAccumulationClustering/{0}/{1}.json".format(title, title), Y.tolist())
    saveclu("EvidenceAccumulationClustering/{0}/pvis/real.clu".format(title), Y)
    return X, title, Y

def main():    
    
    option = menu()    
    while(option!=0):
        if option == 1:
            print('carregando a base de dados - IRIS')
            X, title, Y = loadIris()
            runKMeans(X, title)
        elif option == 2:
            print('carregando a base de dados - TWEETS')
            X, title, Y = loadTweets()
            runKMeans(X, title)
        elif option == 3:
            print('carregando a base de dados - IRIS')
            X, title, Y = loadIris()
            P = loadP(title)
            runEac(P, X , title, Y)
        elif option == 4:
            print('carregando a base de dados - TWEETS')            
            X, title, Y = loadTweets()
            P = loadP(title)
            runEac(P, X ,title, Y, [2,3,4,5,6,7,8,9])
        elif option == 5:
            print('carregando a base de dados - IRIS')
            X, title, Y = loadIris()
            P = runKMeans(X, title)
            runEac(P, X , title, Y)
            print('carregando a base de dados - TWEETS') 
            X, title, Y = loadTweets()
            P = runKMeans(X, title)
            runEac(P, X , title, Y, [2,3,4,5,6,7,8,9])
        elif option == 6:
            runPVis('iris', "KMeans")
            runPVis('iris', "single")
        elif option == 7:
            runPVis('tweets', "KMeans")
            runPVis('tweets', "single")
        option =  menu()
    print("Finished")

if __name__ == '__main__':
    main()
