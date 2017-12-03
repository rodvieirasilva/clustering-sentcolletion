from simplelinkage import SimpleLinkage
from pca import PlotPCA, plotPcaTweets
from scipy.spatial.distance  import pdist
import json
import math
from util import save, inputInt
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, datasets
from random import randrange
import glob

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

def runEac(P, X, prefix):    
    print('Rodando EvidenceAccumulationClustering - {0}'.format(prefix))
    name = "EvidenceAccumulationClustering"
    eca = EvidenceAccumulationCLustering(P=P, X=X, distanceNP=None)    
    kneighbors = len(X)
    k=3
    alg='single'
    title = "{0}/eac-{1}Link-neighbors{2}-k{3}".format(prefix, alg, kneighbors, k)
    eca.step1(kneighbors=kneighbors)  
    y_pred = eca.step2(kneighbors=kneighbors, k=k, title=title, alg=alg)
    pca = PlotPCA(filename=None, data=X)
    pca.plotpca(title=title, words=X, y_pred=y_pred, classnames=set(y_pred))    
    pca.savefig("{0}/{1}_pca.png".format(name, title))
    save("{0}/{1}.json".format(name, title), y_pred.tolist()) 

def runKMeans(X, prefix):
    print('Executando 30 KMeans, k =[2,20] - {0}'.format(prefix))
    P = []
    folder = "EvidenceAccumulationClustering"
    for i in range(0,30):        
        k = randrange(2, 21)
        print('Executando {0} - KMeans, k={1}'.format(i, k))
        model = cluster.KMeans(n_clusters=k, max_iter=1000)        
        model.title = "{0}/KMeans/{1:02.0f}_KMeans_k{2:02.0f}".format(prefix, i, k)
        model.name = "KMeans"
        model.fit(X)
        p = model.labels_
        P.append(p)
        pca = PlotPCA(filename=None, data=X)
        titleFig = "{0} - {1} - KMeans k={2}".format(prefix, i, k)
        pca.plotpca(title=titleFig, words=X, y_pred=p, classnames=set(p))    
        pca.savefig("{0}/{1}_pca.png".format(folder, model.title))
        save("{0}/{1}.json".format(folder, model.title), p.tolist()) 
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
    print('0 - Sair')
    return inputInt('Opção: ', 5)

def loadTweets():
    X = []
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        X = json.load(json_data)
    plotPcaTweets('EvidenceAccumulationClustering/tweets')
    return X, 'tweets'

def loadIris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Mostrando os dados da base IRIS com o PCA
    pca = PlotPCA(filename=None, data=X)
    title = "iris"
    titleFig = "PCA Dataset IRIS"
    pca.plotpca(title=titleFig, words=X, y_pred=y, classnames=iris.target_names) 
    pca.savefig("EvidenceAccumulationClustering/{0}/{1}_pca.png".format(title, titleFig))
    save("EvidenceAccumulationClustering/{0}/{1}.json".format(title, title), y.tolist())    
    return X, title

def main():
    
    option = menu()    
    while(option!=0):
        if option == 1:
            print('carregando a base de dados - IRIS')
            X, title = loadIris()
            runKMeans(X, title)
        elif option == 2:
            print('carregando a base de dados - TWEETS')
            X, title = loadTweets()
            runKMeans(X, title)
        elif option == 3:
            print('carregando a base de dados - IRIS')
            X, title = loadIris()
            P = loadP(title)
            runEac(P, X , title)
        elif option == 4:
            print('carregando a base de dados - TWEETS')            
            X, title = loadTweets()
            P = loadP(title)
            runEac(P, X ,title)
        elif option == 5:
            print('carregando a base de dados - IRIS')
            X, title = loadIris()
            P = runKMeans(X, title)
            runEac(P, X , title)
            print('carregando a base de dados - TWEETS') 
            X, title = loadTweets()
            P = runKMeans(X, title)
            runEac(P, X , title)
        option =  menu()
    print("Finished")

if __name__ == '__main__':
    main()
