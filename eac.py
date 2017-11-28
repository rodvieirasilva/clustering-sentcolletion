from simplelinkage import SimpleLinkage
from pca import PlotPCA
from scipy.spatial.distance  import pdist
import json
import math
from util import save
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, datasets

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
        if (kneighbors is None) or (kneighbors <= 0):
            kneighbors = len(self.distanceNP) - 1
        C = [[0 for i in range(kneighbors)] for j in range(len(self.distanceNP))]
        for p in self.P:            
            for i,x in enumerate(self.distanceNP):
                d = sorted(range(len(x)),key=lambda idx:x[idx])
                cK = 1
                for j in d:                    
                    if j != i:
                        if p[i] == p[j]:                
                            C[i][cK-1] += 1/len(self.P)

                        cK += 1

                    if cK > kneighbors:
                        break        
        self.C = C



        # Calculo da Co-Associação sem verificação do vizinho C(i,j) = nij/N
        V = [[0 for i in range(len(self.P[0]))] for j in range(len(self.P[0]))]
        for p in self.P:            
            for i,x in enumerate(p):
                cK = 1
                for j, l in enumerate(p):
                    if j != i:
                        if p[i] == p[j]:                
                            V[i][cK-1] += 1/len(self.P)

                        cK += 1

                    if cK > kneighbors:
                        break      
        self.C = V



        return self.C

    def step2(self, kneighbors, k, alg='single', title=None):
        distance = pdist(self.C, metric='euclidean')
        singleLink = SimpleLinkage(distance=distance, k=k, alg=alg)
        singleLink.name = "EvidenceAccumulationClustering"
        if(title is None):
            title = "eac-{0}Linkage-neighbors{1}".format(kneighbors, alg)
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

def main():
    print('Escolha uma das opções: ')
    print('1 - Dados da base IRIS')
    print('2 - Partições KMeans da base de Tweets')
    print('0 - Sair')
    option = input('Opção: ')

    if option == "1" or option == "2":
        print('carregando a base de dados')
        name = "EvidenceAccumulationClustering"

        if option == "1":
            # Carregando a Base de Dados Iris
            iris = datasets.load_iris()
            bagofwords = iris.data
            y = iris.target
            
            # Normalizando a base de dados
            minimum = 0
            maximum = 1
            features = len(bagofwords[0,:])
            for i in range(0, features):
                smaller = min(bagofwords[:,i])
                bigger = max(bagofwords[:,i])
                bagofwords[:,i] = (minimum + ((bagofwords[:,i] - smaller)/(bigger-smaller))*(maximum-minimum)) 
                
            # Mostrando os dados da base IRIS com o PCA
            pca = PlotPCA(filename=None, data=bagofwords)
            title = "datasetIRIS"
            titleFig = "EvidenceAccumulationClustering\datasetIRIS"
            pca.plotpca(title=titleFig, words=bagofwords, y_pred=y, classnames=set(y))    
            pca.savefig("{0}/{1}_pca.png".format(name, title))
            save("{0}/{1}.json".format(name, title), y.tolist())    

            P = []
            for k in range(2,10):
                model = cluster.KMeans(n_clusters=k, max_iter=1000);
                model.fit(bagofwords);
                P.append(model.labels_)

        elif option == "2":
            with open('basesjson/sklearn_bagofwords.json') as json_data:
                bagofwords = json.load(json_data)

            PBases = ['KMeans/KMeans_k2.json', 'KMeans/KMeans_k3.json', 'KMeans/KMeans_k4.json', 'KMeans/KMeans_k5.json'
                     ,'KMeans/KMeans_k6.json', 'KMeans/KMeans_k7.json', 'KMeans/KMeans_k8.json', 'KMeans/KMeans_k9.json'
                     , 'KMeans/KMeans_k10.json'

                # , 'KMeans/KMeans_k6.json', 'DBSCAN/DBSCAN_eps_1_min_samples_40.json', 
                # 'WardLink/WardLink_k2.json', 
                # 'WardLink/WardLink_k10.json'
                
                    ]
            P = []
            for base in PBases:
                with open(base) as json_data:
                    P.append(json.load(json_data))

        eca = EvidenceAccumulationCLustering(P=P, X=bagofwords, distanceNP=None)    
        kneighbors = len(bagofwords)
        k=3
        alg='single'
        # title = "KMeans2-5_DBSCAN_eps_1_min_samples_40_WardLink_k2_WardLink_k10_eac-neighbors{0}-k{1}".format(kneighbors, k)
        title = "KMeans2-10_eac-{0}Linkage-neighbors{1}-k{2}".format(alg, kneighbors, k)
        eca.step1(kneighbors=kneighbors)  
        y_pred = eca.step2(kneighbors=kneighbors, k=k, title=title, alg=alg)
        pca = PlotPCA(filename=None, data=bagofwords)
        # titleFig = "KMeans2-5_DBSCAN_eps_1_min_samples_40_\nWardLink_k2_WardLink_k10_eac-neighbors{0}-k{1}".format(kneighbors, k)
        titleFig = "KMeans2-10_eac-{0}Linkage-neighbors{1}-k{2}".format(alg, kneighbors, k)
        pca.plotpca(title=titleFig, words=bagofwords, y_pred=y_pred, classnames=set(y_pred))    
        pca.savefig("{0}/{1}_pca.png".format(name, title))
        save("{0}/{1}.json".format(name, title), y_pred.tolist())    

    print("Finished")

if __name__ == '__main__':
    main()
