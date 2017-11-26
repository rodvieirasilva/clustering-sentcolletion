from simplelinkage import SimpleLinkage
from pca import PlotPCA
from scipy.spatial.distance  import pdist
import json
import math
from util import save
from sklearn.neighbors import kneighbors_graph

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

    def step1(self, kneighbors):
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
        return C

    def step2(self, kneighbors, alg='single', title=None):
        distance = pdist(self.C, metric='euclidean')
        singleLink = SimpleLinkage(distance=distance, k=10, alg=alg)
        singleLink.name = "EvidenceAccumulationClustering"
        if(title is None):
            title = "eac-k{0}".format(kneighbors)
        singleLink.title = title
        singleLink.fit(self.C)
        singleLink.dendrogram()
        return singleLink.labels_

    def binomial(self, x, y):
        
        if y == x:
            return 1
        elif y == 1:         # see georg's comment
            return x
        elif y > x:          # will be executed only if y != 1 and y != x
            return 0
        else:                # will be executed only if y != 1 and y != x and x <= y
            a = math.factorial(x)
            b = math.factorial(y)
            c = math.factorial(x-y)  # that appears to be useful to get the correct result
            div = a // (b * c)
            return div

    def calcM(self, k, n):
        m = 0
        for l in range(1, k+1):
            m += self.binomial(k, l) * ((-1) ** (k-l)) * (l**n)
        return (1.0/math.factorial(k)) * m

def main():
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)

    PBases = ['KMeans/KMeans_k2.json', 'KMeans/KMeans_k3.json', 'KMeans/KMeans_k4.json', 'KMeans/KMeans_k5.json'
                , 'KMeans/KMeans_k6.json', 'DBSCAN/DBSCAN_eps_1_min_samples_40.json', 
                'WardLink/WardLink_k2.json', 
                'WardLink/WardLink_k10.json']
    P = []
    for base in PBases:
        with open(base) as json_data:
            P.append(json.load(json_data))

    eca = EvidenceAccumulationCLustering(P=P, X=bagofwords, distanceNP=None)    
    kneighbors = 2771
    name = "EvidenceAccumulationClustering"
    title = "KMeans2-5_DBSCAN_eps_1_min_samples_40_WardLink_k2_WardLink_k10_eac-k{0}".format(kneighbors)
    eca.step1(kneighbors=kneighbors)  
    y_pred = eca.step2(kneighbors=kneighbors, title=title)
    pca = PlotPCA(filename=None, data=bagofwords)
    titleFig = "KMeans2-5_DBSCAN_eps_1_min_samples_40\nWardLink_k2_WardLink_k10_eac-k{0}".format(kneighbors)
    pca.plotpca(title=titleFig, words=bagofwords, y_pred=y_pred, classnames=set(y_pred))    
    pca.savefig("{0}/{1}_pca.png".format(name, title))
    save("{0}/{1}.json".format(name, title), y_pred.tolist())    
    print("Finished")

if __name__ == '__main__':
    main()
