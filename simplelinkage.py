from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance  import pdist

class SimpleLinkage:
    simpleLinkage = None
    labels_ = None
    distance = None
    k = None
    title = None
    name = None
    alg = None
    
    def __init__(self, distance, k, alg):
        self.k =  k
        self.distance = distance
        self.alg = alg

    def fit(self, data):        
        self.simpleLinkage = linkage(self.distance, self.alg)        
        print("Teste: {0}".format(self.distance.shape))
        self.labels_ = fcluster(self.simpleLinkage, self.k, criterion='maxclust')
    
    def dendrogram(self):
        Z = self.simpleLinkage
        if Z is None:
            Z = linkage(self.distance, self.alg)
        plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, orientation="top", truncate_mode='lastp', p=20, leaf_font_size=20)
        plt.savefig('{0}/{1}-dendrogram-20.png'.format(self.name, self.title))

def main():
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)    
    distance = pdist(bagofwords, metric='euclidean')
    wardLink = SimpleLinkage(distance, 0, 'ward')
    wardLink.dendrogram()    
    singleLink = SimpleLinkage(distance, 0, 'single')
    singleLink.dendrogram()  
    print("Finished")

if __name__ == '__main__':    
    main()


