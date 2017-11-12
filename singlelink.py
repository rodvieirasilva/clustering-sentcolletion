from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance  import pdist

class SingleLink:
    singleLinkage = None
    labels_ = None
    distance = None
    k = None
    
    def __init__(self, distance, k):
        self.k =  k
        self.distance = distance

    def fit(self, data):        
        self.singleLinkage = linkage(self.distance, 'single')        
        self.labels_ = fcluster(self.singleLinkage, self.k, criterion='maxclust')
    
    def dendrogram(self):
        Z = linkage(self.distance, 'single')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, orientation="top", truncate_mode='lastp', p=20, leaf_font_size=20)

def main():
    with open('basesjson/sklearn_bagofwords.json') as json_data:
        bagofwords = json.load(json_data)    
    distance = pdist(bagofwords, metric='euclidean')
    singleLink = SingleLink(distance, 0)
    singleLink.dendrogram()
    plt.savefig('SingleLink/dendrogram-20.png')
    print("Finished")

if __name__ == '__main__':    
    main()


