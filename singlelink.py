from scipy.cluster.hierarchy import linkage, fcluster

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

def main():
     pass
#     Z = linkage(X, 'single')
#     fig = plt.figure(figsize=(25, 10))
#     dn = dendrogram(Z)
#     plt.show()

if __name__ == '__main__':    
    main()


