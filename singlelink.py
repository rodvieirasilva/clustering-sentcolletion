from scipy.cluster.hierarchy import linkage, fcluster

class SingleLink:
    singleLinkag = None
    labels_ = None
    k = None
    
    def __init__(self, k):
        self.k =  k

    def fit(self, data):
        self.singleLinkag = linkage(data, 'single')        
        self.labels_ = fcluster(self.singleLinkag, self.k, criterion='maxclust')

# def main():
#     Z = linkage(X, 'single')
#     fig = plt.figure(figsize=(25, 10))
#     dn = dendrogram(Z)
#     plt.show()

if __name__ == '__main__':
    
    main()


