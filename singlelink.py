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
    print("Finished")

if __name__ == '__main__':    
    main()


