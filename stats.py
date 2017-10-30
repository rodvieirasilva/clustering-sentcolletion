from sklearn import metrics
import matplotlib.pyplot as plt

class StatList(list):

    def __init__(self, complete):
        self.theme = [item['theme'] for item in complete]  
        self.classe = [item['class'] for item in complete]  
        self.themeclasse = ['{0}_{1}'.format(item['theme'], item['class']) for item in complete]

    def add(self, k):
        stat = Stat(k, self.theme, self.classe, self.themeclasse)
        self.append(stat)
        return stat
    
    def plot(self):
        plots = []
        plots.extend(
        [
            {
               "ylabel": "Creation Time",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
               "ylabel": "Execution Time",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "ylabel": "Total Time",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "ylabel": "Indice Rand ajustado com relacao a Classe",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "ylabel": "Indice Rand ajustado com relacao ao Tema",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "ylabel": "Indice Rand ajustado com relacao ao Tema+Classe",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },{
                "ylabel": "Indice Silhueta com relacao a base inicial",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "ylabel": "Indice Variancia Intra-cluster",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            }])

        for stat in self:
            plots[0]["dataX"].append(stat.k)
            plots[0]["dataY"].append(stat.creationTime)
            plots[1]["dataX"].append(stat.k)
            plots[1]["dataY"].append(stat.executionTime)
            plots[2]["dataX"].append(stat.k)
            plots[2]["dataY"].append(stat.totalTime)
            plots[3]["dataX"].append(stat.k)
            plots[3]["dataY"].append(stat.adjusted_rand_score_classe)
            plots[4]["dataX"].append(stat.k)
            plots[4]["dataY"].append(stat.adjusted_rand_score_theme)
            plots[5]["dataX"].append(stat.k)
            plots[5]["dataY"].append(stat.adjusted_rand_score_themeclasse)
            plots[6]["dataX"].append(stat.k)
            plots[6]["dataY"].append(stat.silhouette_score_)
            plots[7]["dataX"].append(stat.k)
            plots[7]["dataY"].append(stat.index_intracluster_variance_)
        for p in plots:
            plt.figure()
            plt.ylabel(p["ylabel"])
            plt.xlabel(p["xlabel"])
            plt.plot(p["dataX"], p["dataY"])
            print(p["dataX"])
            print(p)
        #times
        plt.figure()
        plt.ylabel("Time")
        plt.xlabel("k")
        for plot in plots[0:3]:
            plt.plot(plot["dataX"], plot["dataY"], label=plot["ylabel"])

class Stat:
    beginCreationTime = None
    endCreationTime = None
    creationTime = None
    beginExecutionTime = None
    endExecutionTime = None    
    executionTime = None
    totalTime=None
    adjusted_rand_score_classe = None
    adjusted_rand_score_theme = None
    adjusted_rand_score_themeclasse = None
    silhouette_score_ = None
    index_intracluster_variance_ = None
    theme = None
    classe = None
    themeclasse = None
    k = None

    def __init__(self, k, theme, classe, themeclasse):
        self.theme = theme
        self.classe = classe
        self.themeclasse = themeclasse
        self.k = k

    def calc(self, data, particao):
        self.creationTime = self.endCreationTime - self.beginCreationTime
        self.executionTime = self.endExecutionTime - self.beginExecutionTime
        self.totalTime= self.creationTime + self.executionTime
        self.adjusted_rand_score_classe = metrics.adjusted_rand_score(self.classe, particao)
        self.adjusted_rand_score_theme = metrics.adjusted_rand_score(self.theme , particao)
        self.adjusted_rand_score_themeclasse = metrics.adjusted_rand_score(self.themeclasse, particao)
        try:
            silhouette_score_ = metrics.silhouette_score(data, particao)
        except:
            pass
        
        try:
            index_intracluster_variance_ = self.index_intracluster_variance(data, particao, model.cluster_centers_)
        except:
            pass

    def index_intracluster_variance(data, labels, centroids):
        soma = 0
        n = len(labels)
        for idx, item in enumerate(centroids):
            soma += sum(metrics.pairwise.pairwise_distances(np.array(data)[labels == idx], np.array(item).reshape(1, -1), metric='euclidean'))
        
        return (int(soma)/n)**(1/2)

    def toString(self):
        strstats = '"Tempo Criacao Modelo";"%.2fs"\n' % self.creationTime
        strstats += '"Tempo Execucao Modelo";"%.2fs"\n' % self.executionTime
        strstats += '"Tempo Total Modelo";"%.2fs"\n' % self.totalTime
        strstats += '"Indice Rand ajustado com relacao a Classe";"{0}"\n'.format(self.adjusted_rand_score_classe)
        strstats += '"Indice Rand ajustado com relacao ao Tema";"{0}"\n'.format(self.adjusted_rand_score_theme)
        strstats += '"Indice Rand ajustado com relacao ao Tema+Classe";"{0}"\n'.format(self.adjusted_rand_score_themeclasse)

        try:
            strstats += '"Indice Silhueta com relacao a base inicial";"{0}"\n'.format(self.silhouette_score_)
        except:
            pass
        
        try:
            strstats += '"Indice Variancia Intra-cluster";"{0}"\n'.format(self.index_intracluster_variance_)
        except:
            pass
        return strstats
    def toStringChart(self):
        strstats = 'Tempo Criacao Modelo: %.2fs\n' % self.creationTime
        strstats += 'Tempo Execucao Modelo: %.2fs\n' % self.executionTime
        strstats += 'Tempo Total Modelo: %.2fs\n' % self.totalTime
        strstats += 'Indice Rand ajustado com relacao a Classe: {0}\n'.format(self.adjusted_rand_score_classe)
        strstats += 'Indice Rand ajustado com relacao ao Tema: {0}\n'.format(self.adjusted_rand_score_theme)
        strstats += 'Indice Rand ajustado com relacao ao Tema+Classe: {0}\n'.format(self.adjusted_rand_score_themeclasse)
        try:
            strstats += "Indice Silhueta com relacao a base inicial: {0}\n".format(self.silhouette_score_)
        except:
            pass
        
        try:
            strstats += "Indice Variancia Intra-cluster: {0}\n".format(self.index_intracluster_variance_)
        except:
            pass
        return strstats

def main():   
    print("Finished")

if __name__ == '__main__':
    main()