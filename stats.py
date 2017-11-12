from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph

class StatList(list):
    name=None
    data=None
    A=None
    prefix=None
    def __init__(self, complete, data):
        self.theme = [item['theme'] for item in complete]  
        self.classe = [item['class'] for item in complete]  
        self.themeclasse = ['{0}_{1}'.format(item['theme'], item['class']) for item in complete]
        self.data = data
        self.A = kneighbors_graph(data, len(data), mode='distance', include_self=True)
        self.A = self.A.toarray()

    def add(self, k):
        stat = Stat(k, self.theme, self.classe, self.themeclasse, self.data, self.A)
        self.append(stat)
        return stat
    
    def plot(self):
        plots = []
        plots.extend(
        [
            {
               "name": "creation_time",
               "title": "Creation Time {0}".format(self.name),
               "label": "Creation Time",
               "ylabel": "Time(s)",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
               "name": "execution_time",
               "title": "Execution Time {0}".format(self.name),
               "label": "Execution Time",
               "ylabel": "Time(s)",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "total_time",
                "title": "Total Time {0}".format(self.name),
                "label": "Total Time",
                "ylabel": "Time(s)",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "ind_rand_adj_classe",
                "title": "Indice Rand ajustado com relacao a Classe - {0}".format(self.name),
                "label": "Indice Rand ajustado com relacao a Classe",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "ind_rand_adj_theme",
                "title": "Indice Rand ajustado com relacao ao Tema - {0}".format(self.name),
                "label": "Indice Rand ajustado com relacao ao Tema",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "ind_rand_adj_theme_class",
                "title": "Indice Rand ajustado com relacao ao Tema+Classe - {0}".format(self.name),
                "label": "Indice Rand ajustado com relacao ao Tema+Classe",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },{
                "name": "ind_sil",
                "title": "Indice Silhueta com relacao a base inicial - {0}".format(self.name),
                "label": "Indice Silhueta com relacao a base inicial",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "var_intra_cluster",
                "title": "Indice Variancia Intra-cluster - {0}".format(self.name),
                "label": "Indice Variancia Intra-cluster",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "index_connectivity_28",
                "title": "Índice Conectividade - 28 vizinhos - {0}".format(self.name),
                "label": "Índice Conectividade - 28 vizinhos",
                "ylabel": "Vlr. Indice",
                "xlabel": "K",
                "dataX": [],
                "dataY": []
            },
            {
                "name": "index_connectivity_10",
                "title": "Índice Conectividade - 10 vizinhos - {0}".format(self.name),
                "label": "Índice Conectividade - 10 vizinhos",
                "ylabel": "Vlr. Indice",
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
            plots[8]["dataX"].append(stat.k)
            plots[8]["dataY"].append(stat.index_connectivity_10)
            plots[9]["dataX"].append(stat.k)
            plots[9]["dataY"].append(stat.index_connectivity_28)
        for p in plots:
            plt.figure()
            plt.title(p["title"])
            plt.ylabel(p["ylabel"])
            plt.xlabel(p["xlabel"])
            plt.plot(p["dataX"], p["dataY"], marker='o', linestyle='--')
            plt.savefig("{0}/{1}_stats_{2}.png".format(self.name, self.prefix, p["name"]))
            plt.clf()
            plt.close("all")
        #times
        self.subplot("Tempos de Execução - {0}".format(self.name), "Time(s)", plots[0:3], "{0}times.png")
        #rands
        self.subplot("Índices - {0}".format(self.name), "Vlr. Índice", plots[3:6], "{0}ind_rand_adj.png")
        #connectivity
        self.subplot("Índices - {0}".format(self.name), "Vlr. Índice", plots[8:10], "{0}ind_connectivity.png")
        plt.clf()
        plt.close("all")

    def subplot(self, title, ylabel, plots, filename):
        fig, ax = plt.subplots()
        plt.title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("k")
        for p in plots:
            plt.plot(p["dataX"], p["dataY"], marker='o', linestyle='--', label=p["label"])
        #legend = ax.legend()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.16,
                box.width, box.height * 0.84])
        
        ax.legend(loc='upper center', shadow=False, bbox_to_anchor=(0.5, -0.12),ncol=1)
        plt.savefig(filename.format("{0}/{1}_stats_".format(self.name, self.prefix)))


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
    index_connectivity_ = None
    theme = None
    classe = None
    themeclasse = None
    k = None
    countClusters=None

    def __init__(self, k, theme, classe, themeclasse, data, A):
        self.theme = theme
        self.classe = classe
        self.themeclasse = themeclasse
        self.k = k
        self.data = data
        self.A = A

    def calc(self, labels):
        self.creationTime = self.endCreationTime - self.beginCreationTime
        self.executionTime = self.endExecutionTime - self.beginExecutionTime
        self.totalTime= self.creationTime + self.executionTime
        self.adjusted_rand_score_classe = metrics.adjusted_rand_score(self.classe, labels)
        self.adjusted_rand_score_theme = metrics.adjusted_rand_score(self.theme , labels)
        self.adjusted_rand_score_themeclasse = metrics.adjusted_rand_score(self.themeclasse, labels)
        self.index_connectivity_10 = self.index_connectivity(labels, 10)
        self.index_connectivity_28 = self.index_connectivity(labels, 28)
        try:
            self.silhouette_score_ = metrics.silhouette_score(self.data, labels)
        except:
            pass
        
        try:
            self.index_intracluster_variance_ = self.index_intracluster_variance(labels)
        except:
            pass
        unique, counts = np.unique(labels, return_counts=True)
        self.countClusters = dict(zip(unique, counts))

    def index_intracluster_variance(self, labels):

        # Calculo das Centroides
        clusters = np.unique(labels)
        cluster_centers = []
        for item in clusters:
            cluster_elements = np.array(self.data)[labels == item]
            cluster_center = sum(cluster_elements) / len(cluster_elements)
            cluster_centers.append(cluster_center)

        # Calculo da Variancia Intra-Cluster
        sum_distances = 0
        n = len(labels)
        for idx, item in enumerate(cluster_centers):
            sum_distances += sum(metrics.pairwise.pairwise_distances(np.array(self.data)[labels == clusters[idx]], np.array(item).reshape(1, -1), metric='euclidean'))
        
        return (float(sum_distances)/n)**(0.5)

    def index_connectivity(self, labels, kneighbors):
        result = 0
        for i,x in enumerate(self.A):
            d = sorted(range(len(x)),key=lambda idx:x[idx])
            cK = 1
            for j in d:
                if j != i:
                    if labels[i] != labels[j]:                
                        result += 1/cK

                    cK += 1

                if cK > kneighbors:
                    break
        return result  

    def toString(self):
        strstats = '"Tempo Criacao Modelo";"%.2fs"\n' % self.creationTime
        strstats += '"Tempo Execucao Modelo";"%.2fs"\n' % self.executionTime
        strstats += '"Tempo Total Modelo";"%.2fs"\n' % self.totalTime
        strstats += '"Indice Rand ajustado com relacao a Classe";"{0}"\n'.format(self.adjusted_rand_score_classe)
        strstats += '"Indice Rand ajustado com relacao ao Tema";"{0}"\n'.format(self.adjusted_rand_score_theme)
        strstats += '"Indice Rand ajustado com relacao ao Tema+Classe";"{0}"\n'.format(self.adjusted_rand_score_themeclasse)
        strstats += '"Indice Silhueta com relacao a base inicial";"{0}"\n'.format(self.silhouette_score_)
        strstats += '"Indice Variancia Intra-cluster";"{0}"\n'.format(self.index_intracluster_variance_)
        strstats += '"Indice Conectividade 10 vizinhos";"{0}"\n'.format(self.index_connectivity_10)
        strstats += '"Indice Conectividade 28 vizinhos";"{0}"\n'.format(self.index_connectivity_28)
        strstats += '"Quantidade por cluster";\n'
        for cluster, count in self.countClusters.items():
            strstats += '"{0}";'.format(cluster)
        strstats += "\n"
        for cluster, count in self.countClusters.items():
            strstats += '"{0}";'.format(count)
        return strstats

    def toStringChart(self):
        strstats = 'Tempo Criacao Modelo: %.2fs\n' % self.creationTime
        strstats += 'Tempo Execucao Modelo: %.2fs\n' % self.executionTime
        strstats += 'Tempo Total Modelo: %.2fs\n' % self.totalTime
        strstats += 'Indice Rand ajustado com relacao a Classe: {0}\n'.format(self.adjusted_rand_score_classe)
        strstats += 'Indice Rand ajustado com relacao ao Tema: {0}\n'.format(self.adjusted_rand_score_theme)
        strstats += 'Indice Rand ajustado com relacao ao Tema+Classe: {0}\n'.format(self.adjusted_rand_score_themeclasse)
        strstats += "Indice Silhueta com relacao a base inicial: {0}\n".format(self.silhouette_score_)
        strstats += "Indice Variancia Intra-cluster: {0}\n".format(self.index_intracluster_variance_)
        strstats += "Indice Conectividade 10 vizinhos: {0}\n".format(self.index_connectivity_10)
        strstats += "Indice Conectividade 28 vizinhos: {0}\n".format(self.index_connectivity_28)
        strstats += 'Quantidade por cluster:\n'
        for cluster, count in self.countClusters.items():
            strstats += "{0}:{1}, ".format(cluster, count)
        strstats = strstats[:-2]
        return strstats

def main():   
    teste1()
    print("Finished")

def teste1():
    fig, ax = plt.subplots()
    plt.title('title')
    ax.set_ylabel('ylabel')
    ax.set_xlabel("k")
    plots = [[1,2,3],[0,2,3],[2,2,3]]
    for p in plots:
        plt.plot(p, p, marker='o', linestyle='--', label=p[0])
    #legend = ax.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.16,
            box.width, box.height * 0.84])
    
    ax.legend(loc='upper center', shadow=False, bbox_to_anchor=(0.5, -0.12),ncol=1)
    plt.show()    

if __name__ == '__main__':
    main()