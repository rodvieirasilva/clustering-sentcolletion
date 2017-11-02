from sklearn.decomposition import PCA
import csv
import json
from textdict import TextDict
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class PlotPCA:

        # ["#d2006d",
    #         "#0ea625",
    #         "#d972fa",
    #         "#d4ca3b",
    #         "#015594",
    #         "#ff6954",
    #         "#46ddbb",
    #         "#ff7bd1",
    #         "#844500",
    #         "#99b0ff"]
    basec = ["#2d9590",
            "#da4c23",
            "#4f62d0",
            "#68c243",
            "#8d47b7",
            "#54c76a",
            "#9c6fe8",
            "#abbc35",
            "#db74dc",
            "#4b9b30",
            "#ab3a9b",
            "#53cc91",
            "#e6437d",
            "#369a5b",
            "#d8519e",
            "#3f7821",
            "#8f85e1",
            "#e4a73b",
            "#4481d2",
            "#e17c2e",
            "#4cbee0",
            "#d04133",
            "#5dcebc",
            "#db3750",
            "#48a888",
            "#ae3364",
            "#3e7b39",
            "#a775bb",
            "#7a9533",
            "#605a9c",
            "#beaa40",
            "#6798d1",
            "#ae7b22",
            "#da93cd",
            "#6ead72",
            "#b03939",
            "#2c744d",
            "#e9735b",
            "#a9c277",
            "#984f77",
            "#8f9353",
            "#de6c77",
            "#676722",
            "#e5869c",
            "#caa569",
            "#a14d55",
            "#e3946e",
            "#aa4e1e",
            "#946332",
            "#ac5b49"]
    
    def __init__(self, filename=None, data=None):
        if(filename != None):
            with open(filename) as json_data:
                self.bagofwords = json.load(json_data)
        else:
            self.bagofwords = data
        self.pca = PCA(n_components=2, svd_solver='full')
        self.pca.fit(self.bagofwords)

    def plotpca(self, title, words, y_pred, classnames):
        plt.figure(num=title)
        #plt.gca().set_position((.25, .32, .57, .57))
        plt.title(title)
        XPCA = self.pca.transform(words)
        colors = [self.basec[ (c % len(self.basec)) ] for c in y_pred]
        patches = [mpatches.Patch(color=self.basec[ (i % len(self.basec)) ], label=classn) for i,classn in enumerate(classnames)]
        #plt.legend(handles=patches)
        plt.scatter(XPCA[:, 0], XPCA[:, 1], s=30, color=colors)        
        #plt.legend()

    def show(self):
        plt.show()

    def savefig(self, filename):
        #plt.rcParams["figure.figsize"] = (20,3)
        plt.savefig(filename)

def main():
   
    pca = PlotPCA(filename="basesjson/sklearn_bagofwords.json")

    with open('basesjson/complete.json') as json_data:
        complete = json.load(json_data)

    #{"theme": "game", "product": "archeage", "tweet": "@tweetmee lol atleast i have csgo and archeage to keep me atleast a tad bit busy xp ahah", "class": "1"}
    dictclass = {"-1":0,"1":1}
    dicttheme = {"game":0,"movie":1, "smartphone":2}
    Ys = [ [  dictclass[item["class"]], dicttheme[item["theme"]]  ] for item in complete]
    Yclazz = [item[0] for item in Ys]
    Ytheme = [item[1] for item in Ys]
    dictthemeclass = {0:{0:0,1:1},1:{0:2, 1:3}, 2:{0:4, 1:5}}
    Yclasstheme = [dictthemeclass[item[1]][item[0]] for item in Ys]    
    pca.plotpca("pca_class", pca.bagofwords, Yclazz , ["negative", "positive"])
    pca.plotpca("pca_theme", pca.bagofwords, Ytheme, ["game", "movie", "smartphone"])
    pca.plotpca("pca_classtheme", pca.bagofwords, Yclasstheme , ["negative_game","positive_game","negative_movie","positive_movie", "negative_smartphone", "positive_smartphone"])
    pca.show()
    print("Finish")

if __name__ == '__main__':
    main()