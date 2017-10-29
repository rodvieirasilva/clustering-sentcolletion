from sklearn.decomposition import PCA
import csv
import json
from textdict import TextDict
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PlotPCA:
    basec = ["#d2006d",
            "#0ea625",
            "#d972fa",
            "#d4ca3b",
            "#015594",
            "#ff6954",
            "#46ddbb",
            "#ff7bd1",
            "#844500",
            "#99b0ff"]
    
    def __init__(self, filename, data):
        if(filename != None):
            with open(filename) as json_data:
                self.bagofwords = json.load(json_data)
        else:
            self.bagofwords = data
        self.pca = PCA(n_components=2, svd_solver='full')
        self.pca.fit(self.bagofwords)

    def plotpca(self, title, words, y_pred, classnames):
        plt.figure(num=title)
        plt.title(title)
        XPCA = self.pca.transform(words)
        colors = [self.basec[c] for c in y_pred]
        patches = [mpatches.Patch(color=self.basec[i], label=classn) for i,classn in enumerate(classnames)]
        plt.legend(handles=patches)
        plt.scatter(XPCA[:, 0], XPCA[:, 1], s=30, color=colors)        
        plt.legend()

    def show(self):
        plt.show()

def main():
   
    pca = PlotPCA("sklearn_bagofwords.json")

    with open('complete.json') as json_data:
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