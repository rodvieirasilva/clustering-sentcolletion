from sklearn.decomposition import PCA
import csv
import json
from textdict import TextDict
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys

class PlotPCA:
    
    def __init__(self, filename=None, data=None):
        if(filename != None):
            with open(filename) as json_data:
                self.bagofwords = json.load(json_data)
        else:
            self.bagofwords = data
        self.pca = PCA(n_components=2, svd_solver='full')
        self.pca.fit(self.bagofwords)

    def gencolors(self, n):
        return [colorsys.hsv_to_rgb(x*1.0/n, 0.5, 0.7) for x in range(n)]

    def plotpca(self, title, words, y_pred, classnames):
        plt.figure(num=title)
        plt.title(title)
        XPCA = self.pca.transform(words)
        basec = self.gencolors(len(classnames))
        colors = [basec[ (c % len(basec)) ] for c in y_pred]
        plt.scatter(XPCA[:, 0], XPCA[:, 1], s=30, color=colors)
        if len(classnames) <= 6:
            patches = [mpatches.Patch(color=basec[ (i % len(basec)) ], label=classn) for i,classn in enumerate(classnames)]
            plt.legend(handles=patches)
            plt.legend()

    def show(self):
        plt.show()

    def savefig(self, filename):
        plt.savefig(filename)
        plt.clf()

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
    pca.savefig('pca/pca_class.png')
    pca.plotpca("pca_theme", pca.bagofwords, Ytheme, ["game", "movie", "smartphone"])
    pca.savefig('pca/pca_theme.png')
    pca.plotpca("pca_classtheme", pca.bagofwords, Yclasstheme , ["negative_game","positive_game","negative_movie","positive_movie", "negative_smartphone", "positive_smartphone"])
    pca.savefig('pca/pca_classtheme.png')
    print("Finish")

if __name__ == '__main__':
    main()