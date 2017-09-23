import matplotlib.pyplot as plt
import json
import operator

class ZipfCurve:
    def __init__(self, words):
        self.words = words
        self.calc()

    def calc(self):
        self.frequency = {}
        for word in self.words:
            count = self.frequency.get(word,0)
            self.frequency[word] = count + 1

    def plot(self):
        i=0
        values = list(self.frequency.values())
        values.sort(reverse=True)        
        values = [[i,item] for i, item in enumerate(values)]
        plt.plot([row[0] for row in values], [row[1] for row in values])
        plt.show()
        
    def lunh(self, percent):
        sorted_frequency = sorted(self.frequency.items(), key=lambda x:x[1], reverse=True)
        size = len(sorted_frequency)
        sizePrint = int(size * (percent / 100.0)) 
        for i in range(0, sizePrint):            
            print(sorted_frequency[i])
            print(sorted_frequency[size - i - 1])

def main():
    with open('allwords.json') as json_data:
        allwords = json.load(json_data)
        zipf = ZipfCurve(allwords)
        zipf.lunh(0.1)
        zipf.plot()        
    print("Finish")

if __name__ == '__main__':
    main()