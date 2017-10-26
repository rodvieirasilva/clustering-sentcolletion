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
        self.sorted_frequency = sorted(self.frequency.items(), key=lambda x:x[1], reverse=True)
        

    def plot(self):
        i=0
        values = list(self.frequency.values())
        values.sort(reverse=True)
        values = values[0:999]
        values = [[i,item] for i, item in enumerate(values)]
        plt.plot([row[0] for row in values], [row[1] for row in values])
        plt.ylabel('Frequência do Termo')
        plt.xlabel('Termos')
        plt.title('Curva de Zipf')
        plt.xticks([])
        plt.axis([0, 999, 0, 100])
        plt.yticks([1,2,3,4,5,6,7,100])
        plt.savefig('zipfcurve.png')
        plt.show()
        

    def lunh(self, percent):        
        size = len(self.sorted_frequency)
        sizePrint = int(size * (percent / 100.0)) 
        for i in range(0, sizePrint):            
            print(self.sorted_frequency[i])
            print(self.sorted_frequency[size - i - 1])
    
    def uniques(self):              
        result = [value for key,value in self.frequency.items() if value == 1]
        #print(result)
        print('Size: {}'.format(len(result)))
        return result
            

def main():
    with open('allwordsProcessed.json') as json_data:
        allwords = json.load(json_data)
        zipf = ZipfCurve(allwords)
        zipf.lunh(0.1)
        zipf.plot()   
        zipf.uniques()     
    print("Finish")

if __name__ == '__main__':
    main()