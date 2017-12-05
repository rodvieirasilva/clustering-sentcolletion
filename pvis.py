import requests


class PVis():

    def __init__(self):        
        pass
    
    def post(self, ordmode, nrgrp, filenameRef, filenamesPartitions, filenameOut):
        files = {}
        files['ref'] = open(filenameRef, 'rb')
        for i, f in enumerate(filenamesPartitions):
             files['partitions[{0}]'.format(i)] = open(f,'rb')

        response = requests.post('http://lasid.sor.ufscar.br/pvisproject/pvi_project.php?action=run',
                                files=files,
                                data={
                                    'ordmode':ordmode,
                                    'nrgrp':nrgrp
                                })
        with open(filenameOut, 'wb') as f:
            f.write(response.content)

def main():
    pvis = PVis()
    pvis.post('1', '2', 'EvidenceAccumulationClustering\\tweets\pvis\\realClass.clu', 
        ['EvidenceAccumulationClustering\\tweets\pvis\\partitions\\00-KMeans-k16.clu',
         'EvidenceAccumulationClustering\\tweets\pvis\\partitions\\01-KMeans-k05.clu'], 'teste.pdf')

if __name__ == '__main__':
    main()


# response = requests.post('http://lasid.sor.ufscar.br/pvisproject/pvi_project.php?action=run',
#                          files={
#                              'ref': open('EvidenceAccumulationClustering\\tweets\pvis\\realClass.clu','rb'),
#                              'partitions': open('EvidenceAccumulationClustering\\tweets\pvis\\partitions\\00-KMeans-k16.clu','rb')                             
#                              },
#                         data={
#                             'ordmode':'1',
#                             'nrgrp':'1'

#                         })
# with open('teste.pdf', 'wb') as f:
#     f.write(response.content)