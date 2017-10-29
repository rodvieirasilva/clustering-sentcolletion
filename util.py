import csv
import json
import os

def save(filename, data):
    mkdir(filename)
    with open(filename, 'w') as outfile:
            json.dump(data, outfile)


def savecsv(filename, header, data):
    mkdir(filename)
    with open(filename, 'w', encoding='utf-8') as file:
        for cell in header:
            file.write('"{}"'.format(cell))
            file.write(';')
        file.write('\n')
        for row in data:
            for cell in row:
                file.write(str(cell))
                file.write(';')
            file.write('\n')

def mkdir(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)