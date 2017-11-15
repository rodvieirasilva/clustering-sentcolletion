"""
-- Sent Collection v.1 para análise de agrupamento --
--                  Grupo 1                        --
--Marciele de Menezes Bittencourt                  --
--Rodrigo Vieira da Silva                          --
--Washington Rodrigo Dias da Silva                 --
-----------------------------------------------------
"""
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

def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def inputInt(msg, maxV=None):
    i = input(msg)
    i, success = intTryParse(i)
    success = success and ((maxV is None) or (i <= maxV and i >=0))
    while(not success):
        print("Invalid Value!")
        i = input(msg)        
        i, success = intTryParse(i)   
        success = success and ((maxV is None) or (i <= maxV and i >=0)) 
    return i

def printGreen(prt): print("\033[92m {}\033[00m" .format(prt))      

def printRed(prt): print("\033[92m {}\033[00m" .format(prt)) 