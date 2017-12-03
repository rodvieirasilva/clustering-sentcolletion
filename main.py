"""
-- Sent Collection v.1 para análise de agrupamento --
--                  Grupo 1                        --
--Marciele de Menezes Bittencourt                  --
--Rodrigo Vieira da Silva                          --
--Washington Rodrigo Dias da Silva                 --
-----------------------------------------------------
"""

from algoritmos import main as algmenu
from preprocess import main as premain
from eac import main as eacmain

def menu():
    print("-- Sent Collection v.1 para análise de agrupamento --")
    print("--                  Grupo 1                        --")
    print("--Marciele de Menezes Bittencourt                  --")
    print("--Rodrigo Vieira da Silva                          --")
    print("--Washington Rodrigo Dias da Silva                 --")
    print("-----------------------------------------------------")
    print('Escolha uma das opções: ')
    print('1 - Aplicar Pré-processamento (Atividade 1)')
    print('2 - Aplicar e Analisar algoritmos (Atividade 2)')
    print('3 - Aplicação Evidence Accumulation Clustering (Atividade 3)')
    print('0 - Sair')
    return input('Opção: ')

def main():
    opcao = menu()
    while(opcao!="0"):
        if opcao=="1":
            premain()
        elif opcao=="2":
            algmenu()
        elif opcao=="3":
            eacmain()
        elif(opcao!="0"):
            print("Opção Inválida!")
        opcao = menu()    

if __name__ == '__main__':
    main()