from algoritmos import main as algmain
from preprocess import main as premain

def menu():
    print("-- Sent Collection v.1 para análise de agrupamento --")
    print("--                  Grupo 1                        --")
    print("--Danilo Kaory Yamauti                             --")
    print("--Marciele de Menezes Bittencourt                  --")
    print("--Rodrigo Vieira da Silva                          --")
    print("--Washington Rodrigo Dias da Silva                 --")
    print("-----------------------------------------------------")
    print('Escolha uma das opções: ')
    print('1 - Aplicar Pré-processamento')
    print('2 - Aplicar e Analisar algoritmos')
    print('0 - Sair')
    return int(input('Opção: '))

def main():
    opcao = menu()
    while(opcao!=0):
        if opcao==1:
            premain()
        elif opcao==2:
            algmain()
        elif(opcao<0 or opcao>2):
            print("Opção Inválida!")
        opcao = menu()    

if __name__ == '__main__':
    main()