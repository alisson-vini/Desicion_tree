import random
from copy import deepcopy
import numpy as np
from collections import Counter

class Arvore_decisao():
    def __init__(self):
        self.valor = None # valor daquele Nó
        self.leaves = {} # lista que contem outros Nós

    def aplicar(self, dicionario:dict): # recebe como parámetro umm dict com as chaves sendo os valores da coluna e como valor a linha (como voce quer aplicar é pra ter somente uma linha e N colunas)
        if self.leaves == {}:
            return self.valor
        else:
            valor_linha = dicionario[self.valor]
            if valor_linha in self.leaves:
                return self.leaves[valor_linha].aplicar(dicionario)
            else:
                # valor desconhecido → retorna a folha mais provável
                return 'valor não encontrado'

# TREINAMENTO DA ÁRVORE
def entropia_geral(coluna_target:list):
    resultado = 0
    dict_aparicoes = Counter(coluna_target)
    quantidade_total_elementos = len(coluna_target)

    for quantidade_aparicoes in dict_aparicoes.values():
        pi = quantidade_aparicoes / quantidade_total_elementos
        resultado += pi * np.log2(pi)
    
    return -resultado

def peso(coluna_target:list, valor_target:any):
    dict_frequencia = Counter(coluna_target)
    return dict_frequencia[valor_target] / len(coluna_target)

def ganho_informacao(coluna_target:list, coluna_teste:list):
    entropia_target = entropia_geral(coluna_target)
    dict_filho_entropia = {x:[] for x in coluna_teste}

    for indice in range(len(coluna_teste)):
        dict_filho_entropia[coluna_teste[indice]].append(coluna_target[indice])
        
    frequencia = Counter(coluna_teste)
    lista_filho_peso = [x / len(coluna_teste) for x in frequencia.values()]
    lista_filho_entropia = [entropia_geral(x) for x in dict_filho_entropia.values()]

    soma = 0
    for peso, entropia in zip(lista_filho_peso, lista_filho_entropia):
        soma += peso * entropia

    return entropia_target - soma

# Função que dada uma tabela (dicionário) retorna outras duas tabelas com as mesmas colunas mas mas com os valores distirubuidos alearoriamente (como cortar a tabela em duas e aleatorizar a ordem dados) 
def separar_treino_teste(tabela_real:dict, porcentagem_separacao:int):

    if porcentagem_separacao < 0 or porcentagem_separacao > 100:
        return None

    tabela = deepcopy(tabela_real)
    treino = {}
    teste = {}

    for coluna in tabela.keys():
        treino[coluna] = []
        teste[coluna] = []

    for coluna in tabela.keys():

        i_treino = int(len(tabela[coluna]) * porcentagem_separacao/100)

        for _ in range(i_treino):
            indice = random.randint(0, len(tabela[coluna]) - 1)
            treino[coluna].append(tabela[coluna].pop(indice))

        for i in range(len(tabela[coluna])):
            teste[coluna].append(tabela[coluna].pop(0))
    
    return (treino, teste)

# Função que serve para "treinar" a arvore criando seus nós e colocando os seus respectivos lavores
def arvore_decisao(tabela_treino:dict, no_arvore:Arvore_decisao, coluna_target:str, min_amostras_no:int = None, max_profundidade:int = None, profundidade:int = 1):

    # ----------------------------------------- FUNÇÕES AUXILIARES -------------------------------------------------------------------------------

    # EXOLHA DE VALOR
    def escolher_valor_no(): # olha qual o valor dentro da coluna que mais se reprete e define ele como valor do no
        quantidade_aparicoes = {elemento:0 for elemento in tabela_treino[coluna_target]}
        for elemento in tabela_treino[coluna_target]:
            quantidade_aparicoes[elemento] += 1
        
        no_arvore.valor = max(quantidade_aparicoes, key=quantidade_aparicoes.get) # coloca como valor desse no o elemento que mais aparece (maior propabilidade de sert ele o correto)

    # MANIPULAÇÃO DA TABELA
    def remover_colunas_tabela(tabela:dict, coluna:str): # remove a coluna selecionada (vai ser usado no para tratar a tabela para as proximas chamadas recursivas)
        tabela.pop(coluna)

    def remover_linhas(tabela:dict, coluna_target:str, valor_linhas_permanecidas:any): # remove todas as linhas que não contem um valor x na coluna target (também vai ser usada para tratar a tabela para chamda das proximas funções recursivamente)

         # Descobre quais índices devem ser mantidos
        indices_validos = [
            i for i, v in enumerate(tabela[coluna_target]) if v == valor_linhas_permanecidas
        ]
    
        # Atualiza cada coluna da tabela mantendo apenas os índices válidos
        for chave in tabela:
            tabela[chave] = [tabela[chave][i] for i in indices_validos]

    def transformar_tabela(tabela:dict, coluna_target:str, valor_linha_permanecidas:any): # junta as funções de remoção de coluna e de linhas para criar uma tabela nova
        nova_tabela = deepcopy(tabela) # cria uma tabela cópia para evitar modificações de valores na tabela original

        remover_linhas(nova_tabela, coluna_target, valor_linha_permanecidas)
        remover_colunas_tabela(nova_tabela, coluna_target)
        return nova_tabela

    # ----------------------------------------- DEFINIÇÃO DOS CRITERIOS DE PARADA EXENCIAIS ------------------------------------------------------

    contador_decres_coluna = len(tabela_treino.keys()) - 1

    if contador_decres_coluna == 0: # Parada da recurssão quando não existem mais colunas para fazer separações criando folhas

        escolher_valor_no()        
        return

    elif all(x == tabela_treino[coluna_target][0] for x in tabela_treino[coluna_target]): # caso de parada onde temos uma amostra "pura" (todos os valores target iguais)
        no_arvore.valor = tabela_treino[coluna_target][0]
        return
    
    # ---------------------------------------- DEFINIÇÃO DOS CRITÉRIOS DE PARADA EXTRAS ----------------------------------------------------------

    elif (min_amostras_no != None) and (len(tabela_treino[coluna_target]) <= min_amostras_no):

        escolher_valor_no()
        return
    
    elif (max_profundidade != None) and (profundidade >= max_profundidade):
        escolher_valor_no()
        return

    # ---------------------------------------- Treinamento da Arvore de decisão -------------------------------------------------------------------

    dicionario_ganho_informacao = {x:0 for x in tabela_treino.keys()} # cria um dicionario que contem cada uma das colunas e o ganho de informação como valor

    for coluna in tabela_treino: # itera sobre cada uma das colunas, menos a target
        if coluna != coluna_target:
            dicionario_ganho_informacao[coluna] = ganho_informacao(tabela_treino[coluna_target], tabela_treino[coluna]) # preeche o dicionario com o ganho de informação de cada coluna

    coluna_selecionada = max(dicionario_ganho_informacao, key=dicionario_ganho_informacao.get) # pega o nome da coluna selecionada
    no_arvore.valor = coluna_selecionada # coloca como valor do nó a coluna selecionada

    dicionario_valores_diferentes_colunas = Counter(tabela_treino[coluna_selecionada]) # faz um dicionario que como chave tem cada um dos valores presente na coluna com maior ganho de informação

    for valor in dicionario_valores_diferentes_colunas.keys(): # itera sobre cada um dos elementos filhos da coluna selecionada (cada elemento único é um filho)
        novo_no = Arvore_decisao() # cria um novo no
        no_arvore.leaves[valor] = novo_no # adiciona esse novo no como uma folha da arvore de desição

        nova_tabela = transformar_tabela(tabela_treino, coluna_selecionada, valor)

        arvore_decisao(nova_tabela, novo_no, coluna_target, min_amostras_no, max_profundidade, profundidade+1)

def aplicar_arvore(arvore:Arvore_decisao, tabela:dict): # retorna uma lista com os valores previstos

    lista_dicionario_resposta = [] # lista que vai conter vários dicionarios, um para cada linha, as chaves deles vão ser os nomes das colunas e vão ter somente um valor atrelado (1 dicionario = 1 linha da tabela)

    quantidade_valores_tabela = 0 # esse trecho do código é para verificar se cada coluna tem a mesma quantidade de linha, se não tiver pode ser que existam informações faltantes e daria erro treinar dessa forma
    for coluna in tabela:
        if quantidade_valores_tabela == 0:
            quantidade_valores_tabela = len(tabela[coluna])
        elif quantidade_valores_tabela != quantidade_valores_tabela:
            return 'erro, as colunas não tem a mesma quantidade de linha ou existem valores vazios' # retorna o erro

    # caso não dê o erro citado a cima então o codigo vai proseguir adicionando as "linhas" (dicionarios)
    for indice in range(quantidade_valores_tabela):
        nova_linha = {}
        for coluna in tabela:
            nova_linha[coluna] = tabela[coluna][indice]
        lista_dicionario_resposta.append(nova_linha)
    
    lista_hipoteses = [] # lista para conter as hipoteses de cada linha

    for linha in lista_dicionario_resposta: # preenche a lista de hipoteses aplicando cada linha a arvore de decisão
        hipotese = arvore.aplicar(linha)
        lista_hipoteses.append(hipotese)

    return lista_hipoteses