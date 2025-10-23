import functions
import random

tabela_grande = {
    "febre": [],
    "tosse": [],
    "peso": [],
    "atividade": [],
    "target": []
}

linhas = 300  # quantidade de linhas que você quer

for _ in range(linhas):
    febre = random.choice(["sim", "não"])
    tosse = random.choice(["sim", "não"])
    peso = random.choice(["magro", "normal", "gordo"])
    atividade = random.choice(["baixa", "media", "alta"])
    
    # regra do target
    if febre == "sim" or tosse == "sim":
        target = "doente"
    else:
        target = "saudável"
    
    tabela_grande["febre"].append(febre)
    tabela_grande["tosse"].append(tosse)
    tabela_grande["peso"].append(peso)
    tabela_grande["atividade"].append(atividade)
    tabela_grande["target"].append(target)

tabela_simples = {
    'fumante' : [ 'não', 'não', 'não', 'sim', 'sim', 'sim' ],
    'imc' : [ 'magro', 'médio', 'gordo', 'magro', 'médio', 'gordo' ],
    'target' : [ 'não', 'não', 'não', 'não', 'sim', 'sim' ]
}

tabela_treino, tabela_teste = functions.separar_treino_teste(tabela_grande, 85) # separa em treino (80%) e teste (20%) - é mais útel para casos com muitos dados

#tabela_treino, tabela_teste = tabela_complexa, tabela_complexa # exemplo teste para provar que a rede aprender realmente, quando o conjunto de dados de terino e teste for igual ela sempre (quando possivel) vai ter 100% de acerto

coluna_target = 'target' # COLOQUE AQUI O NOME DA COLUNA TARGET

arvore = functions.Arvore_decisao() # define o modelo
functions.arvore_decisao(tabela_treino, arvore, coluna_target) # treina o modelo com os dados de treino
lista_hipotese = functions.aplicar_arvore(arvore, tabela_teste) # aplica o modelo com os dados de teste (retornando uma tabela)

quantidade_total = len(tabela_teste[coluna_target])
quantidade_acertos = 0

for indice in range(len(tabela_teste[coluna_target])):
    if tabela_teste[coluna_target][indice] == lista_hipotese[indice]:
        quantidade_acertos += 1

print(f'Porcentagem de acerto: {(quantidade_acertos / quantidade_total) * 100}%')