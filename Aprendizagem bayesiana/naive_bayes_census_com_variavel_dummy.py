#Importando a biblioteca Pandas
import pandas as pd

#Importando um arquivo de dados e definindo a variável para a base de dados
base = pd.read_csv('census.csv')

#Definindo os fatores que serão de análises (provisores) e os de classificação (classe)
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
#Tranformando os valores strings para numéricos (usando LabelEncoder)
#Tranformando as variáveis nominais para variáveis do tipo Dummy (OneHotEncoder)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


#Criando as variáveis de teste e treinamento
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#Chamando o algoritmo Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

#Utilizando o classificador para realizar o treinamento
#Lembrando de sempre utilizar a variável de treinamento (previsores e classe)
#Nesse momento é construido a tabela de probabilidade
classificador.fit(previsores_treinamento, classe_treinamento)

#Com a tabela de probabilidade criada, submetemos esse código a base de dados de teste
#para que seja realizada a classificação da mesma
#Com resultado desse código, comparamos com os valores da classe_teste para verificarmos
#se o algoritmo acertou as previsões
previsoes = classificador.predict(previsores_teste)

#Para avaliação da precisão do algorimto, chamamos o accuracy_score que mede a porcentagem
#de acerto do algoritmo
#O confusion_matrix gera uma matriz a qual consegue avaliar em qual classificação houve mais erros e acertos
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
