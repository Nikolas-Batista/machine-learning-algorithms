#Importando a biblioteca Pandas
import pandas as pd
import numpy as np

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

#Realizando o escalonamento dos atributos (deixas os atributos na mesma escala)
#Isso impede algoritmo de definir por si só qual é o atributo de maior peso na equação.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[ : , 102:108] = scaler.fit_transform(previsores[ : , 102:108])

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)



