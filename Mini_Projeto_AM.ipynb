import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
# Lendo os arquivos de treinamento e teste
train = pd.read_csv("train.csv",sep=";")
test = pd.read_csv("test.csv",sep=";")

def extract_features(data):
    # Calcula a intensidade da imagem
    intensity = np.sum(data)/255
    # Calcula a simetria vertical da imagem
    inversa = np.fliplr(data)
    diferenca = np.abs(data - inversa)
    simetria_vertical = np.sum(diferenca)/(255*2)
    # Calcula a simetria horizontal da imagem
    data_hor = np.transpose(data)
    inversa_hor = np.fliplr(data_hor)
    diferenca_hor = np.abs(data_hor - inversa_hor)
    simetria_horizontal = np.sum(diferenca_hor)/(255*2)
    simetria_completa = simetria_horizontal + simetria_vertical

    return intensity, simetria_completa

# Criando novos arquivos de treinamento e teste com informações de intensidade e simetria
train_redu = pd.DataFrame(columns=["label", "intensidade", "simetria"])
test_redu = pd.DataFrame(columns=["label", "intensidade", "simetria"])

for i, row in train.iterrows():
    label = row["label"]
    data = row.drop("label").values.reshape((28, 28))
    intensity, simetry = extract_features(data)
    # Adiciona as informações no novo dataframe
    train_redu.loc[i] = [label,  intensity, simetry]

for i, row in test.iterrows():
    label = row["label"]
    data = row.drop("label").values.reshape((28, 28))
    intensity, simetry = extract_features(data)
    # Adiciona as informações no novo dataframe
    test_redu.loc[i] = [label, intensity, simetry]


# Salvando os novos arquivos de treinamento e teste
train_redu.to_csv("train_redu.csv", index=False)
test_redu.to_csv("test_redu.csv", index=False)

class Perceptron:
  def __init__(self, max_iter=20_000):
        self.max_iter = max_iter

  def fit(self, X, y):
    # Inicializando os pesos
    self.w = [0,0,0]
    self.listaPCI = X
    self.Lista_y =  y
    inter =0
    for _ in range(self.max_iter): #while (len(self.listaPCI) > 0):#for _ in range(self.max_iter):
      inter +=1
      if len(self.listaPCI) == 0:
        break
      index = random.randint(0, len(self.listaPCI)-1)
      pontos = [1,self.listaPCI[index][0],self.listaPCI[index][1]]
      atualiza = [self.Lista_y[index] * x for x in pontos]
      self.w = [x + z for x, z in zip(self.w, atualiza)]
      old_listaPCI = self.listaPCI
      self.listaPCI,self.Lista_y = self.constroiListaPCI(X,y)
      if len(self.listaPCI) < len(old_listaPCI):
        print(f"{len(self.listaPCI)} < {len(old_listaPCI)}")
        self.best_w = self.w
    print(inter)


  def constroiListaPCI(self, X, y):
    l = []  # Lista com os pontos classificador incorretamente.
    new_y = []  # Nova classificação dos pontos incorretos
    for i, x in enumerate(X):
        if np.sign(np.dot ([1,X[i][0],X[i][1]], self.w)) != y[i]:
            l.append(X[i])  # Adiciona o pontos classificado incorretamente
            new_y.append(y[i])
    return l, new_y

  def predict(self, X):
    return np.sign(np.dot ([1,X[0],X[1]], self.best_w))
  def getW(self):
        return self.best_w[0],self.best_w[1],self.best_w[2]
  def calculando_acuracia(self,X,y):
    # Calculando a lista de previsões com base nos pesos w
    # Calculando a quantidade de previsões corretas
    qtd_previsoes_corretas =0
    for i in range(len(X)):
      if np.sign(np.dot ([1,X[i][0],X[i][1]], self.best_w)) == y[i]:
        qtd_previsoes_corretas+=1

    # Calculando a precisão
    precisao = qtd_previsoes_corretas / len(y)
    print("Precisão:", precisao*100)

# Carrega os dados
train_df = pd.read_csv('train_redu.csv')
test_df = pd.read_csv('test_redu.csv')

# Filtra apenas as imagens com label igual a 1 ou 5
train1x5 = train_df[train_df['label'].isin([1, 5])]
test1x5 = test_df[test_df['label'].isin([1, 5])]

# Criando um gráfico de dispersão para os dados de train1x5
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red"}
for label, color in colors.items():
    subset = train1x5[train1x5["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)
ax.set_title(f"Gráfico de dispersão 1 contra 5")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
plt.show()

# Treina os classificadores
X_train = train1x5[['intensidade', 'simetria']].values
y_train = np.where(train1x5['label']==1, 1, -1)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Testa os classificadores
X_test = test1x5[['intensidade', 'simetria']].values
y_test = test1x5['label'].values

# Cria o método de predição do dígito
def predict_digit(classifier, intensidade, simetria):
    y_pred = classifier.predict([intensidade, simetria])
    return int(y_pred)

y_pred_perceptron = [predict_digit(perceptron, intensidade, simetria) for intensidade, simetria in X_test]

modelo = ["Perceptron", "Regressão Linear", "Regressião Logistica "]
# Criando um gráfico de dispersão para os dados de train1x5 com as retas dos calssificadores
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red"}
for label, color in colors.items():
    subset = train1x5[train1x5["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)

for model, classifier, y_pred in zip(modelo, [perceptron],
                                     [y_pred_perceptron]):
    x_plot = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 1])])
    w = classifier.getW()
    b = w[0]
    w1= w[1]
    w2 = w[2]
    y_plot = (-w1 * x_plot - b) / w2
    ax.plot(x_plot, y_plot, label=f"{model}")


ax.set_title("Gráfico de dispersão 1 contra 5 com as retas dos classificadores ")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
ax.set_xlim(40, 120)
ax.set_ylim(40, 180)
plt.show()
