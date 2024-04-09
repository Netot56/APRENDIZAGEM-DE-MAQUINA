import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
import pickle
# Lendo os arquivos de treinamento e teste
train = pd.read_csv("https://raw.githubusercontent.com/Netot56/APRENDIZAGEM-DE-MAQUINA/main/train.csv",sep=";")
test = pd.read_csv("https://raw.githubusercontent.com/Netot56/APRENDIZAGEM-DE-MAQUINA/main/test.csv",sep=";")

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

    #Calculo de simetria completa
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

"""# MODELOS

## Perceptron
"""

class Perceptron:
  def __init__(self, max_iter=1_000_000):
      self.max_iter = max_iter
      self.w = None

  def fit(self, X, y):
      self.w = np.zeros(X.shape[1] + 1)  # Inicializa os pesos
      X_with_bias = X = np.concatenate((np.ones((len(_X),1)), _X), axis=1)  # Adiciona bias aos dados de treinamento
      inter = 0  # Contador de iterações
      pred = np.sign(np.dot(X_with_bias, self.w))  # Calcula as previsões
      ListaPCI_Index = np.where(pred != y)[0]  # Encontra os índices dos pontos classificados incorretamente
      self.Best_PCI_Index = ListaPCI_Index
      self.best_w = np.zeros(X.shape[1] + 1)
      for _ in range(self.max_iter):
          if len(ListaPCI_Index) == 0:  # Todos classificados corretamente, para o treinamento
              break

          random_index = random.choice(ListaPCI_Index)  # Seleciona aleatoriamente um ponto classificado incorretamente
          ListaPCI_point = X_with_bias[random_index]  # Ponto incorretamente classificado
          ListaPCI_label = y[random_index]  # Label do ponto

          self.w += ListaPCI_label * ListaPCI_point  # Atualiza os pesos

          pred = np.sign(np.dot(X_with_bias, self.w))  # Calcula as previsões com o peso atualizado
          ListaPCI_Index = np.where(pred != y)[0]  # Encontra os novos índices dos pontos classificados incorretamente com o peso atualizado

          if len(ListaPCI_Index)<=len(self.Best_PCI_Index):
            abs_weights = np.sum(np.abs(self.w))
            if abs_weights > np.sum(np.abs(self.best_w)):
              self.bestinter = inter
              self.Best_PCI_Index = ListaPCI_Index
              self.best_w = self.w.copy()
          inter += 1
      print(f"Iterações de treinamento: {inter}")

  def predict(self, X):
      X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Adiciona bias aos dados de treinamento
      return np.sign(np.dot (X_with_bias, self.best_w))

  def getW(self):
      return self.best_w
  def getbest(self):
      print(len(self.Best_PCI_Index))
      print(self.bestinter)

  def confussionMatrix(self, pred, y_test):
      #instancia os valores zerados
      TP = 0
      FP = 0
      TN = 0
      FN = 0


      # Calcula a matriz de confusão e as métricas
      for pred_val, true_val in zip(pred, y_test):
          if pred_val == 1 and true_val == 1:
              TP += 1
          elif pred_val == 1 and true_val == -1:
              FP += 1
          elif pred_val == -1 and true_val == 1:
              FN += 1
          elif pred_val == -1 and true_val == -1:
              TN += 1

      # Calcula as métricas de desempenho
      Acc = (TP + TN) / (TP + TN + FP + FN)
      Precisao = TP / (TP + FP) if (TP + FP) != 0 else 0  # Evita divisão por zero
      Recall = TP / (TP + FN) if (TP + FN) != 0 else 0     # Evita divisão por zero
      F1_Score = 2 * (Precisao * Recall) / (Precisao + Recall) if (Precisao + Recall) != 0 else 0  # Evita divisão por zero


      print('================ MATRIZ DE CONFUSÃO PERCEPTRON ==================')
      print('                   Predito Positivo | Predito Negativo')
      print('                   -----------------|------------------')
      print(f'Classe Positiva   | TP = {TP}      | FP = {FP}')
      print(f'Classe Negativa   | FN = {FN}      | TN = {TN}')
      print('--------------------------------------------------------')
      print(f'Acuracia = {Acc*100}%')
      print(f'Precisao = {Precisao*100}%')
      print(f'Recall = {Recall*100}%')
      print(f'F1-Score = {F1_Score*100}%')

"""## Regressão Linear"""

class LinearRegressionClassifier():

    def fit(self, _X, _y):
        self.lista_X = np.array([[1, ponto_X[0], ponto_X[1]] for ponto_X in  _X])
        X_pseudo_inv = np.linalg.inv( self.lista_X.T.dot(self.lista_X)).dot( self.lista_X.T)
        self.w = np.dot(X_pseudo_inv, _y)

    def predict(self, _x):
        X_with_bias = np.c_[np.ones((_x.shape[0], 1)), _x]  # Adiciona bias aos dados de treinamento
        return np.sign(np.dot (X_with_bias, self.w))

    def getW(self):
        return self.w

    def confussionMatrix(self, pred, y_test):
          #instancia os valores zerados
          TP = 0
          FP = 0
          TN = 0
          FN = 0


          # Calcula a matriz de confusão e as métricas
          for pred_val, true_val in zip(pred, y_test):
              if pred_val == 1 and true_val == 1:
                  TP += 1
              elif pred_val == 1 and true_val == -1:
                  FP += 1
              elif pred_val == -1 and true_val == 1:
                  FN += 1
              elif pred_val == -1 and true_val == -1:
                  TN += 1


          # Calcula as métricas de desempenho
          Acc = (TP + TN) / (TP + TN + FP + FN)
          Precisao = TP / (TP + FP) if (TP + FP) != 0 else 0  # Evita divisão por zero
          Recall = TP / (TP + FN) if (TP + FN) != 0 else 0     # Evita divisão por zero
          F1_Score = 2 * (Precisao * Recall) / (Precisao + Recall) if (Precisao + Recall) != 0 else 0  # Evita divisão por zero
          print('================ MATRIZ DE CONFUSÃO REGRESSÃO LINEAR==================')
          print('                   Predito Positivo | Predito Negativo')
          print('                   -----------------|------------------')
          print(f'Classe Positiva   | TP = {TP}      | FP = {FP}')
          print(f'Classe Negativa   | FN = {FN}      | TN = {TN}')
          print('--------------------------------------------------------')
          print(f'Acuracia = {Acc*100}%')
          print(f'Precisao = {Precisao*100}%')
          print(f'Recall = {Recall*100}%')
          print(f'F1-Score = {F1_Score*100}%')

"""## Regressão Logistica"""

class LogisticRegressionClassifier():
    def __init__(self, eta=0.1, tmax=100_000, batch_size=32, lambda_val=0.001):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = batch_size
        self.lambda_val = lambda_val  # Valor de lambda para Weight Decay

    def fit(self, _X, _y):

        X = np.concatenate((np.ones((len(_X),1)), _X), axis=1)
        y = np.array(_y)

        d = X.shape[1]
        N = X.shape[0]
        w = np.zeros(d, dtype=float)
        self.w = []

        for i in range(self.tmax):
            vsoma = np.zeros(d, dtype=float)

            if self.batch_size < N:
                indices = random.sample(range(N), self.batch_size)
                batchX = [X[index] for index in indices]
                batchY = [y[index] for index in indices]
            else:
                batchX = X
                batchY = y

            for xn, yn in zip(batchX, batchY):
                vsoma += (yn * xn) / (1 + np.exp((yn * w).T @ xn))

            # Gradient descent com Weight Decay
            gt = vsoma / self.batch_size + 2 * self.lambda_val * w
            if LA.norm(gt) < 0.0001:
                break
            w = w + (self.eta * gt)

        self.w = w


    #funcao hipotese inferida pela regressa logistica
    def predict_prob(self, X):
        return [(1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) for x in X]

    #Predicao por classificação linear
    def predict(self, X):
        return [1 if (1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) >= 0.5
            else -1 for x in X]



    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]


    def confussionMatrix(self, pred, y_test):
              #instancia os valores zerados
              TP = 0
              FP = 0
              TN = 0
              FN = 0

              # Calcula a matriz de confusão e as métricas
              for pred_val, true_val in zip(pred, y_test):
                  if pred_val == 1 and true_val == 1:
                      TP += 1
                  elif pred_val == 1 and true_val == -1:
                      FP += 1
                  elif pred_val == -1 and true_val == 1:
                      FN += 1
                  elif pred_val == -1 and true_val == -1:
                      TN += 1

              # Calcula as métricas de desempenho
              Acc = (TP + TN) / (TP + TN + FP + FN)
              Precisao = TP / (TP + FP) if (TP + FP) != 0 else 0  # Evita divisão por zero
              Recall = TP / (TP + FN) if (TP + FN) != 0 else 0     # Evita divisão por zero
              F1_Score = 2 * (Precisao * Recall) / (Precisao + Recall) if (Precisao + Recall) != 0 else 0  # Evita divisão por zero

              print('================ MATRIZ DE CONFUSÃO REGRESSÃO LOGISTICA==================')
              print('                   Predito Positivo | Predito Negativo')
              print('                   -----------------|------------------')
              print(f'Classe Positiva   | TP = {TP}      | FP = {FP}')
              print(f'Classe Negativa   | FN = {FN}      | TN = {TN}')
              print('--------------------------------------------------------')
              print(f'Acuracia = {Acc*100}%')
              print(f'Precisao = {Precisao*100}%')
              print(f'Recall = {Recall*100}%')
              print(f'F1-Score = {F1_Score*100}%')

"""# Uso dos modelos

## Carregando e configurando os dados 1x5
"""

# Carrega os dados
train_df = pd.read_csv('train_redu.csv')
test_df = pd.read_csv('test_redu.csv')

# Filtra apenas as imagens com label igual a 1 ou 5
train1x5 = train_df[train_df['label'].isin([1, 5])]
test1x5 = test_df[test_df['label'].isin([1, 5])]

# Criando um gráfico de dispersão para os dados de test1x5
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red"}
for label, color in colors.items():
    subset = test1x5[test1x5["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)
ax.set_title(f"Gráfico de dispersão 1 contra 5")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
plt.show()

# Normalizando os dados
X_train = train1x5[['intensidade', 'simetria']].values
y_train = np.where(train1x5['label']==1, 1, -1) # o que não for 1 se torna -1

X_test = test1x5[['intensidade', 'simetria']].values
y_test = np.where(test1x5['label']==1, 1, -1) # o que não for 1 se torna -1

"""## Perceptron 1x5

"""

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred_perceptron = perceptron.predict(X_test)
perceptron.confussionMatrix(y_pred_perceptron, y_test)

"""## Regressão Linear 1x5"""

LRC = LinearRegressionClassifier()
LRC.fit(X_train, y_train)
y_pred_LRC = LRC.predict(X_test)
LRC.confussionMatrix(y_pred_LRC, y_test)

"""## Regressão Logistica 1x5"""

LoRC = LogisticRegressionClassifier()
LoRC.fit(X_train, y_train)
y_pred_LoRC = LoRC.predict(X_test)
LoRC.confussionMatrix(y_pred_LoRC, y_test)

"""## 1 x 5 - Todos os modelos, comparação

"""

modelo = ["Perceptron", "Regressão Linear", "Regressão Logistica "]
# Criando um gráfico de dispersão para os dados de test1x5 com as retas dos calssificadores
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red"}

for label, color in colors.items():
    subset = test1x5[test1x5["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)

for model, classifier, y_pred in zip(modelo, [perceptron, LRC, LoRC],
                                     [y_pred_perceptron, y_pred_LRC, y_pred_LoRC]):
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

"""## Gráfico de dispersão 1 contra todos com as retas dos classificadores"""

# Filtra as imagens
train_1xtds = train_df[train_df['label'].isin([1, 0, 4, 5])]
test_1xtds = test_df[test_df['label'].isin([1, 0, 4, 5])]

# Treina os classificadores
X_train_1xtds = train_1xtds[['intensidade', 'simetria']].values
y_train_1xtds = np.where(train_1xtds['label']==1, 1, -1)

perceptron_1xtds = Perceptron()
perceptron_1xtds.fit(X_train_1xtds, y_train_1xtds)

linear_regression_1xtds = LinearRegressionClassifier()
linear_regression_1xtds.fit(X_train_1xtds, y_train_1xtds)

logistic_regression_1xtds = LogisticRegressionClassifier()
logistic_regression_1xtds.fit(X_train_1xtds, y_train_1xtds)

# Testa os classificadores
X_test_1xtds = test_1xtds[['intensidade', 'simetria']].values
y_test_1xtds = np.where(test_1xtds['label']==1, 1, -1)

y_pred_perceptron_1xtds = perceptron_1xtds.predict(X_test_1xtds)

y_pred_linear_regression_1xtds = linear_regression_1xtds.predict(X_test_1xtds)

y_pred_logistic_1xtds = logistic_regression_1xtds.predict(X_test_1xtds)

# Criando um gráfico de dispersão para os dados de test_1xtds com as retas dos calssificadores
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red", 0:"purple",4:"green"}
for label, color in colors.items():
    subset = test_1xtds[test_1xtds["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)

for model, classifier, y_pred,color in zip(modelo, [perceptron_1xtds,linear_regression_1xtds,logistic_regression_1xtds], [y_pred_perceptron_1xtds,y_pred_linear_regression_1xtds,y_pred_logistic_1xtds],["black","orange","yellow"]):
    x = np.linspace(50, 160, 100)
    w = classifier.getW()
    b, w1, w2 = w
    y_plot = (-w1 * x_plot - b) / w2
    ax.plot(x_plot, y_plot, color,label=f"{model}")

ax.set_title(f"Gráfico de dispersão {1} contra todos com as retas dos classificadores")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
ax.set_xlim(42, 161)
ax.set_ylim(53, 165)
plt.show()
perceptron_1xtds.confussionMatrix(y_pred_perceptron_1xtds, y_test_1xtds)
linear_regression_1xtds.confussionMatrix(y_pred_linear_regression_1xtds, y_test_1xtds)
logistic_regression_1xtds.confussionMatrix(y_pred_logistic_1xtds, y_test_1xtds)

# Salvar os Classificadores
with open(f'perceptron_{1}_contra_todos.pkl', 'wb') as f:
    pickle.dump(perceptron_1xtds, f)

with open(f'linear_regression_{1}_contra_todos.pkl', 'wb') as f:
    pickle.dump(linear_regression_1xtds, f)

with open(f'logistic_regression_{1}_contra_todos.pkl', 'wb') as f:
    pickle.dump(logistic_regression_1xtds, f)

"""## Gráfico de dispersão 0 contra todos com as retas dos classificadores"""

# Filtra as imagens
train_0xtds = train_df[train_df['label'].isin([0, 4, 5])]
test_0xtds = test_df[test_df['label'].isin([0, 4, 5])]

# Treina os classificadores
X_train_0xtds = train_0xtds[['intensidade', 'simetria']].values
y_train_0xtds = np.where(train_0xtds['label']==0, 1, -1)

perceptron_0xtds = Perceptron(max_iter=7_000_000)
perceptron_0xtds.fit(X_train_0xtds, y_train_0xtds)

linear_regression_0xtds = LinearRegressionClassifier()
linear_regression_0xtds.fit(X_train_0xtds, y_train_0xtds)

logistic_regression_0xtds = LogisticRegressionClassifier()
logistic_regression_0xtds.fit(X_train_0xtds, y_train_0xtds)

# Testa os classificadores
X_test_0xtds = test_0xtds[['intensidade', 'simetria']].values
y_test_0xtds = np.where(test_0xtds['label']==0, 1, -1)

y_pred_perceptron_0xtds = perceptron_0xtds.predict(X_test_0xtds)

y_pred_linear_regression_0xtds = linear_regression_0xtds.predict(X_test_0xtds)

y_pred_logistic_0xtds = logistic_regression_0xtds.predict(X_test_0xtds)

# Criando um gráfico de dispersão para os dados de test_0xtds com as retas dos calssificadores
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red", 0:"purple",4:"green"}
for label, color in colors.items():
    subset = test_0xtds[test_0xtds["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)

for model, classifier, y_pred,color in zip(modelo, [perceptron_0xtds,linear_regression_0xtds,logistic_regression_0xtds], [y_pred_perceptron_0xtds,y_pred_linear_regression_0xtds,y_pred_logistic_0xtds],["black","orange","yellow"]):
    x = np.linspace(50, 160, 100)
    w = classifier.getW()
    b, w1, w2 = w
    y_plot = (-w1 * x_plot - b) / w2
    ax.plot(x_plot, y_plot, color,label=f"{model}")

ax.set_title(f"Gráfico de dispersão 0 contra todos com as retas dos classificadores")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
ax.set_xlim(42, 161)
ax.set_ylim(53, 165)
plt.show()
perceptron_0xtds.confussionMatrix(y_pred_perceptron_0xtds, y_test_0xtds)
linear_regression_0xtds.confussionMatrix(y_pred_linear_regression_0xtds, y_test_0xtds)
logistic_regression_0xtds.confussionMatrix(y_pred_logistic_0xtds, y_test_0xtds)

# Salvar os Classificadores
with open(f'perceptron_{0}_contra_todos.pkl', 'wb') as f:
    pickle.dump(perceptron_0xtds, f)

with open(f'linear_regression_{0}_contra_todos.pkl', 'wb') as f:
    pickle.dump(linear_regression_0xtds, f)

with open(f'logistic_regression_{0}_contra_todos.pkl', 'wb') as f:
    pickle.dump(logistic_regression_0xtds, f)

"""## Gráfico de dispersão 4 contra todos com as retas dos classificadores"""

# Filtra as imagens
train_4xtds = train_df[train_df['label'].isin([4, 5])]
test_4xtds = test_df[test_df['label'].isin([4, 5])]
# Treina os classificadores
X_train_4xtds = train_4xtds[['intensidade', 'simetria']].values
y_train_4xtds = np.where(train_4xtds['label']==4, 1, -1)

perceptron_4xtds = Perceptron(max_iter=7_000_000)
perceptron_4xtds.fit(X_train_4xtds, y_train_4xtds)

linear_regression_4xtds = LinearRegressionClassifier()
linear_regression_4xtds.fit(X_train_4xtds, y_train_4xtds)

logistic_regression_4xtds = LogisticRegressionClassifier()
logistic_regression_4xtds.fit(X_train_4xtds, y_train_4xtds)

# Testa os classificadores
X_test_4xtds = test_4xtds[['intensidade', 'simetria']].values
y_test_4xtds = np.where(test_4xtds['label']==4, 1, -1)
print(len(y_test_4xtds))

y_pred_perceptron_4xtds = perceptron_4xtds.predict(X_test_4xtds)
y_pred_linear_regression_4xtds = linear_regression_4xtds.predict(X_test_4xtds)

y_pred_logistic_4xtds = logistic_regression_4xtds.predict(X_test_4xtds)

# Criando um gráfico de dispersão para os dados de train1x5 com as retas dos calssificadores
fig, ax = plt.subplots(figsize=(8, 6))
colors = {1: "blue", 5: "red", 0:"purple",4:"green"}
for label, color in colors.items():
    subset = test_4xtds[test_4xtds["label"] == label]
    ax.scatter(subset["intensidade"], subset["simetria"], c=color, label=label, alpha=0.5)

for model, classifier, y_pred,color in zip(modelo, [perceptron_4xtds, linear_regression_4xtds,logistic_regression_4xtds], [y_pred_perceptron_4xtds, y_pred_linear_regression_4xtds,y_pred_logistic_4xtds],["black","orange","yellow"]):
    x = np.linspace(50, 160, 100)
    w = classifier.getW()
    b, w1, w2 = w
    y_plot = (-w1 * x_plot - b) / w2
    ax.plot(x_plot, y_plot, color,label=f"{model}")

ax.set_title(f"Gráfico de dispersão 4 contra todos com as retas dos classificadores")
ax.legend()
ax.grid(True)
ax.set_xlabel("Intensidade")
ax.set_ylabel("Simetria")
ax.set_xlim(42, 161)
ax.set_ylim(53, 165)
plt.show()
perceptron_4xtds.confussionMatrix(y_pred_perceptron_4xtds, y_test_4xtds)
linear_regression_4xtds.confussionMatrix(y_pred_linear_regression_4xtds, y_test_4xtds)
logistic_regression_4xtds.confussionMatrix(y_pred_logistic_4xtds, y_test_4xtds)

# Salvar os Classificadores
with open(f'perceptron_{4}_contra_todos.pkl', 'wb') as f:
    pickle.dump(perceptron_4xtds, f)

with open(f'linear_regression_{4}_contra_todos.pkl', 'wb') as f:
    pickle.dump(linear_regression_4xtds, f)

with open(f'logistic_regression_{4}_contra_todos.pkl', 'wb') as f:
    pickle.dump(logistic_regression_4xtds, f)
