# Documentação do Código: Classificação com Random Forest no Conjunto de Dados Iris

## Descrição Geral
Este código utiliza o algoritmo **Random Forest Classifier** para realizar a classificação do conjunto de dados **Iris**. Ele inclui as seguintes etapas:
- Carregamento do conjunto de dados.
- Pré-processamento e padronização.
- Validação cruzada para avaliar o modelo inicial.
- Ajuste de hiperparâmetros com **GridSearchCV**.
- Avaliação final do modelo.

---

## Etapas do Código

### 1. **Importação das Bibliotecas**
As bibliotecas utilizadas são:
- `datasets`: Para carregar o conjunto de dados Iris.
- `train_test_split`: Para dividir os dados em treino e teste.
- `cross_val_score`: Para realizar a validação cruzada.
- `GridSearchCV`: Para ajuste de hiperparâmetros.
- `StandardScaler`: Para padronização dos dados.
- `RandomForestClassifier`: O algoritmo de classificação utilizado.
- `classification_report` e `confusion_matrix`: Para avaliação do desempenho do modelo.

---

### 2. **Carregamento do Conjunto de Dados**
O conjunto de dados **Iris** é carregado com o seguinte código:
```python

iris = datasets.load_iris()
X = iris.data
y = iris.target






