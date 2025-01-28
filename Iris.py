from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar as características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)

# Validação cruzada para avaliar o desempenho do modelo
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f'Média da Acurácia com Validação Cruzada: {cv_scores.mean():.2f}')

# Ajuste de Hiperparâmetros usando GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],  # Número de árvores
    'max_depth': [None, 10, 20, 30],  # Profundidade máxima das árvores
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4]     # Número mínimo de amostras em uma folha
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhor conjunto de parâmetros
print(f'Melhores Parâmetros Encontrados: {grid_search.best_params_}')

# Treinar o modelo com os melhores parâmetros
best_rf = grid_search.best_estimator_

# Fazer previsões com o conjunto de teste
y_pred = best_rf.predict(X_test)

# Avaliar o desempenho do modelo
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))

print('Matriz de Confusão:')
print(confusion_matrix(y_test, y_pred))
