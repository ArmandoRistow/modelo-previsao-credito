from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dados simulados (idade, renda)
X = [
    [25, 2000],
    [40, 8000],
    [35, 5000],
    [28, 3000],
    [50, 10000],
    [23, 1500],
    [45, 9000]
]

# 0 = negado, 1 = aprovado
y = [0, 1, 1, 0, 1, 0, 1]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Criar modelo
modelo = RandomForestClassifier()

# Treinar modelo
modelo.fit(X_train, y_train)

# Fazer previsões
previsoes = modelo.predict(X_test)

# Avaliar
precisao = accuracy_score(y_test, previsoes)

print(f"Precisão do modelo: {precisao:.2f}")
