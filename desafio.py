# Importar bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o Iris Dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (classes)

# Dividir o dataset em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalonar os dados para melhorar o desempenho do modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter labels para formato one-hot encoding
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Criar e compilar o modelo de rede neural
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train_onehot,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test_onehot),
                    verbose=0)

# Fazer previsões
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Calcular a taxa de acerto
accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), y_pred_class)

# Exibir resultados
print(f"Acurácia do modelo: {accuracy:.2f}")
print("Relatório de classificação:")
print(classification_report(np.argmax(y_test_onehot, axis=1), y_pred_class))
print("\nMatriz de confusão:")
print(confusion_matrix(np.argmax(y_test_onehot, axis=1), y_pred_class))

# Plotar curva de aprendizado

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Função de Perda')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
