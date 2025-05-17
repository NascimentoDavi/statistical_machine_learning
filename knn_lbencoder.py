import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATASET = './datasets/loan3000.csv'

# Carregar dados
df = pd.read_csv(DATASET)

# Remover coluna desnecessária
df = df.drop(columns=['Unnamed: 0'])

# Instanciar Label Encoder
label_encod_outcome = LabelEncoder()
label_encod_purpose = LabelEncoder()

# Codificar colunas categóricas utilizando LabelEncoder
df['outcome'] = label_encod_outcome.fit_transform(df['outcome']) #target
df['purpose_'] = label_encod_purpose.fit_transform(df['purpose_']) # feature categórica

# target = variável que modelo deve prever
# outcome = variável utilizada para tentar realizar as previsões

# Separar features e target
x = df.drop(columns=['outcome'])
y = df['outcome']

# Dividir conjunto de dados em duas partes: DADOS DE TREINO | DADOS DE TESTE

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# test_size define que 20% dos dados serão utilizados para teste, e o restante para treino

# Criar modelo KNN com k=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

# Fazer previsões
y_pred = knn.predict(x_test)

# Avaliar acurácia
accuracy = accuracy_score(y_test, y_pred)
# print(f"Acurácia do KNN (k=5): {accuracy:.2f}")

# Gerar matriz de confusão
# Na matriz de confusã comparamos o que o modelo previu com a verdade em si, por isso utilizamos y_test e y_pred

cm = confusion_matrix(y_test, y_pred)

# Mostrar matriz de confusão de forma BUNITA
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encod_outcome.classes_)

disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - KNN")
plt.show()