import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

DATASET = './datasets/loan3000.csv'

# Load data
df = pd.read_csv(DATASET)

# Added to remove a columns that is automatically added whrn saving or reading a CSV File
df = df.drop(columns=['Unnamed: 0'])

# Instantiate LabelEncoder
# Need to be instantiated twice as result of the encoding of two different categorical columns, and each column can have its own unique sets of values
label_encod_outcome = LabelEncoder()
label_encod_purpose = LabelEncoder()

# Encode categorical columns
df['outcome'] = label_encod_outcome.fit_transform(df['outcome']) # Target
df['purpose_'] = label_encod_purpose.fit_transform(df['purpose_']) # Categorical Feature

# Separate features and target
x = df.drop(columns=['outcome'])
y = df['outcome']

# Divide items group into train and test
# Splits the database into 2 parts
# One part is for training the model (learning) | One part is for testing the model (checking how well it performs on new data)
# text_size=0.2 means 20% of the data will go to the test set
# the remaining 80% will be used to train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

# Make predictions
y_pred = dt.predict(x_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia da árvore de Decisão: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=label_encod_outcome.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encod_outcome.classes_)
disp.plot(cmap=plt.cm.Greens)
plt.title("Matriz de Confusão - Decision Tree")
plt.show()

# Save into CSV
report = classification_report(y_test, y_pred, target_names=label_encod_outcome.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('./metrics/dt_classification_report.csv', index=True)
print("Relatório de métricas salvo em: /metrics/dt_classification_report.csv")

# Visualize Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(dt, filled=True, feature_names=x.columns, class_names=label_encod_outcome.classes_)
plt.title("Visualização da Árvore de Decisão")
plt.show()