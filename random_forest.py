import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

DATASET = './datasets/loan3000.csv'

# Load data
df = pd.read_csv(DATASET)

# Remove unnecessary column (e.g., added when saving CSVs)
df = df.drop(columns=['Unnamed: 0'])

# Instantiate LabelEncoders for categorical columns
label_encod_outcome = LabelEncoder()
label_encod_purpose = LabelEncoder()

# Encode categorical columns
df['outcome'] = label_encod_outcome.fit_transform(df['outcome'])  # Target
df['purpose_'] = label_encod_purpose.fit_transform(df['purpose_'])  # Categorical feature

# Separate features and target
x = df.drop(columns=['outcome'])
y = df['outcome']

# Split dataset into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Make predictions
y_pred = rf.predict(x_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do Random Forest: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=label_encod_outcome.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encod_outcome.classes_)
disp.plot(cmap=plt.cm.Oranges)
plt.title("Matriz de Confusão - Random Forest")
plt.show()

# Save metrics report as CSV
report = classification_report(y_test, y_pred, target_names=label_encod_outcome.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('./metrics/rf_classification_report.csv', index=True)
print("Relatório de métricas salvo em: /metrics/rf_classification_report.csv")
