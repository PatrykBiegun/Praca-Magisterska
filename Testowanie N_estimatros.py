import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie pliku
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# Normalizacja kolumny Amount
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Usunięcie kolumny Time
df = df.drop(columns=['Time'])

# Podział na cechy (X) i etykiety (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Metoda wyrównywania liczności klas
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Różne wartości n_estimators
n_estimators_list = [5, 10, 15, 20, 30, 40, 50]

# Wyniki dla różnych konfiguracji
results = []

# Funkcja do trenowania modeli i zapisywania wyników
def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, n_estimators):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    roc_auc = roc_auc_score(y_test, y_pred)
    
    results.append({
        'model': model_name,
        'n_estimators': n_estimators,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    })
    
    print(f"{model_name} Results for n_estimators={n_estimators}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
    print("\n")
    
    feature_importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances_df['Feature'][:10], feature_importances_df['Importance'][:10])
    plt.xlabel('Cechy')
    plt.ylabel('Ważność')
    plt.title(f'Najważniejsze cechy w modelu {model_name} (n_estimators={n_estimators})')
    plt.xticks(rotation=45)
    plt.savefig(f'{model_name.lower()}_feature_importance_{n_estimators}.png')
    plt.close()
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywiste')
    plt.title(f'Macierz Pomyłek - {model_name} (n_estimators={n_estimators})')
    plt.savefig(f'{model_name.lower()}_confusion_matrix_{n_estimators}.png')
    plt.close()

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

for n_estimators in n_estimators_list:
    # Trening modelu RandomForest
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    train_and_evaluate_model('RandomForest', rf, X_train, X_test, y_train, y_test, n_estimators)
    
    # Trening modelu XGBoost
    xgb = XGBClassifier(n_estimators=n_estimators, random_state=42, use_label_encoder=False, eval_metric='logloss')
    train_and_evaluate_model('XGBoost', xgb, X_train, X_test, y_train, y_test, n_estimators)

# Wyświetlenie końcowych wyników jako DataFrame
results_df = pd.DataFrame(results)
print(results_df)
