import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

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

# Różne wartości n_estimators
n_estimators_list = [10, 20, 30, 40 ,50]

# Wyniki dla różnych konfiguracji
results = []

for n_estimators in n_estimators_list:
    # Podział na zestaw treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trening modelu XGBoost
    xgb = XGBClassifier(n_estimators=n_estimators, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # Predykcje
    y_pred = xgb.predict(X_test)

    # Ewaluacja modelu
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    roc_auc = roc_auc_score(y_test, y_pred)

    # Zapis wyników
    results.append({
        'n_estimators': n_estimators,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    })

    # Wyświetlanie wyników w terminalu
    print(f"Results for n_estimators={n_estimators}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print("\n")

    # Wyciąganie ważności cech
    feature_importances = xgb.feature_importances_

    # Tworzenie DataFrame z ważnościami cech
    feature_importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Sortowanie cech według ważności
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Zapis wykresu ważności cech
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances_df['Feature'][:10], feature_importances_df['Importance'][:10])
    plt.xlabel('Cechy')
    plt.ylabel('Ważność')
    plt.title(f'Najważniejsze cechy w modelu XGBoost (n_estimators={n_estimators})')
    plt.xticks(rotation=45)
    plt.savefig(f'xgb_feature_importance_{n_estimators}.png')
    plt.close()

# Wyświetlenie końcowych wyników jako DataFrame
results_df = pd.DataFrame(results)
print(results_df)
