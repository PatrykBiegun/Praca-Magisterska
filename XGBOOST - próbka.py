import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
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

# Metody wyrównywania liczności klas
methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'Undersample': RandomUnderSampler(random_state=42)
}

# Różne wartości n_estimators
n_estimators_list = [10, 20, 30]

# Wyniki dla różnych konfiguracji
results = []

for method_name, method in methods.items():
    # Podział na zestaw treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Przekształcenie zbioru treningowego
    X_res, y_res = method.fit_resample(X_train, y_train)
    
    for n_estimators in n_estimators_list:
        # Trening modelu XGBoost
        xgb = XGBClassifier(n_estimators=n_estimators, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_res, y_res)

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
            'method': method_name,
            'n_estimators': n_estimators,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        })

        # Wyświetlanie wyników w terminalu
        print(f"Results for {method_name} method with n_estimators={n_estimators}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print("\n")

# Wyświetlenie końcowych wyników jako DataFrame
results_df = pd.DataFrame(results)
print(results_df)
