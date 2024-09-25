import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
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

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Zastosowanie SMOTE, ADASYN i Random Undersampling tylko do zbioru treningowego
smote = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)
undersample = RandomUnderSampler(random_state=42)

# List of resampling methods to apply
methods = {
    'SMOTE': smote,
    'ADASYN': adasyn,
    'Undersample': undersample
}

# Różne wartości n_estimators
n_estimators_list = [10, 20, 30]

# Wyniki dla różnych konfiguracji
results = []

for method_name, method in methods.items():
    # Apply resampling method to the training set
    X_train_res, y_train_res = method.fit_resample(X_train, y_train)
    
    for n_estimators in n_estimators_list:
        # Trening modelu RandomForest na przekształconym zbiorze treningowym
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train_res, y_train_res)
        
        # Predykcje na zbiorze testowym
        y_pred = rf.predict(X_test)
        
        # Ewaluacja modelu na zbiorze testowym
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Zapis wyników
        results.append({
            'method': method_name,
            'n_estimators': n_estimators,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })

        # Wyświetlanie wyników w terminalu
        print(f"Results for {method_name} with n_estimators={n_estimators}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Confusion Matrix:\n{cm}")
        print("\n")

        # Wyciąganie ważności cech
        feature_importances = rf.feature_importances_

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
        plt.title(f'Najważniejsze cechy w modelu RandomForest ({method_name}, n_estimators={n_estimators})')
        plt.xticks(rotation=45)
        plt.savefig(f'rf_feature_importance_{method_name}_{n_estimators}.png')
        plt.close()

# Wyświetlenie końcowych wyników jako DataFrame
results_df = pd.DataFrame(results)
print(results_df)
