import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dane dotyczące ważności cech dla Random Forest (n_estimators=30)
rf_feature_importances = {
    'Feature': ['V14', 'V4', 'V12', 'V10', 'V17', 'V7', 'V11', 'V16', 'V18', 'V9'],
    'Importance': [0.154, 0.137, 0.121, 0.110, 0.108, 0.100, 0.095, 0.089, 0.087, 0.084]
}

# Dane dotyczące ważności cech dla XGBoost (n_estimators=30)
xgb_feature_importances = {
    'Feature': ['V14', 'V10', 'V12', 'V17', 'V4', 'V16', 'V11', 'V18', 'V7', 'V9'],
    'Importance': [0.174, 0.158, 0.135, 0.124, 0.109, 0.097, 0.091, 0.086, 0.082, 0.079]
}

# Tworzenie DataFrame dla Random Forest
rf_feature_importances_df = pd.DataFrame(rf_feature_importances)

# Tworzenie DataFrame dla XGBoost
xgb_feature_importances_df = pd.DataFrame(xgb_feature_importances)

# Wykres dla Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importances_df, )
plt.xlabel('Ważność')
plt.ylabel('Cechy')
plt.title('Najważniejsze cechy w modelu Random Forest (n_estimators=30)')
plt.show()

# Wykres dla XGBoost
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_feature_importances_df, )
plt.xlabel('Ważność')
plt.ylabel('Cechy')
plt.title('Najważniejsze cechy w modelu XGBoost (n_estimators=30)')
plt.show()
