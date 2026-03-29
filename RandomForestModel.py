import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)

df = pd.read_csv('cardio_clean.csv')

X = df.drop(columns=['cardio'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  #preserves class proportions
)

print(f"Train size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test class balance : {y_test.value_counts().to_dict()}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]


print("\n" + "=" * 45)
print("  Random Forest — Evaluation Metrics")
print("=" * 45)
print(f"  Accuracy  : {accuracy_score(y_test, rf_pred):.4f}")
print(f"  Precision : {precision_score(y_test, rf_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, rf_pred):.4f}")
print(f"  F1-Score  : {f1_score(y_test, rf_pred):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, rf_prob):.4f}")
print("\nFull Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['No CVD', 'CVD']))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_pred),
    display_labels=['No CVD', 'CVD']
).plot(ax=ax, colorbar=False)
ax.set_title('Random Forest — Confusion Matrix')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: rf_confusion_matrix.png")

importances = rf.feature_importances_
idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(X.columns[idx], importances[idx], color='forestgreen')
ax.set_title('Random Forest — Feature Importances')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('rf_feature_importances.png', dpi=150)
plt.show()
print("Saved: rf_feature_importances.png")