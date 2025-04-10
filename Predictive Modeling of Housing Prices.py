import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

data_train = pd.read_csv('/content/train.csv')
df1 = pd.read_csv('/content/sample_submission.csv')
data_test = pd.read_csv('/content/test.csv')

numeric_cols_train = data_train.select_dtypes(include=[np.number]).columns
categorical_cols_train = data_train.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols_train:
    data_train[col].fillna(data_train[col].median(), inplace=True)

for col in categorical_cols_train:
    data_train[col].fillna(data_train[col].mode()[0], inplace=True)

data_encoded = pd.get_dummies(data_train, columns=categorical_cols_train)

X_train = data_encoded.drop('SalePrice', axis=1)
y_train = data_encoded['SalePrice']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

numeric_cols_test = data_test.select_dtypes(include=[np.number]).columns
categorical_cols_test = data_test.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols_test:
    data_test[col].fillna(data_train[col].median(), inplace=True)

for col in categorical_cols_test:
    data_test[col].fillna(data_train[col].mode()[0], inplace=True)

test_encoded = pd.get_dummies(data_test, columns=categorical_cols_test)

missing_cols = set(X_train.columns) - set(test_encoded.columns)
for col in missing_cols:
  test_encoded[col] = 0  
    
extra_cols = set(test_encoded.columns) - set(X_train.columns)
test_encoded = test_encoded.drop(columns=extra_cols)  
    
test_encoded = test_encoded[X_train.columns]

X_test_scaled = scaler.transform(test_encoded)

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train_final, y_train_final)

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

cv_scores = cross_val_score(model, X_train_scaled, y_train, scoring='neg_root_mean_squared_error', cv=5)
print(f"Cross Validation RMSE Scores: {-cv_scores}")
print(f"Mean CV RMSE: {-cv_scores.mean():.2f}")

y_test_pred = model.predict(X_test_scaled)

predictions_df = pd.DataFrame({'Id': data_test['Id'], 'Predicted_SalePrice': y_test_pred})
comparison_df = df1.merge(predictions_df, on='Id', how='inner')
rmse_test = np.sqrt(mean_squared_error(comparison_df['SalePrice'], comparison_df['Predicted_SalePrice']))

print(f"Test RMSE: {rmse_test:.2f}")
print(comparison_df.head(10))