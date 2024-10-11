import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("G:/My Drive/SMT5/BANGKIT/PORTFOLIO/AI/UAS/updated_kos_dataset.csv")

# Check for missing values and display data info
print(df.isnull().sum())
print(df.tail())
df.info()
df.head()

# One-hot encoding for categorical features
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['Jenis_kos', 'Meja_belajar', 'Wifi', 'AC', 'Kamar_mandi_dalam', 'Include_listrik', 'Akses_24_jam', 'Daerah']]).toarray()
encoded_feature_names = encoder.get_feature_names_out(['Jenis_kos', 'Meja_belajar', 'Wifi', 'AC', 'Kamar_mandi_dalam', 'Include_listrik', 'Akses_24_jam', 'Daerah'])
df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine encoded features with the target variable
df = pd.concat([df_encoded, df[['Harga']]], axis=1)

# Define features and target variable
X = df.drop(['Harga'], axis=1)
y = df['Harga']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Algorithm
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
print('Random Forest Algorithm Results:')
print("Model Score:", round(rf_regressor.score(X_test, y_test) * 100, 2))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Feature importance
importance = rf_regressor.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importance)[::-1]
top_features = feature_names[sorted_indices]
print("Feature Importance:")
for feature in top_features:
    print(feature)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), importance[sorted_indices], align='center')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance using Random Forest')
plt.show()

# Prediction on new data
input_values = {
    'Jenis_kos': ['putra'],
    'Meja_belajar': ['ada'],
    'Wifi': ['ada'],
    'AC': ['ada'],
    'Kamar_mandi_dalam': ['ada'],
    'Include_listrik': ['tidak'],
    'Akses_24_jam': ['iya'],
    'Daerah': ['Mulyorejo']
}
input_df = pd.DataFrame(input_values)

# One-hot encode the new data using the same encoder
input_encoded = encoder.transform(input_df).toarray()
input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)

# Ensure the new data frame has the same columns as the training data
input_encoded_df = input_encoded_df.reindex(columns=X.columns, fill_value=0)

predicted_prices = rf_regressor.predict(input_encoded_df)
print("Predicted Kos Price:")
for price in predicted_prices:
    print(price)

# Plotting predicted prices
plt.figure(figsize=(8, 6))
plt.plot(predicted_prices, marker='o', linestyle='-', color='b', label='Predicted Prices')
plt.title('Predicted Kos Prices')
plt.xlabel('Data Index')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
