

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load data
df = pd.read_csv(r"c:\Users\Richard\Documents\New folder (3)\2023_nba_player_stats.csv")


# Column names like Min, FGA, 3PA, FTA, REB, AST, PTS are in your CSV.
X = df[["Min", "FGA", "3PA", "FTA", "REB", "AST"]]
y = df["PTS"]

# Drop rows with missing values in these columns
data = pd.concat([X, y], axis=1).dropna()
X = data[["Min", "FGA", "3PA", "FTA", "REB", "AST"]]
y = data["PTS"]

# 3. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model 1: Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_test_pred_lin = lin_model.predict(X_test)
r2_lin = r2_score(y_test, y_test_pred_lin)
mae_lin = mean_absolute_error(y_test, y_test_pred_lin)

print("Linear Regression:")
print("  Test R^2:", round(r2_lin, 3))
print("  Test MAE:", round(mae_lin, 2))

# 5. Model 2: k-Nearest Neighbours Regressor (second algorithm)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_test_pred_knn = knn_model.predict(X_test)
r2_knn = r2_score(y_test, y_test_pred_knn)
mae_knn = mean_absolute_error(y_test, y_test_pred_knn)

print("\nk-Nearest Neighbours (k=5):")
print("  Test R^2:", round(r2_knn, 3))
print("  Test MAE:", round(mae_knn, 2))
