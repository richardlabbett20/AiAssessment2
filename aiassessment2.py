import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error




df = pd.read_csv(r"c:\Users\Richard\Documents\New folder (3)\2023_nba_player_stats.csv")

# select input features x and the target variable y
#    here we use basic box-score stats as features and total points  as the target
X = df[["Min", "FGA", "3PA", "FTA", "REB", "AST"]]
y = df["PTS"]

# combine x and y then drop any rows with missing values in these columns
#    to avoid errors when fitting the models
data = pd.concat([X, y], axis=1).dropna()
X = data[["Min", "FGA", "3PA", "FTA", "REB", "AST"]]
y = data["PTS"]

# split the data into training and test sets
#    80% is used for training and 20% for testing
#    randomstate=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# nodel 1 linear regression
#    ceate a linear regression model and fit it on the training data
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# make predictions on the test set using the linear regression model
#    and calculate R2 and mae to evaluate its performance
y_test_pred_lin = lin_model.predict(X_test)
r2_lin = r2_score(y_test, y_test_pred_lin)
mae_lin = mean_absolute_error(y_test, y_test_pred_lin)

print("Linear Regression:")
print("  Test R^2:", round(r2_lin, 3))   # proportion of variance explained
print("  Test MAE:", round(mae_lin, 2))  # average absolute error in points

#  model 2 k nearest neighbours regressor
#    create a knn regressor with k=5 and fit it on the training data
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# make predictions on the test set using the k-nn model
#    and calculate R2 and MAE for comparison with linear regression
y_test_pred_knn = knn_model.predict(X_test)
r2_knn = r2_score(y_test, y_test_pred_knn)
mae_knn = mean_absolute_error(y_test, y_test_pred_knn)

print("\nk-Nearest Neighbours (k=5):")
print("  Test R^2:", round(r2_knn, 3))
print("  Test MAE:", round(mae_knn, 2))

