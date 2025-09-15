import xgboost as xgb
import sklearn
from sklearn.multioutput import MultiOutputRegressor

# MultiOutputRegressor pozwala trenować jeden model na wiele targetów
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror'
))

model.fit(X_train, y_train)