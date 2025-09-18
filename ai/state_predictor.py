import joblib
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

class RaceRegressor:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        """Initialization of the regressor model."""
        # We use MultiOutputRegressor to predict multiple values simultaneously
        self.model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective='reg:squarederror',
                n_jobs=-1,
                tree_method='gpu_hist'  # Use GPU if available
            )
        )
        self.is_trained = False

    def train(self, X, y, test_size=0.2, random_state=42):
        """Training the model with a train-test split for validation."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        score = self.model.score(X_val, y_val)
        print(f"Validation R^2 score: {score:.4f}")
        return score

    def predict(self, X):
        """Prediction for new data"""
        if not self.is_trained:
            raise ValueError("Model has not been trained or loaded from file.")
        return self.model.predict(X)

    def save(self, path="race_regressor.pkl"):
        """Save the model to a file"""
        if not self.is_trained:
            raise ValueError("Not saving because the model is not trained.")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path="race_regressor.pkl"):
        """Load the model from a file"""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")