from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_evaluate_models():
    # Load preprocessed data and preprocessor
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load('preprocessed_data.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
    }

    best_model = None
    best_mse = float('inf')

    for name, model in models.items():
        # Define full pipeline with model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        print(f"{name} - Validation MSE: {mse}, R2: {r2}")

        # Save the best model
        if mse < best_mse:
            best_mse = mse
            best_model = pipeline

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    print("Best model saved!")

if __name__ == "__main__":
    train_and_evaluate_models()

    