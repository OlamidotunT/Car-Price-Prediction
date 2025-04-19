from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    try:
        # Load preprocessed data and best model
        logging.info("Loading preprocessed data and best model...")
        _, _, X_test, _, _, y_test = joblib.load('preprocessed_data.pkl')
        best_model = joblib.load('best_model.pkl')

        # Evaluate on test set
        logging.info("Evaluating the model on the test set...")
        y_test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        logging.info(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")

        # Optionally return the metrics
        return mse, r2

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()


    