import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path, sample_size=None):
    try:
        # Load the dataset
        logging.info("Loading dataset...")
        df = pd.read_csv(file_path)

        # Sample the data if sample_size is specified
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)

        if df.empty:
            logging.error("Dataset is empty after sampling.")
            return None

        # Define features and target
        X = df.drop('price', axis=1)
        y = df['price']

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

        # Define preprocessing pipelines
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing pipelines
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        # Split data into train, validation, and test sets
        logging.info("Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Save preprocessed data and preprocessor
        logging.info("Saving preprocessed data and preprocessor...")
        joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test), 'preprocessed_data.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        logging.info("Data preprocessed and saved successfully!")

        # Optionally return preprocessed objects
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

    except FileNotFoundError:
        logging.error("File not found. Please check the file path.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    load_and_preprocess_data('advert.csv', sample_size=40000)


