from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and columns
model = joblib.load('car_price_model.pkl')
columns = joblib.load('columns.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    mileage = request.form['mileage']
    year_of_registration = request.form['year_of_registration']
    standard_make = request.form['standard_make']
    standard_model = request.form['standard_model']
    fuel_type = request.form['fuel_type']
    body_type = request.form['body_type']

    # Create input vector with correct order
    input_data = {
        'mileage': float(mileage),
        'year_of_registration': float(year_of_registration),
        'standard_make': standard_make,
        'standard_model': standard_model,
        'fuel_type': fuel_type,
        'body_type': body_type
    }

    # Convert input data into the format expected by the model
    input_array = np.zeros(len(columns))
    for i, col in enumerate(columns):
        if col in input_data:
            if isinstance(input_data[col], str):
                # Handle categorical data
                if col == input_data[col]:
                    input_array[i] = 1
            else:
                # Handle numerical data
                input_array[i] = input_data[col]

    # Predict the price
    predicted_price = model.predict([input_array])[0]

    return jsonify({
        'predicted_price': f"{predicted_price:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)



