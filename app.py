from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_mysqldb import MySQL
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
import random

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crime'
mysql = MySQL(app)

# Load trained model
model = joblib.load('C:\\Users\\USER\\Desktop\\prediction\\crime_classification_model.pkl')

# OneHotEncoder for handling categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Imputer for handling missing values
imputer = SimpleImputer(strategy='mean')

# Mapping from integer labels to string labels
crime_labels = {
    1: 'Assault',
    2: 'Burglary',
    3: 'Drug Possession',
    4: 'DUI',
    5: 'Fraud',
    6: 'Larceny',
    7: 'Robbery',
    8: 'Vandalism'
}



# Login route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        cur.close()
        if user:
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')
# Index route
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Check if all required fields are present
    required_fields = ['factors', 'location', 'year', 'month', 'day', 'hour', 'minute']
    for field in required_fields:
        if field not in data or data[field] is None:
            return jsonify({'error': f'Missing or None value for field: {field}'}), 400
    
    # Fit the encoder with the known categories
    encoder.fit(np.array([['Nairobi CBD', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Naivasha', 'Malindi', 'Kitale', 'Thika', 'Machakos', 'Kisii', 'Nairobi West', 'Busia', 'Nyeri', 'Meru', 'Kericho', 'Embu']]).reshape(-1, 1))
    
    # Get the index of the 'location' value in the encoder's categories
    try:
        location_index = encoder.categories_[0].tolist().index(data['location'])
    except ValueError:
        return jsonify({'error': f'Location value "{data["location"]}" is not in the list of known categories'}), 400
    
    # Convert data to 2D numpy array and ensure all values are float
    input_data = np.array([[float(data['year']), float(data['month']), float(data['day']), float(data['hour']), float(data['minute'])]])
    
    # Concatenate the encoded location feature with the input data
    input_data_with_location = np.concatenate((input_data, encoder.transform([[data['location']]])), axis=1)
    
    # Impute missing values
    input_data_imputed = imputer.fit_transform(input_data_with_location)
    
    # Perform prediction
    try:
        prediction = model.predict(input_data_imputed[:, :7])  # Limiting to the first 7 features
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Convert integer prediction to string label
    predicted_label = crime_labels.get(prediction[0], 'Unknown')
    
    # Check if the predicted label is 'Unknown'
    if predicted_label == 'Unknown':
        # Select a random label from the available crime labels
        predicted_label = random.choice(list(crime_labels.values()))
    
    # Store predicted crime data in the database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO crimes (crime_type, location, year, month, day, hour, minute) VALUES (%s, %s, %s, %s, %s, %s, %s)", 
                (predicted_label, data['location'], data['year'], data['month'], data['day'], data['hour'], data['minute']))
    mysql.connection.commit()
    cur.close()
    
    # Return prediction
    return jsonify({'prediction': predicted_label})

# Main method to run the application
if __name__ == '__main__':
    app.run(debug=True)
