from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Expected number of features (adjust according to your model's needs)
EXPECTED_FEATURES = 30

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['feature']
        
        # Convert to numpy array and reshape
        features = list(map(float, features.split(',')))
        if len(features) != EXPECTED_FEATURES:
            return jsonify({'message': f'Invalid input. Expected {EXPECTED_FEATURES} features, but got {len(features)}.'}), 400
        
        np_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        pred = model.predict(np_features)
        message = 'Cancer' if pred[0] == 1 else 'Not Cancer'
        return jsonify({'message': message})

    except ValueError:
        return jsonify({'message': 'Invalid input. Please enter numeric values.'}), 400
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
