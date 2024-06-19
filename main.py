from flask import Flask, request, jsonify
import mongo_portfolio
import price_prediction 
import pandas as pd

app = Flask(__name__)
# Import portfolio
collection = mongo_portfolio.create_portfolio()

# Calcule model 
model = price_prediction.create_model()

# Route for the API root
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Portfolio Management API"})

# Route to add entries to the portfolio 
@app.route('/data/', methods=['POST'])
def add_data():
    data = request.get_json()  # Get JSON data from the request
    try:
        collection.insert_many(data)  # Insert the data into the MongoDB collection
        return jsonify({"message": "Data added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return an error message if something goes wrong

# Route to get the portfolio
@app.route('/portfolio/', methods=['GET'])
def get_portfolio():
    try:
        data = list(collection.find({}, {"_id": 0}))  # Retrieve all data from the collection, excluding the _id field
        return jsonify(data)  # Return the data as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return an error message if something goes wrong

# Route to get recommendations
@app.route('/recommendations/', methods=['GET'])
def get_recommendations():
    try:
        # Get the JSON request
        data = request.get_json()
        
        # Convert the JSON to a DataFrame
        df = pd.DataFrame(data)
        time_step = 60
        
        # Call the predict_future function
        recommendations = price_prediction.predict_future(model, time_step, df)
        
        return jsonify({"recommendations": recommendations.tolist()})  # Return the recommendations as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return an error message if something goes wrong

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on host 0.0.0.0 and port 5000
