
from pymongo import MongoClient

def create_portfolio():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['finance_db']
    collection = db['financial_data']

    # Create example entries in MongoDB (if they do not exist) simuliting a portfolio 
    if collection.count_documents({}) == 0:
        example_data = [
            {"date": "2023-01-01", "symbol": "AAPL", "close": 150},
            {"date": "2023-01-02", "symbol": "AAPL", "close": 152},
            {"date": "2023-01-03", "symbol": "AAPL", "close": 151}
        ]
        collection.insert_many(example_data)  # Insert the example data into the collection

    return collection  # Return the collection object
