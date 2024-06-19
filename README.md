Sure, here's the README in English with the added information about using Dask for parallel processing in data reading:

---

# Price Prediction and Portfolio Recommendations Project

## Description

This project aims to develop a system that predicts financial asset prices and provides recommendations on which assets to add to a portfolio. It uses TensorFlow for predictions and MongoDB to manage portfolio information. Dask is used for parallel processing in data reading to enhance performance.

## Project Structure

The project is organized into several main files:

1. **main.py**: The main file that executes the project workflow.
2. **mongo_portfolio.py**: Contains functions necessary to interact with the MongoDB database.
3. **price_prediction.py**: Contains the price prediction model using TensorFlow.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.7+
- TensorFlow
- pymongo
- numpy
- pandas
- dask

You can install the dependencies using pip:

```bash
pip install tensorflow pymongo numpy pandas dask
```

## Configuration

### Database

The project uses MongoDB to store and manage portfolio information. Make sure you have a running instance of MongoDB. You can configure the connection in the `mongo_portfolio.py` file.

### Execution

To run the project, simply execute the `main.py` file:

```bash
python main.py
```

## Features

### Price Prediction

The `price_prediction.py` file contains a price prediction model based on neural networks using TensorFlow. You can adjust the model and parameters as needed.

### Portfolio Management

The `mongo_portfolio.py` file manages CRUD (Create, Read, Update, Delete) operations for the portfolio using MongoDB. Make sure to configure your database connection properly.

### Recommendations

Based on the price predictions, the system will provide recommendations on which assets to add to the portfolio. These recommendations are based on predefined criteria that you can adjust according to your needs.

### Parallel Data Processing

The project uses Dask for parallel processing in data reading to improve performance. This allows handling large datasets more efficiently.

## Contribution

If you wish to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

