# Learning Notes

## What I learned

This project helped me understand the full path between a dataset, a machine learning model and an API.

I learned how to:

- load and explore a dataset with pandas;
- clean missing values before training a model;
- separate input features from the target column;
- split data into training and test sets;
- train a supervised classification model with RandomForestClassifier;
- evaluate a model with accuracy;
- train an unsupervised anomaly detection model with IsolationForest;
- save and reload trained models with joblib;
- expose machine learning models through FastAPI endpoints;
- validate user input with Pydantic;
- test API endpoints with the automatic FastAPI documentation;
- containerize a small API project with Docker.
- create a `/health` endpoint to check if the API is running;
- create a `/metrics` endpoint to expose simple monitoring information;
- write automated API tests with pytest and FastAPI TestClient;

## Difficulties I encountered

The most difficult parts for me were:

- understanding the difference between a DataFrame and a Series;
- understanding why `X` contains the input data and `y` contains the expected answer;
- understanding why the target `y` must be one-dimensional;
- understanding why a `.joblib` file is binary and cannot be read like a text file;
- understanding the difference between `predict`, `predict_proba` and `decision_function`;
- understanding why API input must be converted into a DataFrame before prediction;
- understanding how Docker runs the API outside my local virtual environment.

## What I would improve next

- add a Streamlit dashboard to make the predictions easier to use;
- improve automated test coverage;
- improve the `/health` endpoint if the API becomes more complex;
- improve the `/metrics` endpoint with more monitoring information;
- improve Docker configuration;
- add CI/CD with GitHub Actions;
- add more model evaluation metrics;
- add visualizations for anomaly detection;
- compare RandomForestClassifier with another classification model.