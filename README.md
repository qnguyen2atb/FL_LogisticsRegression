# FL_LogisticsRegression


To run the pipeline:  python main_FL.py 


## Overview
This is a prototype of a simple federated learning system using logistic regression as the main ML model at clients and server sides. Aggregation is done with weighted averaging. This contains 3 parts:

## Data source
ETL data from the ATB hackathon: select the relevant features

### ML model
For this prototype, we used two versions of the logistic regression models. The logistic regression model from scikit-learn is used to train local models locally and a customized version of logistic regression that we re-wrote using [this reference](
https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/) to train the global model.

## Architecture
### Clients
Data from data model is used to simulate 10 local clients. Each client corresponds to a geographical area

### Aggregation
This module performs weighted averaging at the server side to generte the global model

## Prediction
### Global prediction
This module performs prediction at the local clients using the global model

### Local prediction
This module performs training and predicting at the local clients
