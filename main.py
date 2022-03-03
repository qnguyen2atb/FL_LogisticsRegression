

# Train from scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression()
from data import x_train, x_test, y_train, y_test

lr = CustomLogisticRegression()
lr.fit( x_train, x_test, epochs=150)
