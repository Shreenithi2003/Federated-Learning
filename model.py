from sklearn.linear_model import LogisticRegression
import numpy as np

class SimpleModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        
    def get_weights(self):
        # Return model parameters or initial values if not trained
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            return [self.model.coef_, self.model.intercept_]
        else:
            # Return dummy initial values - 6 features
            return [np.zeros((1, 6)), np.array([0.0])]
    
    def set_weights(self, weights):
        # Set model parameters
        self.model.coef_ = weights[0]
        self.model.intercept_ = weights[1]
        return self
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score, log_loss
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        loss = log_loss(y, probabilities)
        accuracy = accuracy_score(y, predictions)
        return loss, accuracy