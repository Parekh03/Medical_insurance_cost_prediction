import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(model,X_test,y_test):
    """Picks any random data(cost) from the given y_test,
    and displays the corroesponding cost prediction made by the model(from sklearn)
    
    and,
    
    Evaluates the model on the basis of:
    1.Mean Absolute Error (MAE)
    2.R2 Score
    3.Residual Distribution plot (KDE plot)
    
    input: Model, test features and test labels"""
    y_pred = model.predict(X_test)
    random_index = random.randint(0,y_pred.size)
    print(f"Actual Cost : $ {y_test[random_index]:.2f}")
    print(f"Predicted Cost : $ {y_pred[random_index]:.2f}")
    mae = mean_absolute_error(y_test,y_pred)
    print("\n")
    print(f"Mean Absolute Error : $ {mae:.2f}")
    r2 = r2_score(y_test,y_pred)
    print(f'R2 Score : {r2:.2f}')
    print("\n")
    residual = y_test - y_pred
    sns.kdeplot(residual,fill=True)
    plt.title('Residual Distribution Plot')
    
def evaluate_ann_model(model,X_test,y_test):
    """Picks any random data(cost) from the given y_test,
    and displays the corroesponding cost prediction made by the ANN model
    
    and,
    
    Evaluates the model on the basis of:
    1.Mean Absolute Error (MAE)
    2.R2 Score
    3.Residual Distribution plot (KDE plot)
    
    input: Model, test features and test labels"""
    y_pred = np.squeeze(model.predict(X_test))
    random_index = random.randint(0,y_pred.size)
    print(f"Actual Cost : $ {y_test[random_index]:.2f}")
    print(f"Predicted Cost : $ {y_pred[random_index]:.2f}")
    mae = mean_absolute_error(y_test,y_pred)
    print("\n")
    print(f"Mean Absolute Error : $ {mae:.2f}")
    r2 = r2_score(y_test,y_pred)
    print(f'R2 Score : {r2:.2f}')
    print("\n")
    residual = y_test - y_pred
    sns.kdeplot(residual,fill=True)
    plt.title('Residual Distribution Plot')
    
def plot_loss_curves(history):
    """Plots the loss curves (Training loss and Validation loss) for the input history object of the model;
    """
    #Training loss curve
    plt.title('Training loss curve')
    plt.plot(pd.DataFrame(history.history['loss']), label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.legend()
    plt.show()
    
    #Validation loss curve
    plt.title('Validation loss curve')
    plt.plot(pd.DataFrame(history.history['val_loss']), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.legend()
    plt.show()
    