# Medical_insurance_cost_prediction

## Project Description
This project involves using the concepts of Machine Learning, to predict the cost of Medical Insurance, on the basis of features/attributes like: <br>
1. Age of a person
2. Gender/Sex
3. BMI
4. Number of children
5. Smoker (yes/no)
6. Region of residence

### About the Dataset
The dataset used for this project is 'Medical Cost Personal Dataset' which is publicly available on Kaggle. <br>
You can find the Dataset [here](https://www.kaggle.com/datasets/mirichoi0218/insurance)

### Steps involved
1. <b>Data Loading</b> - This includes loading the data (csv file), along with importing the necessary libraries.
2. <b>Data Exploration</b> - This includes viewing the contents of the loaded data, and inspecting the features/attributes in the data. There are 1338 number of instances and 7 attributes present.
3. <b>Data Preprocessing</b> - This is one of the most important steps in Machine Learning, as it involves converting the raw data into a format which the machine can understand and learn patterns.
<br><br>The first step is to <b>handle the null values</b> present in the data (if any).The data used in this projects does not have any null values.
<br><br>The next step performed was to create <b>Features and Labels</b>. The label consists of the 'charges', and features consists of various attributes like 'age', 'sex', 'bmi', 'children' ,'smoker' and 'region'.
<br><br> To handle the categorical features, I have performed <b>One-Hot Encoding</b> technique, rather than Label Encoding because, the categorical features present in the data are nominal (not ordinal).
<br> Moreover, <b>Normalization</b> was performed on the numerical features using the MinMaxScaler from sklearn.
<br><br>After that, I have split the features and labels into <b>Training, Testing and Validation(For ANN model) sets</b>, which would be necessary for creating models.

4. <b>Creating Models</b> - I have created multiple models for medical insurance cost prediction, which are as follows: <br>
     * <i>Linear Regression Model</i><br>
   * <i>K-Nearest Neighbour(KNN) Model</i><br>
   * <i>Radnom Forest Model</i><br>
   * <i>Artificial Neural Network Model</i><br>
5.  <b>Evaluating the Models</b> - To evaluate the models, I have used the following metrics: <br>
    * <i>Mean Absolute Error (MAE)</i><br>
     * <i>R2 Score</i><br>
     * <i>Residual Distribution Plot</i><br>
      * <i>Comparing the model's prediction by picking any random data(cost) from the testing set, and displaying the corroesponding cost prediction made by the model.</i><br><br>
     <b>Note: </b>To implement the above evaluation, I have created seperate helper function for models from sklearn, and ANN. You can view them in <i><b>helper.py</b></i>. They are also available in the main file.

### Summary of the models
#### Linear Regression Model -
* Mean Absolute Error (MAE) - $4000 (approx)
* R2 Score - 76%

#### KNN Model - 
* Mean Absolute Error (MAE) - $3500 (approx)
* R2 Score - 79%

#### Random Forest Model - 
* Mean Absolute Error (MAE) - $2500 (approx)
* R2 Score - 86%

#### Artificial Neural Network Model -
* Mean Absolute Error (MAE) - $1500 (approx)
* R2 Score - 87%
### Guide to run the code
Download the dataset from [here](https://www.kaggle.com/datasets/mirichoi0218/insurance). Then replace the path in <b><i>'pd.read_csv("___")'</i></b> with the actual path of your downloaded dataset. After that, you are good to go!. You can run all the cells.
