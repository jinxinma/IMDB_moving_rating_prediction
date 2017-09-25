# IMDB Movie Rating Prediction

For this project, I implemented an end-to-end machine learning model using the IMDB movie data. The goal is to use the state-of-art machine learning algorithms to best predict IMDB movie rating.

### 1. Repository Structure
There are several scripts in this repository:
* `IMDB_EDA.ipynb`: This jupyter notebook has all the exploratory data analysis (EDA) including summary table, plots, and key insights. The notebook organized in a way that each step of EDA is explained.
* `IMDB_Modeling.ipynb`: This jupyter notebook demonstrates the modeling process including feature engineering, hypterparameter tuning, and saving the final model. EDA is also performed here as it's key to test whether newly created features are insightful before testing them in the model. The notebook also provides brief explanation on how the model is progressively built and should be self-contained
* `modeling.py`: User can run this script using the command line tool. The script outputs the RMSE's from both the training and the test sets. The script will also save the predictions into a `txt` file and the model into a `pickle` file
* `user_defined_functions folder`: This folder includes two main python scripts: eda.py and modeling.py. These scripts have all the functions used throughout the EDA and the end-to-end modeling process
  * `eda_functions.py` has code for creating EDA plots and summary statistics
  * `model_functions.py` includes code for the modeling 
  
### 2. Tools Used
* `Data Preprocessing`: **Pandas** + **Scikit learn**. Pandas has various built-in function that can readily preprocess the data and produces insights. Scikit learn has a Pipeline class which allows for automating the data wrangling process. When using these tools together, data wrangling becomes much easier. Note that users can also define their own classes for data preprocessing in Scikit learn. The class needs to have a `fit` method and a `tranform` method in order to be used in the pipeline. I defined my own class to subset data and select most important features. I also borrowed a `CategoricalEncoder`  class implemented by Scikit learn's author to one-hot-encode the categorical features in pipeline. The `CategoricalEncoder` class will be implemented in Scikit learn's future versions
* `Data Visualization`: **Pandas**, **Seaborn**, and **matplotlib**. The plotting functions of Pandas and Seaborn are essentially build on top of matplotlib. Pandas incorporating plotting functions into its dataframe. Seaborn improves aesthetics upon matplotlib. For this project, I chose to use static plots as they are sufficient to derive insights from the data. **Plotly** is a good option to create interactive plots when necessary
* `Machine Learning`: **Scikit learn**, **XGBoost**, and **Keras**. I mainly focused on Scikit learn and XGBoost for this project because for small to medium dataset, they prove to be very powerful. These two libraries are also nicely maintained and documented. I also tried out Keras which is a deep learning library. When handling small datasets, it's advisible to turn to neural networks when genral machine learning algorithms don't work well. For this project, Keras has similar performance to XGBoost and therefore I chose XGBoost to train the final model
