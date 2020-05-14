import numpy as np  # used for handling numbers
import pandas as pd  # used for handling the datasets

from sklearn.impute import SimpleImputer  # used for handling missing data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # used for encoding categorical data

from sklearn.model_selection import train_test_split  # used for splitting training and testting data

from sklearn.preprocessing import StandardScaler  # used for feature scaling

dataset = pd.read_csv('Data.csv')  # to import dataset into a variable

# splitting the attributes into independent and dependent attributes
X = dataset.iloc[:, :-1].values  # attributes to determine dependent variable / class
Y = dataset.iloc[:, -1].values  # dependent variables / class
