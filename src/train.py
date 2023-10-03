# Imports
# DO NOT CHANGE THESE LINES.
import os

import json
import pandas as pd
import warnings
import pymc3 as pm
import theano
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder
from joblib import dump

warnings.filterwarnings('ignore')

# Paths
# DO NOT CHANGE THESE LINES.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR, 'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
INPUT_SCHEMA_DIR = os.path.join(INPUT_DIR, "schema")
DATA_DIR = os.path.join(INPUT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")
MODEL_PATH = os.path.join(MODEL_INPUTS_OUTPUTS, "model")
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
OHE_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'ohe.joblib')
PREDICTOR_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "predictor")
PREDICTOR_FILE_PATH = os.path.join(PREDICTOR_DIR_PATH, "predictor.joblib")
IMPUTATION_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'imputation.joblib')
LABEL_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'label_encoder.joblib')
SCALER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'scaler.joblib')
if not os.path.exists(MODEL_ARTIFACTS_PATH):
    os.makedirs(MODEL_ARTIFACTS_PATH)
if not os.path.exists(PREDICTOR_DIR_PATH):
    os.makedirs(PREDICTOR_DIR_PATH)

# Reading the schema
"""
The schema contains metadata about the datasets. We will use the schema to get information about the type of each 
feature (NUMERIC or CATEGORICAL) and the id and target features, this will be helpful in preprocessing stage. 
"""

file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('json')][0]
schema_path = os.path.join(INPUT_SCHEMA_DIR, file_name)
with open(schema_path, "r", encoding="utf-8") as file:
    schema = json.load(file)
features = schema['features']

numeric_features = []
categorical_features = []
nullable_features = []
for f in features:
    if f['dataType'] == 'CATEGORICAL':
        categorical_features.append(f['name'])
    else:
        numeric_features.append(f['name'])
    if f['nullable']:
        nullable_features.append(f['name'])

id_feature = schema['id']['name']
target_feature = schema['target']['name']

# Reading training data
file_name = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TRAIN_DIR, file_name)
df = pd.read_csv(file_path)

# Data Preprocessing
"""
Data preprocessing is very important before training the model, as the data may contain missing values in some 
cells. Moreover, most of the learning algorithms cannot work with categorical data, thus the data has to be encoded. 
In this section we will impute the missing values and encode the categorical features. Afterwards the data will be 
ready to train the model. 

Imputing missing data The median value will be used to impute missing values of the numeric features and the mode 
will be used to impute categorical features. 
You can add your own preprocessing steps such as: 
Normalization, Outlier removal, Dropping or adding features 

Important note: Saving the values used for imputation during training step is crucial. These values will be used to 
impute missing data in the testing set. This is very important to avoid the well known problem of data leakage. 
During testing, you should not make any assumptions about the data in hand, alternatively anything needed during the 
testing phase should be learned from the training phase. This is why we are creating a dictionary of values used 
during training to reuse these values during testing.
"""

# Imputing missing data
imputation_values = {}
for column in nullable_features:
    if column in numeric_features:
        value = df[column].median()
    else:
        value = df[column].mode()[0]

    df[column].fillna(value, inplace=True)
    imputation_values[column] = value
dump(imputation_values, IMPUTATION_FILE)


# Comment the above code and write you own imputation code here


# Encoding Categorical features
"""
The id column is just an identifier for the training example, so we will exclude it during the encoding phase.<br>
Target feature will be label encoded in the next step.
"""


# Saving the id and target columns in a different variable.
ids = df[id_feature]
target = df[target_feature]

# Dropping the id and target from the dataframe
df.drop(columns=[id_feature, target_feature], inplace=True)

# Ensure that all categorical columns are stored as str type.
# This is to ensure that even if the categories are numbers (1, 2, ...), they still get encoded.
for c in categorical_features:
    df[c] = df[c].astype(str)

# Encoding the categorical features if exist
if categorical_features:
    encoder = OneHotEncoder(top_categories=6)
    encoder.fit(df)
    df = encoder.transform(df)

    # Saving the encoder to use it on the testing dataset
    dump(encoder, OHE_ENCODER_FILE)

# Scaling numeric features if exist
if numeric_features:
    scaler = StandardScaler()
    df_numeric = df[numeric_features]
    df[numeric_features] = scaler.fit_transform(df_numeric)
    dump(scaler, SCALER_FILE_PATH)


x = df.values
y = target.values
n_features = x.shape[1]

# Training the bayesian regressor
if __name__ == '__main__':

    x_shared = theano.shared(x)
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=n_features)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Likelihood
        mu = alpha + pm.math.dot(x_shared, beta)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(2000, step=pm.Slice())
    # Saving the model
    with model:
        dump({'model': model, 'trace': trace, 'x_shared': x_shared}, PREDICTOR_FILE_PATH)

