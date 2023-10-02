# Imports
# DO NOT CHANGE THESE LINES.
import os
import pandas as pd
import json
import warnings
import pymc3 as pm
import theano
import numpy as np
from joblib import load
warnings.filterwarnings('ignore')


# Paths
# DO NOT CHANGE THESE LINES.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR, 'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
OUTPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "outputs")
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
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
LABEL_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'label_encoder.joblib')
SCALER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'scaler.joblib')

if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)


# Reading the schema
file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('.json')][0]
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

# Reading test data.
file_name = [f for f in os.listdir(TEST_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TEST_DIR, file_name)
df = pd.read_csv(file_path)


# Data preprocessing
"""
Note that when we work with testing data, we have to impute using the same values learned during
training. This is to avoid data leakage. 
"""

imputation_values = load(IMPUTATION_FILE)
for column in nullable_features:
    df[column].fillna(imputation_values[column], inplace=True)


# Encoding
# We encode the data using the same encoder that we saved during training.

# Saving the id column in a different variable.
ids = df[id_feature]

# Dropping the id from the dataframe
df.drop(columns=[id_feature], inplace=True)

# Encoding the rest of the features if exist
if os.path.exists(OHE_ENCODER_FILE):
    encoder = load(OHE_ENCODER_FILE)
    df = encoder.transform(df)

# Scaling the numeric features if exist
if os.path.exists(SCALER_FILE_PATH):
    scaler = load(SCALER_FILE_PATH)
    df[numeric_features] = scaler.transform(df[numeric_features])

x = df.values

model = load(PREDICTOR_FILE_PATH)

loaded_model = model['model']
loaded_trace = model['trace']
x_shared = model['x_shared']

x_shared.set_value(x)


with loaded_model:
    # Generate samples from the posterior predictive distribution
    posterior_predictive_test = pm.sample_posterior_predictive(loaded_trace, samples=2000, model=loaded_model)

# 'posterior_predictive' is a dictionary. Extract the samples for the observed node (e.g., 'y' or whatever name you used)
predictions_samples_test = posterior_predictive_test['y']

mean_predictions_test = predictions_samples_test.mean(axis=0)


# Compute point estimates and uncertainty estimates from the samples

prediction_df = pd.DataFrame({id_feature: ids, "prediction": mean_predictions_test})

prediction_df.to_csv(PREDICTIONS_FILE)
