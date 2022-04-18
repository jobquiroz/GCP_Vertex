
# package import
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
from google.cloud import bigquery
import argparse
import os
import sys
import hypertune

# import argument to local variables
parser = argparse.ArgumentParser()
# the passed param, dest: a name for the param, default: if absent fetch this param from the OS, type: type to convert to, help: description of argument
parser.add_argument('--epochs', dest = 'epochs', default = 10, type = int, help = 'Number of Epochs')
parser.add_argument('--batch_size', dest = 'batch_size', default = 32, type = int, help = 'Batch Size')
parser.add_argument('--var_target', dest = 'var_target', type=str)
parser.add_argument('--var_omit', dest = 'var_omit', type=str, nargs='*')
parser.add_argument('--project_id', dest = 'project_id', type=str)
parser.add_argument('--dataname', dest = 'dataname', type=str)
parser.add_argument('--region', dest = 'region', type=str)
parser.add_argument('--notebook', dest = 'notebook', type=str)
# hyperparameters
parser.add_argument('--lr',dest='learning_rate', required=True, type=float, help='Learning Rate')
parser.add_argument('--m',dest='momentum', required=True, type=float, help='Momentum')
args = parser.parse_args()

# built in parameters for data source:
PROJECT_ID = args.project_id
DATANAME = args.dataname
REGION = args.region
NOTEBOOK = args.notebook

# clients
bigquery = bigquery.Client(project = PROJECT_ID)

# get schema from bigquery source
query = f"SELECT * FROM {DATANAME}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{DATANAME}_prepped'"
schema = bigquery.query(query).to_dataframe()

# get number of classes from bigquery source
nclasses = bigquery.query(query = f'SELECT DISTINCT {args.var_target} FROM {DATANAME}.{DATANAME}_prepped WHERE {args.var_target} is not null').to_dataframe()
nclasses = nclasses.shape[0]

# Make a list of columns to omit
OMIT = args.var_omit + ['splits']

# use schema to prepare a list of columns to read from BigQuery
selected_fields = schema[~schema.column_name.isin(OMIT)].column_name.tolist()

# all the columns in this data source are either float64 or int64
output_types = [dtypes.float64 if x=='FLOAT64' else dtypes.int64 for x in schema[~schema.column_name.isin(OMIT)].data_type.tolist()]

# remap input data to Tensorflow inputs of features and target
def transTable(row_dict):
    target=row_dict.pop(args.var_target)
    target = tf.one_hot(tf.cast(target,tf.int64), nclasses)
    target = tf.cast(target, tf.float32)
    return(row_dict, target)

# function to setup a bigquery reader with Tensorflow I/O
def bq_reader(split):
    reader = BigQueryClient()

    training = reader.read_session(
        parent = f"projects/{PROJECT_ID}",
        project_id = PROJECT_ID,
        table_id = f"{DATANAME}_prepped",
        dataset_id = DATANAME,
        selected_fields = selected_fields,
        output_types = output_types,
        row_restriction = f"splits='{split}'",
        requested_streams = 3
    )
    
    return training

train = bq_reader('TRAIN').parallel_read_rows().prefetch(1).map(transTable).shuffle(args.batch_size*10).batch(args.batch_size)
validate = bq_reader('VALIDATE').parallel_read_rows().prefetch(1).map(transTable).batch(args.batch_size)
test = bq_reader('TEST').parallel_read_rows().prefetch(1).map(transTable).batch(args.batch_size)

# Logistic Regression

# model input definitions
feature_columns = {header: tf.feature_column.numeric_column(header) for header in selected_fields if header != args.var_target}
feature_layer_inputs = {header: tf.keras.layers.Input(shape = (1,), name = header) for header in selected_fields if header != args.var_target}

# feature columns to a Dense Feature Layer
feature_layer_outputs = tf.keras.layers.DenseFeatures(feature_columns.values())(feature_layer_inputs)

# batch normalization then Dense with softmax activation to nclasses
layers = tf.keras.layers.BatchNormalization()(feature_layer_outputs)
layers = tf.keras.layers.Dense(nclasses, activation = tf.nn.softmax)(layers)

# the model
model = tf.keras.Model(
    inputs = feature_layer_inputs,
    outputs = layers
)

#opt = tf.keras.optimizers.SGD() #SGD or Adam
opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)

loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer = opt,
    loss = loss,
    metrics = ['accuracy', tf.keras.metrics.AUC(curve='PR')]
)

# setup tensorboard logs and train
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'], histogram_freq=1)
history = model.fit(train, epochs = args.epochs, callbacks = [tensorboard_callback], validation_data = validate)

# output the model save files
model.save(os.getenv("AIP_MODEL_DIR"))

# report hypertune info back to Vertex AI Training > Hyperparamter Tuning Job
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag = 'loss',
    metric_value = history.history['loss'][-1],
    global_step = 1)
