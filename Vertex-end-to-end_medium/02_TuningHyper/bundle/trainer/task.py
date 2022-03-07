import tensorflow as tf
from tensorflow.keras import layers, models, losses
import argparse
import hypertune

GCS_PATH_FOR_DATA = 'gs://ma-mx-presales-lab-bucket/vertex-end-to-end/'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=5)
parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.1)

args = parser.parse_args()


def extract(example):
    data = tf.io.parse_example(
        example,
        # Schema of the example.
        {
          'image': tf.io.FixedLenFeature(shape=(32, 32, 3), dtype=tf.float32),
          'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        )
    return data['image'], data['label']

def get_dataset(filename):
    return tf.data.TFRecordDataset([GCS_PATH_FOR_DATA + filename]).map(extract, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1024).batch(128).cache().prefetch(tf.data.experimental.AUTOTUNE)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(args.dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


train_dataset = get_dataset('train.tfrecord')
val_dataset = get_dataset('val.tfrecord')
test_dataset = get_dataset('test.tfrecord')


# A distributed strategy to take advantage of available hardward.
# No-op otherwise.
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = create_model()
    # Restore from the latest checkpoint if available.
    latest_ckpt = tf.train.latest_checkpoint('gs://ma-mx-presales-lab-bucket/vertex-end-to-end/checkpoints/')
    if latest_ckpt:
        model.load_weights(latest_ckpt)

# Create a callback to store a check at the end of each epoch.
#ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
#    filepath='gs://ma-mx-presales-lab-bucket/vertex-end-to-end/checkpoints/', #+ 'val/',
#    monitor='val_loss',
#    save_weights_only=True
#    )

#model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[ckpt_callback])

hpt = hypertune.HyperTune()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag='val_accuracy',
          metric_value=logs['val_accuracy'],
          global_step=epoch
        )
custom_callback = CustomCallback()

model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback])

model.evaluate(test_dataset, verbose=2)

# Export the model to GCS.
model.save("gs://ma-mx-presales-lab-bucket/vertex-end-to-end/models/")
