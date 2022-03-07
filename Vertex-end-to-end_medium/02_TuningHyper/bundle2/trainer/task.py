import tensorflow as tf
from tensorflow.keras import layers, models, losses
import datetime
import argparse

GCS_PATH_FOR_DATA = 'gs://ma-mx-presales-lab-bucket/vertex-end-to-end/'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=5)
parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.7272)

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
    latest_ckpt = tf.train.latest_checkpoint('gs://ma-mx-presales-lab-bucket/vertex-end-to-end/checkpoints')
    if latest_ckpt:
        model.load_weights(latest_ckpt)

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "gs://ma-mx-presales-lab-bucket/vertex-end-to-end/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S/")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[tensorboard_callback])

model.evaluate(test_dataset, verbose=2)

# Export the model to GCS.
model.save("gs://ma-mx-presales-lab-bucket/vertex-end-to-end/models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S/") )
