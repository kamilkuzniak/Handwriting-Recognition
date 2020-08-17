import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


class CnnModel(Model):
    BATCH_SIZE = 32
    EPOCHS = 5

    def __init__(self):
        super(CnnModel, self).__init__()
        # getting the train and test data from the mnist dataset
        self.mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.mnist.load_data()
        self.train_images = (np.expand_dims(self.train_images, axis=-1) / 255.).astype(np.float32)
        self.train_labels = self.train_labels.astype(np.int64)
        self.test_images = (np.expand_dims(self.test_images, axis=-1) / 255.).astype(np.float32)
        self.test_labels = self.test_labels.astype(np.int64)

        # setting up the cnn model with 2 convolutional layers, 2 max pooling layers, flatten layer and 2 dense layers
        self.conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu)
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

    def build_model(self):
        """
        Function that builds the specified model
        :return: Built model
        """
        self.build(self.train_images.shape)
        return self

    def compile_train_model(self):
        """
        Utility function that compiles the model, fits it to the training data and saves the obtained weights
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        self.fit(self.train_images, self.train_labels, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)

        self.save_weights('./data/model_weights')
