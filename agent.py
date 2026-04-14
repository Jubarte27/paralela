# import libraries
import os
import random
from contextlib import redirect_stdout
import sys

import tensorflow as tf

class Agent():
    # load fashion mnist dataset
    def __init__(self):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        # normalize data
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # reshape data
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

        # convert labels to categorical
        # there are 10 classes or labels in the data set
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        (self.X_train, self.y_train), (self.X_test, self.y_test) = (X_train, y_train), (X_test, y_test)
    

    # step 1 - fitness function
    def fitness_function(self, individual):
        # decode individuals
        layers = int(individual[0])
        neurons = int(individual[1])
        learning_rate = individual[2]
        batch_size = int(individual[3])
        activation = ['relu', 'tanh', 'sigmoid'][int(individual[4])]

        # print hyperparameters for reference
        print(f"Training with layers: {layers}, neurons: {neurons}, learning_rate: {learning_rate}, batch_size: {batch_size}, activation: {activation}")

        # validate hyperparameters
        if neurons <= 0:
            print("Invalid number of neurons. Setting to 32.")
            neurons = 32
        if learning_rate <= 0:
            print("Invalid learning rate. Setting to 0.001.")
            learning_rate = 0.001
        if batch_size <= 0 or batch_size > len(self.X_train):
            print(f"Invalid batch size. Setting to 32.")
            batch_size = 32

        # build the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(784,)))
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # compile the model
        # using adam as optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        try:
            # train the model for 5 epochs - hopefully won't take long
            history = model.fit(
                self.X_train, self.y_train,
                epochs=5,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0
            )
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return 0  # default fitness value when error

        # validation accuracy of the last epoch
        val_accuracy = history.history['val_accuracy'][-1]
        return val_accuracy

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(1)

    # Parse arguments sent from C11
    layers = int(sys.argv[1])
    neurons = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    batch_size = int(sys.argv[4])
    activation = int(sys.argv[5])
    
    individual = [layers, neurons, learning_rate, batch_size, activation]

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    
    with open(os.devnull, 'w') as out_null:
        with redirect_stdout(out_null):
            agent = Agent()
            accuracy = agent.fitness_function(individual)
    
    # Print exclusively the float value so C's `fscanf` can read it cleanly
    print(f"{accuracy}")