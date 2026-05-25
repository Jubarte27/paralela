# import libraries
import os
import random
from contextlib import redirect_stdout
import sys
import select
import uuid
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, or 3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, timeout_seconds):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.start_time is None:
            return
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout_seconds:
            print(f"\nReached timeout of {self.timeout_seconds}s. Stopping training...")
            self.model.stop_training = True

class Agent():
    data_loaded = False
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    # load fashion mnist dataset
    def __init__(self):
        tf.random.set_seed(42)
        random.seed(42)

        if not Agent.data_loaded:
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

            subset = 10000
            X_train = X_train[:subset]
            y_train = y_train[:subset]

            (Agent.X_train, Agent.y_train), (Agent.X_test, Agent.y_test) = (X_train, y_train), (X_test, y_test)
            Agent.data_loaded = True
            
        self.X_train = Agent.X_train
        self.y_train = Agent.y_train
        self.X_test = Agent.X_test
        self.y_test = Agent.y_test
    

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
        if layers > 3 or layers < 1:
            print("Invalid number of layers. Setting to 1.")
            layers = 1
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

        current_neurons = neurons
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(current_neurons, activation=activation))
            if current_neurons <= 4:
                pass
            elif current_neurons <= 8:
                current_neurons -= 2
            elif current_neurons <= 16:
                current_neurons -= 4
            else:
                current_neurons = current_neurons // 2
        
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # compile the model
        # using adam as optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )

        timeout = TimeoutCallback(timeout_seconds=60) # 5-minute limit

        try:
            # train the model for 5 epochs - hopefully won't take long
            history = model.fit(
                self.X_train, self.y_train,
                epochs=5,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0,
                callbacks=[early_stop, timeout]
            )
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return 0  # default fitness value when error

        # validation accuracy of the last epoch
        val_accuracy = max(history.history['val_accuracy'])
        return val_accuracy

def read_stdin_with_timeout(timeout_seconds=5, write_on_err=None):
    ready, _, _ = select.select([sys.stdin.fileno()], [], [], timeout_seconds)
    
    if ready:
        return sys.stdin.readline().rstrip()
    else:
        if write_on_err is not None:
            print(sys.stdin.read())
            # with open(write_on_err, 'a') as err_file:
            #     err_file.write(f"Timeout occurred while waiting for input: \n{}.\n")
        raise BufferError()

def main():
    agent = Agent()
    write_on_err=f"{uuid.uuid4()}-err.log"
    with open(os.devnull, 'w') as out_null:
        while (line := read_stdin_with_timeout(write_on_err=write_on_err)) != "exit":
            args = line.split(" ")
            layers = int(args[0])
            neurons = int(args[1])
            learning_rate = float(args[2])
            batch_size = int(args[3])
            activation = int(args[4])
            
            individual = [layers, neurons, learning_rate, batch_size, activation]

            with redirect_stdout(out_null):
                accuracy = agent.fitness_function(individual)
            print(f"{accuracy:.17f}")
            sys.stdout.flush()

if __name__ == "__main__":
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

    main()

