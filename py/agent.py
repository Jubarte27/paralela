import os
import sys
import random
import time
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SINGLE_THREAD = (sys.argv[1].lower() == "true") if len(sys.argv) >= 2 else False
if SINGLE_THREAD:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import tensorflow as tf


@dataclass
class Individual:
    layers: int
    neurons: int
    learning_rate: float
    batch_size: int
    activation: int

    @staticmethod
    def decode(string: str):
        args = string.split(" ")
        layers = int(args[0])
        neurons = int(args[1])
        learning_rate = float(args[2])
        batch_size = int(args[3])
        activation = int(args[4])

        return Individual(layers, neurons, learning_rate, batch_size, activation)


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


class Agent:

    def __init__(self, X_train, y_train, X_test, y_test):
        tf.random.set_seed(42)
        random.seed(42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fitness_function(self, individual: Individual):
        activation = ["relu", "tanh", "sigmoid"][individual.activation]

        # build the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(784,)))

        current_neurons = individual.neurons
        for _ in range(individual.layers):
            model.add(tf.keras.layers.Dense(current_neurons, activation=activation))
            if current_neurons <= 4:
                pass
            elif current_neurons <= 8:
                current_neurons -= 2
            elif current_neurons <= 16:
                current_neurons -= 4
            else:
                current_neurons = current_neurons // 2

        model.add(tf.keras.layers.Dense(10, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=individual.learning_rate)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        )

        timeout = TimeoutCallback(timeout_seconds=60)  # 1-minute limit

        try:
            history = model.fit(
                self.X_train,
                self.y_train,
                epochs=20,
                batch_size=individual.batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0,
                callbacks=[early_stop, timeout],
            )
        except Exception as e:
            return 0.0

        val_accuracy = max(history.history["val_accuracy"])
        return val_accuracy


# Global variable designated for each separate worker process
worker_agent = None


def init_worker(X_train, y_train, X_test, y_test):
    if SINGLE_THREAD:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    global worker_agent
    worker_agent = Agent(X_train, y_train, X_test, y_test)


def worker_task(line: str) -> str:
    global worker_agent
    if not worker_agent:
        raise RuntimeError("Worker agent not initialized.")
    args = line.split(" ")
    task_id = int(args[0])
    individual = Individual.decode(" ".join(args[1:6]))

    try:
        with open(os.devnull, "w") as out_null:
            with redirect_stdout(out_null):
                accuracy = worker_agent.fitness_function(individual)
        return f"{task_id} {accuracy:.17f}"

    except Exception as e:
        return f"{task_id} -inf"


def load_and_preprocess_data():
    """Loads and preprocesses the dataset once for the main process."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # reshape data
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    subset = 10000
    X_train = X_train[:subset]
    y_train = y_train[:subset]

    return X_train, y_train, X_test, y_test


def main(max_workers=8):
    print_lock = threading.Lock()

    def output_result(future):
        """Callback to handle outputs centrally in the main process."""
        result = future.result()
        with print_lock:
            print(result, flush=True)

    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(X_train, y_train, X_test, y_test),
    ) as executor:
        while True:
            try:
                line = input()
                # with open(f"{uuid.uuid4()}.txt", '+a') as f:
                #     f.write(f"{line}\n{threading.active_count()}\n")
                #     f.flush()

                if not line:
                    continue
                if line == "exit":
                    break

                executor.submit(worker_task, line).add_done_callback(output_result)

            except EOFError:
                break
            except BufferError:
                # Timeout waiting for stdin, just loop and listen again
                continue
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    max_workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    main(max_workers)
