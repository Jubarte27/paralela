from enum import StrEnum
import os
import sys
import random
import time
import threading
from dataclasses import dataclass, fields
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import numpy as np

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


class Dataset(StrEnum):
    MNIST_SMALL = "small"
    MNIST = "full"


class LayerPattern(StrEnum):
    HALVE = "halve"
    SAME = "same"


class Optimizer(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"


@dataclass
class Individual:
    activation: int
    layers: int
    neurons: int
    layer_pattern: LayerPattern
    learning_rate: float
    decay: float
    optimizer: Optimizer
    batch_size: int

    @staticmethod
    def decode(string: str):
        args = string.split(" ")
        required, got = len(fields(Individual)), len(args)
        if required > got:
            raise RuntimeError(f"Missing {required - got} arguments {string}")
        i = 0

        def next_arg():
            nonlocal i
            arg = args[i]
            i += 1
            return arg

        activation = int(next_arg())
        layers = int(next_arg())
        neurons = int(next_arg())
        layer_pattern = LayerPattern(next_arg())
        learning_rate = float(next_arg())
        decay = float(next_arg())
        optimizer = Optimizer(next_arg())
        batch_size = int(next_arg())

        return Individual(
            layers=layers,
            neurons=neurons,
            learning_rate=learning_rate,
            batch_size=batch_size,
            activation=activation,
            layer_pattern=layer_pattern,
            decay=decay,
            optimizer=optimizer,
        )


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

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        tf.random.set_seed(42)
        random.seed(42)

        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test

        self.shape = X_test.shape[1:]
        if len(self.shape) < 3:
            self.shape = (*self.shape, 1)

    @staticmethod
    def conv_layer(filters, kernel_size=(3, 3)):
        return (
            tf.keras.layers.Conv2D(filters, kernel_size, activation='gelu', padding='same'),
            # tf.keras.layers.GaussianDropout(0.1),
            # tf.keras.layers.Conv2D(filters, kernel_size, activation='gelu', padding='same'),

            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GaussianDropout(0.25)
        )

    @staticmethod
    def dense_layer(layer_pattern, layers, neurons, activation):
        if layer_pattern == LayerPattern.HALVE:
            return [
                tf.keras.layers.Dense(neurons // (2**i), activation=activation)
                for i in range(layers)
            ]

        if layer_pattern == LayerPattern.SAME:
            return [
                tf.keras.layers.Dense(neurons, activation=activation)
                for _ in range(layers)
            ]

        raise KeyError(f"Invalid layer_pattern: {layer_pattern}")

    @staticmethod
    def optimizer(name, learning_rate, decay):
        if decay > 0:
            rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
            )
        else:
            rate = learning_rate

        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=rate)
        if name == "adamw":
            return tf.keras.optimizers.AdamW(learning_rate=rate)

        raise KeyError(f"Invalid optimizer: {name}")

    def fitness_function(self, individual: Individual):
        activation = ["relu", "tanh", "sigmoid"][individual.activation]

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=self.shape),
                tf.keras.layers.BatchNormalization(),
                # *Agent.conv_layer(8), # demais para apenas 1 core (em um tempo rasoavel)
                tf.keras.layers.Flatten(),
                *Agent.dense_layer(
                    individual.layer_pattern,
                    individual.layers,
                    individual.neurons,
                    activation,
                ),
                tf.keras.layers.GaussianDropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        optimizer = Agent.optimizer(
            individual.optimizer, individual.learning_rate, individual.decay
        )
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        )
        timeout = TimeoutCallback(timeout_seconds=5 * 60)  # 5m

        try:
            history = model.fit(
                self.X_train,
                self.y_train,
                epochs=10,
                batch_size=individual.batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0,
                callbacks=[early_stop, timeout],
            )
        except Exception as e:
            raise e
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
    individual = Individual.decode(" ".join(args[1:]))

    try:
        with open(os.devnull, "w") as out_null:
            with redirect_stdout(out_null):
                accuracy = worker_agent.fitness_function(individual)
        return f"{task_id} {accuracy:.17f}"

    except Exception as e:
        raise e
        return f"{task_id} -inf"


SMALL_DATASETS = [Dataset.MNIST]

def load_and_preprocess_data(dataset: Dataset = Dataset.MNIST_SMALL):
    if dataset == Dataset.MNIST:
        (X_train, y_train), (X_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    elif dataset == Dataset.MNIST_SMALL:
        (X_train, y_train), (X_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
        subset = 10000
        X_train = X_train[:subset]
        y_train = y_train[:subset]
    else:
        raise KeyError(f"Invalid dataset: {dataset}")

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test


def main(max_workers: int = -1, dataset: Dataset = Dataset.MNIST_SMALL):
    print_lock = threading.Lock()

    def output_result(future):
        result = future.result()
        with print_lock:
            print(result, flush=True)

    X_train, y_train, X_test, y_test = load_and_preprocess_data(dataset)

    with ProcessPoolExecutor(
        max_workers=max_workers if max_workers > 0 else None,
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


def read_arg(i: int, *, default: str) -> str:
    return sys.argv[i] if len(sys.argv) >= i + 1 else default


if __name__ == "__main__":
    max_workers = int(read_arg(2, default="-1"))
    dataset = Dataset(read_arg(3, default=Dataset.MNIST_SMALL.value))
    main(max_workers, dataset)
