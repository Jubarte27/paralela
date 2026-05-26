from enum import Enum
import os
import sys
import random
import time
import threading
from dataclasses import dataclass
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

class NeuronPattern(Enum):
    HALVE = "halve"
    SAME = "same"

@dataclass
class Individual:
    layers: int
    neurons: int
    neuron_pattern: NeuronPattern
    learning_rate: float
    batch_size: int
    activation: int
    decay: int

    @staticmethod
    def decode(string: str):
        args = string.split(" ")
        i = 0
        layers = int(args[i])
        i += 1
        neurons = int(args[i])
        i += 1
        learning_rate = float(args[i])
        i += 1
        batch_size = int(args[i])
        i += 1
        activation = int(args[i])
        i += 1

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
        
        self.shape = X_test.shape()[1:]

    
    @staticmethod
    def conv_layer(filters, kernel_size=(3, 3)):
        return (
            tf.keras.layers.Conv2D(filters, kernel_size, activation='gelu', padding='same'),
            tf.keras.layers.GaussianDropout(0.1),
            tf.keras.layers.Conv2D(filters, kernel_size, activation='gelu', padding='same'),

            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GaussianDropout(0.25)
        )
    
    @staticmethod
    def dense_layer(neuron_pattern, layers, neurons, activation):
        if neuron_pattern == NeuronPattern.HALVE:
            return [tf.keras.layers.Dense(neurons // (2**i), activation=activation) for i in range(layers)]
        
        if neuron_pattern == NeuronPattern.SAME:
            return [tf.keras.layers.Dense(neurons, activation=activation) for _ in range(layers)]
        
        raise KeyError(f"Invalid neuron_pattern: {neuron_pattern}")
    
    @staticmethod
    def optimizer(name, learning_rate, decay):
        if decay > 0:
            rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True
            )
        else:
            rate = learning_rate

        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=rate)
        if name  == "adamw":
            return tf.keras.optimizers.AdamW(learning_rate=rate)
        
        raise KeyError(f"Invalid optimizer: {name}")


    def fitness_function(self, individual: Individual):
        activation = ["relu", "tanh", "sigmoid"][individual.activation]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.shape),
            *Agent.conv_layer(32),
            *Agent.dense_layer(individual.neuron_patter, individual.layers, individual.neurons, activation),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        
        optimizer = Agent.optimizer(individual.optimizer, individual.learning_rate, individual.decay)
        model.compile( optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
        timeout = TimeoutCallback(timeout_seconds=5*60)  # 5m

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

SMALL_DATASETS = ["fashion"]

def load_and_preprocess_data(dataset="cifar10"):
    if dataset == "fashion":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # reshape data
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # reshape data
        X_train = X_train.reshape(-1, 32 * 32, 3)
        X_test = X_test.reshape(-1, 32 * 32, 3)
    else:
        raise KeyError(f"Invalid dataset: {dataset}")

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    if dataset in SMALL_DATASETS:
        subset = 10000
        X_train = X_train[:subset]
        y_train = y_train[:subset]

    return X_train, y_train, X_test, y_test


def main(max_workers=8, dataset="cifar10"):
    print_lock = threading.Lock()

    def output_result(future):
        """Callback to handle outputs centrally in the main process."""
        result = future.result()
        with print_lock:
            print(result, flush=True)

    X_train, y_train, X_test, y_test = load_and_preprocess_data(dataset)

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


def read_arg(i: int, *, default: str):
    return sys.argv[i] if len(sys.argv) >= i + 1 else default


if __name__ == "__main__":
    max_workers = int(read_arg(2, default="8"))
    dataset = read_arg(3, default="cifar10")
    main(max_workers, dataset)
