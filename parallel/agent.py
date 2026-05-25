# import libraries
from io import TextIOWrapper
import os
import random
from contextlib import redirect_stdout
import sys
import select
from typing import TextIO
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


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

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )

        timeout = TimeoutCallback(timeout_seconds=60) # 1-minute limit

        try:
            history = model.fit(
                self.X_train, self.y_train,
                epochs=5,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0,
                callbacks=[early_stop, timeout]
            )
        except Exception as e:
            return 0.0  # default fitness value when error

        val_accuracy = max(history.history['val_accuracy'])
        return val_accuracy

def read_stdin_with_timeout(timeout_seconds=5, write_on_err=None):
    ready, _, _ = select.select([sys.stdin.fileno()], [], [], timeout_seconds)
    
    if ready:
        return sys.stdin.readline().rstrip()
    else:
        raise BufferError()

# --- WORKER FUNCTION ---
def worker_task(agent, line, print_lock, print_file: TextIO):
    """Executes a single evaluation in a background thread."""
    args = line.split(" ")
    try:
        task_id = int(args[0])
        layers = int(args[1])
        neurons = int(args[2])
        learning_rate = float(args[3])
        batch_size = int(args[4])
        activation = int(args[5])
        
        individual = [layers, neurons, learning_rate, batch_size, activation]

        with open(os.devnull, 'w') as out_null:
            with redirect_stdout(out_null):
                accuracy = agent.fitness_function(individual)
        
        with print_lock:
            print(f"{task_id} {accuracy:.17f}", file=print_file)
            with open(f"{uuid.uuid4()}.txt", '+a') as f:
                print(f"wrote", file=f, flush=True)

    except Exception as e:
        with print_lock:
            print_file.write(f"{args[0]} -inf\n") 
            print_file.flush()

def main(max_workers=8):
    agent = Agent()
    print_lock = threading.Lock()
    print_file = sys.stdout
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                
                # Dispatch the request to a background thread
                executor.submit(worker_task, agent, line, print_lock, print_file)

            except BufferError:
                # Timeout waiting for stdin, just loop and listen again
                continue
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

    max_workers = int(sys.argv[1]) if len(sys.argv) >= 2 else 8

    main()
