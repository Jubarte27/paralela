# import libraries
import os
import random
from contextlib import redirect_stdout

from tensorflow import keras
import sys
import tensorflow as tf

class Agent():
    # load fashion mnist dataset
    def __init__(self):
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # normalize data
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # reshape data
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

        # convert labels to categorical
        # there are 10 classes or labels in the data set
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        (self.X_train, self.y_train), (self.X_test, self.y_test) = (X_train, y_train), (X_test, y_test)
    

    # step 1 - fitness function
    def fitness_function(self, individual):
        # decode individuals
        neurons = int(individual[0])
        learning_rate = individual[1]
        batch_size = int(individual[2])
        activation = ['relu', 'tanh', 'sigmoid'][int(individual[3])]

        # print hyperparameters for reference
        print(f"Training with neurons: {neurons}, learning_rate: {learning_rate}, batch_size: {batch_size}, activation: {activation}")

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
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(neurons, activation=activation, input_shape=(784,)))
        model.add(keras.layers.Dense(10, activation='softmax'))

        # compile the model
        # using adam as optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
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

class GeneticAlgorithm:
    # step 2 - generate population
    def generate_population(self, size):
        population = []
        for _ in range(size):
            neurons = random.randint(32, 256)
            learning_rate = 10 ** random.uniform(-4, -1)
            batch_size = random.choice([32, 64, 128])
            activation = random.randint(0, 2)
            individual = [neurons, learning_rate, batch_size, activation]
            population.append(individual)
        return population

    # step 3 - selection (returns parents)
    def selection(self, population, fitness_scores, num_parents):
        parents = []
        for _ in range(num_parents):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitness_scores[idx1] > fitness_scores[idx2]:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])
        return parents

    # step 4 - crossover (returns offsprings)
    def crossover(self, parents, offspring_size):
        offspring = []
        for _ in range(offspring_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, len(parent1)-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)
        return offspring

    # step 5 - mutation (adds randomness)
    def mutation(self, offspring):
        for individual in offspring:
            if random.random() < 0.1:
                mutation_index = random.randint(0, len(individual)-1)
                if mutation_index == 0:
                    individual[mutation_index] = random.randint(32, 256)
                elif mutation_index == 1:
                    individual[mutation_index] = 10 ** random.uniform(-4, -1)
                elif mutation_index == 2:
                    individual[mutation_index] = random.choice([32, 64, 128])
                elif mutation_index == 3:
                    individual[mutation_index] = random.randint(0, 2)
        return offspring

    # step 6 - ga optimization
    def genetic_algorithm(self):

        agent = Agent()

        num_generations = 5
        population_size = 10
        num_parents = 5

        best_accuracies = []    
        average_accuracies = []

        population = self.generate_population(population_size)
        best_individual = None
        best_accuracy = 0

        for generation in range(num_generations):
            print(f"nGeneration {generation}")

            # evaluate fitness
            fitness_scores = []
            for idx, individual in enumerate(population):
                print(f"Evaluating Individual {idx+1}/{len(population)}")
                accuracy = agent.fitness_function(individual)
                fitness_scores.append(accuracy)
                print(f"Validation Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_individual = individual

            # record metrics
            best_accuracies.append(max(fitness_scores))
            average_accuracies.append(sum(fitness_scores) / len(fitness_scores))

            # selection
            parents = self.selection(population, fitness_scores, num_parents)

            # crossover
            offspring_size = population_size - len(parents)
            offspring = self.crossover(parents, offspring_size)

            # mutation
            offspring = self.mutation(offspring)

            # next generation of population
            population = parents + offspring
        
        return best_accuracies, best_accuracy, best_individual

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(1)

    # Parse arguments sent from C11
    neurons = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    activation = int(sys.argv[4])
    
    individual = [neurons, learning_rate, batch_size, activation]

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