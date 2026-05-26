#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <subprocess.h>

#include "args.h"
#include "types.h"
#include "rand.h"

// Step 2: Generate population
void generate_population(Individual* population, int size)
{
    int batch_choices[] = { 32, 64, 128 };
    for (int i = 0; i < size; i++)
    {
        population[i].layers = random_randint(1, 4); // 1 to 4 layers
        population[i].neurons = random_randint(32, 256);
        population[i].learning_rate = pow(10, random_between(-4.0, -1.0));
        population[i].batch_size = batch_choices[random_randint(0, 2)];
        population[i].activation = random_randint(0, 2);
    }
}

void ensure_zero(int result, const char* operation) {
    if (result != 0) { // an error occurred!
        fprintf(stderr, "Error on %s: %d\n", operation, result);
        exit(EXIT_FAILURE);
    }
}

double evaluate_fitness(Individual ind, struct subprocess_s* subprocess) {
    FILE* p_stdin = subprocess_stdin(subprocess);
    FILE* p_stdout = subprocess_stdout(subprocess);

    fprintf(p_stdin, "%d %d %d %.17g %d %d\n",
        0,
        ind.layers,
        ind.neurons,
        ind.learning_rate,
        ind.batch_size,
        ind.activation);
    fflush(p_stdin);

    double accuracy = 0.0;

    char string[100]; // Buffer to store the extracted string
    if (fgets(string, 100, p_stdout) == NULL) {
        fprintf(stderr, "Failed to read from stdout: \"%s\"\n", string);
        exit(EXIT_FAILURE);
    }

    if (sscanf(string, "0 %lf", &accuracy) != 1) {
        fprintf(stderr, "Failed to read accuracy from agent: \"%s\"\n", string);
        exit(EXIT_FAILURE);
    }

    return accuracy;
}

// Step 3: Selection (Tournament selection of size 2)
void selection(Individual* population, double* fitness_scores, int pop_size, Individual* parents, int num_parents)
{
    for (int i = 0; i < num_parents; i++)
    {
        int idx1 = random_randint(0, pop_size - 1);
        int idx2 = random_randint(0, pop_size - 1);

        // Ensure distinct indices
        while (idx1 == idx2)
        {
            idx2 = random_randint(0, pop_size - 1);
        }

        if (fitness_scores[idx1] > fitness_scores[idx2])
        {
            parents[i] = population[idx1];
        }
        else
        {
            parents[i] = population[idx2];
        }
    }
}

// Step 4: Crossover
void crossover(Individual* parents, int num_parents, Individual* offspring, int offspring_size)
{
    for (int i = 0; i < offspring_size; i++)
    {
        int p1_idx = random_randint(0, num_parents - 1);
        int p2_idx = random_randint(0, num_parents - 1);

        while (p1_idx == p2_idx && num_parents > 1)
        {
            p2_idx = random_randint(0, num_parents - 1);
        }

        Individual p1 = parents[p1_idx];
        Individual p2 = parents[p2_idx];
        Individual child;

        // Random crossover point between 1 and 4 (inclusive) for 5 genes
        int cp = random_randint(1, 4);

        child.layers = (cp > 0) ? p1.layers : p2.layers;
        child.neurons = (cp > 1) ? p1.neurons : p2.neurons;
        child.learning_rate = (cp > 2) ? p1.learning_rate : p2.learning_rate;
        child.batch_size = (cp > 3) ? p1.batch_size : p2.batch_size;
        child.activation = (cp > 4) ? p1.activation : p2.activation;

        offspring[i] = child;
    }
}

// Step 5: Mutation
void mutation(Individual* offspring, int offspring_size)
{
    int batch_choices[] = { 32, 64, 128 };
    for (int i = 0; i < offspring_size; i++)
    {
        if ((double)rand() / RAND_MAX < 0.1) // 10% mutation chance
        {
            int mutation_index = random_randint(0, 4);
            if (mutation_index == 0)
            {
                offspring[i].layers = random_randint(1, 4);
            }
            else if (mutation_index == 1)
            {
                offspring[i].neurons = random_randint(32, 256);
            }
            else if (mutation_index == 2)
            {
                offspring[i].learning_rate = pow(10, random_between(-4.0, -1.0));
            }
            else if (mutation_index == 3)
            {
                offspring[i].batch_size = batch_choices[random_randint(0, 2)];
            }
            else if (mutation_index == 4)
            {
                offspring[i].activation = random_randint(0, 2);
            }
        }
    }
}

// Step 6: GA Optimization
int main(int argc, char* argv[]) {
    read_args(argc, argv);
    apply_args();
    // Initialize random seed
    srand((unsigned int)time(NULL));

    int offspring_size = POP_SIZE - NUM_PARENTS;

    // Memory allocation
    Individual* population = malloc(POP_SIZE * sizeof(Individual));
    Individual* parents = malloc(NUM_PARENTS * sizeof(Individual));
    Individual* offspring = malloc(offspring_size * sizeof(Individual));
    Individual* next_population = malloc(POP_SIZE * sizeof(Individual));
    double* fitness_scores = calloc(POP_SIZE, sizeof(double));

    Individual best_individual;
    double best_accuracy = -1.0;

    generate_population(population, POP_SIZE);

    // Start Python ONCE ----
    const char* command_line[] = { "python3", "py/agent.py", "false", "8", NULL };
    struct subprocess_s subprocess;
    ensure_zero(subprocess_create(command_line, subprocess_option_inherit_environment | subprocess_option_search_user_path, &subprocess), "create");

    FILE* p_stdin = subprocess_stdin(&subprocess);
    FILE* p_stdout = subprocess_stdout(&subprocess);

    for (int generation = 0; generation < NUM_GENERATIONS; generation++)
    {
        printf("\nGeneration %d\n", generation + 1);

        double max_fitness = -1.0;
        double sum_fitness = 0.0;

        // Evaluate fitness
        for (int i = 0; i < POP_SIZE; i++)
        {
            printf("Evaluating Individual %d/%d...\n", i + 1, POP_SIZE);

            double accuracy = evaluate_fitness(population[i], &subprocess);
            fitness_scores[i] = accuracy;

            printf("Validation Accuracy: %.17g\n", accuracy);

            sum_fitness += accuracy;
            if (accuracy > max_fitness)
                max_fitness = accuracy;

            if (accuracy > best_accuracy)
            {
                best_accuracy = accuracy;
                best_individual = population[i];
            }
        }

        printf("Best Gen Accuracy: %.17g | Avg Gen Accuracy: %.17g\n",
            max_fitness, sum_fitness / POP_SIZE);

        // Selection, Crossover, Mutation
        selection(population, fitness_scores, POP_SIZE, parents, NUM_PARENTS);
        crossover(parents, NUM_PARENTS, offspring, offspring_size);
        mutation(offspring, offspring_size);

        // Construct next generation
        for (int i = 0; i < NUM_PARENTS; i++) { next_population[i] = parents[i]; }
        for (int i = 0; i < offspring_size; i++) { next_population[NUM_PARENTS + i] = offspring[i]; }

        // Swap pointers for the next generation
        Individual* temp = population;
        population = next_population;
        next_population = temp;
    }

    printf("\n=== Optimization Complete ===\n");
    printf("Best Accuracy: %.4f\n", best_accuracy);
    printf("Best Hyperparameters: Layers=%d, Neurons=%d, LR=%lf, Batch=%d, Act=%d\n",
        best_individual.layers, best_individual.neurons, best_individual.learning_rate,
        best_individual.batch_size, best_individual.activation);

    // ---- Kill Python Safely ----
    fprintf(p_stdin, "exit\n");
    fflush(p_stdin);

    int process_return;
    ensure_zero(subprocess_join(&subprocess, &process_return), "join");
    ensure_zero(subprocess_destroy(&subprocess), "destroy");

    free(population);
    free(parents);
    free(offspring);
    free(next_population);
    free(fitness_scores);

    return 0;
}