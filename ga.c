#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include <subprocess.h>

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// Define the structure of an Individual
typedef struct
{
    int layers;
    int neurons;
    double learning_rate;
    int batch_size;
    int activation;
} Individual;

typedef struct
{
    Individual * individual;
    double accuracy;
} IndividualAccuracy;

#pragma omp declare reduction(pick_best : IndividualAccuracy : \
    (omp_out = omp_in.accuracy > omp_out.accuracy ? omp_in : omp_out)) initializer(omp_priv =  {.accuracy = -INFINITY})

// Utility function for random uniform double between min and max
double random_uniform(double min, double max)
{
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Utility function for random integer between min and max (inclusive)
int random_randint(int min, int max)
{
    return min + rand() % (max - min + 1);
}

// Step 2: Generate population
void generate_population(Individual *population, int size)
{
    int batch_choices[] = {32, 64, 128};
    for (int i = 0; i < size; i++)
    {
        population[i].layers = random_randint(1, 3);
        population[i].neurons = random_randint(8, 256 / population[i].layers);
        population[i].learning_rate = pow(10, random_uniform(-4.0, -1.0));
        population[i].batch_size = batch_choices[random_randint(0, 2)];
        population[i].activation = random_randint(0, 2);
    }
}

void ensure_zero(int result, const char *operation) {
    if (result != 0) { // an error occurred!
        fprintf(stderr, "Error on %s: %d\n", operation, result);
        exit(EXIT_FAILURE);
    }
}

char* dump_stdout(FILE* p_stdout) {
    
    fseek(p_stdout, 0, SEEK_END); 
    long fileSize = ftell(p_stdout);
    
    char* buff;
    buff = malloc(fileSize * sizeof(char) + 1);

    buff[fileSize] = '\0';

    return buff;
}

// Subprocess call to the Python agent
double evaluate_fitness(Individual ind)
{
    char layers[2], neurons[4], learning_rate[30], batch_size[4], activation[2];

    sprintf((char * restrict) &layers, "%d", ind.layers);
    sprintf((char * restrict) &neurons, "%d", ind.neurons);
    sprintf((char * restrict) &learning_rate, "%.17lg", ind.learning_rate);
    sprintf((char * restrict) &batch_size, "%d", ind.batch_size);
    sprintf((char * restrict) &activation, "%d", ind.activation);

    const char *command_line[] = {"python3", "agent.py", (const char *) &layers, (const char *) &neurons, (const char *) &learning_rate, (const char *) &batch_size, (const char *) &activation, NULL};
    struct subprocess_s subprocess;
    ensure_zero(subprocess_create(command_line, subprocess_option_inherit_environment | subprocess_option_search_user_path, &subprocess), "create");

    int process_return;
    int result = subprocess_join(&subprocess, &process_return);

    // FILE* p_stderr = subprocess_stderr(&subprocess);
    // char *err = dump_stdout(p_stderr);
    // fprintf(stderr, "subprocess err: \"%s\"\n", err);
    // free(err);
    // p_stderr = subprocess_stdout(&subprocess);
    // err = dump_stdout(p_stderr);
    // fprintf(stderr, "subprocess out: \"%s\"\n", err);
    // free(err);

    ensure_zero(process_return, "subprocess");
    ensure_zero(result, "join");

    FILE* p_stdout = subprocess_stdout(&subprocess);

    double accuracy = 0.0;

    char string[100]; // Buffer to store the extracted string
    if (fgets(string, 100, p_stdout) == NULL) {
        char *err = dump_stdout(p_stdout);
        fprintf(stderr, "Failed to read from stdout: \"%s\"\n", err);
        free(err);
        exit(EXIT_FAILURE);
    }

    if (sscanf(string, "%lg", &accuracy) != 1) {
        fprintf(stderr, "Failed to read accuracy from agent: \"%s\"\n", string);
        exit(EXIT_FAILURE);
    }

    ensure_zero(subprocess_destroy(&subprocess), "destroy");
    
    return accuracy;
}

void select_random_distinct(int *arr, int n, int k) {
    if (k > n) return; // Cannot pick more elements than exist

    for (int i = 0; i < k; i++) {
        // Pick a random index from the remaining pool
        int j = i + random_randint(0, n - 1);
        
        // Swap selected element arr[j] with current position arr[i]
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;

        // arr[i] now contains a unique random element
        printf("%d ", arr[i]);
    }
}

// Step 3: Selection (Tournament selection of size 2)
void selection(Individual *population, double *fitness_scores, int pop_size, Individual *parents, int num_parents)
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
void crossover(Individual *parents, int num_parents, Individual *offspring, int offspring_size)
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

        int cp = random_randint(0, 5);

        child.layers        = (cp > 0) ? p1.layers : p2.layers;
        child.neurons       = (cp > 1) ? p1.neurons : p2.neurons;
        child.learning_rate = (cp > 2) ? p1.learning_rate : p2.learning_rate;
        child.batch_size    = (cp > 3) ? p1.batch_size : p2.batch_size;
        child.activation    = (cp > 4) ? p1.activation : p2.activation;

        offspring[i] = child;
    }
}

// Step 5: Mutation
void mutation(Individual *offspring, int offspring_size)
{
    int batch_choices[] = {32, 64, 128};
    for (int i = 0; i < offspring_size; i++)
    {
        if ((double)rand() / RAND_MAX < 0.1)
        { // 10% mutation chance
            int mutation_index = random_randint(0, 4);
            if (mutation_index == 0)
            {
                offspring[i].layers = random_randint(1, 3);
                offspring[i].neurons = max(offspring[i].neurons, 256 / offspring[i].layers);
            }
            else if (mutation_index == 1)
            {
                offspring[i].neurons = random_randint(8, 256 / offspring[i].layers);
            }
            else if (mutation_index == 2)
            {
                offspring[i].learning_rate = pow(10, random_uniform(-4.0, -1.0));
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
int main()
{
    // Initialize random seed
    srand((unsigned int)time(NULL));

    int num_generations = 10;
    int population_size = 16;
    int num_parents = 5;
    int offspring_size = population_size - num_parents;

    // Memory allocation
    Individual *population = malloc(population_size * sizeof(Individual));
    Individual *parents = malloc(num_parents * sizeof(Individual));
    Individual *offspring = malloc(offspring_size * sizeof(Individual));
    Individual *next_population = malloc(population_size * sizeof(Individual));
    double *fitness_scores = malloc(population_size * sizeof(double));

    IndividualAccuracy best_individual_accuracy; 

    generate_population(population, population_size);
    
    // omp_set_num_threads(10);

    for (int generation = 0; generation < num_generations; generation++)
    {
        printf("\nGeneration %d\n", generation);

        double max_fitness = -1.0;
        double sum_fitness = 0.0;

        // Evaluate fitness
        #pragma omp parallel for reduction(+:sum_fitness) reduction(max:max_fitness) reduction(pick_best:best_individual_accuracy)
        for (int i = 0; i < population_size; i++)
        {
            printf("Evaluating Individual %d/%d...\n", i + 1, population_size);
            
            IndividualAccuracy individual_accuracy = {&population[i], evaluate_fitness(population[i])};

            fitness_scores[i] = individual_accuracy.accuracy;

            printf("Validation Accuracy: %.17g\n", individual_accuracy.accuracy);

            sum_fitness += individual_accuracy.accuracy;

            if (individual_accuracy.accuracy > max_fitness)
                max_fitness = individual_accuracy.accuracy;

            if (individual_accuracy.accuracy > best_individual_accuracy.accuracy)
            {
                best_individual_accuracy.accuracy = individual_accuracy.accuracy;
                best_individual_accuracy.individual = individual_accuracy.individual;
            }
        }

        printf("Best Gen Accuracy: %.17g | Avg Gen Accuracy: %.17g\n", max_fitness, sum_fitness / population_size);

        // Selection, Crossover, Mutation
        selection(population, fitness_scores, population_size, parents, num_parents);
        crossover(parents, num_parents, offspring, offspring_size);
        mutation(offspring, offspring_size);

        // Construct next generation
        for (int i = 0; i < num_parents; i++)
        {
            next_population[i] = parents[i];
        }
        for (int i = 0; i < offspring_size; i++)
        {
            next_population[num_parents + i] = offspring[i];
        }

        // Swap pointers for the next generation
        Individual *temp = population;
        population = next_population;
        next_population = temp;
    }

    printf("\n=== Optimization Complete ===\n");
    printf("Best Accuracy: %lg\n", best_individual_accuracy.accuracy);
    printf("Best Hyperparameters: Layers=%d, Neurons=%d, LR=%lf, Batch=%d, Act=%d\n",
           best_individual_accuracy.individual->layers,
           best_individual_accuracy.individual->neurons,
           best_individual_accuracy.individual->learning_rate,
           best_individual_accuracy.individual->batch_size,
           best_individual_accuracy.individual->activation);

    // Free memory
    free(population);
    free(parents);
    free(offspring);
    free(next_population);
    free(fitness_scores);

    return 0;
}