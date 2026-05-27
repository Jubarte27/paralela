#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <subprocess.h>

#define VERSION_NAME "Sequential"

#include "args.hpp"
#include "base.hpp"
#include "python_proc.h"
#include "rand.hpp"
#include "types.hpp"

void generate_population(Individual *population, int size);
double evaluate_fitness(Individual ind, struct subprocess_s *subprocess);

int main(int argc, char *argv[]) {
  read_args(argc, argv);
  apply_args();
  int offspring_size = POP_SIZE - NUM_PARENTS;
  Individual *population = alloc_pop();
  Individual *parents = alloc_parents();
  Individual *offspring = alloc_offspring();
  Individual *next_population = alloc_pop();
  double *fitness_scores = alloc_scores();

  IndividualAccuracy best_individual_accuracy = {.accuracy = -INFINITY};

  generate_population(population, POP_SIZE);

  // Start Python ONCE ----
  const char *command_line[] = python_command_line("false", "-1");
  struct subprocess_s subprocess;
  if (subprocess_create(command_line,
                         subprocess_option_inherit_environment |
                             subprocess_option_search_user_path,
                         &subprocess))
    fatal(error_code_t::CREATE, "create");

  FILE *p_stdin = subprocess_stdin(&subprocess);
  FILE *p_stdout = subprocess_stdout(&subprocess);

  for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
    printf("\nGeneration %d\n", generation + 1);

    double max_fitness = -1.0;
    double sum_fitness = 0.0;

    // Evaluate fitness
    for (int i = 0; i < POP_SIZE; i++) {
      printf("Evaluating Individual %d/%d...\n", i + 1, POP_SIZE);

      double accuracy = evaluate_fitness(population[i], &subprocess);
      fitness_scores[i] = accuracy;

      report(population, i, accuracy);

      sum_fitness += accuracy;
      if (accuracy > max_fitness)
        max_fitness = accuracy;

      if (accuracy > best_individual_accuracy.accuracy) {
        best_individual_accuracy.accuracy = accuracy;
        best_individual_accuracy.individual = population[i];
      }
    }

    printf("Best Gen Accuracy: %.17g | Avg Gen Accuracy: %.17g\n", max_fitness,
           sum_fitness / POP_SIZE);

    for (int i = 0; i < NUM_PARENTS; i++)
      selection(population, fitness_scores, POP_SIZE, parents, i);
    for (int i = 0; i < offspring_size; i++)
      crossover(parents, NUM_PARENTS, offspring, i);
    for (int i = 0; i < offspring_size; i++)
      mutation(offspring, i);

    // Construct next generation
    for (int i = 0; i < NUM_PARENTS; i++) {
      next_population[i] = parents[i];
    }
    for (int i = 0; i < offspring_size; i++) {
      next_population[NUM_PARENTS + i] = offspring[i];
    }

    // Swap pointers for the next generation
    Individual *temp = population;
    population = next_population;
    next_population = temp;
  }

  // ---- Kill the snake----
  cleanup(error_code_t::SUCCESS, "success");
  goodbye(best_individual_accuracy);

  free(population);
  free(parents);
  free(offspring);
  free(next_population);
  free(fitness_scores);

  return 0;
}

void generate_population(Individual *population, int size) {
  for (int i = 0; i < size; i++) {
    generate_individual(population[i]);
  }
}

double evaluate_fitness(Individual ind, struct subprocess_s *subprocess) {
  FILE *p_stdin = subprocess_stdin(subprocess);
  FILE *p_stdout = subprocess_stdout(subprocess);

  fprintf(p_stdin, python_printf_string, python_args_in_order(0, ind));
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