#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <subprocess.h>
#include <sys/stat.h>
#include <time.h>

typedef enum {
  UNKNOWN = -1,
  SUCCESS = 0,
  TERMINATE = 1,
  DESTROY = 2,
  READ_FIFO = 3,
  READ_STD = 4,
  CREATE = 5,
} error_code_t;

typedef struct kill_me_on_exit_t {
  struct subprocess_s* process;
  int fd;
  char fifo_path[256];
  char* temp_dir;
} kill_me_on_exit_t;

kill_me_on_exit_t kill_me_on_exit = {
    .process = NULL, .fd = -1, .fifo_path = "", .temp_dir = NULL};
#pragma omp threadprivate(kill_me_on_exit)

void fatal(error_code_t err);
void cleanup(error_code_t err);

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

typedef struct {
  int layers;
  int neurons;
  double learning_rate;
  int batch_size;
  int activation;
} Individual;

typedef struct {
  Individual individual;
  double accuracy;
} IndividualAccuracy;

#pragma omp declare reduction(pick_best:IndividualAccuracy : (            \
        omp_out = omp_in.accuracy > omp_out.accuracy ? omp_in : omp_out)) \
    initializer(omp_priv = {.accuracy = -INFINITY})

// Utility function for random uniform double between min and max
double random_uniform(double min, double max) {
  return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Utility function for random integer between min and max (inclusive)
int random_randint(int min, int max) { return min + rand() % (max - min + 1); }

void generate_population(Individual* population, int size) {
  int batch_choices[] = {32, 64, 128};
  for (int i = 0; i < size; i++) {
    population[i].layers = random_randint(1, 3);
    population[i].neurons = random_randint(8, 256 / population[i].layers);
    population[i].learning_rate = pow(10, random_uniform(-4.0, -1.0));
    population[i].batch_size = batch_choices[random_randint(0, 2)];
    population[i].activation = random_randint(0, 2);
  }
}

void wait_for_fifo(int fd) {
  struct pollfd poll_fd = {fd, POLLIN, 0};
  if (poll(&poll_fd, 1, 30000) <= 0) fatal(READ_FIFO);
}

bool read_from_fifo(int fd, char* buffer, size_t size) {
  wait_for_fifo(fd);
  ssize_t bytes_read = read(fd, buffer, size - 1);
  if (bytes_read < 0) fatal(READ_FIFO);
  buffer[bytes_read] = '\0';
  return true;
}

void prepare_subprocess() {
  char template[] = "/tmp/paralela-fifo-XXXXXX";
  char fifo[256];
  kill_me_on_exit.temp_dir = mkdtemp(template);

  snprintf(kill_me_on_exit.fifo_path, sizeof(kill_me_on_exit.fifo_path),
           "%s/fifo", kill_me_on_exit.temp_dir);
  if (mkfifo(kill_me_on_exit.fifo_path, 0666) == -1) fatal(READ_FIFO);

  kill_me_on_exit.fd = open(kill_me_on_exit.fifo_path, O_RDWR);

  if (kill_me_on_exit.fd == -1) fatal(READ_FIFO);

  kill_me_on_exit.process = malloc(sizeof(struct subprocess_s));
  const char* command_line[] = {"python3", "agent.py",
                                kill_me_on_exit.fifo_path, NULL};
  if (subprocess_create(command_line,
                        subprocess_option_inherit_environment |
                            subprocess_option_search_user_path |
                            subprocess_option_enable_async,
                        kill_me_on_exit.process))
    fatal(CREATE);
}

// Subprocess call to the Python agent
double evaluate_fitness(Individual ind) {
  char layers[16], neurons[16], learning_rate[32], batch_size[16],
      activation[16];

  snprintf(layers, sizeof(layers), "%d", ind.layers);
  snprintf(neurons, sizeof(neurons), "%d", ind.neurons);
  snprintf(learning_rate, sizeof(learning_rate), "%.17lg", ind.learning_rate);
  snprintf(batch_size, sizeof(batch_size), "%d", ind.batch_size);
  snprintf(activation, sizeof(activation), "%d", ind.activation);

  FILE* p_stdin = subprocess_stdin(kill_me_on_exit.process);

  char fifo_command[32];
  read_from_fifo(kill_me_on_exit.fd, fifo_command, sizeof(fifo_command));
  if (strcmp(fifo_command, "listening") != 0) fatal(READ_FIFO);

  // Send hyperparameters
  fprintf(p_stdin, "%d %d %.17lg %d %d\n", ind.layers, ind.neurons,
          ind.learning_rate, ind.batch_size, ind.activation);
  fflush(p_stdin);

  wait_for_fifo(kill_me_on_exit.fd);

  char result[64];
  double accuracy = 0.0;
  FILE* p_stdout = subprocess_stdout(kill_me_on_exit.process);

  if (fgets(result, sizeof(result), p_stdout) == NULL ||
      sscanf(result, "%lg", &accuracy) != 1)
    fatal(READ_STD);

  return accuracy;
}

void select_random_distinct(int* arr, int n, int k) {
  if (k > n) return;  // Cannot pick more elements than exist

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
void selection(Individual* population, double* fitness_scores, int pop_size,
               Individual* parents, int num_parents) {
  for (int i = 0; i < num_parents; i++) {
    int idx1 = random_randint(0, pop_size - 1);
    int idx2 = random_randint(0, pop_size - 1);

    // Ensure distinct indices
    while (idx1 == idx2) idx2 = random_randint(0, pop_size - 1);

    if (fitness_scores[idx1] > fitness_scores[idx2])
      parents[i] = population[idx1];
    else
      parents[i] = population[idx2];
  }
}

void crossover(Individual* parents, int num_parents, Individual* offspring,
               int offspring_size) {
  for (int i = 0; i < offspring_size; i++) {
    int p1_idx = random_randint(0, num_parents - 1);
    int p2_idx = random_randint(0, num_parents - 1);

    while (p1_idx == p2_idx && num_parents > 1)
      p2_idx = random_randint(0, num_parents - 1);

    Individual p1 = parents[p1_idx];
    Individual p2 = parents[p2_idx];
    Individual child;

    int cp = random_randint(0, 5);

    child.layers = (cp > 0) ? p1.layers : p2.layers;
    child.neurons = (cp > 1) ? p1.neurons : p2.neurons;
    child.learning_rate = (cp > 2) ? p1.learning_rate : p2.learning_rate;
    child.batch_size = (cp > 3) ? p1.batch_size : p2.batch_size;
    child.activation = (cp > 4) ? p1.activation : p2.activation;

    offspring[i] = child;
  }
}

void mutation(Individual* offspring, int offspring_size) {
  int batch_choices[] = {32, 64, 128};
  for (int i = 0; i < offspring_size; i++) {
    if ((double)rand() / RAND_MAX < 0.1) {  // 10% mutation chance
      int mutation_index = random_randint(0, 4);
      if (mutation_index == 0) {
        offspring[i].layers = random_randint(1, 3);
        offspring[i].neurons =
            max(offspring[i].neurons, 256 / offspring[i].layers);
      } else if (mutation_index == 1)
        offspring[i].neurons = random_randint(8, 256 / offspring[i].layers);
      else if (mutation_index == 2)
        offspring[i].learning_rate = pow(10, random_uniform(-4.0, -1.0));
      else if (mutation_index == 3)
        offspring[i].batch_size = batch_choices[random_randint(0, 2)];
      else if (mutation_index == 4)
        offspring[i].activation = random_randint(0, 2);
    }
  }
}

int main() {
  // Initialize random seed
  srand((unsigned int)time(NULL));

  int num_generations = 5;
  int population_size = max(omp_get_num_procs() - 2, 1);
  int num_parents = 5;
  int offspring_size = population_size - num_parents;

  // Memory allocation
  Individual* population = malloc(population_size * sizeof(Individual));
  Individual* parents = malloc(num_parents * sizeof(Individual));
  Individual* offspring = malloc(offspring_size * sizeof(Individual));
  Individual* next_population = malloc(population_size * sizeof(Individual));
  double* fitness_scores = malloc(population_size * sizeof(double));

  IndividualAccuracy best_individual_accuracy = {.accuracy = -INFINITY};

  generate_population(population, population_size);

  omp_set_num_threads(10);

  double max_fitness = -1.0;
  double sum_fitness = 0.0;
#pragma omp parallel shared(max_fitness, sum_fitness)
  {
    prepare_subprocess();
    for (int generation = 0; generation < num_generations; generation++) {
#pragma omp single
      {
        printf("\nGeneration %d\n", generation);
        max_fitness = -1.0;
        sum_fitness = 0.0;
      }

// Evaluate fitness
#pragma omp for reduction(+ : sum_fitness) reduction(max : max_fitness) \
    reduction(pick_best : best_individual_accuracy)
      for (int i = 0; i < population_size; i++) {
        printf("Evaluating Individual %d/%d...\n", i + 1, population_size);

        IndividualAccuracy individual_accuracy = {
            population[i], evaluate_fitness(population[i])};

        fitness_scores[i] = individual_accuracy.accuracy;

        printf("Validation Accuracy: %.17g\n", individual_accuracy.accuracy);

        sum_fitness += individual_accuracy.accuracy;

        if (individual_accuracy.accuracy > max_fitness)
          max_fitness = individual_accuracy.accuracy;

        if (individual_accuracy.accuracy > best_individual_accuracy.accuracy) {
          best_individual_accuracy.accuracy = individual_accuracy.accuracy;
          best_individual_accuracy.individual = individual_accuracy.individual;
        }
      }

#pragma omp single
      {
        printf("Best Gen Accuracy: %.17g | Avg Gen Accuracy: %.17g\n",
               max_fitness, sum_fitness / population_size);

        selection(population, fitness_scores, population_size, parents,
                  num_parents);
        crossover(parents, num_parents, offspring, offspring_size);
        mutation(offspring, offspring_size);

        // Construct next generation
        for (int i = 0; i < num_parents; i++) next_population[i] = parents[i];
        for (int i = 0; i < offspring_size; i++)
          next_population[num_parents + i] = offspring[i];

        // Prepare for the next generation
        Individual* temp = population;
        population = next_population;
        next_population = temp;
        printf("\n=== Optimization Complete ===\n");
        printf("Best Accuracy: %lg\n", best_individual_accuracy.accuracy);
        printf(
            "Best Hyperparameters: Layers=%d, Neurons=%d, LR=%lf, Batch=%d, "
            "Act=%d\n",
            best_individual_accuracy.individual.layers,
            best_individual_accuracy.individual.neurons,
            best_individual_accuracy.individual.learning_rate,
            best_individual_accuracy.individual.batch_size,
            best_individual_accuracy.individual.activation);
      }
    }
  }

  free(population);
  free(parents);
  free(offspring);
  free(next_population);
  free(fitness_scores);

  return 0;
}

void fatal(error_code_t err) {
  cleanup(err);
  exit(err);
}

char* dump_stdout(FILE* p_stdout) {
  fseek(p_stdout, 0, SEEK_END);
  long fileSize = ftell(p_stdout);
  rewind(p_stdout);

  char* buff;
  buff = malloc(fileSize * sizeof(char) + 1);

  fread(buff, 1, fileSize, p_stdout);
  buff[fileSize] = '\0';

  return buff;
}

void cleanup(error_code_t err) {
  if (kill_me_on_exit.process) {
    switch (err) {
      case READ_STD:
        char* err = dump_stdout(subprocess_stdout(kill_me_on_exit.process));
        fprintf(stderr, "Failed to read from stdout: \"%s\"\n", err);
        free(err);
        break;

      case READ_FIFO:
        fprintf(stderr, "Error reading from FIFO\n");
        break;

      case TERMINATE:
        fprintf(stderr, "Failed to terminate subprocess\n");
        break;

      case DESTROY:
        fprintf(stderr, "Failed to destroy subprocess\n");
        break;

      case UNKNOWN:
        fprintf(stderr, "No sei\n");
        break;

      default:
        break;
    }
    fprintf(subprocess_stdin(kill_me_on_exit.process), "exit\n");
    if (subprocess_terminate(kill_me_on_exit.process) != 0) {
      fprintf(stderr, "Failed to terminate subprocess: %d\n",
              kill_me_on_exit.process->child);
    }
    if (subprocess_destroy(kill_me_on_exit.process) != 0) {
      fprintf(stderr, "Failed to destroy subprocess: %d\n",
              kill_me_on_exit.process->child);
    }

    free(kill_me_on_exit.process);
    kill_me_on_exit.process = NULL;
  }
  if (strlen(kill_me_on_exit.fifo_path) > 0) {
    unlink(kill_me_on_exit.fifo_path);
    kill_me_on_exit.fifo_path[0] = '\0';
  }
  if (kill_me_on_exit.temp_dir) {
    rmdir(kill_me_on_exit.temp_dir);
    kill_me_on_exit.temp_dir = NULL;
  }
  if (kill_me_on_exit.fd != -1) {
    close(kill_me_on_exit.fd);
    kill_me_on_exit.fd = -1;
  }
}
