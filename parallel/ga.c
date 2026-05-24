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


typedef struct kill_me_on_exit_t {
  struct subprocess_s *process;
  char* fifo_path;
  char* temp_dir;
} kill_me_on_exit_t;


kill_me_on_exit_t kill_me_on_exit = {.process = NULL, .fifo_path = NULL, .temp_dir = NULL};
#pragma omp threadprivate(kill_me_on_exit)

void fatal(int err);
void cleanup();

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

void ensure_zero(int result, const char* operation) {
  if (result != 0) {  // an error occurred
    fprintf(stderr, "Error on %s: %d\n", operation, result);
    fatal(EXIT_FAILURE);
  }
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

void wait_for_fifo(int fd) {
  struct pollfd poll_fd = {fd, POLLIN, 0};
  if (poll(&poll_fd, 1, 30000) <= 0) {
    fprintf(stderr, "Timeout or error while waiting for FIFO input\n");
    fatal(EXIT_FAILURE);
  }
}

bool read_from_fifo(int fd, char* buffer, size_t size) {
  wait_for_fifo(fd);
  ssize_t bytes_read = read(fd, buffer, size - 1);
  if (bytes_read == -1) {
    fprintf(stderr, "Error reading from FIFO\n");
    fatal(EXIT_FAILURE);
  }
  if (bytes_read == 0) {
    fprintf(stderr, "End of file reached while reading from FIFO\n");
    fatal(EXIT_FAILURE);
  }
  buffer[bytes_read] = '\0';
  return true;
}

void prepare_subprocess() {

}

// Subprocess call to the Python agent
double evaluate_fitness(Individual ind) {
  char layers[16], neurons[16], learning_rate[32], batch_size[16],
      activation[16], fifo[256];

  snprintf(layers, sizeof(layers), "%d", ind.layers);
  snprintf(neurons, sizeof(neurons), "%d", ind.neurons);
  snprintf(learning_rate, sizeof(learning_rate), "%.17lg", ind.learning_rate);
  snprintf(batch_size, sizeof(batch_size), "%d", ind.batch_size);
  snprintf(activation, sizeof(activation), "%d", ind.activation);
  
  char template[] = "/tmp/paralela-fifo-XXXXXX";
  char *temp_dir = mkdtemp(template);
  kill_me_on_exit.temp_dir = temp_dir;

  snprintf(fifo, sizeof(fifo), "%s/fifo", temp_dir);
  if (mkfifo(fifo, 0666) == -1) {
    perror("mkfifo");
    fatal(EXIT_FAILURE);
  }
  kill_me_on_exit.fifo_path = fifo;
  int fd = open(fifo, O_RDWR);
  if (fd == -1) {
    perror("open fifo");
    fatal(EXIT_FAILURE);
  }

  const char* command_line[] = {"python3", "agent.py", fifo, NULL};
  ensure_zero(subprocess_create(command_line,
                                subprocess_option_inherit_environment |
                                    subprocess_option_search_user_path |
                                    subprocess_option_enable_async,
                                &kill_me_on_exit.process),
              "create");
  FILE* p_stdout = subprocess_stdout(&kill_me_on_exit.process);
  FILE* p_stdin = subprocess_stdin(&kill_me_on_exit.process);

  int TIMEOUT_SECONDS = 30;
  time_t end_time = time(NULL) + TIMEOUT_SECONDS;
  char fifo_command[512];
  char result[100];

  double accuracy = 0.0;

  while (time(NULL) < end_time &&
         read_from_fifo(fd, fifo_command, sizeof(fifo_command))) {
    if (strcmp(fifo_command, "exit\n") == 0) {
      break;
    }

    if (strcmp(fifo_command, "listening") == 0) {
      // Send hyperparameters to the agent
      fprintf(p_stdin, "%d %d %.17lg %d %d\n", ind.layers, ind.neurons,
              ind.learning_rate, ind.batch_size, ind.activation);
      fflush(p_stdin);

      wait_for_fifo(fd);

      if (fgets(result, sizeof(result), p_stdout) == NULL) {
        char* err = dump_stdout(p_stdout);
        fprintf(stderr, "Failed to read from stdout: \"%s\"\n", err);
        free(err);
        fatal(EXIT_FAILURE);
      }

      if (sscanf(result, "%lg", &accuracy) != 1) {
        fprintf(stderr, "Failed to read accuracy from agent: \"%s\"\n", result);
        fatal(EXIT_FAILURE);
      }

      // continue;
      fprintf(p_stdin, "exit\n");
      break;
    }

    fprintf(stderr, "I don't know what they want: \"%s\"\n", fifo_command);
    fatal(EXIT_FAILURE);
  }

  int process_return;

  ensure_zero(subprocess_join(&kill_me_on_exit.process, &process_return), "join");
  ensure_zero(process_return, "subprocess");
  ensure_zero(subprocess_destroy(&kill_me_on_exit.process), "destroy");

  kill_me_on_exit.process = NULL;
  cleanup();

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

// #pragma omp parallel {
//   #pragma omp single
  for (int generation = 0; generation < num_generations; generation++) {
    printf("\nGeneration %d\n", generation);

    double max_fitness = -1.0;
    double sum_fitness = 0.0;

// Evaluate fitness
#pragma omp parallel for reduction(+ : sum_fitness) \
    reduction(max : max_fitness)                    \
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

    printf("Best Gen Accuracy: %.17g | Avg Gen Accuracy: %.17g\n", max_fitness,
           sum_fitness / population_size);

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
  }

  printf("\n=== Optimization Complete ===\n");
  printf("Best Accuracy: %lg\n", best_individual_accuracy.accuracy);
  printf(
      "Best Hyperparameters: Layers=%d, Neurons=%d, LR=%lf, Batch=%d, Act=%d\n",
      best_individual_accuracy.individual.layers,
      best_individual_accuracy.individual.neurons,
      best_individual_accuracy.individual.learning_rate,
      best_individual_accuracy.individual.batch_size,
      best_individual_accuracy.individual.activation);

  free(population);
  free(parents);
  free(offspring);
  free(next_population);
  free(fitness_scores);

  return 0;
}
// }

void fatal(int err) {
  if (kill_me_on_exit.process && !subprocess_alive(kill_me_on_exit.process)) {
    fprintf(stderr, "He dead: %d\n", kill_me_on_exit.process->return_status);
    char stderr_buff[512];
    
    const int fd = fileno(kill_me_on_exit.process->stderr_file);
    const ssize_t bytes_read = read(fd, stderr_buff, sizeof(stderr_buff));
    if (bytes_read < 0) {
      fprintf(stderr, "Error reading from subprocess stdout\n");
    } else {
      stderr_buff[bytes_read] = '\0';
      printf("Subprocess stderr:\n%s\n", stderr_buff);
    }
  }
  cleanup();
  exit(err);
}

void cleanup() {
  if (kill_me_on_exit.process) {
    subprocess_terminate(kill_me_on_exit.process);
    subprocess_destroy(kill_me_on_exit.process);
    kill_me_on_exit.process = NULL;
  }
  if (kill_me_on_exit.fifo_path) {
    unlink(kill_me_on_exit.fifo_path);
    kill_me_on_exit.fifo_path = NULL;
  }
  if (kill_me_on_exit.temp_dir) {
    rmdir(kill_me_on_exit.temp_dir);
    kill_me_on_exit.temp_dir = NULL;
  }
}