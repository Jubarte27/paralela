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

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#define MINUTES_TO_SECONDS 60
#define SECONDS_TO_MILLISECONDS 1000
#define TIMEOUT_MS (1 * MINUTES_TO_SECONDS * SECONDS_TO_MILLISECONDS)

const char TEMPLATE_DIR[] = "/tmp/paralela-fifo-XXXXXX";
const char FIFO_SUFFIX[] = "/fifo";
const int batch_choices[] = {32, 64, 128};

typedef enum {
  UNKNOWN = -1,
  SUCCESS = 0,
  TERMINATE = 1,
  DESTROY = 2,
  READ_FIFO = 3,
  READ_STD = 4,
  CREATE = 5,
} error_code_t;

typedef struct thread_state_t {
  struct subprocess_s* process;
  int fd;
  char temp_dir[sizeof(TEMPLATE_DIR) + 1];
  char fifo_path[sizeof(TEMPLATE_DIR) + sizeof(FIFO_SUFFIX) + 1];
} thread_state_t;

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

void fatal(error_code_t err, const char* where);
void cleanup(error_code_t err, const char* where);

//--------------- PROCESS ---------------

void wait_for_fifo(int fd);
bool read_from_fifo(int fd, char* buffer, size_t size);
void prepare_subprocess();

//--------------- RANDS ---------------

// inclusive
double random_between(double min, double max);
// inclusive
int random_randint(int min, int max);
int random_choice(const int* choices, int num_choices);
void select_random_distinct(int* arr, int n, int k);

//--------------- GA ---------------

void generate_population(Individual* population, int size);
double evaluate_fitness(Individual ind);
void selection(Individual* population, double* fitness_scores, int pop_size,
               Individual* parents, int num_parents);
void crossover(Individual* parents, int num_parents, Individual* offspring,
               int offspring_size);
void mutation(Individual* offspring, int offspring_size);

thread_state_t thread_state = {NULL, -1, "", ""};
#pragma omp threadprivate(thread_state)

#pragma omp declare reduction(pick_best:IndividualAccuracy : (            \
        omp_out = omp_in.accuracy > omp_out.accuracy ? omp_in : omp_out)) \
    initializer(omp_priv = {.accuracy = -INFINITY})

int main() {
  srand((unsigned int)time(NULL));
  omp_set_num_threads(max(omp_get_num_procs() - 2, 1));
  printf("Using %d threads\n", omp_get_max_threads());

  int num_generations = 10;
  int population_size = omp_get_max_threads();
  int num_parents = population_size / 4;
  int offspring_size = population_size - num_parents;

  // Memory allocation
  Individual* population = malloc(population_size * sizeof(Individual));
  Individual* parents = malloc(num_parents * sizeof(Individual));
  Individual* offspring = malloc(offspring_size * sizeof(Individual));
  Individual* next_population = malloc(population_size * sizeof(Individual));
  double* fitness_scores = malloc(population_size * sizeof(double));

  IndividualAccuracy best_individual_accuracy = {.accuracy = -INFINITY};

  generate_population(population, population_size);

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

#pragma omp for reduction(+ : sum_fitness) reduction(max : max_fitness) \
    reduction(pick_best : best_individual_accuracy)
      for (int i = 0; i < population_size; i++) {
        printf("%2d/%2d.start\n", i + 1, population_size);

        IndividualAccuracy individual_accuracy = {
            population[i], evaluate_fitness(population[i])};

        fitness_scores[i] = individual_accuracy.accuracy;

        printf("%2d/%2d.acc=%.17g\n", i + 1, population_size, individual_accuracy.accuracy);

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

void generate_population(Individual* population, int size) {
  for (int i = 0; i < size; i++) {
    population[i].layers = random_randint(1, 3);
    population[i].neurons = random_randint(8, 256 / population[i].layers);
    population[i].learning_rate = pow(10, random_between(-4.0, -1.0));
    population[i].batch_size = random_choice(batch_choices, 3);
    population[i].activation = random_randint(0, 2);
  }
}

// Subprocess call to the Python agent
double evaluate_fitness(Individual ind) {
  struct subprocess_s* p = thread_state.process;
  char result[64];
  double accuracy = 0.0;

  read_from_fifo(thread_state.fd, result, sizeof(result));
  if (strcmp(result, "listening") != 0) fatal(READ_FIFO, "evaluate_fitness_1");

  // Send hyperparameters
  fprintf(p->stdin_file, "%d %d %.17lg %d %d\n", ind.layers, ind.neurons,
          ind.learning_rate, ind.batch_size, ind.activation);
  fflush(p->stdin_file);

  wait_for_fifo(thread_state.fd);

  if (fgets(result, sizeof(result), p->stdout_file) == NULL ||
      sscanf(result, "%lg", &accuracy) != 1)
    fatal(READ_STD, "evaluate_fitness_2");

  return accuracy;
}

// Step 3: Selection (Tournament selection of size 2)
void selection(Individual* population, double* fitness_scores, int pop_size,
               Individual* parents, int num_parents) {
  for (int i = 0; i < num_parents; i++) {
    int idx1 = random_randint(0, pop_size - 1);
    int idx2 = random_randint(0, pop_size - 1);

    // while (idx1 == idx2) may get stuck
    if (pop_size <= 1) fatal(UNKNOWN, "selection");

    while (idx1 == idx2) idx2 = random_randint(0, pop_size - 1);

    if (fitness_scores[idx1] > fitness_scores[idx2]) {
      parents[i] = population[idx1];
    } else {
      parents[i] = population[idx2];
    }
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
  for (int i = 0; i < offspring_size; i++) {
    if ((double)rand() / RAND_MAX < 0.1) {  // 10% mutation chance
      Individual* ind = offspring + i;
      int mutation_index = random_randint(0, 4);
      switch (mutation_index) {
        case 0:
          ind->layers = random_randint(1, 3);
          ind->neurons = max(ind->neurons, 256 / ind->layers);
          break;
        case 1:
          ind->neurons = random_randint(8, 256 / ind->layers);
          break;
        case 2:
          ind->learning_rate = pow(10, random_between(-4.0, -1.0));
          break;
        case 3:
          ind->batch_size = random_choice(batch_choices, 3);
          break;
        case 4:
          ind->activation = random_randint(0, 2);
          break;
      }
    }
  }
}

void fatal(error_code_t err, const char* where) {
  cleanup(err, where);
  exit(err);
}

char* dump_stdout(FILE* p_stdout) {
  fseek(p_stdout, 0, SEEK_END);
  long fileSize = ftell(p_stdout);
  rewind(p_stdout);

  char* buff;
  buff = malloc(fileSize * sizeof(char) + 1);

  if (fread(buff, 1, fileSize, p_stdout) != fileSize)
    fatal(READ_STD, "dump_stdout");
  buff[fileSize] = '\0';

  return buff;
}

void clean_process(error_code_t err, const char* where);

void cleanup(error_code_t err, const char* where) {
  if (thread_state.process) {
    clean_process(err, where);
  }
  if (thread_state.fifo_path[0] != '\0') {
    unlink(thread_state.fifo_path);
    thread_state.fifo_path[0] = '\0';
  }
  if (thread_state.temp_dir[0] != '\0') {
    rmdir(thread_state.temp_dir);
    thread_state.temp_dir[0] = '\0';
  }
  if (thread_state.fd != -1) {
    close(thread_state.fd);
    thread_state.fd = -1;
  }
}

void clean_process(error_code_t err, const char* where) {
  struct subprocess_s* p = thread_state.process;
  switch (err) {
    case READ_STD:
      char* err = dump_stdout(p->stdout_file);
      fprintf(stderr, "Failed to read from stdout: \"%s\"\n", err);
      free(err);
      break;

    case READ_FIFO:
      fprintf(stderr, "Error reading from FIFO: %s\n", where);
      break;

    case TERMINATE:
      fprintf(stderr, "Failed to terminate subprocess: %s\n", where);
      break;

    case DESTROY:
      fprintf(stderr, "Failed to destroy subprocess: %s\n", where);
      break;

    case UNKNOWN:
      fprintf(stderr, "No sei: %s\n", where);
      break;

    default:
      fprintf(stderr, "Unknown error: %s\n", where);
      break;
  }
  fprintf(p->stdin_file, "exit\n");
  if (subprocess_terminate(p) != 0) {
    fprintf(stderr, "Failed to terminate subprocess: %d\n", p->child);
  }
  if (subprocess_destroy(p) != 0) {
    fprintf(stderr, "Failed to destroy subprocess: %d\n", p->child);
  }

  free(thread_state.process);
  thread_state.process = NULL;
}

// FIFOS

void wait_for_fifo(int fd) {
  struct pollfd poll_fd = {fd, POLLIN, 0};
  if (poll(&poll_fd, 1, TIMEOUT_MS) <= 0) fatal(READ_FIFO, "wait_for_fifo");
}

bool read_from_fifo(int fd, char* buffer, size_t size) {
  wait_for_fifo(fd);
  ssize_t bytes_read = read(fd, buffer, size - 1);
  if (bytes_read < 0) fatal(READ_FIFO, "read_from_fifo");
  buffer[bytes_read] = '\0';
  return true;
}

void prepare_subprocess() {
  strcpy(thread_state.temp_dir, TEMPLATE_DIR);
  if (!mkdtemp(thread_state.temp_dir)) {
    fatal(READ_FIFO, "prepare_subprocess_mkdtemp");
  }

  snprintf(thread_state.fifo_path, sizeof(thread_state.fifo_path), "%s%s",
           thread_state.temp_dir, FIFO_SUFFIX);
  if (mkfifo(thread_state.fifo_path, 0666) == -1)
    fatal(READ_FIFO, "prepare_subprocess_mkfifo");

  thread_state.fd = open(thread_state.fifo_path, O_RDWR);

  if (thread_state.fd == -1) fatal(READ_FIFO, "prepare_subprocess_open");

  thread_state.process = malloc(sizeof(struct subprocess_s));
  const char* command_line[] = {"python3", "agent.py", thread_state.fifo_path,
                                NULL};
  if (subprocess_create(command_line,
                        subprocess_option_inherit_environment |
                            subprocess_option_search_user_path |
                            subprocess_option_enable_async,
                        thread_state.process))
    fatal(CREATE, "prepare_subprocess_subprocess_create");
}

// RANDS

inline double random_double() { return ((double)rand() / (double)RAND_MAX); }

double random_between(double min, double max) {
  return min + random_double() * (max - min);
}

int random_randint(int min, int max) { return min + rand() % (max - min + 1); }

int random_choice(const int* choices, int num_choices) {
  return choices[random_randint(0, num_choices - 1)];
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