#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <subprocess.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#define min(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

const char EXIT[] = "exit\n";
const char LISTENING[] = "listening\n";
#define ACCURACY_SIZE 20  // 0.84200000762939450\n
#define TIMEOUT_MS (2 * 60 * 1000)

typedef enum {
  UNKNOWN,
  SUCCESS,
  TERMINATE,
  DESTROY,
  READ_FILDES,
  READ_STD,
  CREATE,
  WRITE_FILDES,
  READ_FILDES_TIMEOUT
} error_code_t;

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

typedef struct double_range_t {
  double min;
  double max;
} double_range_t;

typedef struct int_range_t {
  int min;
  int max;
} int_range_t;

typedef struct thread_state_t {
  struct subprocess_s* process;
  char reusable_buffer[256];
} thread_state_t;

typedef struct limits_t {
  int_range_t layers;
  double_range_t learning_rate;
  int_range_t activation;
} limits_t;

thread_state_t thread_state = {NULL, ""};
#pragma omp threadprivate(thread_state)

const limits_t default_limits = {
    .layers = {1, 3},
    .learning_rate = {-4.0, -1.0},
    .activation = {0, 2},
};
const int_range_t default_neuron_limits[] = {{32, 256}, {8, 32}, {4, 16}};
const int default_batch_choices[] = {32, 64, 128};

const limits_t limits = default_limits;

const int_range_t* neuron_limits = default_neuron_limits;
int neuron_limits_len = sizeof(default_neuron_limits);

const int* batch_choices = default_batch_choices;
int batch_choices_len = sizeof(default_batch_choices);

//--------------- ERROR ---------------

void fatal(error_code_t err, const char* where);
void cleanup(error_code_t err, const char* where);

//--------------- PROCESS ---------------

bool wait_for_fildes(int fd);
bool wait_and_read(int fd, char* buffer, size_t size);
void prepare_subprocess();
int subprocess_join_timeout(pid_t child_pid, int timeout_ms, int* exit_status);

//--------------- RANDS ---------------

double random_between(double min, double max);  // inclusive
int random_randint(int min, int max);           // inclusive
int random_choice(const int* choices, int num_choices);
void select_random_distinct(int* arr, int n, int k);

int random_layers();
int random_neurons(int layers);
double random_learning_rate();
int random_batch_size();
int random_activation();

int clamp_neurons(int neurons, int layer);

//--------------- UNUSED BUT MAYBE USEFULL ---------------

__attribute__((unused)) static void dump_fd_to_stdout(int fd);
__attribute__((unused)) static void print_stack_trace();
__attribute__((unused)) static int get_cursor_row();

//--------------- GA ---------------

void generate_population(Individual* population, int size);
double evaluate_fitness(Individual ind);
void selection(Individual* population, double* fitness_scores, int pop_size,
               Individual* parents, int num_parents);
void crossover(Individual* parents, int num_parents, Individual* offspring,
               int offspring_size);
void mutation(Individual* offspring, int offspring_size);

#pragma omp declare reduction(pick_best:IndividualAccuracy : (            \
        omp_out = omp_in.accuracy > omp_out.accuracy ? omp_in : omp_out)) \
    initializer(omp_priv = {.accuracy = -INFINITY})

int main() {
  srand((unsigned int)time(NULL));
  omp_set_num_threads(max(omp_get_num_procs() / 2, 1));
  printf("Using %d threads\n", omp_get_max_threads());

  int num_generations = 10;
  // int pop_size = max(omp_get_max_threads() * 4, 4);
  int pop_size = 10;
  int num_parents = pop_size / 2;
  int offspring_size = pop_size - num_parents;

  Individual* population = malloc(pop_size * sizeof(Individual));
  Individual* next_population = malloc(pop_size * sizeof(Individual));
  Individual* parents = malloc(num_parents * sizeof(Individual));
  Individual* offspring = malloc(offspring_size * sizeof(Individual));
  double* fitness_scores = malloc(pop_size * sizeof(double));

  IndividualAccuracy best_individual_accuracy = {.accuracy = -INFINITY};

  generate_population(population, pop_size);

  double max_fitness = -1.0;
  double sum_fitness = 0.0;
#pragma omp parallel shared(max_fitness, sum_fitness)
  {
    for (int generation = 0; generation < num_generations; generation++) {
#pragma omp single
      {
        printf("\nGeneration %2d/%2d\n", generation + 1, num_generations);
        max_fitness = -1.0;
        sum_fitness = 0.0;
      }

#pragma omp for reduction(+ : sum_fitness) reduction(max : max_fitness) \
    reduction(pick_best : best_individual_accuracy) schedule(dynamic)
      for (int i = 0; i < pop_size; i++) {
        printf("!");
        fflush(stdout);

        IndividualAccuracy individual_accuracy = {
            population[i], evaluate_fitness(population[i])};

        fitness_scores[i] = individual_accuracy.accuracy;
        sum_fitness += individual_accuracy.accuracy;

        if (individual_accuracy.accuracy > max_fitness)
          max_fitness = individual_accuracy.accuracy;

        if (individual_accuracy.accuracy > best_individual_accuracy.accuracy) {
          best_individual_accuracy.accuracy = individual_accuracy.accuracy;
          best_individual_accuracy.individual = individual_accuracy.individual;
        }
        if (thread_state.process == NULL) {
          // timeout
          printf("<layer:%d|neurons:%d>", population[i].layers,
                 population[i].neurons);
        } else {
          printf(".");
        }
        fflush(stdout);
      }

#pragma omp single
      {
        printf("\n");
        for (int i = 0; i < pop_size; i++) {
          printf("%2d/%2d.acc=%.17f\n", i + 1, pop_size, fitness_scores[i]);
        }

        printf(
            "Best Gen Accuracy: %.17f | Avg Gen Accuracy: %.17f | Best so far: "
            "%.17f\n",
            max_fitness, sum_fitness / pop_size,
            best_individual_accuracy.accuracy);

        selection(population, fitness_scores, pop_size, parents, num_parents);
        crossover(parents, num_parents, offspring, offspring_size);
        mutation(offspring, offspring_size);

        // Construct next generation
        for (int i = 0; i < num_parents; i++) {
          next_population[i] = parents[i];
        }
        for (int i = 0; i < offspring_size; i++) {
          next_population[num_parents + i] = offspring[i];
        }

        // Prepare for the next generation
        Individual* temp = population;
        population = next_population;
        next_population = temp;
      }
    }
    cleanup(SUCCESS, "parallel_end");
  }
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

  free(population);
  free(parents);
  free(offspring);
  free(next_population);
  free(fitness_scores);

  return 0;
}

double evaluate_fitness(Individual ind) {
  if (thread_state.process == NULL) {
    prepare_subprocess();
  }

  struct subprocess_s* p = thread_state.process;
  double accuracy = 0.0;

  // Send hyperparameters
  int size;
  if ((size = sprintf(thread_state.reusable_buffer, "%d %d %.17lg %d %d\n",
                      ind.layers, ind.neurons, ind.learning_rate,
                      ind.batch_size, ind.activation)) <= 0) {
    fatal(UNKNOWN, "tudo errado");
  }
  if (write(p->stdin_fd, thread_state.reusable_buffer, size) <= 0)
    fatal(WRITE_FILDES, "write_params");

  thread_state.reusable_buffer[0] = '\0';

  int err;
  if (!wait_and_read(p->stdout_fd, thread_state.reusable_buffer,
                     ACCURACY_SIZE + 1)) {
    return -INFINITY;
  }
  if ((err = sscanf(thread_state.reusable_buffer, "%lf\n", &accuracy)) != 1) {
    cleanup(READ_FILDES, "read_accuracy");
    return -INFINITY;
  }

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

void generate_population(Individual* population, int size) {
  for (int i = 0; i < size; i++) {
    population[i].layers = random_layers();
    population[i].neurons = random_neurons(population[i].layers);
    population[i].learning_rate = random_learning_rate();
    population[i].batch_size = random_batch_size();
    population[i].activation = random_activation();
  }
}

void mutation(Individual* offspring, int offspring_size) {
  for (int i = 0; i < offspring_size; i++) {
    if ((double)rand() / RAND_MAX < 0.1) {  // 10% mutation chance
      Individual* ind = offspring + i;
      int mutation_index = random_randint(0, 4);
      switch (mutation_index) {
        case 0:
          ind->layers = random_layers();
          ind->neurons = clamp_neurons(ind->neurons, ind->layers);
          break;
        case 1:
          ind->neurons = random_neurons(ind->layers);
          break;
        case 2:
          ind->learning_rate = random_learning_rate();
          break;
        case 3:
          ind->batch_size = random_batch_size();
          break;
        case 4:
          ind->activation = random_activation();
          break;
      }
    }
  }
}
// errors

void clean_process(error_code_t err, const char* where);

void fatal(error_code_t err, const char* where) {
  cleanup(err, where);
  exit(err);
}

void cleanup(error_code_t err, const char* where) {
  if (thread_state.process) {
    clean_process(err, where);
  }
}

void kill_child() {
  struct subprocess_s* p = thread_state.process;
  if (subprocess_terminate(p) != 0) {
    fprintf(stderr, "Failed to terminate subprocess: %d\n", p->child);
  }
}

void clean_process(error_code_t err, const char* where) {
  struct subprocess_s* p = thread_state.process;
  char buff[256];
  switch (err) {
    case READ_STD:
      fprintf(stderr, "Failed to read from stdout\n");
      // dump_fd_to_stdout(p->stdout_fd);
      break;

    case READ_FILDES:
      fprintf(stderr, "Error reading from fildes: %s\n", where);
      // dump_fd_to_stdout(p->stderr_fd);
      // dump_fd_to_stdout(p->stdout_fd);
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

    case SUCCESS:
    case READ_FILDES_TIMEOUT:
      // silent
      break;

    default:
      fprintf(stderr, "Unknown error: %s\n", where);
      break;
  }

  int result;
  if (write(p->stdin_fd, EXIT, strlen(EXIT)) < 1) {
    fprintf(stderr, "Failed to write to subprocess stdin");
    kill_child();
  } else if (!subprocess_join_timeout(
                 p->child, 500,
                 &result)) {  // dead is ok, success is ok, timeout is not
    // you were given a chance, now you'll be killed
    kill_child();
  } else {
    // all is ok
  }

  if (subprocess_destroy(p) != 0) {
    fprintf(stderr, "Failed to destroy subprocess: %d\n", p->child);
  }

  if (thread_state.process) {
    free(thread_state.process);
    thread_state.process = NULL;
  }
}

// fildes

bool wait_for_fildes(int fd) {
  struct pollfd poll_fd = {fd, POLLIN, 0};
  int result = poll(&poll_fd, 1, TIMEOUT_MS);
  if (result < 0) {
    fatal(READ_FILDES, "wait_for_fildes");
  } else if (result == 0) {
    cleanup(READ_FILDES_TIMEOUT, "timeout");
    return false;
  }
  return true;
}

bool wait_and_read(int fd, char* buffer, size_t size) {
  if (!wait_for_fildes(fd)) return false;

  ssize_t bytes_read = read(fd, buffer, size - 1);
  if (bytes_read < 0) fatal(READ_FILDES, "wait_and_read");
  if (bytes_read == 0) fatal(READ_FILDES, "fd is dead");
  buffer[bytes_read] = '\0';

  return true;
}

// subproc

void prepare_subprocess() {
  thread_state.process = malloc(sizeof(struct subprocess_s));
  const char* command_line[] = {"python3", "agent.py", NULL};
  if (subprocess_create(command_line,
                        subprocess_option_inherit_environment |
                            subprocess_option_search_user_path |
                            subprocess_option_enable_async,
                        thread_state.process))
    fatal(CREATE, "prepare_subprocess_subprocess_create");
}

int subprocess_join_timeout(pid_t child_pid, int timeout_ms, int* exit_status) {
  int status;
  int elapsed_ms = 0;
  const int sleep_interval_ms = 10;  // Check every 10ms

  while (elapsed_ms < timeout_ms) {
    pid_t wpid = waitpid(child_pid, &status, WNOHANG);

    if (wpid == child_pid) {
      if (exit_status)
        *exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
      return 1;  // Success
    } else if (wpid == -1 && errno == ECHILD) {
      return -1;  // Child is dead
    }

    // Sleep for 10ms and yield CPU, then check again
    usleep(sleep_interval_ms * 1000);
    elapsed_ms += sleep_interval_ms;
  }

  return 0;  // Timeout
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
    int j = i + random_randint(0, n - i - 1);

    // Swap selected element arr[j] with current position arr[i]
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;

    // arr[i] now contains a unique random element
    printf("%d ", arr[i]);
  }
}

double random_learning_rate() {
  return pow(
      10, random_between(limits.learning_rate.min, limits.learning_rate.max));
}
int random_layers() {
  return random_randint(limits.layers.min, limits.layers.max);
}
int random_batch_size() {
  return random_choice(batch_choices, batch_choices_len);
}
int random_activation() {
  return random_randint(limits.activation.min, limits.activation.max);
}

int random_neurons(int layers) {
  if (layers > limits.layers.max || layers < limits.layers.min) {
    fatal(UNKNOWN, "layers");
  }

  int i = layers - limits.layers.min;
  int_range_t limit = neuron_limits[i];

  return random_randint(limit.min, limit.max);
}

int clamp_neurons(int neurons, int layer) {
  if (layer > limits.layers.max || layer < limits.layers.min) {
    fatal(UNKNOWN, "layer");
  }
  int i = layer - limits.layers.min;

  return min(max(neurons, neuron_limits[i].min), neuron_limits[i].max);
}

// unused

void dump_fd_to_stdout(int fd) {
  char buffer[4096];
  ssize_t bytes_read;
  // #pragma omp critical(stdout_lock)
  {
    // Read from the file descriptor in chunks until EOF (returns 0) or error
    // (returns -1)
    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
      ssize_t bytes_written = 0;

      // Ensure all bytes read are completely written to stdout
      while (bytes_written < bytes_read) {
        ssize_t written = write(STDOUT_FILENO, buffer + bytes_written,
                                bytes_read - bytes_written);

        if (written < 0) {
          perror("Error writing to stdout");
          break;
        }
        bytes_written += written;
      }
    }
  }

  // If read returns a negative value, an error occurred
  if (bytes_read < 0) {
    perror("Error reading from file descriptor");
  }
}

void print_stack_trace() {
  void* buffer[100];
  // Capture up to 100 stack frames
  int nptrs = backtrace(buffer, 100);

  // Convert addresses to human-readable strings
  char** strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }

  printf("Stack Trace (%d frames):\n", nptrs);
  for (int i = 0; i < nptrs; i++) {
    printf("%s\n", strings[i]);
  }
  fflush(stdout);

  free(strings);  // Free the list of strings
}

int get_cursor_row() {
  struct termios old_term, new_term;
  char buf[32];
  int i = 0;
  int row = 0, col = 0;

  // 1. Save current terminal settings
  if (tcgetattr(STDIN_FILENO, &old_term) != 0) {
    fatal(UNKNOWN, "get_cursor_row_tcgetattr");
    return -1;
  }
  new_term = old_term;

  // 2. Disable canonical mode and echo. This lets us read the terminal's
  //    response immediately without it showing up on the screen.
  new_term.c_lflag &= ~(ICANON | ECHO);
  if (tcsetattr(STDIN_FILENO, TCSANOW, &new_term) != 0) {
    fatal(UNKNOWN, "get_cursor_row_tcsetattr");
    return -1;
  }

  // 3. Request cursor position using the ANSI code \033[6n
  if (write(STDOUT_FILENO, "\033[6n", 4) != 4) {
    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    fatal(UNKNOWN, "Failed to write cursor position request");
    return -1;
  }

  // 4. Read the response from the terminal (Format: \033[ROW;COLR)
  while (i < sizeof(buf) - 1) {
    if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
    if (buf[i] == 'R') break;
    i++;
  }
  buf[i] = '\0';

  // 5. Restore the original terminal settings immediately
  tcsetattr(STDIN_FILENO, TCSANOW, &old_term);

  // 6. Parse the row out of the terminal's response
  if (sscanf(buf, "\033[%d;%dR", &row, &col) == 2) {
    return row;
  }

  return -1;  // Return error if parsing fails
}