#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <subprocess.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "types.hpp"
#include "rand.hpp"
#include "args.hpp"

const char EXIT[] = "exit\n";

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

#define TIMEOUT_MS (5 * 60 * 1000) // 2m
#define SMALL_TIMEOUT_MS (500)

struct subprocess_s global_agent;
bool agent_running = false;

static char io_buffer[4096];
static ssize_t io_buffer_len = 0;
static ssize_t io_buffer_pos = 0;

//--------------- ERROR ---------------

void fatal(error_code_t err, const char *where);
void cleanup(error_code_t err, const char *where);

//--------------- PROCESS ---------------

void prepare_subprocess();
int subprocess_join_timeout(pid_t child_pid, int timeout_ms, int *exit_status);
int read_line_timeout(int fd, char *buffer, size_t max_size);

//--------------- GA ---------------

void send_evaluation_request(int id, Individual ind);

void generate_population(Individual *population, int i);
void selection(Individual *population, double *fitness_scores, int pop_size,
               Individual *parents, int i);
void crossover(const Individual *parents, int num_parents,
               Individual *offspring, int i);
void mutation(Individual *offspring, int i);

int main(int argc, char *argv[]) {
  read_args(argc, argv);
  apply_args();

  srand((unsigned int)time(NULL));

  int offspring_size = POP_SIZE - NUM_PARENTS;

  Individual *population = (Individual *)malloc(POP_SIZE * sizeof(Individual));
  Individual *next_population =
      (Individual *)malloc(POP_SIZE * sizeof(Individual));
  Individual *parents = (Individual *)malloc(NUM_PARENTS * sizeof(Individual));
  Individual *offspring =
      (Individual *)malloc(offspring_size * sizeof(Individual));
  double *fitness_scores = (double *)malloc(POP_SIZE * sizeof(double));

  IndividualAccuracy best_individual_accuracy = {.accuracy = -INFINITY};

  prepare_subprocess();
#pragma omp parallel shared(best_individual_accuracy, population,              \
                                fitness_scores, io_buffer, io_buffer_len,      \
                                io_buffer_pos, global_agent, agent_running)
  {
#pragma omp master
    {
      for (int i = 0; i < POP_SIZE; i++) {
        generate_population(population, i);
      }
      for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        printf("\nGeneration %2d/%2d\n", generation + 1, NUM_GENERATIONS);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < POP_SIZE; i++) {
          send_evaluation_request(i, population[i]);
          printf("?");
          fflush(stdout);
        }

        for (int i = 0; i < POP_SIZE; i++) {
          char *task_buffer = (char *)malloc(256);
          if (read_line_timeout(global_agent.stdout_fd, task_buffer, 256) ==
              1) {
#pragma omp task firstprivate(task_buffer) shared(fitness_scores)
            {
              int response_id;
              double response_accuracy;

              if (sscanf(task_buffer, "%d %lf", &response_id,
                         &response_accuracy) == 2) {
                fitness_scores[response_id] = response_accuracy;
                printf(".");
                fflush(stdout);
              } else {
                fatal(UNKNOWN, "sscanf");
              }
              free(task_buffer);
            }
          } else {
            free(task_buffer);
            fatal(READ_FILDES, "Agent failed to respond or pipe broke.");
          }
        }

#pragma omp taskwait
        printf("\n");
        double max_fitness = -INFINITY;
        double sum_fitness = 0.0;
        for (int i = 0; i < POP_SIZE; i++) {
          printf("%2d/%2d.acc=%.17f\n", i + 1, POP_SIZE, fitness_scores[i]);
          double score = fitness_scores[i];
          sum_fitness += score;
          if (score > max_fitness)
            max_fitness = score;

          if (score > best_individual_accuracy.accuracy) {
            best_individual_accuracy.accuracy = score;
            best_individual_accuracy.individual = population[i];
          }
        }

        printf(
            "Best Gen Accuracy: %.17f | Avg Gen Accuracy: %.17f | Best so far: "
            "%.17f\n",
            max_fitness, sum_fitness / POP_SIZE,
            best_individual_accuracy.accuracy);

        for (int i = 0; i < NUM_PARENTS; i++) {
          // someone may end up procreating with themselves
          selection(population, fitness_scores, POP_SIZE, parents, i);
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < POP_SIZE; i++) {
          if (i < NUM_PARENTS) {
            next_population[i] = parents[i];
          } else {
            crossover(parents, NUM_PARENTS, next_population,
                      i);                 // set next_population[i]
            mutation(next_population, i); // mutate next_population[i]
          }
        }

        // reuse buffers, "round robin"
        Individual *temp = population;
        population = next_population;
        next_population = temp;
      }
    }
  }
  cleanup(SUCCESS, "success");

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

void send_evaluation_request(int id, Individual ind) {
  if (!agent_running)
    fatal(UNKNOWN, "Agent is dead.");

  char buffer[256];
  int size = snprintf(buffer, sizeof(buffer), "%d %d %d %.17lg %d %d\n", id,
                      ind.layers, ind.neurons, ind.learning_rate,
                      ind.batch_size, ind.activation);
  if (size <= 0)
    fatal(UNKNOWN, "String format failed");

    // Prevent multiple threads from writing to the pipe at the exact same
    // microsecond
#pragma omp critical(pipe_write)
  {
    if (write(global_agent.stdin_fd, buffer, size) <= 0) {
      fatal(WRITE_FILDES, "send_evaluation_request");
    }
  }
}

bool wait_for_it(int fd, int timeout) {
  struct pollfd poll_fd = {fd, POLLIN, 0};
  int result = poll(&poll_fd, 1, timeout);
  if (result < 0) {
    return false;
  }
  if (result == 0)
    return false;
  return true;
}

// 0: buffer is small  1: success  -1: error
int read_line_timeout(int fd, char *buffer, size_t max_size) {
  size_t i = 0;

  int ret = 0;
  while (i < max_size - 1) {
    // from buffer first
    if (io_buffer_pos < io_buffer_len) {
      char c = io_buffer[io_buffer_pos++];
      buffer[i++] = c;
      if (c == '\n') {
        ret = 1;
        break;
      }
      continue;
    }

    if (!wait_for_it(fd, TIMEOUT_MS)) {
      // no newline
      return -1;
    }

    io_buffer_len = read(fd, io_buffer, sizeof(io_buffer));
    if (io_buffer_len <= 0)
      return false; // EOF or Pipe broken
    io_buffer_pos = 0;
  }

  buffer[i] = '\0';
  return true;
}

//--------------- GA ---------------

void selection(Individual *population, double *fitness_scores, int pop_size,
               Individual *parents, int i) {
  int idx1 = random_randint(0, pop_size - 1);
  int idx2 = random_randint(0, pop_size - 1);

  if (pop_size <= 1)
    fatal(UNKNOWN, "selection");

  while (idx1 == idx2)
    idx2 = random_randint(0, pop_size - 1);

  if (fitness_scores[idx1] > fitness_scores[idx2]) {
    parents[i] = population[idx1];
  } else {
    parents[i] = population[idx2];
  }
}

void crossover(const Individual *parents, int num_parents,
               Individual *offspring, int i) {
  int p1_idx = random_randint(0, num_parents - 1);
  int p2_idx = random_randint(0, num_parents - 1);

  while (p1_idx == p2_idx && num_parents > 1)
    p2_idx = random_randint(0, num_parents - 1);

  const Individual *p1 = parents + p1_idx;
  const Individual *p2 = parents + p2_idx;
  Individual child;

  int cp = random_randint(1, 4); // no clones
  child.layers = (cp > 0) ? p1->layers : p2->layers;
  child.neurons = (cp > 1) ? p1->neurons : p2->neurons;
  child.learning_rate = (cp > 2) ? p1->learning_rate : p2->learning_rate;
  child.batch_size = (cp > 3) ? p1->batch_size : p2->batch_size;
  child.activation = (cp > 4) ? p1->activation : p2->activation;

  offspring[i] = child;
}

void generate_population(Individual *population, int i) {
  population[i].layers = random_layers();
  population[i].neurons = random_neurons(population[i].layers);
  population[i].learning_rate = random_learning_rate();
  population[i].batch_size = random_batch_size();
  population[i].activation = random_activation();
}

void mutation(Individual *offspring, int i) {
  if ((double)rand() / RAND_MAX < 0.1) {
    Individual *ind = offspring + i;
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

//--------------- SUBPROC ---------------

void prepare_subprocess() {
  char n_workers[4];
  snprintf(n_workers, sizeof(n_workers), "%d", OUR_THREADS);
  const char *command_line[] = {"python3", "-u",      "py/agent.py",
                                "true",    n_workers, NULL};
  if (subprocess_create(command_line,
                        subprocess_option_inherit_environment |
                            subprocess_option_search_user_path |
                            subprocess_option_enable_async,
                        &global_agent)) {
    fatal(CREATE, "Failed to spawn Python agent");
  }
  // int fd = global_agent.stdout_fd;
  // int flags = fcntl(fd, F_GETFL, 0);       // Get current flags
  // fcntl(fd, F_SETFL, flags | O_NONBLOCK); // Add non-blocking flag
  agent_running = true;
}

int subprocess_join_timeout(pid_t child_pid, int timeout_ms, int *exit_status) {
  int status;
  int elapsed_ms = 0;
  const int sleep_interval_ms = 10;

  while (elapsed_ms < timeout_ms) {
    pid_t wpid = waitpid(child_pid, &status, WNOHANG);

    if (wpid == child_pid) {
      if (exit_status)
        *exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
      return 1;
    } else if (wpid == -1 && errno == ECHILD) {
      return -1;
    }

    usleep(sleep_interval_ms * 1000);
    elapsed_ms += sleep_interval_ms;
  }
  return 0;
}

void kill_child() {
  if (subprocess_terminate(&global_agent) != 0) {
    fprintf(stderr, "Failed to terminate subprocess.\n");
  }
}

void clean_process(error_code_t err, const char *where) {
  if (!agent_running)
    return;

  if (err != SUCCESS && err != READ_FILDES_TIMEOUT) {
    fprintf(stderr, "Fatal Error [%d] at %s\n", err, where);
  }

  int result;
  if (write(global_agent.stdin_fd, EXIT, strlen(EXIT)) < 1) {
    kill_child();
  } else if (!subprocess_join_timeout(global_agent.child, 500, &result)) {
    kill_child();
  }

  subprocess_destroy(&global_agent);
  agent_running = false;
}

void fatal(error_code_t err, const char *where) {
  cleanup(err, where);
  exit(err);
}

void cleanup(error_code_t err, const char *where) { clean_process(err, where); }
