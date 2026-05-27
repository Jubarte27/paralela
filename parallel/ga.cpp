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

#define VERSION_NAME "Parallel"

#include "args.hpp"
#include "base.hpp"
#include "python_proc.h"
#include "rand.hpp"
#include "types.hpp"

#define TIMEOUT_MS (1 * 60 * 1000) // 5m
#define SMALL_TIMEOUT_MS (500)

static char io_buffer[4096];
static ssize_t io_buffer_len = 0;
static ssize_t io_buffer_pos = 0;

//--------------- PROCESS ---------------

void prepare_subprocess();
int subprocess_join_timeout(pid_t child_pid, int timeout_ms, int *exit_status);
int read_line_timeout(int fd, char *buffer, size_t max_size);

//--------------- GA ---------------

void send_evaluation_request(int id, Individual ind);

void selection(Individual *population, double *fitness_scores, int pop_size,
               Individual *parents, int i);
void crossover(const Individual *parents, int num_parents,
               Individual *offspring, int i);
void mutation(Individual *offspring, int i);

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

  prepare_subprocess();
#pragma omp parallel shared(best_individual_accuracy, population,              \
                                fitness_scores, io_buffer, io_buffer_len,      \
                                io_buffer_pos, global_agent, agent_running)
  {
#pragma omp master
    {
      for (int i = 0; i < POP_SIZE; i++) {
        generate_individual(population[i]);
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
          int e;
          if ((e = read_line_timeout(global_agent.stdout_fd, task_buffer,
                                     256)) == 1) {
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
                fatal(error_code_t::UNKNOWN, "sscanf");
              }
              free(task_buffer);
            }
          } else {
            free(task_buffer);
            fatal(error_code_t::READ_FILDES, "Agent failed to respond or pipe broke.");
          }
        }

#pragma omp taskwait
        printf("\n");
        double max_fitness = -INFINITY;
        double sum_fitness = 0.0;
        for (int i = 0; i < POP_SIZE; i++) {
          report(population, i, fitness_scores[i]);
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
  cleanup(error_code_t::SUCCESS, "success");
  goodbye(best_individual_accuracy);

  free(population);
  free(parents);
  free(offspring);
  free(next_population);
  free(fitness_scores);

  return 0;
}

void send_evaluation_request(int id, Individual ind) {
  if (!agent_running)
    fatal(error_code_t::UNKNOWN, "Agent is dead.");

  char buffer[256];
  int size = snprintf(buffer, sizeof(buffer), python_printf_string,
                      python_args_in_order(id, ind));
  if (size <= 0)
    fatal(error_code_t::UNKNOWN, "String format failed");

  if (write(global_agent.stdin_fd, buffer, size) <= 0)
    fatal(error_code_t::WRITE_FILDES, "send_evaluation_request");
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

//--------------- SUBPROC ---------------

void prepare_subprocess() {
  char n_workers[4];
  snprintf(n_workers, sizeof(n_workers), "%d", OUR_THREADS);
  const char *command_line[] = python_command_line("true", n_workers);
  if (subprocess_create(command_line,
                        subprocess_option_inherit_environment |
                            subprocess_option_search_user_path |
                            subprocess_option_enable_async,
                        &global_agent)) {
    fatal(error_code_t::CREATE, "Failed to spawn Python agent");
  }
  // int fd = global_agent.stdout_fd;
  // int flags = fcntl(fd, F_GETFL, 0);       // Get current flags
  // fcntl(fd, F_SETFL, flags | O_NONBLOCK); // Add non-blocking flag
  agent_running = true;
}
