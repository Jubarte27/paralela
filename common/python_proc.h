#ifndef GA_PYTHON_PROC_H
#define GA_PYTHON_PROC_H

#include <stdlib.h>
#include <errno.h>

#include "subprocess.h"
#include "types.hpp"

#define python_command_line(force_single_thread_model, n_workers)              \
  {"python3",                                                                  \
   "-u",                                                                       \
   "py/agent.py",                                                              \
   force_single_thread_model,                                                  \
   n_workers,                                                                  \
   DATASETS[DATASET].data(),                                                   \
   NULL};

#define python_printf_string "%d %d %d %d %s %.17g %.17g %s %d\n"
#define python_args_in_order(id, dat) \
    id, \
    dat.activation, \
    dat.layers, \
    dat.neurons, \
    PATTERNS[dat.layer_pattern].data(), \
    dat.learning_rate, \
    dat.decay, \
    OPTIMIZERS[dat.optimizer].data(), \
    dat.batch_size

const char EXIT[] = "exit\n";

struct subprocess_s global_agent;
bool agent_running = false;

void kill_child() {
  if (subprocess_terminate(&global_agent) != 0) {
    fprintf(stderr, "Failed to terminate subprocess.\n");
  }
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

void clean_process(error_code_t err, const char *where) {
  if (!agent_running)
    return;

  if (err != error_code_t::SUCCESS && err != error_code_t::READ_FILDES_TIMEOUT) {
    fprintf(stderr, "Fatal Error [%d] at %s\n", static_cast<int>(err), where);
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


void fatal(error_code_t err, const char *where) {
  if (global_agent.stderr_file != NULL) dump_fd_to_stdout(global_agent.stderr_fd);
  cleanup(err, where);
  exit(static_cast<int>(err));
}

void cleanup(error_code_t err, const char *where) {
  clean_process(err, where);
}

#endif // GA_PYTHON_PROC_H