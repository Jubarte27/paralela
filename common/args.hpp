#ifndef GA_ARGS_H
#define GA_ARGS_H

#include <initializer_list>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_map>

template <typename K, typename V>
std::unordered_map<V, K> reversed(const std::unordered_map<K, V> &original) {
  std::unordered_map<V, K> flipped;
  for (const auto &[key, value] : original)
    flipped.insert({value, key});
  return flipped;
}

template <typename T> struct converter_t {
  std::unordered_map<T, std::string> from_type;
  std::unordered_map<std::string, T> to_type;
  std::vector<T> original;

  converter_t(std::initializer_list<std::pair<T, std::string>> type_to_param)
      : from_type(type_to_param.begin(), type_to_param.end()) {
        to_type = reversed(from_type);
        original = std::vector<T>();
        original.reserve(type_to_param.size());
        for (const std::pair<T, std::string>& par : type_to_param) {
          original.emplace_back(par.first);
        }
      }
  converter_t(std::initializer_list<std::pair<std::string, T>> param_to_type)
      : to_type(param_to_type.begin(), param_to_type.end()) {
        from_type = reversed(to_type);
        original = std::vector<T>();
        original.reserve(param_to_type.size());
        for (const std::pair<std::string, T>& par : param_to_type) {
          original.emplace_back(par.second);
        }
      }

  const std::string &operator[](const T &k) const { return from_type.at(k); }
  const T &operator[](const std::string &k) const { return to_type.at(k); }

  bool contains(const T &k) const { return from_type.contains(k); }
  bool contains(const std::string &k) const { return to_type.contains(k); }
};

// will use half the trainig set for fasion_mnist and full for cifar_10
typedef enum { MNIST, MNIST_SMALL } dataset_t;
typedef enum { HALVE, SAME } layer_pattern_t;
typedef enum { ADAM, ADAMW } optimizer_t;

const converter_t<dataset_t> DATASETS{{"full", MNIST}, {"small", MNIST_SMALL}};
const converter_t<layer_pattern_t> PATTERNS{{"halve", HALVE}, {"same", SAME}};
const converter_t<optimizer_t> OPTIMIZERS{{"adam", ADAM}, {"adamw", ADAMW}};

//for python
int OUR_THREADS;
dataset_t DATASET;

#ifndef VERSION_NAME
#define VERSION_NAME "GENERIC"
#endif

//for ga
int NUM_GENERATIONS;
int POP_SIZE;
int NUM_PARENTS;

void reset_args() {
  OUR_THREADS = 4;
  DATASET = MNIST;
  NUM_GENERATIONS = 10;
  POP_SIZE = 10;
  NUM_PARENTS = POP_SIZE / 2;
}

void apply_args() {
  omp_set_num_threads(OUR_THREADS);
  printf("Using: " VERSION_NAME " version, seed 42, %d threads, %s dataset, %d generations, %d population size, "
         "%d parents per generation\n",
         OUR_THREADS, DATASETS[DATASET].data(), NUM_GENERATIONS, POP_SIZE,
         NUM_PARENTS);
  srand(42);
}

void fail_arg(char *arg, int index) {
  fprintf(stderr, "Parameter %d is invalid: %s\n", index, arg);
  exit(1);
}

int read_positive_int(char *arg, int index) {
  int number_read = (int)strtol(arg, NULL, 0);
  if (number_read < 1)
    fail_arg(arg, index);
  return number_read;
}

int read_int(char *arg, int index) {
  int number_read = (int)strtol(arg, NULL, 0);
  return number_read;
}

#define next_arg(i, argc) if ((i) >= (argc)) return; (i)++; (arg)++;
#define read_u(dest, i, argc) if (strlen(*(arg)) > 0) (dest) = read_positive_int(*(arg), (i));
#define read_i(dest, i, argc) if (strlen(*(arg)) > 0) (dest) = read_int(*(arg), (i));
void read_args(int argc, char *argv[]) {
  reset_args();
  char **arg = argv;
  char **end = argv + argc;
  int i = 0;
  next_arg(i, argc); // ignore file name
  if (strlen(*arg) > 0 && (DATASETS.contains(*arg))) DATASET = DATASETS[*arg];
  else fail_arg(*arg, i);

  next_arg(i, argc);
  read_i(OUR_THREADS, i, argc);
  if (OUR_THREADS > omp_get_num_procs()) fail_arg(*arg, i);

  next_arg(i, argc);
  read_u(NUM_GENERATIONS, i, argc);

  next_arg(i, argc);
  read_u(POP_SIZE, i, argc);

  next_arg(i, argc);
  read_u(NUM_PARENTS, i, argc);
  if (NUM_PARENTS > POP_SIZE) fail_arg(*arg, i);
}

#endif // GA_ARGS_H