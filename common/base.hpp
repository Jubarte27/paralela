#ifndef GA_BASE_H
#define GA_BASE_H

#include <stdlib.h>

#include <boost/preprocessor/punctuation/paren.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "args.hpp"
#include "rand.hpp"
#include "types.hpp"

#define cast(T) BOOST_PP_LPAREN() T *BOOST_PP_RPAREN()
#define cast_ind() cast(Individual)
#define cast_double() cast(double)

#define malloc_something(num, something)                                       \
  cast(something) malloc((num) * sizeof(something))
#define malloc_ind(num) malloc_something(num, Individual)

#define alloc_pop() malloc_ind(POP_SIZE)
#define alloc_parents() malloc_ind(NUM_PARENTS)
#define alloc_offspring() malloc_ind((POP_SIZE - NUM_PARENTS))
#define alloc_scores() cast_double() calloc(POP_SIZE, sizeof(double))

#define crossover_cut(_, data, target)                                         \
  individual.target =                                                          \
      (BOOST_PP_TUPLE_ELEM(0, data) > BOOST_PP_TUPLE_ELEM(1, data)++           \
           ? BOOST_PP_TUPLE_ELEM(2, data)                                      \
           : BOOST_PP_TUPLE_ELEM(3, data))                                     \
          ->target;
#define full_crossover_cut(data, args...)                                      \
  BOOST_PP_SEQ_FOR_EACH(crossover_cut, data, BOOST_PP_VARIADIC_TO_SEQ(args))

void fatal(error_code_t err, const char *where);
void cleanup(error_code_t err, const char *where);

void generate_individual(Individual &population) {
  population.activation = random_activation();
  population.layers = random_layers();
  population.neurons = random_neurons(population.layers);

  population.layer_pattern = random_layer_pattern();

  population.learning_rate = random_learning_rate();
  population.decay = random_decay();

  population.optimizer = random_optimizer();

  population.batch_size = random_batch_size();

  int activation;
  int layers;
  int neurons;
  layer_pattern_t layer_pattern;
  float learning_rate;
  float decay;
  optimizer_t optimizer;
  int batch_size;
}

void selection(Individual *population, double *fitness_scores, int pop_size,
               Individual *parents, int i) {
  int idx1 = random_randint(0, pop_size - 1);
  int idx2 = random_randint(0, pop_size - 1);

  if (pop_size <= 1)
    fatal(error_code_t::UNKNOWN, "selection");

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
  Individual &child = offspring[i];
  Individual &individual = offspring[i];

  int cut_point = random_randint(1, 4); // no clones
  int index = 0;

  full_crossover_cut((cut_point, index, p1, p2), layers, neurons, learning_rate,
                     batch_size, activation);
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

void print_individual(const Individual *i, double acc) {
  printf("acc=%.17f,Layers=%d,Neurons=%d,LR=%lf,DecayRate=%lf,Batch=%d,Act="
         "%d,LayerPattern=%s,Optimizer=%s",
         acc, i->layers, i->neurons, i->learning_rate, i->decay, i->batch_size,
         i->activation, PATTERNS[i->layer_pattern].data(),
         OPTIMIZERS[i->optimizer].data());
}

void report(const Individual *pop, int index, double acc) {
  const Individual *i = pop + index;
  printf("%2d/%2d,", index + 1, POP_SIZE);
  print_individual(i, acc);
  printf("\n");
}

void goodbye(IndividualAccuracy best_individual_accuracy) {
  printf("\n=== Optimization Complete ===\n");
  Individual *i = &best_individual_accuracy.individual;
  double acc = best_individual_accuracy.accuracy;
  printf("Best:");
  print_individual(i, acc);
  printf("\n");
}

#endif // GA_BASE_H