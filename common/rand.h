#ifndef GA_RAND_H
#define GA_RAND_H

#include <stdlib.h>
#include <math.h>
#include "types.h"

const limits_t default_limits = {
    .layers = {1, 3},
    .learning_rate = {-4.0, -1.0},
    .activation = {0, 2},
};
const int_range_t default_neuron_limits[] = { {32, 256}, {16, 128}, {8, 64} };
const int default_batch_choices[] = { 32, 64, 128 };

const limits_t limits = default_limits;

const int_range_t* neuron_limits = default_neuron_limits;
int neuron_limits_len = sizeof(default_neuron_limits);

const int* batch_choices = default_batch_choices;
int batch_choices_len = sizeof(default_batch_choices);

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

inline double random_double() { return ((double)rand() / (double)RAND_MAX); }
double random_between(double min, double max) {
    return min + random_double() * (max - min);
}
int random_randint(int min, int max) { return min + rand() % (max - min + 1); }
int random_choice(const int* choices, int num_choices) {
    return choices[random_randint(0, num_choices - 1)];
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
    int i = layers - limits.layers.min;
    return random_randint(neuron_limits[i].min, neuron_limits[i].max);
}

int clamp_neurons(int neurons, int layer) {
    int i = layer - limits.layers.min;

    return min(max(neurons, neuron_limits[i].min), neuron_limits[i].max);
}

#endif //GA_RAND_H