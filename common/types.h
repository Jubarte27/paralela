#ifndef GA_TYPES_H
#define GA_TYPES_H

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

typedef struct limits_t {
  int_range_t layers;
  double_range_t learning_rate;
  int_range_t activation;
} limits_t;

#endif // GA_TYPES_H