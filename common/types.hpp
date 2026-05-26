#ifndef GA_TYPES_H
#define GA_TYPES_H

struct Individual {
  int activation;
  int layers;
  int neurons;
  layer_pattern_t layer_pattern;
  float learning_rate;
  float decay;
  optimizer_t optimizer;
  int batch_size;
};

struct IndividualAccuracy {
  Individual individual;
  double accuracy;
};

struct double_range_t {
  double min;
  double max;
};

struct int_range_t {
  int min;
  int max;
};

struct limits_t {
  int_range_t activation;
  int_range_t layers;
  double_range_t learning_rate;
  double_range_t decay;
};

enum class error_code_t {
  UNKNOWN,
  SUCCESS,
  TERMINATE,
  DESTROY,
  READ_FILDES,
  READ_STD,
  CREATE,
  WRITE_FILDES,
  READ_FILDES_TIMEOUT
};


#endif // GA_TYPES_H