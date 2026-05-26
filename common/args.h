#ifndef GA_ARGS_H
#define GA_ARGS_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef enum {
    SMALL,
    LARGE
} dataset_size_t;

int OUR_THREADS;
dataset_size_t DATASET_SIZE;
int NUM_GENERATIONS;
int POP_SIZE;
int NUM_PARENTS;


void reset_args() {
    OUR_THREADS = 4;
    DATASET_SIZE = LARGE;
    NUM_GENERATIONS = 10;
    POP_SIZE = 10;
    NUM_PARENTS = POP_SIZE / 2;
}

void apply_args() {
    omp_set_num_threads(OUR_THREADS);
    printf("Using: %d threads, %s dataset, %d generations, %d population size, %d parents per generation\n", OUR_THREADS, DATASET_SIZE == SMALL ? "small" : "large", NUM_GENERATIONS, POP_SIZE, NUM_PARENTS);
}

void fail_arg(char* arg, int index) {
    fprintf(stderr, "Parameter %d is invalid: %s", index, arg);
    exit(1);
}

int read_positive_int(char* arg, int index) {
    int number_read = (int)strtol(arg, NULL, 0);
    if (number_read < 1) fail_arg(arg, index);
    return number_read;
}

void read_args(int argc, char* argv[]) {
    reset_args();
    char** arg = argv;
    char** end = argv + argc;
    int i = 1;

    if (i >= argc) return; i++; arg++; // ignore file name
    if (strlen(*arg) > 0) {
        if (strcmp(*arg, "small") == 0) DATASET_SIZE = SMALL;
        else if (strcmp(*arg, "large") == 0) DATASET_SIZE = LARGE;
        else fail_arg(*arg, i);
    }

    if (i >= argc) return; i++; arg++;
    if (strlen(*arg) > 0 && (OUR_THREADS = read_positive_int(*arg, i)) > omp_get_num_procs()) fail_arg(*arg, i);

    if (i >= argc) return; i++; arg++;
    if (strlen(*arg) > 0) NUM_GENERATIONS = read_positive_int(*arg, i);

    if (i >= argc) return; i++; arg++;
    if (strlen(*arg) > 0) POP_SIZE = read_positive_int(*arg, i);

    if (i >= argc) return; i++; arg++;
    if (strlen(*arg) > 0 && (NUM_PARENTS = read_positive_int(*arg, i)) > POP_SIZE) fail_arg(*arg, i);
}

#endif //GA_ARGS_H