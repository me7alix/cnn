#ifndef NN_H_
#define NN_H_

/*
	Saving nn to a file
	1. 784,32,16,10  // the struct of nn
	2. RELU, RELU, SIGM  // activation functions
	3. 0.3213,0.95114,0.0484...  // weights of first layer
	4. 0.8423,0.1929,0.6411...  // biases of first layer
	5. etc
*/

#include "matrix.h"

#define NN_RELU_PARAM 0.01f

#define ARR_LEN(a) sizeof((a))/sizeof((a[0]))
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#define NN_SET_INPUT(nn, mt) mat_copy(NN_INPUT((nn)), (mt))

typedef enum {
	ACT_SIGM,
	ACT_RELU,
} Act;

typedef struct {
	size_t count;
	Mat *ws;
	Mat *bs;
	Mat *as; // The amount of activations is count+1
	Act *actf;
} NN;

void nn_forward(NN nn);
void nn_rand(NN nn, float low, float high);
NN nn_new(size_t* arch, Act* actf, size_t arch_count);
void nn_print(NN nn, const char* name);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_zero(NN nn);
void nn_finite_diff(NN m, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif