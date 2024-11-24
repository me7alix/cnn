#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include <math.h>


float actf(float x, Act actf){
	switch (actf) {
		case ACT_RELU:
			return x > 0 ? x : x*NN_RELU_PARAM;
		case ACT_SIGM:
			return 1.f / (1.f + expf(-x));
	}
}

float dactf(float y, Act actf){
	switch (actf) {
		case ACT_RELU:
			return y >= 0 ? 1 : NN_RELU_PARAM;
		//case ACT_SIGM:
		//	return y * (1.0 - y);
		case ACT_SIGM:
			return 1.f / (1.f + expf(-y)) * (1.0 - (1.f / (1.f + expf(-y))));
	}
}

void mat_act(Mat a, Act f){
	for(size_t i = 0; i < a.rows; i++){
		for(size_t j = 0; j < a.cols; j++){
			MAT_AT(a, i, j) = actf(MAT_AT(a, i, j), f);
		}
	}
}

NN nn_new(size_t* arch, Act* actf, size_t arch_count){
	assert(arch_count > 0);

	NN nn;
	nn.count = arch_count - 1;
	nn.ws = malloc(sizeof(*nn.ws) * nn.count);
	assert(nn.ws != NULL);
	nn.bs = malloc(sizeof(*nn.bs) * nn.count);
	assert(nn.bs != NULL);
	nn.as = malloc(sizeof(*nn.as) * arch_count);
	assert(nn.as != NULL);

	nn.actf = malloc(sizeof(Act) * nn.count);

	nn.as[0] = mat_alloc(1, arch[0]);

	for(size_t i = 1; i < arch_count; i++){
		nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
		nn.bs[i-1] = mat_alloc(1, arch[i]);
		nn.as[i] = mat_alloc(1, arch[i]);

		nn.actf[i-1] = actf[i-1];
	}
	return nn;
}


void nn_print(NN nn, const char* name){
	printf("%s = [\n", name);
	for(size_t i = 0; i < nn.count; i++){
		mat_print(nn.ws[i], "ws", 1);
		mat_print(nn.bs[i], "bs", 1);
	}
	printf("]\n");
}

void nn_rand(NN nn, float low, float high){
	for(size_t i = 0; i < nn.count; i++){
		mat_rand(nn.ws[i], low, high);
		mat_rand(nn.bs[i], low, high);
	}
}

void nn_forward(NN nn){
	for(size_t i = 0; i < nn.count; i++){
		mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
		mat_sum(nn.as[i+1], nn.bs[i]);
		mat_act(nn.as[i+1], nn.actf[i]);
	}
}

float nn_cost(NN nn, Mat ti, Mat to){
	assert(ti.rows == to.rows);
	assert(to.cols == NN_OUTPUT(nn).cols);
	size_t n = ti.rows;
	
	float c = 0;
	for(size_t i = 0; i < n; i++){
		Mat x = mat_row(ti, i);
		Mat y = mat_row(to, i);

		mat_copy(NN_INPUT(nn), x);
		nn_forward(nn);

		size_t q = to.cols;
		for(size_t j = 0; j < q; j++){
			float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
			c += d*d;
		}
	}

	return c/n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to){
	float saved, c = nn_cost(nn, ti, to);
	for(size_t i = 0; i < nn.count; i++){
		for(size_t j = 0; j < nn.ws[i].rows; j++){
            for(size_t k = 0; k < nn.ws[i].cols; k++){
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }
		
		for(size_t j = 0; j < nn.bs[i].rows; j++){
			for(size_t k = 0; k < nn.bs[i].cols; k++){
				saved = MAT_AT(nn.bs[i], j, k);
				MAT_AT(nn.bs[i], j, k) += eps;
				MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
				MAT_AT(nn.bs[i], j, k) = saved;
			}
		}
	}
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to){
	assert(ti.rows == to.rows);
	assert(NN_OUTPUT(nn).cols == to.cols);
	size_t n = ti.rows;

	nn_zero(g);

	// i - current sample 
	// l - current layer
	// j - current activation
	// k - previous activation

	for(size_t i = 0; i < n; i++){
		mat_copy(NN_INPUT(nn), mat_row(ti, i));
		nn_forward(nn);

		for(size_t j = 0; j <= nn.count; j++){
			mat_zero(g.as[j]);
		}

		for(size_t j = 0; j < to.cols; j++){
			MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
		}

		for(size_t l = nn.count; l > 0; l--){
			for(size_t j = 0; j < nn.as[l].cols; j++){
				float a = MAT_AT(nn.as[l], 0, j);
				float da = MAT_AT(g.as[l], 0, j);
				float qa = dactf(a, nn.actf[l-1]);
				MAT_AT(g.bs[l-1], 0, j) += 2 * da * qa;
				for(size_t k = 0; k < nn.as[l-1].cols; k++){
					float pa = MAT_AT(nn.as[l-1], 0, k);
					float w = MAT_AT(nn.ws[l-1], k, j);
					MAT_AT(g.ws[l-1], k, j) += 2*da*qa*pa;
					MAT_AT(g.as[l-1], 0, k) += 2*da*qa*w;
				}
			}
		}
	}

	if(n == 1) return;
	for(size_t i = 0; i < g.count; i++){
		for(size_t j = 0; j < g.ws[i].rows; j++){
			for(size_t k = 0; k < g.ws[i].cols; k++){
				MAT_AT(g.ws[i], j, k) /= n;
			}
		}
		for(size_t j = 0; j < g.bs[i].rows; j++){
			for(size_t k = 0; k < g.bs[i].cols; k++){
				MAT_AT(g.bs[i], j, k) /= n;
			}
		}
	}
}

void nn_learn(NN nn, NN g, float rate){
	for(size_t i = 0; i < nn.count; i++){
		for(size_t j = 0; j < nn.ws[i].rows; j++){
			for(size_t k = 0; k < nn.ws[i].cols; k++){
				MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
			}
		}

		for(size_t j = 0; j < nn.bs[i].rows; j++){
			for(size_t k = 0; k < nn.bs[i].cols; k++){
				MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
			}
		}
	}	
}

void nn_zero(NN nn){
	for(size_t i = 0; i < nn.count; i++){
		mat_zero(nn.ws[i]);
		mat_zero(nn.bs[i]);
		mat_zero(nn.as[i]);
	}
	mat_zero(nn.as[nn.count]);
}