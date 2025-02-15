#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

// #define NN_BACKPROP_TRADITIONAL

/*
	Saving nn to a file
	1. 784,32,16,10  // the struct of nn
	2. RELU, RELU, SIGM  // activation functions
	3. 0.3213,0.95114,0.0484...  // weights of first layer
	4. 0.8423,0.1929,0.6411...  // biases of first layer
	5. etc
*/

typedef struct {
    float* es;
    size_t rows;
    size_t cols;
    size_t rc;
    size_t pad;
} Mat;


#define MAT_AT(m, i, j) (m).es[(i)*(m).rc+(j)+(m).pad]
Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat a);
void mat_sum(Mat a, Mat b);
void mat_dot(Mat res, Mat a, Mat b);
Mat mat_submatrix(Mat a, int r1, int c1, int r2, int c2);
Mat mat_get_rows(Mat src, int r, int cnt);
void mat_print(Mat a, const char* name, size_t padding);
void mat_f(Mat a, float (*func)(float el));
void mat_rand(Mat a, float low, float high);
void mat_zero(Mat a);
Mat mat_row(Mat a, size_t r);
void mat_copy(Mat dst, Mat src);
#define MAT_PRINT(m) mat_print(m, #m, 0)

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

void nn_copy(NN nn, NN cnn);
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

#ifdef NN_IMPLEMENTATION

Mat mat_alloc(size_t rows, size_t cols){
    return (Mat) {
        .es = malloc(sizeof(float) * rows * cols),
        .rows = rows,
        .cols = cols,
        .rc = cols,
        .pad = 0
    };
}

void mat_free(Mat a){
    free(a.es);
}

Mat mat_submatrix(Mat a, int c1, int r1, int c2, int r2){
    int nr = r2 - r1 + 1;
    int nc = c2 - c1 + 1;

    return (Mat) {
        .rows = nr,
        .cols = nc,
        .rc = a.cols,
        .pad = c1,
        .es = &MAT_AT(a, r1, 0)
    };
}

void mat_sum(Mat a, Mat b){
    assert(a.rows == b.rows && a.cols == b.cols);
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            MAT_AT(a, i, j) += MAT_AT(b, i, j);
        }
    }
}

void mat_dot(Mat res, Mat a, Mat b){
    assert(a.cols == b.rows);
    assert(a.rows == res.rows && b.cols == res.cols);
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < b.cols; j++){
            MAT_AT(res, i, j) = 0;
            for(size_t o = 0; o < a.cols; o++){
                MAT_AT(res, i, j) += MAT_AT(a, i, o) * MAT_AT(b, o, j);
            }
        }
    }
}

void mat_print(Mat a, const char* name, size_t padding){
    for(size_t c = 0; c < padding; c++) printf("\t");
    printf("%s = [\n", name);
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            for(size_t c = 0; c < padding; c++) printf("\t");
            printf("\t%f", MAT_AT(a, i, j));
        }
        printf("\n");
    }
    for(size_t c = 0; c < padding; c++) printf("\t");
    printf("]\n");
}

void mat_f(Mat a, float (*func)(float el)){
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            MAT_AT(a, i, j) = func(MAT_AT(a, i, j));
        }
    }
}

void mat_zero(Mat a){
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            MAT_AT(a, i, j) = 0;
        }
    }
}

void mat_rand(Mat a, float low, float high){
    for(size_t i = 0; i < a.rows; i++){
        for(size_t j = 0; j < a.cols; j++){
            MAT_AT(a, i, j) = (float)rand()/(float)(RAND_MAX) * (high - low) + low;
        }
    }
}

Mat mat_row(Mat a, size_t r){
    return (Mat){
        .rows = 1,
        .cols = a.cols,
        .rc = a.cols,
        .pad = 0,
        .es = &MAT_AT(a, r, 0),
    };
}

void mat_copy(Mat dst, Mat src){
    assert(dst.rows == src.rows && dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}



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

void nn_copy(NN nn, NN cnn){
	for(size_t i = 0; i < nn.count; i++){
		mat_copy(nn.ws[i], cnn.ws[i]);
		mat_copy(nn.bs[i], cnn.bs[i]);
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

#endif