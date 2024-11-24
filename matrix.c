#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "matrix.h"

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

