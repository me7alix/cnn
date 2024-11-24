#ifndef MATRIX_H_
#define MATRIX_H_

#define MAT_AT(m, i, j) (m).es[(i)*(m).rc+(j)+(m).pad]

typedef struct {
    float* es;
    size_t rows;
    size_t cols;
    size_t rc;
    size_t pad;
} Mat;

// Only this method allocates memory
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

#endif
