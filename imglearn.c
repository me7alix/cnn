#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <raylib.h>
#include "csv_parser.c"

#define NN_IMPLEMENTATION
#include "nn.h"

void draw_mat(Mat m, Vector2 pos, int tile){
  for (int i = 0; i < 28 * 28; i++) {
    float c = MAT_AT(m, 0, i) * 255.0;
    DrawRectangle(pos.x + (i % 28) * tile, pos.y + (i / 28) * tile, tile, tile, (Color){c, c, c, 255});
  }
}

int main() {
  srand(time(0));
  Mat mat = parse_csv_to_mat("./digitrec/train.csv");

  size_t layers[] = {3, 7, 12, 13, 10, 6, 1};
  Act *actf = (Act[]){ACT_SIGM, ACT_RELU, ACT_RELU, ACT_RELU, ACT_RELU, ACT_SIGM};
  NN nn = nn_new(layers, actf, ARR_LEN(layers));
  NN g = nn_new(layers, actf, ARR_LEN(layers));
  nn_rand(nn, -0.5, 0.5);

  // initializing the data
  Mat imgs = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);

  for (size_t i = 0; i < imgs.rows; i++) {
    for (size_t j = 0; j < imgs.cols; j++) {
      MAT_AT(imgs, i, j) /= 255.0;
    }
  }

  Mat img1 = mat_submatrix(imgs, 0, 19, 28*28-1, 2);
  Mat img2 = mat_submatrix(imgs, 0, 8, 28*28-1, 8);
  Mat img3 = mat_submatrix(imgs, 0, 5, 28*28-1, 7);

  Mat ti = mat_alloc(1, 3);
  Mat to = mat_alloc(1, 1);

  Vector2 pos = {320, 320};
  float tile = 5;

  float slider = 0.0;
  float lrs = 0.4;

  // init window 
  InitWindow(900, 620, "Window");

  while (!WindowShouldClose()) {
    if(IsMouseButtonDown(MOUSE_BUTTON_LEFT))
      slider = fmin(1.0, fmaxf(GetMouseX()/900.0, 0));

    if(IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
      lrs = fmin(1.0, fmaxf(GetMouseX()/900.0, 0));

    BeginDrawing();
    ClearBackground(DARKGRAY);

    for(int i = 0; i < 28; i++) {
      for(int j = 0; j < 28; j++) { 
        MAT_AT(ti, 0, 0) = j / 28.0;
        MAT_AT(ti, 0, 1) = i / 28.0;
        MAT_AT(ti, 0, 2) = 0;
        MAT_AT(to, 0, 0) = MAT_AT(img1, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.01 * lrs);

        MAT_AT(ti, 0, 0) = j / 28.0;
        MAT_AT(ti, 0, 1) = i / 28.0;
        MAT_AT(ti, 0, 2) = 0.5;
        MAT_AT(to, 0, 0) = MAT_AT(img2, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.01 * lrs);       
        MAT_AT(ti, 0, 0) = j / 28.0;
        MAT_AT(ti, 0, 1) = i / 28.0;
        MAT_AT(ti, 0, 2) = 1;
        MAT_AT(to, 0, 0) = MAT_AT(img3, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.01 * lrs);
      }
    }

    float pixels = 280.0/3.0;
    for(int i = 0; i < pixels; i++) {
      for(int j = 0; j < pixels; j++) { 
        MAT_AT(NN_INPUT(nn), 0, 0) = j / pixels;
        MAT_AT(NN_INPUT(nn), 0, 1) = i / pixels;
        MAT_AT(NN_INPUT(nn), 0, 2) = slider;
        nn_forward(nn);

        float c = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.0;
        DrawRectangle(pos.x + j*3, pos.y + i*3, 3, 3, (Color){c, c, c, 255});
      }
    }

    draw_mat(img1, (Vector2){20, 20}, 10);
    draw_mat(img2, (Vector2){320, 20}, 10);
    draw_mat(img3, (Vector2){620, 20}, 10);

    DrawRectangle(lrs * 900 - 10, 600, 20, 20, RED);
    DrawRectangle(slider * 900 - 10, 600, 20, 20, BLUE);

    EndDrawing();
  }

  CloseWindow();

  return 0;
}
