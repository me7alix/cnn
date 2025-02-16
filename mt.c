#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv_parser.c"
#define NN_IMPLEMENTATION
#include "nn.h"

#include <pthread.h>
#include <unistd.h>

#include <raylib.h>

size_t layers[] = (size_t[]){28 * 28, 128, 32, 10};
Act *actfs;

Mat ti, to, cti, cto;
NN *gs;
NN *nns;
NN nn;

void paint(NN nn);
void mt_learn();

int main() {
  srand(time(0));
  Mat mat = parse_csv_to_mat("./digitrec/train.csv");
  actfs = (Act[]){ACT_RELU, ACT_RELU, ACT_SIGM};

  nn = nn_new(layers, actfs, ARR_LEN(layers));
  NN g = nn_new(layers, actfs, ARR_LEN(layers));
  nn_rand(nn, -0.5, 0.5);

  // initializing the data

  Mat nto = mat_submatrix(mat, 0, 0, 0, mat.rows - 1);
  ti = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);
  to = mat_alloc(mat.rows, 10);
  cti = mat_submatrix(mat, 1, 0, mat.cols - 1, 5000);
  cto = mat_submatrix(to, 0, 0, to.cols - 1, 5000);

  mat_zero(to);

  // preparing the data

  for (size_t i = 0; i < mat.rows; i++) {
    MAT_AT(to, i, (int)MAT_AT(nto, i, 0)) = 1.0;
  }

  for (size_t i = 0; i < ti.rows; i++) {
    for (size_t j = 0; j < ti.cols; j++) {
      MAT_AT(ti, i, j) /= 255;
    }
  }

  printf("cost before training = %f\n", nn_cost(nn, cti, cto));

  mt_learn();

  printf("cost after training = %f\n", nn_cost(nn, cti, cto));

  // paint(nn);
  return 0;
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_main = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_threads = PTHREAD_COND_INITIALIZER;

// Shared flags
bool threads_ready = false; // Flag to signal threads to start processing
bool terminate = false;     // Flag to signal threads to terminate
bool *thread_ready_flags;   // Array of flags for each thread

// Thread function
void *thread_function(void *arg) {
  int thread_num = *(int *)arg;
  free(arg);

  while (true) {
    // Wait for the signal to start processing
    pthread_mutex_lock(&mutex);
    while (!threads_ready && !terminate) {
      pthread_cond_wait(&cond_threads, &mutex);
    }

    // Check if termination was signaled
    if (terminate) {
      pthread_mutex_unlock(&mutex);
      break;
    }

    pthread_mutex_unlock(&mutex);

    size_t s = 6;
    size_t pos = rand() % (ti.rows - s);
    Mat gti = mat_submatrix(ti, 0, pos, ti.cols - 1, pos + s);
    Mat gto = mat_submatrix(to, 0, pos, to.cols - 1, pos + s);
    nn_backprop(nns[thread_num], gs[thread_num], gti, gto);

    // Signal completion
    pthread_mutex_lock(&mutex);
    thread_ready_flags[thread_num] = true;
    pthread_cond_signal(&cond_main);
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

void mt_learn() {
  int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
  if (num_threads <= 0) {
    perror("Failed to determine the number of processors");
    return;
  }

  pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
  if (!threads) {
    perror("Failed to allocate memory for threads");
    exit(EXIT_FAILURE);
  }

  thread_ready_flags = calloc(num_threads, sizeof(bool));
  if (!thread_ready_flags) {
    perror("Failed to allocate memory for thread flags");
    free(threads);
    exit(EXIT_FAILURE);
  }

  gs = malloc(sizeof(NN) * num_threads);
  nns = malloc(sizeof(NN) * num_threads);
  if (!gs || !nns) {
    perror("Failed to allocate memory for neural networks");
    free(threads);
    free(thread_ready_flags);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_threads; i++) {
    gs[i] = nn_new(layers, actfs, sizeof(layers) / sizeof(layers[0]));
    nns[i] = nn_new(layers, actfs, sizeof(layers) / sizeof(layers[0]));
  }

  for (int i = 0; i < num_threads; i++) {
    nn_copy(nns[i], nn);
  }

  // Create worker threads
  for (int i = 0; i < num_threads; i++) {
    int *thread_num_ptr = malloc(sizeof(int));
    if (!thread_num_ptr) {
      perror("Failed to allocate memory for thread number");
      exit(EXIT_FAILURE);
    }
    *thread_num_ptr = i;
    if (pthread_create(&threads[i], NULL, thread_function, thread_num_ptr) !=
        0) {
      perror("Failed to create thread");
      free(thread_num_ptr);
      exit(EXIT_FAILURE);
    }
  }

  int cnt = 0;
  while (true) {
    // Signal threads to start processing
    pthread_mutex_lock(&mutex);
    threads_ready = true;
    pthread_cond_broadcast(&cond_threads);
    pthread_mutex_unlock(&mutex);

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
      pthread_mutex_lock(&mutex);
      while (!thread_ready_flags[i]) {
        pthread_cond_wait(&cond_main, &mutex);
      }
      thread_ready_flags[i] = false; // Reset flag for next iteration
      pthread_mutex_unlock(&mutex);
    }

    // Reset the threads_ready flag for the next iteration
    pthread_mutex_lock(&mutex);
    threads_ready = false;
    pthread_mutex_unlock(&mutex);

    // Aggregate gradients and update the main neural network
    for (int i = 0; i < num_threads; i++) {
      nn_copy(nns[i], nn);
    }
    for (int i = 0; i < num_threads; i++) {
      nn_learn(nn, gs[i], 0.017f);
    }

    cnt += num_threads;
    if (cnt % num_threads == 0) { // Simplified condition
      float tc = nn_cost(nn, ti, to);
      printf("cost %d - %f\n", cnt, tc);
    }

    // Optional: Add a termination condition to break the loop
    // For example, after a certain number of iterations
    // if (cnt >= MAX_ITERATIONS) {
    //     break;
    // }
  }

  // Clean up (This part will not be reached in the current infinite loop)
  pthread_mutex_lock(&mutex);
  terminate = true;
  pthread_cond_broadcast(&cond_threads);
  pthread_mutex_unlock(&mutex);

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  free(threads);
  free(thread_ready_flags);
  free(gs);
  free(nns);
}

// visual part

Mat ConvertToMatrix(float pixels[28][28]) {
  Mat matrix = mat_alloc(1, 28 * 28);
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      MAT_AT(matrix, 0, x + y * 28) = pixels[y][x];
    }
  }
  return matrix;
}

void paint(NN nn) {
  const int tiles = 28;
  const int tile = 20;
  const int screenWidth = tile * tiles;
  const int screenHeight = tile * tiles + 200;

  InitWindow(screenWidth, screenHeight,
             "Drawing with cross brush on 28x28 canvas");

  float pixels[28][28];

  SetTargetFPS(60);
  int cnt = 0;

  while (!WindowShouldClose()) {
    Vector2 mousePosition = GetMousePosition();
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      int x = mousePosition.x / tile;
      int y = mousePosition.y / tile;
      if (x >= 0 && x < tiles && y >= 0 && y < tiles) {
        float b = 0.4;
        pixels[y][x] += b;
        if (x > 0)
          pixels[y][x - 1] += b / 2.0;
        if (x < tiles - 1)
          pixels[y][x + 1] += b / 2.0;
        if (y > 0)
          pixels[y - 1][x] += b / 2.0;
        if (y < tiles - 1)
          pixels[y + 1][x] += b / 2.0;
      }
      for (int i = 0; i < tiles; i++) {
        for (int j = 0; j < tiles; j++) {
          pixels[i][j] = pixels[i][j] > 1.0 ? 1.0 : pixels[i][j];
        }
      }
    }

    if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
      for (int y = 0; y < tiles; y++) {
        for (int x = 0; x < tiles; x++) {
          pixels[y][x] = 0.0;
        }
      }
    }

    char buf[128];
    char buf2[128];
    if (cnt++ % 6 == 0) {
      Mat img = ConvertToMatrix(pixels);
      NN_SET_INPUT(nn, img);
      nn_forward(nn);
      float max = 0.0;
      int mval = 0;
      for (size_t j = 0; j < 10; j++) {
        if (MAT_AT(NN_OUTPUT(nn), 0, j) > max) {
          max = MAT_AT(NN_OUTPUT(nn), 0, j);
          mval = j;
        }
      }
      sprintf(buf, "Number - %i", mval);
      mat_free(img);
    }

    BeginDrawing();

    ClearBackground(RAYWHITE);

    DrawText("Click RMB to clear the screen", 228, 580, 20, DARKGRAY);
    DrawText(buf, 20, 580, 20, DARKGRAY);

    for (int i = 0; i < 10; i++) {
      float k = MAT_AT(NN_OUTPUT(nn), 0, i);
      int t = tile * tiles + 160;
      float l = (tile * tiles - 60) / 9;
      DrawRectangle(20 + i * l, t - k * 117, 20, k * 117, DARKGRAY);
      sprintf(buf2, "%d", i);
      DrawText(buf2, 20 + i * l + 5, t + 10, 20, DARKGRAY);
    }

    for (int y = 0; y < tiles; y++) {
      for (int x = 0; x < tiles; x++) {
        float v = (int)(pixels[y][x] * 255.0f);
        Color c = {v, v, v, 255};
        DrawRectangle(x * tile, y * tile, tile, tile, c);
      }
    }

    EndDrawing();
  }

  CloseWindow();
}
