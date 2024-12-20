#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "csv_parser.h"
#include "nn.h"
#include <unistd.h>

#include <raylib.h>


void paint(NN nn);

int main(){
	srand(time(0));
	Mat mat = parse_csv_to_mat("./digit-recognizer/train.csv");

    size_t layers[] = {28*28, 64, 32, 10};
    Act* actf = (Act[]){ACT_RELU, ACT_RELU, ACT_SIGM};
	NN nn = nn_new(layers, actf, ARR_LEN(layers));
	NN g = nn_new(layers, actf, ARR_LEN(layers));
	nn_rand(nn, -0.5, 0.5);
    //NN_PRINT(nn);

    // allocating the date

	Mat nto = mat_submatrix(mat, 0, 0, 0, mat.rows-1);
	Mat ti = mat_submatrix(mat, 1, 0, mat.cols-1, mat.rows-1);
	Mat to = mat_alloc(mat.rows, 10);
    Mat cti = mat_submatrix(mat, 1, 0, mat.cols-1, 5000);
    Mat cto = mat_submatrix(to, 0, 0, to.cols-1, 5000);

	mat_zero(to);

    // preparing the data
	for(size_t i = 0; i < mat.rows; i++){
		MAT_AT(to, i, (int)MAT_AT(nto, i, 0)) = 1.0;
	}
	for(size_t i = 0; i < ti.rows; i++){
		for(size_t j = 0; j < ti.cols; j++){
			MAT_AT(ti, i, j) /= 255;
		}
	}
    
	printf("cost before training = %f\n", nn_cost(nn, ti, to));

    // learning process
	for(size_t i = 0; true; i++){
		size_t s = 6;
		size_t pos = (rand()) % (ti.rows - s);
		Mat gti = mat_submatrix(ti, 0, pos, ti.cols-1, pos+s);
		Mat gto = mat_submatrix(to, 0, pos, to.cols-1, pos+s);
		nn_backprop(nn, g, gti, gto);
		nn_learn(nn, g, 0.017);
        if(i % 2000 == 0){
            float tc = nn_cost(nn, cti, cto);
            if(tc < 0.5) break;
            printf("%zu - %f\n",(int) i, tc);
        }
	}

	printf("cost after learning = %f\n", nn_cost(nn, cti, cto));

    paint(nn);
    return 0;
}


// visual part

Mat ConvertToMatrix(float pixels[28][28]) {
    Mat matrix = mat_alloc(1, 28 * 28);
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            MAT_AT(matrix, 0, x + y*28) = pixels[y][x];
        }
    }
    return matrix;
}

void paint(NN nn) {
    const int tiles = 28;
    const int tile = 20;
    const int screenWidth = tile * tiles;
    const int screenHeight = tile * tiles + 200;
    
    InitWindow(screenWidth, screenHeight, "Drawing with cross brush on 28x28 canvas");
    
    float pixels[28][28];
    
    SetTargetFPS(60);
    int cnt = 0;

    while(!WindowShouldClose()){
        Vector2 mousePosition = GetMousePosition();
        if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
            int x = mousePosition.x / tile;
            int y = mousePosition.y / tile;
            if (x >= 0 && x < tiles && y >= 0 && y < tiles) {
                pixels[y][x] += 0.5; // Центральный пиксель
                if(x > 0) pixels[y][x - 1] += 0.25;
                if(x < tiles-1) pixels[y][x + 1] += 0.25;
                if(y > 0) pixels[y - 1][x] += 0.25;
                if(y < tiles-1) pixels[y + 1][x] += 0.25;
            }
            for(int i = 0; i < tiles; i++){
                for(int j = 0; j < tiles; j++){
                    pixels[i][j] = pixels[i][j] > 1.0 ? 1.0 : pixels[i][j];
                }
            }
        }
        
        if(IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)){
            for(int y = 0; y < tiles; y++){
                for(int x = 0; x < tiles; x++){
                    pixels[y][x] = 0.0;
                }
            }
        }

        char buf[128];
        char buf2[128];
        if(cnt++ % 6 == 0) {
            Mat img = ConvertToMatrix(pixels);
            NN_SET_INPUT(nn, img);
            nn_forward(nn);
            float max = 0.0;
            int mval = 0;
            for(size_t j = 0; j < 10; j++){
                if(MAT_AT(NN_OUTPUT(nn), 0, j) > max){
                    max = MAT_AT(NN_OUTPUT(nn), 0, j);
                    mval = j;
                }
            }
            sprintf(buf, "Number - %i", mval);
            mat_free(img);
        }

        BeginDrawing();

        DrawText("Click RMB to clear the screen!", 225, 580, 20, DARKGRAY);
        DrawText(buf, 20, 580, 20, DARKGRAY);

        for(int i = 0; i < 10; i++){
            float k = MAT_AT(NN_OUTPUT(nn), 0, i);
            int t = tile*tiles+160;
            float l = (tile*tiles-60)/9;
            DrawRectangle(20 + i * l, t-k*117, 20, k*117, DARKGRAY);
            sprintf(buf2, "%d", i);
            DrawText(buf2, 20 + i * l + 5, t + 10, 20, DARKGRAY);
        }

        ClearBackground(RAYWHITE);
        
        for(int y = 0; y < tiles; y++){
            for(int x = 0; x < tiles; x++){
                float v = (int)(pixels[y][x] * 255.0f);
                Color c = {v, v, v, 255};
                DrawRectangle(x * tile, y * tile, tile, tile, c);
            }
        }
        
        EndDrawing();
    }
    
    CloseWindow();
}
