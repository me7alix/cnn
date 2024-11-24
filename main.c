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

    size_t layers[] = {28*28, 32, 16, 10};
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
            if(tc < 0.04) break;
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
    const int screenWidth = 20 * 28;
    const int screenHeight = 600;
    
    InitWindow(screenWidth, screenHeight, "Drawing with cross brush on 28x28 canvas");
    
    float pixels[28][28];
    
    SetTargetFPS(60);
    int cnt = 0;

    int tile = 28;

    while(!WindowShouldClose()){
        Vector2 mousePosition = GetMousePosition();
        if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
            int x = mousePosition.x / 20;
            int y = mousePosition.y / 20;
            if (x >= 0 && x < tile && y >= 0 && y < tile) {
                pixels[y][x] += 0.5; // Центральный пиксель
                if(x > 0) pixels[y][x - 1] += 0.25;
                if(x < 27) pixels[y][x + 1] += 0.25;
                if(y > 0) pixels[y - 1][x] += 0.25;
                if(y < 27) pixels[y + 1][x] += 0.25;
            }
            for(int i = 0; i < tile; i++){
                for(int j = 0; j < tile; j++){
                    pixels[i][j] = pixels[i][j] > 1.0 ? 1.0 : pixels[i][j];
                }
            }
        }
        
        if(IsKeyPressed(KEY_C)){
            for(int y = 0; y < tile; y++){
                for(int x = 0; x < tile; x++){
                    pixels[y][x] = 0.0;
                }
            }
        }

        char buf[128];
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

        DrawText(buf, 10, 580, 20, DARKGRAY);
        ClearBackground(RAYWHITE);
        
        for(int y = 0; y < tile; y++){
            for(int x = 0; x < tile; x++){
                float v = (int)(pixels[y][x] * 255.0f);
                Color c = {v, v, v, 255};
                DrawRectangle(x * 20, y * 20, 20, 20, c);
            }
        }
        
        EndDrawing();
    }
    
    CloseWindow();
}
