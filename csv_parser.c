#include "csv_parser.h"

void count_rows_cols(FILE* file, size_t* rows, size_t* cols) {
    char line[65536];
    *rows = 0;
    *cols = 0;

    // Подсчитываем количество столбцов по первой строке
    if (fgets(line, sizeof(line), file)) {
        (*rows)++;
        char* token = strtok(line, ",");
        while (token) {
            (*cols)++;
            token = strtok(NULL, ",");
        }
    }

    // Подсчитываем оставшиеся строки
    while (fgets(line, sizeof(line), file)) {
        (*rows)++;
    }
    rewind(file); // Возвращаемся в начало файла
}

Mat parse_csv_to_mat(const char* filename) {
    FILE* file = fopen(filename, "r");

    size_t rows, cols;
    count_rows_cols(file, &rows, &cols);

    Mat mat = mat_alloc(rows, cols);

    char line[65536];
    size_t row = 0;
    
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (size_t col = 0; col < cols && token; col++) {
            MAT_AT(mat, row, col) = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
        row++;
    }

    fclose(file);
    return mat;
}