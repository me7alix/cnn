#ifndef CSV_PARSER_H_
#define CSV_PARSER_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn.h"

Mat parse_csv_to_mat(const char* filename);

#endif