mkdir -p build
cc -g demos/image_learning.c -o build/image_learning -lm -lraylib
cc -g demos/digits_recognition.c -o build/digits_recognition -lm -lraylib
cc -g demos/car_racing.c -o build/car_racing -lm -lraylib
