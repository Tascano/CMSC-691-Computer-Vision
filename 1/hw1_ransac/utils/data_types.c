#include "data_types.h"

void vec_init(Vector *vector){
    vector->size = 0;
    vector->capacity = VECTOR_INITIAL_CAPACITY;
    vector->data = (float*)malloc(sizeof(float) * vector->capacity);
}

void vec_resize(Vector *vector){
    if(vector->size >= vector->capacity){
        vector->capacity *= 2;
        vector->data = (float*)realloc(vector->data, sizeof(float) * vector->capacity);
    }
}

void vec_append(Vector *vector, float value){
    vec_resize(vector);
    vector->data[vector->size++] = value;
}

void vec_prepend(Vector *vector, float value){
    vec_set(vector, 0, value);
    vector->size++;
}

float vec_get(Vector *vector, int index){
    if(index > vector->capacity || index < 0){
        printf("Index %d out of bounds for vector of size %d\n", index, vector->size);
        exit(1);
    }
    return vector->data[index];
}

void vec_set(Vector *vector, int index, float value){
    while(index >= vector->size){
        vec_append(vector, 0);
    }

    vector->data[index] = value;
}

float vec_pop(Vector *vector){
    int data = vector->data[vector->size - 2];
    vec_set(vector, vector->size - 1, 0);
    vector->size = vector->size - 1;
    return data;
}

void vec_delete_index(Vector *vector, int index){
    for(int i = 0; i < index; i++){
        vector->data[index + i] = vector->data[index + i + 1];
    }
    vector->size = vector->size - 1;
}

void vec_delete_value(Vector *vector, float value){
    for(int i = 0; i < vector->size; i++){
        if(fabs(vector->data[i] - value) <= 0.000001){
            vec_delete_index(vector, i);
        }
    }
}

int vec_find_value(Vector *vector, float value){
    for(int i = 0; i < vector->size; i++){
        if(fabs(vector->data[i] - value) <= 0.000001){
            return i;
        }
    }
    return -1;
}

int vec_size(Vector *vector){
    return vector->size;
}

int vec_capacity(Vector *vector){
    return vector->capacity;
}

bool vec_is_empty(Vector *vector){
    return vector->size == 0;
}

void vec_free_memory(Vector *vector){
    if (vector->data)
        free(vector->data);
}
