#ifndef data_types_h
#define data_types_h

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define uchar unsigned char
#define uchar16 unsigned short
#define char16 signed short
#define int32 int
#define word float
#define dword double

#define VECTOR_INITIAL_CAPACITY 16
#define M_PI 3.14159265358979323846

#define MAX_BRIGHTNESS 255

typedef struct Mat8U
{
    int width;
    int height;
    int depth;
    uchar *data;
} Mat8U;

typedef struct Mat16S
{
    int width;
    int height;
    int depth;
    char16 *data;
} Mat16S;

typedef struct Mat16U
{
    int width;
    int height;
    int depth;
    uchar16 *data;
} Mat16U;

typedef struct Mat32F
{
    int width;
    int height;
    int depth;
    float *data;
} Mat32F;

typedef struct KernelData
{
    float sigma;
    int kernel_size;
    float *data;
} KernelData;

typedef struct pt2f
{
    float x;
    float y;
} pt2f;

typedef struct pt2i
{
    int x;
    int y;
} pt2i;

typedef struct _TPoint {
  double x;
  double y;
} TPoint;

typedef struct {
    int size;
    int capacity;
    float *data;
} Vector;

typedef struct ArrPoint2f {
    Vector x;
    Vector y;
} ArrPoint2f;

void vec_init(Vector *vector);
void vec_append(Vector *vector, float value);
void vec_prepend(Vector *vector, float value);
void vec_delete_index(Vector *vector, int index);
void vec_delete_value(Vector *vector, float value);
void vec_set(Vector *vector, int index, float value);
float vec_get(Vector *vector, int index);
void vec_resize(Vector *vector);
void vec_free_memory(Vector *vector);
float vec_pop(Vector *vector);
int vec_find_value(Vector *vector, float value);
int vec_size(Vector *vector);
int vec_capacity(Vector *vector);

bool is_empty(Vector *vector);

ArrPoint2f* init_point2f();
void free_point2f(ArrPoint2f *pt);
ArrPoint2f* append_point2f(ArrPoint2f* pt, float x, float y);

#endif