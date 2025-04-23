#include <iostream>
#include <math.h>
using namespace std;

const int N = 1000;

void print_matrix(int mat[N * N])
{
    cout << "=======================" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << mat[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << "=======================" << endl;
}

void matrix_mul_sequentially(int r[N * N], int a[N * N], int b[N * N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            r[i * N + j] = sum;
        }
    }
}

void matrix_mul_kernels(int r[N * N], int a[N * N], int b[N * N])
{
#pragma acc kernels
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            r[i * N + j] = sum;
        }
    }
}

void matrix_mul_parallel_loop(int r[N * N], int a[N * N], int b[N * N])
{
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            r[i * N + j] = sum;
        }
    }
}

int get_matrix(int result[N * N], int min, int max)
{
    int range = max - min + 1;
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            result[i * N + j] = rand() % range + min;
    }
    return 0;
}

int main(int argc, char **argv)
{
    int *r = new int[N * N];
    int *a = new int[N * N];
    int *b = new int[N * N];
    int min = 0;
    int max = 1;
    get_matrix(a, min, max);
    get_matrix(b, min, max);

    clock_t start_sequentially = clock();
    matrix_mul_sequentially(r, a, b);
    clock_t end_sequentially = clock();
    printf("Time without parallelization: %ld sec \n", (end_sequentially - start_sequentially) / CLOCKS_PER_SEC);

    clock_t start_kernels = clock();
    matrix_mul_kernels(r, a, b);
    clock_t end_kernels = clock();
    printf("Time for kernels: %ld sec \n", (end_kernels - start_kernels) / CLOCKS_PER_SEC);

    clock_t start_parallel_loop = clock();
    matrix_mul_parallel_loop(r, a, b);
    clock_t end_parallel_loop = clock();
    printf("Time for kernels: %ld sec \n", (end_parallel_loop - start_parallel_loop) / CLOCKS_PER_SEC);
}
