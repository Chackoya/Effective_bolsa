#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>
#include <chrono>
#include <thread>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#define NUM_THREADS	8
#define DIMS 3
#define RUNS 100
#define DIM0 4096
#define	DIM1 4096
#define DIM2 4
#define MIN_DOMAIN -1.0f
#define MAX_DOMAIN 1.0f
#define DOMAIN_DELTA (1.0f / (MAX_DOMAIN - MIN_DOMAIN))
#define SIZE (DIM0 * DIM1 * DIM2)
#define BYTES (SIZE * sizeof(float))
#define TIND(x) (x * SIZE)
#define DEBUG 0
#define SIZE_PER_THREAD (SIZE / NUM_THREADS)

void init_x(float *a, const int N);
void init_y(float *a, const int N);
void init_z(float *a, const int N);
void print_firstn_array(float *a, const int N);
void randbits_init_array(float *a, int bits, const int N);
void zero_out_array(float *a, const int N);
void add_array(float *a,  float *b);
void warp(float *tensors, float *image, float *result, float *dimensions);
void warp_avx2(float *tensors, float *image, float *result, float *dimensions);
void warp_avx2_mt(float *tensors, float *image, float *result, float *dimensions, int tind);
int array_equal(float *a, float *b, const int N);


int main() {

	// Allocate memory on host
	float *tensors = (float *)malloc(DIMS * BYTES); // tensor array
	float *image = (float *)malloc(BYTES);			// image to warp
	float *result = (float *)malloc(BYTES);			// where to store the result of the warp operator
	float dimensions[DIMS] = { DIM0, DIM1, DIM2 };	// dimensions
	float durations = 0.0f;

	// Initialize stuff
	init_x(tensors + TIND(0), SIZE);
	init_y(tensors + TIND(1), SIZE);
	init_z(tensors + TIND(2), SIZE);
	zero_out_array(result, SIZE);
	randbits_init_array(image, 8, SIZE);

	/*
	printf("X tensor: \n");
	print_firstn_array(tensors, SIZE);
	printf("Y tensor: \n");
	print_firstn_array(tensors + TIND(1), SIZE);
	printf("Z tensor: \n");
	print_firstn_array(tensors + TIND(2), SIZE);
	printf("Image: \n");
	print_firstn_array(image, SIZE);
	*/

	printf("Tensor resolution:\t[%d, %d, %d]\n", DIM0, DIM1, DIM2);
	printf("Evaluation points:\t%d\n", (DIM0 * DIM1 * DIM2));


	printf("Number of threads:\t%d\n", NUM_THREADS); // t
	std::thread myThreads[NUM_THREADS];


	printf("\n");

	for (int n = 0; n < RUNS; ++n) {
		auto start = std::chrono::high_resolution_clock::now();
		//warp(tensors, image, result, dimensions);
		//warp_avx2(tensors, image, result, dimensions);
		for (int i = 0; i < NUM_THREADS; ++i) myThreads[i] = std::thread(warp_avx2_mt, tensors, image, result, dimensions, i); // t
		for (int i = 0; i < NUM_THREADS; i++) myThreads[i].join(); // t

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		durations += (float)(duration.count() / 1000.0);
	}

	printf("Number of runs  :\t%d\n", RUNS);
	printf("Total time taken:\t%.6f ms.\n", durations);
	printf("Avg warp op took:\t%.6f ms.\n", (durations / RUNS));

	int errors = array_equal(image, result, SIZE);
	printf("\nWarp identity: %s (%d errors)\n", ((errors == 0) ? "Passed" : "Failed"), errors);

	printf("\nImage: \n");
	print_firstn_array(image, 10);

	printf("\nResult: \n");
	print_firstn_array(result, 10);

	return 0;
}

int array_equal(float *a, float *b, const int N) {
	int count = 0;
	for (int i = 0; i < N; ++i) {
		if (a[i] != b[i]) {
			++count;
		}
	}
	return count;
}

void init_x(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (float)(i / ((DIM0 - 1.0) * 0.5)) - 1.0;
			}
		}
	}
}

void init_y(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (j / ((DIM1 - 1.0) * 0.5)) - 1.0;
			}
		}
	}
}

void init_z(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (k / ((DIM2 - 1) * 0.5)) - 1.0;
			}
		}
	}
}

void print_firstn_array(float *a, const int N) {
	for (int i = 0; i < N; ++i) {
		printf("%.3f ", a[i]);
	}
	printf("\n");
}

void randbits_init_array(float *a, int bits, const int N) {
	srand(1234567890);
	int mask = (1 << bits) - 1;
	for (int i = 0; i < N; ++i) {
		a[i] = (float)(rand() & mask);
	}
}

void zero_out_array(float *a, const int N) {
	for (int i = 0; i < N; ++i) {
		a[i] = 0.0;
	}
}

void add_array(float *a, float *b) {
	for (int i = 0; i < SIZE; ++i) {
		a[i] += b[i];
	}
}


#define __COORD(x) lroundf((fmaxf(fminf(tensors[TIND(x) + i], MAX_DOMAIN), MIN_DOMAIN) - MIN_DOMAIN) * (dimensions[x] * DOMAIN_DELTA - DOMAIN_DELTA))

// tested
void warp(float *tensors, float *image, float *result, float *dimensions) {
	for (int i = 0; i < SIZE; ++i) {
		int temp = __COORD(0);
		for (int j = 1; j < DIMS; ++j) {
			temp = (int)dimensions[j] * temp + __COORD(j);
		}
		result[i] = image[temp];
	}
}


#define __MM256_COOORS(x) _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_fmsub_ps(_mm256_set1_ps(dimensions[x]), _domain_delta, _domain_delta), _mm256_sub_ps(_mm256_max_ps(_mm256_min_ps(_mm256_load_ps(tensors + TIND(x) + i), _max_domain), _min_domain), _min_domain)))

void warp_avx2(float *tensors, float *image, float *result, float *dimensions) {
	__m256 _min_domain = _mm256_set1_ps(MIN_DOMAIN);
	__m256 _max_domain = _mm256_set1_ps(MAX_DOMAIN);
	__m256 _domain_delta = _mm256_set1_ps(DOMAIN_DELTA);

	for (int i = 0; i < SIZE; i += 8) {

		__m256i _temp = __MM256_COOORS(0);
		for (int j = 1; j < DIMS; ++j) {
			_temp = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtps_epi32(_mm256_set1_ps(dimensions[j])), _temp), __MM256_COOORS(j));
		}
		for (int k = 0; k < 8; ++k) {
			result[i + k] = image[_temp.m256i_i32[k]];
		}

	}
}

void warp_avx2_mt(float *tensors, float *image, float *result, float *dimensions, int tind) {
	__m256 _min_domain = _mm256_set1_ps(MIN_DOMAIN);
	__m256 _max_domain = _mm256_set1_ps(MAX_DOMAIN);
	__m256 _domain_delta = _mm256_set1_ps(DOMAIN_DELTA);

	for (int i = SIZE_PER_THREAD * tind; i < SIZE_PER_THREAD * (tind + 1); i += 8) {

		__m256i _temp = __MM256_COOORS(0);
		for (int j = 1; j < DIMS; ++j) {
			_temp = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtps_epi32(_mm256_set1_ps(dimensions[j])), _temp), __MM256_COOORS(j));
		}
		for (int k = 0; k < 8; ++k) {
			result[i + k] = image[_temp.m256i_i32[k]];
		}

	}
}
