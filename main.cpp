#include <iostream>
#include<emmintrin.h>
#include<time.h>
#include<Windows.h>
#include <immintrin.h>
using namespace std;
const int N = 1000;
float elm[N][N] = {0};

void reset(float **test)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            test[i][j] = elm[i][j];
        }
    }
    return;
}

void gaussian_naive(float **m, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = i + 1; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            for(int k = i + 1; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

// SSE SIMD optimization
void gaussian_sse(float **m, int n)
{
    __m128 t1, t2, t3;
    for(int i = 0; i < n; i++)
    {
        float t[4] = {m[i][i], m[i][i], m[i][i], m[i][i]};
        t2 = _mm_loadu_ps(t);
        int j = i + 1;
        j = j / 4 * 4; // align
        for(j; j < n - 4; j += 4)
        {
            t1 = _mm_loadu_ps(m[i] + j);
            t3 = _mm_div_ps(t1, t2);
            _mm_storeu_ps(m[i] + j, t3);
        }
        for(j; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            float temp2[4] = {m[j][i], m[j][i], m[j][i], m[j][i]};
            t2 = _mm_loadu_ps(temp2);
            int k = i + 1;
            k = k / 4 * 4; // align
            for(k; k < n - 4; k+=4)
            {
                t1 = _mm_loadu_ps(m[i] + k);
                t1 = _mm_mul_ps(t1, t2);
                t3 = _mm_loadu_ps(m[j] + k);
                t3 = _mm_sub_ps(t3, t1);
                _mm_storeu_ps(m[j] + k, t3);
            }
            for(k; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

// SSE Align SIMD optimization
void gaussian_sse_align(float **m, int n)
{
    __m128 t1, t2, t3;
    for(int i = 0; i < n; i++)
    {
        float t[4] = {m[i][i], m[i][i], m[i][i], m[i][i]};
        t2 = _mm_loadu_ps(t);
        int j = i + 1;
        for(j; j < n - 4; j += 4)
        {
            t1 = _mm_loadu_ps(m[i] + j);
            t3 = _mm_div_ps(t1, t2);
            _mm_storeu_ps(m[i] + j, t3);
        }
        for(j; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            float temp2[4] = {m[j][i], m[j][i], m[j][i], m[j][i]};
            t2 = _mm_loadu_ps(temp2);
            int k = i + 1;
            for(k; k < n - 4; k+=4)
            {
                t1 = _mm_loadu_ps(m[i] + k);
                t1 = _mm_mul_ps(t1, t2);
                t3 = _mm_loadu_ps(m[j] + k);
                t3 = _mm_sub_ps(t3, t1);
                _mm_storeu_ps(m[j] + k, t3);
            }
            for(k; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

// AVX SIMD optimization
void gaussian_avx(float **m, int n)
{
    __m256 t1, t2, t3;
    for(int i = 0; i < n; i++)
    {
        float t[8] = {m[i][i], m[i][i], m[i][i], m[i][i], m[i][i], m[i][i], m[i][i], m[i][i]};
        t2 = _mm256_loadu_ps(t);
        int j = i + 1;
        for(j; j < n - 8; j += 8)
        {
            t1 = _mm256_loadu_ps(m[i] + j);
            t3 = _mm256_div_ps(t1, t2);
            _mm256_storeu_ps(m[i] + j, t3);
        }
        for(j; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            float temp2[8] = {m[j][i], m[j][i], m[j][i], m[j][i]};
            t2 = _mm256_loadu_ps(temp2);
            int k = i + 1;
            for(k; k < n - 8; k+=8)
            {
                t1 = _mm256_loadu_ps(m[i] + k);
                t1 = _mm256_mul_ps(t1, t2);
                t3 = _mm256_loadu_ps(m[j] + k);
                t3 = _mm256_sub_ps(t3, t1);
                _mm256_storeu_ps(m[j] + k, t3);
            }
            for(k; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

int main()
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            elm[i][j] = (rand() % 100);
        }
    }
    float** test = new float*[N];
    for (int i = 0; i < N; i++)
    {
        test[i] = new float[N];
    }

    reset(test);

    srand(time(NULL));
    LARGE_INTEGER timeStart;
    LARGE_INTEGER timeEnd;

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    double quadpart = (double)frequency.QuadPart;

    //naive
    QueryPerformanceCounter(&timeStart);
    gaussian_naive(test, N);
    QueryPerformanceCounter(&timeEnd);
    double _Simple = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    printf("Simple:%f\n", _Simple);
//    for(int i =0; i < 10; i++)
//    {
//        for(int j = 0; j < 10; j++)
//            cout << test[i][j] << " ";
//        cout <<endl;
//    }
    cout << endl;
    reset(test);

    //SSE
    QueryPerformanceCounter(&timeStart);
    gaussian_sse(test, N);
    QueryPerformanceCounter(&timeEnd);
    double _SSE_Gauss = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    printf("SSE_Gauss:%f\n", _SSE_Gauss);
    cout << endl;
    reset(test);

    //AVX
    QueryPerformanceCounter(&timeStart);
    gaussian_avx(test, N);
    QueryPerformanceCounter(&timeEnd);
    double _AVX_Gauss = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    printf("AVX_Gauss:%f\n", _AVX_Gauss);
//    for(int i =0; i < 10; i++)
//    {
//        for(int j = 0; j < 10; j++)
//            cout << test[i][j] << " ";
//        cout <<endl;
//    }
    cout << endl;
    reset(test);

    system("pause");
    return 0;
}

