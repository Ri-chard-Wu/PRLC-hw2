// g++ vector_normalization.cpp -o vector_normalization -msse2 -mavx -std=c++17

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <valarray>
#include <vector>
// https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/immintrin.h
#include <immintrin.h>
#include <type_traits>

template <typename T>
std::vector<std::vector<T>> generate_vectors(const std::size_t num_vectors,
                                             const std::size_t vector_size,
                                             const unsigned long random_seed)
{
    std::mt19937 gen{random_seed};
    std::uniform_real_distribution<> dist{-1.0, 1.0};

    std::vector<std::vector<T>> data(num_vectors);
    for (std::size_t i = 0; i < num_vectors; i++)
    {
        std::vector<T> vec(vector_size);
        for (std::size_t j = 0; j < vector_size; j++)
        {
            vec.at(j) = dist(gen);
        }
        data.at(i) = vec;
    }
    return data;
}

/*
template<typename T, size_t N>
union f32xN
{
    T m;
    float f[N];
};

// warning: ignoring attributes on template argument
typedef f32xN<__m256, 8> f32x8;
typedef f32xN<__m128, 4> f32x4;
*/

union f32x4
{
    __m128 m;
    float f[4];
};

union f32x8
{
    __m256 m;
    float f[8];
};

template <typename T, size_t N>
void set_f32xN(T& vec, const float* v)
{
    for (std::size_t i = 0; i < N; i++)
    {
        vec.f[i] = v[i];
    }
}

template <typename T, size_t N>
class FloatMatrixN
{
public:
    FloatMatrixN(const std::size_t h, const std::size_t w) : m_h(h), m_w(w)
    {
        m_num_arrays = static_cast<std::size_t>(
            std::ceil(static_cast<float>(m_w) / static_cast<float>(N)));
        m_w_actual = m_num_arrays * N;
        T vec;
        const std::vector<float> zeros(N, 0);
        set_f32xN<T, N>(vec, zeros.data());
        std::vector<T> array(m_h, vec);
        m_data = std::vector<std::vector<T>>(m_num_arrays, array);
    }

    void set_value(const std::size_t i, const std::size_t j, const float val)
    {
        if (i < 0 || i >= m_h)
        {
            throw std::runtime_error("Index i is out of bound.");
        }
        if (j < 0 || j >= m_w)
        {
            throw std::runtime_error("Index j is out of bound.");
        }
        m_data[j / N][i].f[j % N] = val;
    }

    float get_value(const std::size_t i, const std::size_t j) const
    {
        return m_data[j / N][i].f[j % N];
    }

    void normalize()
    {
        const std::vector<float> zeros(N, 0);
        T zero_vec;
        set_f32xN<T, N>(zero_vec, zeros.data());
        for (std::size_t i = 0; i < m_num_arrays; i++)
        {
            std::vector<T>& f32_vec = m_data[i];
            T vec = zero_vec;
            for (std::size_t j = 0; j < m_h; j++)
            {
                if constexpr (std::is_same_v<T, f32x4>)
                {
                    vec.m += _mm_mul_ps(f32_vec[j].m, f32_vec[j].m);
                }
                else if constexpr (std::is_same_v<T, f32x8>)
                {
                    vec.m += _mm256_mul_ps(f32_vec[j].m, f32_vec[j].m);
                }
                else
                {
                    throw std::runtime_error{"Unsupported"};
                }
            }
            if constexpr (std::is_same_v<T, f32x4>)
            {
                vec.m = 1.0f / _mm_sqrt_ps(vec.m);
            }
            else if constexpr (std::is_same_v<T, f32x8>)
            {
                vec.m = 1.0f / _mm256_sqrt_ps(vec.m);
            }
            for (std::size_t j = 0; j < m_h; j++)
            {
                f32_vec[j].m *= vec.m;
            }
        }
    }

private:
    std::vector<std::vector<T>> m_data;
    std::size_t m_h;
    std::size_t m_w;
    std::size_t m_num_arrays;
    std::size_t m_w_actual; // Multiples of N
};

typedef FloatMatrixN<f32x8, 8> FloatMatrix8;
typedef FloatMatrixN<f32x4, 4> FloatMatrix4;

template <typename T>
T convert_vectors_to_matrix(const std::vector<std::vector<float>>& data,
                            const std::size_t num_vectors,
                            const std::size_t vector_size)
{
    T matrix{vector_size, num_vectors};
    for (std::size_t i = 0; i < num_vectors; i++)
    {
        for (std::size_t j = 0; j < vector_size; j++)
        {
            matrix.set_value(j, i, data[i][j]);
        }
    }
    return matrix;
}

template <typename T>
std::vector<std::vector<float>>
convert_matrix_to_vectors(const T& matrix, const std::size_t num_vectors,
                          const std::size_t vector_size)
{
    std::vector<std::vector<float>> converted_data(num_vectors);
    for (std::size_t i = 0; i < num_vectors; i++)
    {
        std::vector<float> row(vector_size);
        for (std::size_t j = 0; j < vector_size; j++)
        {
            row[j] = matrix.get_value(j, i);
        }
        converted_data.at(i) = row;
    }
    return converted_data;
}

template <typename T>
inline void normalize(std::vector<std::vector<T>>& data)
{
    for (std::size_t i = 0; i < data.size(); i++)
    {
        std::vector<T>& row = data[i];
        float square_sum = 0;
        for (std::size_t j = 0; j < row.size(); j++)
        {
            square_sum += row[j] * row[j];
        }
        for (std::size_t j = 0; j < row.size(); j++)
        {
            row[j] /= std::sqrt(square_sum);
        }
    }
}

template <typename T>
std::vector<std::valarray<T>>
convert_vectors_to_valarray(const std::vector<std::vector<T>>& data,
                            const std::size_t num_vectors,
                            const std::size_t vector_size)
{
    std::vector<std::valarray<T>> converted_data(vector_size);
    for (std::size_t i = 0; i < vector_size; i++)
    {
        std::valarray<T> row(num_vectors);
        for (std::size_t j = 0; j < num_vectors; j++)
        {
            row[j] = data[j][i];
        }
        converted_data.at(i) = row;
    }
    return converted_data;
}

template <typename T>
std::vector<std::vector<T>>
convert_valarray_to_vectors(const std::vector<std::valarray<T>>& data,
                            const std::size_t num_vectors,
                            const std::size_t vector_size)
{
    std::vector<std::vector<T>> converted_data(num_vectors);
    for (std::size_t i = 0; i < num_vectors; i++)
    {
        std::vector<T> row(vector_size);
        for (std::size_t j = 0; j < vector_size; j++)
        {
            row[j] = data[j][i];
        }
        converted_data.at(i) = row;
    }
    return converted_data;
}

template <typename T>
inline void normalize(std::vector<std::valarray<T>>& data)
{
    std::valarray<T> square_sum(0.0f, data[0].size());
    for (std::size_t i = 0; i < data.size(); i++)
    {
        square_sum += data[i] * data[i];
    }
    std::valarray<T> multiplier = 1.0f / std::sqrt(square_sum);
    for (std::size_t i = 0; i < data.size(); i++)
    {
        data[i] *= multiplier;
    }
}

template <typename T>
bool is_equivalent(const std::vector<std::vector<T>>& data_1,
                   const std::vector<std::vector<T>>& data_2, const T& atol)
{
    if (data_1.size() != data_2.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < data_1.size(); i++)
    {
        const std::vector<T>& row_1 = data_1.at(i);
        const std::vector<T>& row_2 = data_2.at(i);
        if (row_1.size() != row_2.size())
        {
            return false;
        }
        for (std::size_t j = 0; j < row_1.size(); j++)
        {
            if (std::abs(row_1.at(j) - row_2.at(j)) > atol)
            {
                std::cout << row_1.at(j) << " " << row_2.at(j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main()
{
    const unsigned long random_seed{0};

    const std::size_t num_vectors{25601};
    const std::size_t vector_size{33};
    const float atol{1e-7};

    std::cout << "a\n";

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    // Data for experiments.
    const std::vector<std::vector<float>> vectors =
        generate_vectors<float>(num_vectors, vector_size, random_seed);

    // Make a copy of the data.
    // num_vectors x vector_size
    std::vector<std::vector<float>> normalized_vectors = vectors;
    // Normalize.
    start = std::chrono::high_resolution_clock::now();
    normalize(normalized_vectors);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "Baseline Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << " ns" << std::endl;

    // vector_size x num_vectors
    std::vector<std::valarray<float>> normalized_valarray =
        convert_vectors_to_valarray(vectors, num_vectors, vector_size);
    start = std::chrono::high_resolution_clock::now();
    normalize(normalized_valarray);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "Valarray Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << " ns" << std::endl;
    assert(is_equivalent(normalized_vectors,
                         convert_valarray_to_vectors(normalized_valarray,
                                                     num_vectors, vector_size),
                         atol));

    // vector_size x num_vectors
    FloatMatrix4 normalized_matrix4 = convert_vectors_to_matrix<FloatMatrix4>(
        vectors, num_vectors, vector_size);
    start = std::chrono::high_resolution_clock::now();
    normalized_matrix4.normalize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "SSE Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << " ns" << std::endl;
    assert(is_equivalent(normalized_vectors,
                         convert_matrix_to_vectors<FloatMatrix4>(
                             normalized_matrix4, num_vectors, vector_size),
                         atol));

    // vector_size x num_vectors
    FloatMatrix8 normalized_matrix8 = convert_vectors_to_matrix<FloatMatrix8>(
        vectors, num_vectors, vector_size);
    start = std::chrono::high_resolution_clock::now();
    normalized_matrix8.normalize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "AVX Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << " ns" << std::endl;
    assert(is_equivalent(normalized_vectors,
                         convert_matrix_to_vectors<FloatMatrix8>(
                             normalized_matrix8, num_vectors, vector_size),
                         atol));
}