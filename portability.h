/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018 Junekey Jeon                                                                   ///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this software     ///
/// and associated documentation files(the "Software"), to deal in the Software without restriction,  ///
/// including without limitation the rights to use, copy, modify, merge, publish, distribute,         ///
/// sublicense, and / or sell copies of the Software, and to permit persons to whom the Software is   ///
/// furnished to do so, subject to the following conditions :                                         ///
///                                                                                                   ///
/// The above copyright notice and this permission notice shall be included in all copies or          ///
/// substantial portions of the Software.                                                             ///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING     ///
/// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND        ///
/// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,       ///
/// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    ///
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           ///
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/// This header file defines a number of preprocessor to match some compiler/platform specific 
/// attributes and etc..

#pragma once

/// Brought from http://www.codeproject.com/Tips/363337/Some-Handy-Visual-Cplusplus-Pre-Processor-Macros
/// + some modifications

// Cross-compilation helpers
#if defined(_MSC_VER)
#  define LIB_EXPORT __declspec(dllexport)
#  define LIB_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#  define LIB_EXPORT /* */
#  define LIB_IMPORT extern
#endif
#
#if defined(__cplusplus)
#  define EXTERN_C extern "C"
#else
#  define EXTERN_C /* */
#endif

// Calling convention helpers
#if defined(_MSC_VER)
#  define STDCALL __stdcall
#  define CDECL __cdecl
#  define FASTCALL __fastcall
#elif defined(__GNUC__)
#  define STDCALL  __attribute__((stdcall))
#  define CDECL /* */
#  define FASTCALL __attribute__((fastcall))
#endif 

// Function inlining
#if defined(__CUDA_ARCH__)
#  define FORCEINLINE __forceinline__
#elif defined(_MSC_VER)
#  define FORCEINLINE __forceinline
#elif defined(__GNUC__)
#  define FORCEINLINE __attribute__((always_inline))
#endif 

// Memory alignment
#if defined(__cplusplus)
#if __cplusplus >= 201103L
#  define ALIGNAS(x) alignas(x)
#elif defined(__CUDACC__)
#  define ALIGNAS(x) __align__(x)
#elif defined(_MSC_VER)
#  define ALIGNAS(x) __declspec(align(x))
#elif defined(__GNUC__)
#  define ALIGNAS(x) __attribute__((aligned(x)))
#endif
#else
#if __STDC_VERSION__ >= 201112L
#  define ALIGNAS(x) _Alignas(x)
#elif defined(__CUDACC__)
#  define ALIGNAS(x) __align__(x)
#elif defined(_MSC_VER)
#  define ALIGNAS(x) __declspec(align(x))
#elif defined(__GNUC__)
#  define ALIGNAS(x) __attribute__((aligned(x)))
#endif
#endif

// Visual C++ by default generates suboptimal memory layout for
// classes with multiple inheritances from empty base classes.
// As they couldn't fix this default behavior in order to maintain compatibility,
// they added an attribute that modifies this.
#if defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 180030501
#define EMPTY_BASE_OPT __declspec(empty_bases)
#else
#define EMPTY_BASE_OPT
#endif

// MSVC's templated function call operator has a bug
// See https://connect.microsoft.com/VisualStudio/feedback/details/1184914/rejected-template-keyword-qualifier-on-template-operator-instanciation
#if defined(_MSC_VER) && _MSC_VER <= 1900 && !defined(__clang__) && !defined(__CUDACC__)
#  define TEMPLATE_FUNCTION_CALL_OPERATOR operator()
#else
#  define TEMPLATE_FUNCTION_CALL_OPERATOR template operator()
#endif


// Portable way to silence a warning about unused variable
// See http://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables
#ifdef __cplusplus
template <typename... Args> void UNUSED(Args&&...) {}
#else
void UNUSED(...) {}
#endif
// Sample usage:
// int main(int argc, char* argv[])
// {
// 	UNUSED(argc, argv);
// 	return 0;
// }


// Feature testing for generalized constexpr
#if defined(__cpp_constexpr)
#if __cpp_constexpr >= 201603L
#define CONSTEXPR_LAMBDA constexpr
#define HAS_CONSTEXPR_LAMBDA
#else
#define CONSTEXPR_LAMBDA
#endif
#if __cpp_constexpr >= 201304L
#define GENERALIZED_CONSTEXPR constexpr
#define NONCONST_CONSTEXPR constexpr
#define HAS_GENERALIZED_CONSTEXPR
#else
#define GENERALIZED_CONSTEXPR
#define NONCONST_CONSTEXPR
#endif
#else
#if defined(__cplusplus)
#if defined(_MSC_VER)
#if _MSC_VER <= 1900
#define CONSTEXPR_LAMBDA
#define GENERALIZED_CONSTEXPR
#define NONCONST_CONSTEXPR
#elif _MSC_VER <= 1910
#define CONSTEXPR_LAMBDA
#define GENERALIZED_CONSTEXPR constexpr
#define NONCONST_CONSTEXPR constexpr
#define HAS_GENERALIZED_CONSTEXPR
#else
#define CONSTEXPR_LAMBDA constexpr
#define GENERALIZED_CONSTEXPR constexpr
#define NONCONST_CONSTEXPR constexpr
#define HAS_CONSTEXPR_LAMBDA
#define HAS_GENERALIZED_CONSTEXPR
#endif
#else
#define CONSTEXPR_LAMBDA constexpr
#define GENERALIZED_CONSTEXPR constexpr
#define NONCONST_CONSTEXPR constexpr
#define HAS_CONSTEXPR_LAMBDA
#define HAS_GENERALIZED_CONSTEXPR
#endif
#endif
#endif


// A feature test for std::void_t
#if defined(__cplusplus) && __cplusplus >= 201703L
#define VOID_T ::std::void_t
#elif defined(_MSC_VER) && _MSC_VER >= 1900
#define VOID_T ::std::void_t
#else
namespace jkl {
	template <class...>
	using void_t = void;
}
#define VOID_T ::jkl::void_t
#endif


// To make some simple functions available in CUDA
#ifdef __NVCC__
#include <cuda_runtime.h>
#define JKL_GPU_EXECUTABLE __host__ __device__
#define JKL_GPU_ONLY __device__
#else
#define JKL_GPU_EXECUTABLE
#define JKL_GPU_ONLY
#endif