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

#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include "../portability.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Boost::Hana-style compile-time integer
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <class T, T x>
		struct compile_time_number;

		template <bool b>
		using compile_time_bool = compile_time_number<bool, b>;

		template <class T, T x>
		struct compile_time_number : std::integral_constant<T, x> {
			JKL_GPU_EXECUTABLE constexpr auto operator+() const noexcept {
				return compile_time_number<decltype(+x), +x>{};
			}
			JKL_GPU_EXECUTABLE constexpr auto operator-() const noexcept {
				return compile_time_number<decltype(-x), -x>{};
			}
			JKL_GPU_EXECUTABLE constexpr auto operator~() const noexcept {
				return compile_time_number<decltype(~x), ~x>{};
			}

		#define JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(op) template <class U, U y>\
			JKL_GPU_EXECUTABLE constexpr auto operator op(compile_time_number<U, y> that) const noexcept {\
				return compile_time_number<decltype(x op y), (x op y)>{};\
			}

			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(+);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(-);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(*);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(/);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(%);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(&);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(|);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(^);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(==);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(!=);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(<=);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(>=);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(<);
			JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR(>);

		#undef JKL_COMPILE_TIME_NUMBER_DEFINE_BINARY_OPERATOR
		};

		template <bool b>
		JKL_GPU_EXECUTABLE constexpr auto operator!(compile_time_bool<b>) noexcept {
			return compile_time_bool<!b>{};
		}

		template <bool b1, bool b2>
		JKL_GPU_EXECUTABLE constexpr auto operator&&(compile_time_bool<b1>, compile_time_bool<b2>) noexcept {
			return compile_time_bool<b1 && b2>{};
		}

		template <bool b1, bool b2>
		JKL_GPU_EXECUTABLE constexpr auto operator||(compile_time_bool<b1>, compile_time_bool<b2>) noexcept {
			return compile_time_bool<b1 || b2>{};
		}
	}
}