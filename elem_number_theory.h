/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2017 Junekey Jeon                                                                   ///
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
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "portability.h"

namespace jkl {
	namespace math {
		/// Calculate integer square root using Newton's method
		/// Note that for most of the cases this is much slower than UIntType(std::sqrt(n)).
		/// Use this function only when floating point operations should be avoided or
		/// can be error-prone, e.g., when n is extremely large.
		template <class UIntType>
		JKL_GPU_EXECUTABLE UIntType isqrt(UIntType const& n) noexcept
			(noexcept(std::declval<UIntType>() + std::declval<UIntType>()) &&
				noexcept(std::declval<UIntType>() / std::declval<UIntType>()))
		{
			UIntType x = n;
			UIntType x_new;
			while( true ) {
				x_new = (x + (n / x)) / 2;
				if( x_new == x || x_new == x + 1 )
					return x;

				x = std::move(x_new);
			}
		}
	}
}