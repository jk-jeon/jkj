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
#include <type_traits>
#include "portability.h"

namespace jkl {
	// pseudo_ptr is a pointer-like class mainly used for implementations of operator-> of proxy iterators
	// When T is a reference type, it acts essentially like a real pointer
	template <class T>
	class pseudo_ptr {
		T value;
		using rvalue_reference = std::add_rvalue_reference_t<T>;

	public:
		using pointer = std::remove_reference_t<T>*;
		using element_type = T;

		// Move on-the-fly generated value into pseudo_ptr
		constexpr explicit pseudo_ptr(rvalue_reference value) noexcept(std::is_nothrow_move_constructible_v<T>)
			: value(static_cast<rvalue_reference>(value)) {}

		// Dereference operator
		NONCONST_CONSTEXPR T operator*() && noexcept(std::is_nothrow_move_constructible_v<T>) {
			return static_cast<rvalue_reference>(value);
		}
		
		// Member access operator
		constexpr decltype(auto) operator->() noexcept {
			return &value;
		}
	};
}