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

/// This header file defines the class rvalue_ptr, which is the "rvalue pointer",
/// which is the concept corresponding to that of rvalue reference.

#pragma once
#include <cstddef>
#include <iterator>
#include <type_traits>
#include "portability.h"

namespace jkj {
	// rvalue_ptr is a pointer that is dereferenced to an rvalue reference
	// There is no special treatment for array types
	// T should not be a reference
	template <class T>
	class rvalue_ptr {
		T* ptr;
	public:
		using pointer = T*;
		using element_type = T;

		// Default constructor - initialize to a garbage value
		rvalue_ptr() noexcept = default;
		// Initialize from nullptr
		constexpr rvalue_ptr(std::nullptr_t) noexcept : ptr{ nullptr } {}
		// Explicit conversion from an lvalue pointer
		constexpr explicit rvalue_ptr(T* ptr) noexcept : ptr{ ptr } {}
		// Initialize from an rvalue pointer of different type
		template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
		constexpr rvalue_ptr(rvalue_ptr<U> that) noexcept : ptr{ that.ptr } {}

		// Assignment from nullptr
		NONCONST_CONSTEXPR rvalue_ptr& operator=(std::nullptr_t) noexcept {
			ptr = nullptr;
			return *this;
		}
		// Assignment from an rvalue pointer to a different type
		template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
		NONCONST_CONSTEXPR rvalue_ptr& operator=(rvalue_ptr<U> that) noexcept {
			ptr = that.ptr;
			return *this;
		}

		// Dereference operator
		constexpr T&& operator*() const noexcept {
			return std::move(*ptr);
		}

		// Subscript operator
		constexpr T&& operator[](std::ptrdiff_t pos) const noexcept {
			return std::move(ptr[pos]);
		}

		// Member access operator
		// This operator cannot be implemented as there is no "real" rvalue pointer
		//constexpr T* operator->() const noexcept {
		//	return ptr;
		//}

		// Increment operators
		NONCONST_CONSTEXPR rvalue_ptr& operator++() noexcept {
			++ptr;
			return *this;
		}
		NONCONST_CONSTEXPR rvalue_ptr operator++(int) noexcept {
			return rvalue_ptr{ ptr++ };
		}
		NONCONST_CONSTEXPR rvalue_ptr& operator+=(std::ptrdiff_t n) noexcept {
			ptr += n;
			return *this;
		}
		// Decrement operators
		NONCONST_CONSTEXPR rvalue_ptr& operator--() noexcept {
			--ptr;
			return *this;
		}
		NONCONST_CONSTEXPR rvalue_ptr operator--(int) noexcept {
			return rvalue_ptr{ ptr-- };
		}
		NONCONST_CONSTEXPR rvalue_ptr& operator-=(std::ptrdiff_t n) noexcept {
			ptr -= n;
			return *this;
		}

		// Plus
		constexpr rvalue_ptr operator+() const noexcept {
			return *this;
		}
		constexpr rvalue_ptr operator+(std::ptrdiff_t n) const noexcept {
			return rvalue_ptr{ ptr + n };
		}

		// Minus
		constexpr rvalue_ptr operator-(std::ptrdiff_t n) const noexcept {
			return rvalue_ptr{ ptr - n };
		}
		constexpr std::ptrdiff_t operator-(rvalue_ptr that) const noexcept {
			return ptr - that.ptr;
		}
		constexpr std::ptrdiff_t operator-(T* that) const noexcept {
			return ptr - that;
		}
		friend constexpr std::ptrdiff_t operator-(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 - ptr2.ptr;
		}

		// Unary not
		constexpr bool operator!() const noexcept {
			return !ptr;
		}

		// Relations
		constexpr bool operator==(rvalue_ptr that) const noexcept {
			return ptr == that.ptr;
		}
		constexpr bool operator==(T* that) const noexcept {
			return ptr == that;
		}
		friend constexpr bool operator==(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 == ptr2.ptr;
		}


		constexpr bool operator!=(rvalue_ptr that) const noexcept {
			return ptr != that.ptr;
		}
		constexpr bool operator!=(T* that) const noexcept {
			return ptr != that;
		}
		friend constexpr bool operator!=(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 != ptr2.ptr;
		}

		constexpr bool operator<(rvalue_ptr that) const noexcept {
			return ptr < that.ptr;
		}
		constexpr bool operator<(T* that) const noexcept {
			return ptr < that;
		}
		friend constexpr bool operator<(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 < ptr2.ptr;
		}

		constexpr bool operator>(rvalue_ptr that) const noexcept {
			return ptr > that.ptr;
		}
		constexpr bool operator>(T* that) const noexcept {
			return ptr > that;
		}
		friend constexpr bool operator>(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 > ptr2.ptr;
		}

		constexpr bool operator<=(rvalue_ptr that) const noexcept {
			return ptr <= that.ptr;
		}
		constexpr bool operator<=(T* that) const noexcept {
			return ptr <= that;
		}
		friend constexpr bool operator<=(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 <= ptr2.ptr;
		}

		constexpr bool operator>=(rvalue_ptr that) const noexcept {
			return ptr >= that.ptr;
		}
		constexpr bool operator>=(T* that) const noexcept {
			return ptr >= that;
		}
		friend constexpr bool operator>=(T* ptr1, rvalue_ptr ptr2) noexcept {
			return ptr1 >= ptr2.ptr;
		}
	};

	// Helper
	template <class T>
	constexpr rvalue_ptr<T> make_rvalue_ptr(T* ptr) noexcept {
		return rvalue_ptr<T>{ ptr };
	}
}

template <class T>
struct std::iterator_traits<jkj::rvalue_ptr<T>> {
	using difference_type = std::ptrdiff_t;
	using value_type = std::remove_cv_t<T>;
	using pointer = T*;
	using reference = T&&;
	using iterator_category = std::random_access_iterator_tag;
};