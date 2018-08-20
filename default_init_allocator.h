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
#include <memory>
#include <type_traits>

namespace jkl {
	// Removed all members deprecated in C++17
	template <class T, class BaseAllocator = std::allocator<T>>
	class default_init_allocator : public BaseAllocator {
	public:
		using BaseAllocator::BaseAllocator;
	};
}

namespace std {
	template <class T, class BaseAllocator>
	class allocator_traits<jkl::default_init_allocator<T, BaseAllocator>>
		: public allocator_traits<BaseAllocator>
	{
		using base_traits = allocator_traits<BaseAllocator>;

	public:
		using allocator_type = typename base_traits::allocator_type;

		template <class U>
		using rebind_alloc = jkl::default_init_allocator<U,
			typename base_traits::template rebind_alloc<U>>;

		template <class U>
		using rebind_traits = allocator_traits<rebind_alloc<U>>;

		template <class U, class... Args>
		static void construct(allocator_type& a, U* p, Args&&... args)
			noexcept(noexcept(base_traits::construct(
				static_cast<BaseAllocator&>(a), p, std::forward<Args>(args)...)))
		{
			base_traits::construct(
				static_cast<BaseAllocator&>(a), p, std::forward<Args>(args)...);
		}

		template <class U>
		static void construct(allocator_type&, U* p)
			noexcept(std::is_nothrow_default_constructible_v<U>)
		{
			new(p) U;
		}
	};
}