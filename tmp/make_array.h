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
#include <array>
#include <type_traits>
#include <utility>

namespace jkj {
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// std::make_array of C++20 (brought from http://en.cppreference.com/w/cpp/experimental/make_array)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {
		template<class> struct is_ref_wrapper : std::false_type {};
		template<class T> struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type {};

		template <class D, class...> struct return_type_helper { using type = D; };
		template <class FirstType, class... RemainingTypes>
		struct return_type_helper<void, FirstType, RemainingTypes...> :
			std::common_type<FirstType, RemainingTypes...> {
			static constexpr bool check = !is_ref_wrapper<FirstType>::value &&
				return_type_helper<void, RemainingTypes...>::check;
			static_assert(check, "jkj::tmp: Types cannot contain reference_wrappers when D is void");
		};
		template <>
		struct return_type_helper<void> {
			static constexpr bool check = true;
		};

		template <class D, class... Types>
		using return_type = std::array<typename return_type_helper<D, Types...>::type, sizeof...(Types)>;
	}

	template < class D = void, class... Types>
	constexpr detail::return_type<D, Types...> make_array(Types&&... t) {
		return{ std::forward<Types>(t)... };
	}
}