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

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Brace-initialization version of std::is_constructible
		// Minor modification of:
		// https://stackoverflow.com/questions/20860535/is-convertible-for-multiple-arguments
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <typename To_, typename... From_>
			struct is_braces_constructible {
			private:
				template <typename To, typename... From>
				struct tag {};

				template <typename To, typename... From>
				static auto test(tag<To, From...>)
					-> decltype(To{ std::declval<From>()... }, std::true_type());
				static auto test(...)->std::false_type;

			public:
				static constexpr bool value = decltype(test(tag<To_, From_...>()))::value;
			};
		}
		template <typename To, typename... From>
		struct is_braces_constructible : std::integral_constant<bool,
			detail::is_braces_constructible<To, From...>::value> {};

		template <typename To, typename... From>
		static constexpr bool is_braces_constructible_v = is_braces_constructible<To, From...>::value;

		namespace detail {
			template <bool is_constructible, typename To, typename... From>
			struct is_nothrow_braces_constructible : std::false_type {};

			template <typename To, typename... From>
			struct is_nothrow_braces_constructible<true, To, From...>
				: std::integral_constant<bool, noexcept(To{ std::declval<From>()... })> {};
		}
		template <typename To, typename... From>
		struct is_nothrow_braces_constructible :
			detail::is_nothrow_braces_constructible<is_braces_constructible_v<To, From...>, To, From...> {};

		template <typename To, typename... From>
		static constexpr bool is_nothrow_braces_constructible_v = is_nothrow_braces_constructible<To, From...>::value;
	}
}