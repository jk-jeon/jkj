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

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Multiple argument version of std::is_convertible
		// This can be necessary for detecting if a constructor is explicit or not, because
		// std::is_constructible does not care explicit-ness.
		// This implementation is brought from
		// https://stackoverflow.com/questions/20860535/is-convertible-for-multiple-arguments
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename To_, typename... From_>
		struct is_convertible {
		private:
			template <typename To>
			struct indirector {
				indirector(To);
			};

			template <typename To, typename... From>
			struct tag {};

			template <typename To, typename... From>
			static auto test(tag<To, From...>)
				-> decltype(indirector<To>({ std::declval<From>()... }), std::true_type());
			static auto test(...)
				->std::false_type;

		public:
			static constexpr bool value = decltype(test(tag<To_, From_...>()))::value;
		};

		template <typename To, typename... From>
		static constexpr bool is_convertible_v = is_convertible<To, From...>::value;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Check if static_cast is possible
		// This implementation is a minor modification of the above is_convertible.
		// I think the convention <From, To> is terrible and <To, From> is far better, but
		// std::is_convertible already uses that convention...
		// NOTE: The result of is_explicitly_convertible and std::is_constructible are in general, NOT same.
		//       For example, std::is_constructible_v<int&&, int&> is false, while
		//       is_explicitly_convertible<int&, int&&> is true.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename From_, typename To_>
		struct is_explicitly_convertible {
		private:
			template <typename To, typename From>
			struct tag {};

			template <typename To, typename From>
			static auto test(tag<To, From>)
				-> decltype(static_cast<To>(std::declval<From>()), std::true_type());
			static auto test(...)
				->std::false_type;

		public:
			static constexpr bool value = decltype(test(tag<To_, From_>()))::value;
		};

		template <typename From, typename To>
		static constexpr bool is_explicitly_convertible_v = is_explicitly_convertible<From, To>::value;
	}
}