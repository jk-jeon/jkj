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
#include <tuple>
#include "is_complete.h"
#include "remove_cvref.h"
#include "../portability.h"

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Check if a type supports std::tuple_element and std::tuple_size
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		// How std::get-like functionality is provided?
		// Note that even if there is not "get", is_tuple concludes that T is indeed tuple-like,
		// if std::tuple_element and std::tuple are both specialized.
		// If real tuple-like access is neccessary, the static member which_get should be inspected.
		// Or, you can use jkl::tmp::get() function that does that for you.
		// Be carefult that jkl::tmp::get() will not work if the given tuple-like class is defined inside
		// the namespace jkl::tmp and is relying on ADL.
		enum class which_get_t { none, member, adl };

		template <typename T>
		struct is_tuple {
		private:
			template <bool has_size, class = void>
			struct impl {
				static constexpr std::size_t size = std::tuple_size<T>::value;

				template <bool length_zero, class = void>
				struct check_get {
					template <class U, class = decltype(std::declval<U>().template get<0>())>
					static constexpr which_get_t check(int) { return which_get_t::member; }

					struct no_adl_get {};
					template <std::size_t I, class U>
					no_adl_get get(U&&);

					template <class U>
					static constexpr which_get_t check(float) {
						using std::get;
						using result_type = decltype(get<0>(std::declval<U>()));
						return std::conditional_t<std::is_same<result_type, no_adl_get>::value,
							std::integral_constant<which_get_t, which_get_t::none>,
							std::integral_constant<which_get_t, which_get_t::adl>>::value;
					}

					static constexpr which_get_t value = check<T>(0);
				};

				template <class dummy>
				struct check_get<true, dummy> {
					static constexpr which_get_t value = which_get_t::none;
				};

				template <bool length_zero, class dummy = void>
				struct check_tuple_element {
					static constexpr bool value = is_complete<std::tuple_element<0, T>>::value;
				};

				template <class dummy>
				struct check_tuple_element<true, dummy> {
					static constexpr bool value = true;
				};

				static constexpr bool value = check_tuple_element<size == 0>::value;
				static constexpr which_get_t which_get = check_get<size == 0>::value;
			};

			template <class dummy>
			struct impl<false, dummy> {
				static constexpr bool value = false;
				static constexpr which_get_t which_get = which_get_t::none;
			};

		public:
			static constexpr bool value = impl<is_complete<std::tuple_size<T>>::value>::value;
			static constexpr which_get_t which_get = impl<is_complete<std::tuple_size<T>>::value>::which_get;
		};

		template <class T>
		static constexpr bool is_tuple_v = is_tuple<T>::value;

		template <std::size_t I, class T, class = std::enable_if_t<
			is_tuple<remove_cvref_t<T>>::which_get == which_get_t::member>>
		FORCEINLINE constexpr decltype(auto) get(T&& x)
			noexcept(noexcept(std::forward<T>(x).template get<I>()))
		{
			return std::forward<T>(x).template get<I>();
		}

		namespace detail {
			template <std::size_t I, class T>
			struct is_adl_get_noexcept {
				static constexpr bool inspect() noexcept {
					using std::get;
					return noexcept(get<I>(std::declval<T>()));
				}
			};
		}

		template <std::size_t I, class T, class = void, class = std::enable_if_t<
			is_tuple<remove_cvref_t<T>>::which_get == which_get_t::adl>>
		FORCEINLINE constexpr decltype(auto) get(T&& x) noexcept(detail::is_adl_get_noexcept<I, T>::inspect())
		{
			using std::get;
			return get<I>(std::forward<T>(x));
		}
	}
}