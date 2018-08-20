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
#include <functional>
#include <type_traits>
#include <utility>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// This header file is a collection of utilities that are related to forwarding in a generic context.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// If the argument is an lvalue reference, wrap it into std::reference_wrapper.
		// Otherwise, do nothing. Main purpose is to be used in a generic context
		// where an argument should be passed as a reference to functions such as std::async
		// which deliberately remove reference qualifiers.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <typename T, bool = std::is_lvalue_reference<T>::value>
			struct ref_if_lvalue_helper;

			template <typename T>
			struct ref_if_lvalue_helper<T, true>
			{
				static decltype(auto) get(T&& x) noexcept {
					return std::ref(x);
				}
			};

			template <typename T>
			struct ref_if_lvalue_helper<T, false>
			{
				static decltype(auto) get(T&& x) noexcept {
					return std::forward<T>(x);
				}
			};

			template <typename T, bool b>
			struct ref_if_lvalue_helper<std::reference_wrapper<T>, b> {
				static decltype(auto) get(std::reference_wrapper<T> x) noexcept {
					return x;
				}
			};

			template <typename T, bool = std::is_lvalue_reference<T>::value>
			struct cref_if_lvalue_helper;

			template <typename T>
			struct cref_if_lvalue_helper<T, true> {
				static decltype(auto) get(T&& x) noexcept {
					return std::cref(x);
				}
			};

			template <typename T>
			struct cref_if_lvalue_helper<T, false> {
				static decltype(auto) get(T&& x) noexcept {
					return std::forward<T>(x);
				}
			};

			template <typename T, bool b>
			struct cref_if_lvalue_helper<std::reference_wrapper<T const>, b> {
				static decltype(auto) get(std::reference_wrapper<T const> x) noexcept {
					return x;
				}
			};
		}
		template <typename T>
		decltype(auto) ref_if_lvalue(T&& x) noexcept {
			return detail::ref_if_lvalue_helper<T>::get(std::forward<T>(x));
		}
		template <typename T>
		decltype(auto) cref_if_lvalue(T&& x) noexcept {
			return detail::cref_if_lvalue_helper<T>::get(std::forward<T>(x));
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// To prevent "too perfect" behaviour of perfect forwarding;
		// See Scott Meyers' Effective Modern C++, Item 27
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class Target, class Arg>
		using prevent_too_perfect_fwd = std::enable_if_t<
			!std::is_base_of<std::decay_t<Target>, std::decay_t<Arg>>::value
		>;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Result holder
		// Useful when one wishes to deduce the return type of a function by using the return value of 
		// another function, especially when it is necessary to delay returning that result which prevent 
		// you to just use auto to deduce the return type.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		struct result_holder {
			T result_;
			result_holder() = default;
			template <typename Callable, typename... Args, class = prevent_too_perfect_fwd<result_holder, Callable>>
			result_holder(Callable&& callable, Args&&... args)
				noexcept(std::is_nothrow_constructible<
					T, decltype(std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...))>::value &&
					noexcept(std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...)))
				: T(std::invoke(std::forward<Callable>(callable),
					std::forward<Args>(args)...)) {}
			T result() && noexcept(std::is_nothrow_move_constructible<T>::value) {
				return std::move(result_);
			}
			T& result() & noexcept {
				return result_;
			}
		};

		template <>
		struct result_holder<void> {
			result_holder() = default;
			template <typename Callable, typename... Args, class = prevent_too_perfect_fwd<result_holder, Callable>>
			result_holder(Callable&& callable, Args&&... args)
				noexcept(noexcept(std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...))) {
				std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...);
			}
			void result() noexcept {}
		};

		namespace detail {
			// If the return value is an rvalue reference, it is a temporary result that will dangle after the
			// function call. Hence, we should move it to a non-reference value and keep it.
			template <typename Callable, typename... Args>
			struct result_holder_type_impl {
				using result_type = decltype(std::declval<Callable>()(std::declval<Args>()...));
				using result_holder_type = result_holder<std::conditional_t<
					std::is_rvalue_reference<result_type>::value,
					std::remove_reference_t<result_type>,
					result_type>>;
			};
		}
		template <typename Callable, typename... Args>
		using result_holder_type = typename detail::result_holder_type_impl<Callable, Args...>::result_holder_type;

		template <typename Callable, typename... Args>
		auto hold_result(Callable&& callable, Args&&... args)
			noexcept(std::is_nothrow_constructible<result_holder_type<Callable, Args...>, Callable, Args...>::value)
		{
			return result_holder_type<Callable, Args...>{ std::forward<Callable>(callable), std::forward<Args>(args)... };
		}
	}
}