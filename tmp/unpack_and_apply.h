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
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include "is_tuple.h"
#include "../portability.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// If the std::decay_t<Arg> is a tuple (that is, is_tuple evaluates to true), then unpack it and 
		// apply the results to the functor. Otherwise, apply arg to the functor directly.
		// Possibly much more simplified with std::apply of C++17
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <typename Callable, typename Arg, bool is_tuple>
			struct unpack_and_apply_helper;

			template <typename Callable, typename Arg>
			struct unpack_and_apply_helper<Callable, Arg, true> {
				template <std::size_t... I>
				FORCEINLINE static decltype(auto) do_work_impl(Callable&& callable, Arg&& arg, std::index_sequence<I...>)
					noexcept(noexcept(std::invoke(std::forward<Callable>(callable), std::get<I>(std::forward<Arg>(arg))...)))
				{
					return std::invoke(std::forward<Callable>(callable), std::get<I>(std::forward<Arg>(arg))...);
				}

				FORCEINLINE static decltype(auto) do_work(Callable&& callable, Arg&& arg)
					noexcept(noexcept(do_work_impl(std::forward<Callable>(callable), std::forward<Arg>(arg),
						std::make_index_sequence<std::tuple_size<std::decay_t<Arg>>::value>{})))
				{
					return do_work_impl(std::forward<Callable>(callable), std::forward<Arg>(arg),
						std::make_index_sequence<std::tuple_size<std::decay_t<Arg>>::value>{});
				}
			};

			template <typename Callable, typename Arg>
			struct unpack_and_apply_helper<Callable, Arg, false> {
				FORCEINLINE static decltype(auto) do_work(Callable&& callable, Arg&& arg)
					noexcept(noexcept(std::invoke(std::forward<Callable>(callable), std::forward<Arg>(arg))))
				{
					return std::invoke(std::forward<Callable>(callable), std::forward<Arg>(arg));
				}
			};
		}

		template <typename Callable, typename Arg>
		FORCEINLINE decltype(auto) unpack_and_apply(Callable&& callable, Arg&& arg)
			noexcept(noexcept(detail::unpack_and_apply_helper<Callable, Arg, is_tuple<std::decay_t<Arg>>::value>::do_work(
				std::forward<Callable>(callable), std::forward<Arg>(arg))))
		{
			return detail::unpack_and_apply_helper<Callable, Arg, is_tuple<std::decay_t<Arg>>::value>::do_work(
				std::forward<Callable>(callable), std::forward<Arg>(arg));
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Call chain
		// Forward the result of a callable f to another callable g, according to the following rule:
		//  1. If f() is void, call g().
		//  2. If f() is a tuple, unpack it and apply to g.
		//  3. Otherwise, call g(f()).
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <class ResultType>
			struct chain_call_impl {
				template <typename OutCallable, typename InCallable, typename... InArgs>
				static decltype(auto) chain_call(OutCallable&& out_callable, InCallable&& in_callable, InArgs&&... in_args)
					noexcept(noexcept(unpack_and_apply(std::forward<OutCallable>(out_callable),
						std::forward<InCallable>(in_callable)(std::forward<InArgs>(in_args)...))))
				{
					return unpack_and_apply(std::forward<OutCallable>(out_callable),
						std::forward<InCallable>(in_callable)(std::forward<InArgs>(in_args)...));
				}
			};

			template <>
			struct chain_call_impl<void> {
				template <typename OutCallable, typename InCallable, typename... InArgs>
				static decltype(auto) chain_call(OutCallable&& out_callable, InCallable&& in_callable, InArgs&&... in_args)
					noexcept(noexcept(std::forward<InCallable>(in_callable)(std::forward<InArgs>(in_args)...)) &&
						noexcept(std::forward<OutCallable>(out_callable)()))
				{
					std::forward<InCallable>(in_callable)(std::forward<InArgs>(in_args)...);
					return std::forward<OutCallable>(out_callable)();
				}
			};
		}

		template <typename OutCallable, typename InCallable, typename... InArgs>
		decltype(auto) chain_call(OutCallable&& out_callable, InCallable&& in_callable, InArgs&&... in_args)
			noexcept(noexcept(detail::chain_call_impl<decltype(std::forward<InCallable>(in_callable)(
				std::forward<InArgs>(in_args)...))>::chain_call(std::forward<OutCallable>(out_callable),
					std::forward<InCallable>(in_callable), std::forward<InArgs>(in_args)...)))
		{
			return detail::chain_call_impl<decltype(std::forward<InCallable>(in_callable)(
				std::forward<InArgs>(in_args)...))>::chain_call(std::forward<OutCallable>(out_callable),
					std::forward<InCallable>(in_callable), std::forward<InArgs>(in_args)...);
		}
	}
}