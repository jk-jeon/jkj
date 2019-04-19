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
#include <utility>
#include "../portability.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Static branching
		// static_if      : execute the callable with the given argument when condition = true
		// static_if_else : execute the callable_if when condition = true, while 
		//                  execute the callable_else when condition = false
		// Note that for most of the cases, if the condition inside an if( ... ) statement is 
		// statically evaluable, then the compiler may detect that and perform appropriate optimization.
		// This static_if & static_if_else may not improve the code performance but even can be harmful 
		// by preventing the compiler to catch the right semantics. 
		// Furtheremore, callable may not be inlined even when appropriate.
		// The purpose of this function is to overcome the incomplete support of constexpr.
		// MSVC often calls a constexpr function even when it can be evaluated at compile-time, and this 
		// prevents branching statement reduction. For example, for a function 
		//
		// constexpr bool f(int arg) { ... }, 
		//
		// the branching
		//
		// if( f(arg) ) { do_something1(); }
		// else { do_something2(); }
		//
		// should turn into either
		//
		// do_something1();
		//
		// or 
		//
		// do_something2();
		//
		// without any branching statement whenever arg is a compile-time constant, but 
		// MSVC often fails to do that because it mistakenly thinks f(arg) as a non-compile-time constant.
		// This static_if will become completely useless after the support of "if constexpr" statements of C++17
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <bool condition, class Callable, class... Args>
			struct static_if_helper {
				FORCEINLINE static decltype(auto) do_branch(Callable&& callable, Args&&... args)
					noexcept(noexcept(std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...)))
				{
					return std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...);
				}
			};
			template <class Callable, class... Args>
			struct static_if_helper<false, Callable, Args...> {
				using return_type = decltype(std::declval<Callable>()(std::declval<Args>()...));
				FORCEINLINE static return_type do_branch(Callable&& callable, Args&&... args) noexcept {
					return return_type();
				}
			};
			template <bool condition, class CallableIf, class CallableElse, class... Args>
			struct static_if_else_helper {
				FORCEINLINE static decltype(auto) do_branch(CallableIf&& callable_if, CallableElse&&, Args&&... args)
					noexcept(noexcept(std::invoke(std::forward<CallableIf>(callable_if), std::forward<Args>(args)...)))
				{
					return std::invoke(std::forward<CallableIf>(callable_if), std::forward<Args>(args)...);
				}
			};
			template <class CallableIf, class CallableElse, class... Args>
			struct static_if_else_helper<false, CallableIf, CallableElse, Args...> {
				FORCEINLINE static decltype(auto) do_branch(CallableIf&&, CallableElse&& callable_else, Args&&... args)
					noexcept(noexcept(std::invoke(std::forward<CallableElse>(callable_else), std::forward<Args>(args)...)))
				{
					return std::invoke(std::forward<CallableElse>(callable_else), std::forward<Args>(args)...);
				}
			};
		}
		template <bool condition, class Callable, class... Args>
		FORCEINLINE decltype(auto) static_if(Callable&& callable, Args&&... args)
			noexcept(!condition || noexcept(std::invoke(std::forward<Callable>(callable), std::forward<Args>(args)...)))
		{
			return detail::static_if_helper<condition, Callable, Args...>::do_branch(
				std::forward<Callable>(callable), std::forward<Args>(args)...);
		}
		template <bool condition, class CallableIf, class CallableElse, class... Args>
		FORCEINLINE auto static_if_else(CallableIf&& callable_if, CallableElse&& callable_else, Args&&... args)
			noexcept(noexcept(detail::static_if_else_helper<condition, CallableIf, CallableElse, Args...>::do_branch(
				std::forward<CallableIf>(callable_if), std::forward<CallableElse>(callable_else),
				std::forward<Args>(args)...)))
		{
			return detail::static_if_else_helper<condition, CallableIf, CallableElse, Args...>::do_branch(
				std::forward<CallableIf>(callable_if), std::forward<CallableElse>(callable_else),
				std::forward<Args>(args)...);
		}
	}
}