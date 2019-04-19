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
#include <utility>
#include "../portability.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// static_for, breakable_static_for
		// Unroll the loop "for( CounterType counter_ = counter; Test<counter_>::value; Next<counter_>::value ) { ... }"
		// Test and Next should be template types having the member variable "value".
		// Functor must have the function call operator with one template argument of type CounterType.
		// Thus, ordinary functors (like lambdas) cannot be used as functors.
		// Instead, such a functor can be converted to a usable form by passing it to the function make_loop_functor().
		// Passing the loop counter as a template parameter rather than a usual function argument has an 
		// advantage that it is guaranteed to be evaluated at compile-time.
		//
		// Note that there is no guarantee that the generated code is perfectly the same as manual unrolling, 
		// mainly due to inlining failure and other compiler details.
		// "#pragma unroll" kind of things do the same (even sometimes better), 
		// but this template-metaprogramming approach may harmonize better with other stuffs and is portable.
		//
		// Be careful: loop unrolling not always result in better performance especially for simple and short loops.
		// Compiler may often effectively understand the semantics inside the loop and perform various 
		// optimization techniques including but not limited to loop unrolling, but complex template mechanisms 
		// like static_for & breakable_static_for often prevent the compiler to catch the semantics. 
		// For example, the following code may run slower if static_for is used instead of for:
		//
		// int a[10], b[10];
		// for( int i=0; i<10; i++ )
		//     a[i] = b[i];
		//
		// Compiler may generate assembly codes performing vectorized mov if possible, and if not possible, 
		// it may generate codes calling memcpy() or may unroll the loop depending on the situations.
		// There is absolutely no benefit of static_for here.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <typename CounterType, CounterType counter, template <CounterType> class Test,
				template <CounterType> class Next, class Functor, bool condition = Test<counter>::value>
			struct static_for_helper {
				template <class... Args>
				FORCEINLINE static void do_loop(Functor&& functor, Args&&... args)
					// noexcept operator does not work with template function call operator in Visual Studio 2017
					// NVCC also seems not properly working
					// It seems that the issue is resolved since MSVC 2017 Update 3 & CUDA 9.0
#if (!defined(__CUDACC__) && (!defined(_MSC_VER) || _MSC_VER <= 1900 || _MSC_VER > 1910 || defined(__clang__))) || (__CUDACC__VER_MAJOR__ >= 9)
					noexcept(noexcept(functor.TEMPLATE_FUNCTION_CALL_OPERATOR<counter>(args...)) &&
						noexcept(static_for_helper<CounterType, Next<counter>::value, Test, Next, Functor>::do_loop(
							std::forward<Functor>(functor), std::forward<Args>(args)...)))	// Please give me noexcept(auto)!!!
#endif
				{
					functor.TEMPLATE_FUNCTION_CALL_OPERATOR<counter>(args...);
					static_for_helper<CounterType, Next<counter>::value, Test, Next, Functor>::do_loop(
						std::forward<Functor>(functor), std::forward<Args>(args)...);
				}
			};

			template <typename CounterType, CounterType counter, template <CounterType> class Test,
				template <CounterType> class Next, class Functor>
			struct static_for_helper<CounterType, counter, Test, Next, Functor, false> {
				template <class... Args>
				FORCEINLINE static void do_loop(Functor&&, Args&&...) noexcept {}
			};
		}

		template <typename CounterType, CounterType init_value, template <CounterType> class Test,
			template <CounterType> class Next, class Functor, class... Args>
		FORCEINLINE void static_for(Functor&& functor, Args&&... args)
			noexcept(noexcept(detail::static_for_helper<CounterType, init_value, Test, Next, Functor>::do_loop(
				std::forward<Functor>(functor), std::forward<Args>(args)...)))
		{
			detail::static_for_helper<CounterType, init_value, Test, Next, Functor>::do_loop(
				std::forward<Functor>(functor), std::forward<Args>(args)...);
		}

		namespace detail {
			template <typename CounterType, CounterType counter, template <CounterType> class Test,
				template <CounterType> class Next, class Functor, bool condition = Test<counter>::value>
			struct breakable_static_for_helper {
				template <class... Args>
				FORCEINLINE static bool do_loop(Functor&& functor, Args&&... args)
					// noexcept operator does not work with template function call operator in Visual Studio 2017
#if !defined(__CUDACC__) && (!defined(_MSC_VER) || _MSC_VER <= 1900 || defined(__clang__))
					noexcept(noexcept(functor.TEMPLATE_FUNCTION_CALL_OPERATOR<counter>(args...)) &&
						noexcept(breakable_static_for_helper<CounterType, Next<counter>::value, Test, Next, Functor>::do_loop(
							std::forward<Functor>(functor), std::forward<Args>(args)...)))	// Please give me noexcept(auto)!!!
#endif
				{
#ifdef __CUDACC__
					if( functor.template operator()<counter>(args...) ) {
#else
					if( functor.TEMPLATE_FUNCTION_CALL_OPERATOR<counter>(args...) ) {
#endif
						return breakable_static_for_helper<CounterType, Next<counter>::value, Test, Next, Functor>::do_loop(
							std::forward<Functor>(functor), std::forward<Args>(args)...);
					}
					return false;
					}
				};

			template <typename CounterType, CounterType counter, template <CounterType> class Test,
				template <CounterType> class Next, class Functor>
			struct breakable_static_for_helper<CounterType, counter, Test, Next, Functor, false> {
				template <class... Args>
				FORCEINLINE static bool do_loop(Functor&& functor, Args&&... args) noexcept {
					return true;
				}
			};
			}

		template <typename CounterType, CounterType init_value, template <CounterType> class Test,
			template <CounterType> class Next, class Functor, class... Args>
		FORCEINLINE bool breakable_static_for(Functor&& functor, Args&&... args)
			noexcept(noexcept(detail::breakable_static_for_helper<CounterType, init_value, Test, Next, Functor>::do_loop(
				std::forward<Functor>(functor), std::forward<Args>(args)...)))
		{
			return detail::breakable_static_for_helper<CounterType, init_value, Test, Next, Functor>::do_loop(
				std::forward<Functor>(functor), std::forward<Args>(args)...);
		}

		// Helper classes
		template <typename CounterType, CounterType y>
		struct less_than {
			template <CounterType x>
			struct test {
				static constexpr bool value = x < y;
			};
		};
		template <typename CounterType, CounterType y>
		struct less_than_or_equal_to {
			template <CounterType x>
			struct test {
				static constexpr bool value = x <= y;
			};
		};
		template <typename CounterType, CounterType y>
		struct greater_than {
			template <CounterType x>
			struct test {
				static constexpr bool value = x > y;
			};
		};
		template <typename CounterType, CounterType y>
		struct greater_than_or_equal_to {
			template <CounterType x>
			struct test {
				static constexpr bool value = x >= y;
			};
		};
		template <typename CounterType>
		struct increment {
			template <CounterType x>
			struct next {
				static constexpr CounterType value = x + (CounterType)1;
			};
		};
		template <typename CounterType>
		struct decrement {
			template <CounterType x>
			struct next {
				static constexpr CounterType value = x - (CounterType)1;
			};
		};
		template <typename CounterType, CounterType y>
		struct increment_by {
			template <CounterType x>
			struct next {
				static constexpr CounterType value = x + y;
			};
		};
		template <typename CounterType, CounterType y>
		struct decrement_by {
			template <CounterType x>
			struct next {
				static constexpr CounterType value = x - y;
			};
		};

		// Do loop unrolling of "for( CounterType counter=a; counter<b; counter++ ) { ... }"; [a,b) from left to right
		template <typename CounterType, CounterType a, CounterType b, class Functor, class... Args>
		FORCEINLINE void asc_static_for(Functor&& functor, Args&&... args)
			noexcept(noexcept(static_for<CounterType, a,
				less_than<CounterType, b>::template test,
				increment<CounterType>::template next>(
					std::forward<Functor>(functor), std::forward<Args>(args)...)))
		{
			static_for<CounterType, a, less_than<CounterType, b>::template test,
				increment<CounterType>::template next>(
					std::forward<Functor>(functor), std::forward<Args>(args)...);
		}
		// Do loop unrolling of "for( int counter=b-1; counter>=a; counter-- ) { ... }"; [a,b) from right to left
		template <typename CounterType, CounterType b, CounterType a, class Functor, class... Args>
		FORCEINLINE void des_static_for(Functor&& functor, Args&&... args)
			noexcept(noexcept(static_for<CounterType, b - 1,
				greater_than_or_equal_to<CounterType, a>::template test,
				decrement<CounterType>::template next>(
					std::forward<Functor>(functor), std::forward<Args>(args)...)))
		{
			static_for<CounterType, b - 1,
				greater_than_or_equal_to<CounterType, a>::template test,
				decrement<CounterType>::template next>(
					std::forward<Functor>(functor), std::forward<Args>(args)...);
		}

		// Make an ordinary functor into a functor having template function call operator
		namespace detail {
			template <typename CounterType, class Functor>
			struct loop_functor {
				Functor functor;
				loop_functor(Functor&& functor) noexcept(std::is_nothrow_constructible<Functor, Functor&&>::value)
					: functor(std::forward<Functor>(functor)) {}
				template <CounterType counter, typename... Args>
				FORCEINLINE void operator()(Args&&... args) noexcept(noexcept(functor(counter, std::forward<Args>(args)...))) {
					functor(counter, std::forward<Args>(args)...);
				}
			};
		}
		template <typename CounterType, class Functor>
		FORCEINLINE auto make_loop_functor(Functor&& functor) noexcept(std::is_nothrow_constructible<Functor, Functor&&>::value) {
			return detail::loop_functor<CounterType, Functor>{ std::forward<Functor>(functor) };
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// std::tuple version for std::for_each
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <class Tuple, std::size_t counter, bool end>
			struct static_for_each_impl;

			template <class Tuple, std::size_t counter>
			struct static_for_each_impl<Tuple, counter, false> {
				template <class Functor, class... AdditionalArgs>
				FORCEINLINE static void work(Tuple&& t, Functor&& f, AdditionalArgs&&... additional_args) {
					using std::get;
					f(get<counter>(std::forward<Tuple>(t)), additional_args...);
					static_for_each_impl<Tuple, counter + 1, std::tuple_size<std::decay_t<Tuple>>::value == counter + 1>::work(
						std::forward<Tuple>(t), std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
				}
			};

			template <class Tuple, std::size_t counter>
			struct static_for_each_impl<Tuple, counter, true> {
				template <class Functor, class... AdditionalArgs>
				FORCEINLINE static void work(Tuple&&, Functor&&, AdditionalArgs&&...) {}
			};
		}

		template <class Tuple, class Functor, class... AdditionalArgs>
		FORCEINLINE void static_for_each(Tuple&& t, Functor&& f, AdditionalArgs&&... additional_args) {
			detail::static_for_each_impl<Tuple, 0, std::tuple_size<std::decay_t<Tuple>>::value == 0>::work(
				std::forward<Tuple>(t), std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
		}
	}
}