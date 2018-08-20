/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2017 Junekey Jeon                                                                   ///
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
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include "portability.h"

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// An empty struct
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		struct empty_type {};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Multiple argument version of std::is_convertible
		/// This can be necessary for detecting if a constructor is explicit or not, because
		/// std::is_constructible does not care explicit-ness.
		/// This implementation is brought from
		/// https://stackoverflow.com/questions/20860535/is-convertible-for-multiple-arguments
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
		/// Check if static_cast is possible
		/// This implementation is a minor modification of the above is_convertible.
		/// I think the convention <From, To> is terrible and <To, From> is far better, but
		/// std::is_convertible already uses that convention...
		/// NOTE: The result of is_explicitly_convertible and std::is_constructible are in general, NOT same.
		///       For example, std::is_constructible_v<int&&, int&> is false, while
		///       is_explicitly_convertible<int&, int&&> is true.
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Assert helper
		/// There may be a template types such that any attempt to instantiate those types are 
		/// certainly an error. In those cases, one may want to use static_assert to report an error message, 
		/// but static_assert(false, ...) always produces an error even if the type is never instantiated.
		/// In that case, use assert_helper.
		///
		/// [Usage example]
		/// template <int N, class dummy = void>
		/// struct a_type {};
		/// template <class dummy>
		/// struct a_type<-1, dummy> { // a_type must not be instantitated with N = -1
		///   static_assert(assert_helper<a_type<-1, dummy>>::value, "N should not be negative!");
		/// };
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename... T>
		struct assert_helper : std::false_type {};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Check if a type is a complete type
		/// Brought from https://stackoverflow.com/questions/44229676/how-to-decide-if-a-template-specialization-exist
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <class T, class = char(*)[sizeof(T)]>
			char is_complete_impl(std::nullptr_t);
			template <class>
			long is_complete_impl(...);
		}

		template <class T>
		using is_complete = std::integral_constant<bool, sizeof(detail::is_complete_impl<T>(nullptr)) == sizeof(char)>;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// List of types
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename...>
		struct typelist;

		template <typename... Lists>
		struct merge_typelist;

		template <typename... FirstListTypes, typename... RemainingLists>
		struct merge_typelist<typelist<FirstListTypes...>, RemainingLists...> {
			using type = typename merge_typelist<typelist<FirstListTypes...>, typename merge_typelist<RemainingLists...>::type>::type;
		};

		template <typename... FirstListTypes, typename... SecondListTypes>
		struct merge_typelist<typelist<FirstListTypes...>, typelist<SecondListTypes...>> {
			using type = typelist<FirstListTypes..., SecondListTypes...>;
		};

		template <typename... FirstListTypes>
		struct merge_typelist<typelist<FirstListTypes...>> {
			using type = typelist<FirstListTypes...>;
		};

		template <>
		struct merge_typelist<> {
			using type = typelist<>;
		};

		template <typename... Lists>
		using merge_typelist_t = typename merge_typelist<Lists...>::type;

		template <typename First, typename... Remainings>
		struct typelist<First, Remainings...> {
		private:
			template <std::size_t index, class dummy = void>
			struct type_helper {
				static_assert(index <= sizeof...(Remainings), "typelist: type index out of range!");
				using type = typename typelist<Remainings...>::template type<index - 1>;
			};

			template <class dummy>
			struct type_helper<0, dummy> {
				using type = First;
			};

			template <std::size_t from, std::size_t to, class dummy = void>
			struct sublist_helper {
				static_assert(from <= to && to <= sizeof...(Remainings)+1, "typelist: invalid sublist specification!");
				using type = typename typelist<Remainings...>::template sublist<from - 1, to - 1>;
			};

			template <std::size_t to, class dummy>
			struct sublist_helper<0, to, dummy> {
				static_assert(to <= sizeof...(Remainings)+1, "typelist: invalid sublist specification!");
				using type = typename merge_typelist<typelist<First>, typename typelist<Remainings...>::template sublist<0, to - 1>>::type;
			};

			template <class dummy>
			struct sublist_helper<0, 0, dummy> {
				using type = typelist<>;
			};

		public:
			static constexpr std::size_t length = sizeof...(Remainings)+1;

			// Access the type of specified index
			template <std::size_t index>
			using type = typename type_helper<index>::type;

			// std::tuple corresponding to the given typelist
			using tuple_type = std::tuple<First, Remainings...>;

			// [from, to)
			template <std::size_t from, std::size_t to>
			using sublist = typename sublist_helper<from, to>::type;

			// Find a specific type in the list
			// Returns the length of the list if the given type is not in the list
			template <typename Type>
			static constexpr std::size_t find() noexcept {
				return std::is_same<Type, First>::value ? 0 :
					typelist<Remainings...>::template find<Type>() + 1;
			}
			template <typename Type>
			static constexpr bool has() noexcept {
				return find<Type>() < length;
			}
		};

		template <>
		struct typelist<> {
		private:
			template <std::size_t from, std::size_t to>
			struct sublist_helper {
				static_assert(from == to && to == 0, "typelist: invalid sublist specification!");
				using type = typelist<>;
			};

		public:
			static constexpr std::size_t length = 0;

			template <std::size_t index>
			struct type;

			using tuple_type = std::tuple<>;

			template <std::size_t from, std::size_t to>
			using sublist = typename sublist_helper<from, to>::type;

			template <typename Type>
			static constexpr std::size_t find() noexcept { return 0; }
			template <typename Type>
			static constexpr bool has() noexcept { return false; }
		};

		// Extract the corresponding typelist from tuple
		template <class Tuple>
		struct tuple_to_typelist;

		template <typename... Types>
		struct tuple_to_typelist<std::tuple<Types...>> {
			using type = typelist<Types...>;
		};

		template <class Tuple>
		using tuple_to_typelist_t = typename tuple_to_typelist<Tuple>::type;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Remove multiplicity from a typelist
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class Typelist>
		struct remove_multiplicity;

		template <typename First, typename... Remainings>
		struct remove_multiplicity<typelist<First, Remainings...>> {
			using type = std::conditional_t<
				typelist<Remainings...>::template has<First>(),
				typename remove_multiplicity<typelist<Remainings...>>::type,
				merge_typelist_t<typelist<First>, typename remove_multiplicity<typelist<Remainings...>>::type>
			>;
		};

		template <>
		struct remove_multiplicity<typelist<>> {
			using type = typelist<>;
		};

		template <class Typelist>
		using remove_multiplicity_t = typename remove_multiplicity<Typelist>::type;
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Return directly what is passed into
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		struct identity_functor {
			template <class T>
			constexpr T&& operator()(T&& x) const noexcept {
				return std::forward<T>(x);
			}
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// std::make_array of C++17 (brought from http://en.cppreference.com/w/cpp/experimental/make_array)
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template<class> struct is_ref_wrapper : std::false_type {};
			template<class T> struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type {};
			
			template <class D, class...> struct return_type_helper { using type = D; };
			template <class FirstType, class... RemainingTypes>
			struct return_type_helper<void, FirstType, RemainingTypes...> : 
				std::common_type<FirstType, RemainingTypes...> {
				static constexpr bool assert = !is_ref_wrapper<FirstType>::value &&
					return_type_helper<void, RemainingTypes...>::assert;
				static_assert(assert, "Types cannot contain reference_wrappers when D is void");
			};
			template <>
			struct return_type_helper<void> {
				static constexpr bool assert = true;
			};

			template <class D, class... Types>
			using return_type = std::array<typename return_type_helper<D, Types...>::type, sizeof...(Types)>;
		}

		template < class D = void, class... Types>
		constexpr detail::return_type<D, Types...> make_array(Types&&... t) {
			return{ std::forward<Types>(t)... };
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Static branching
		/// static_if      : execute the callable with the given argument when condition = true
		/// static_if_else : execute the callable_if when condition = true, while 
		///                  execute the callable_else when condition = false
		/// Note that for most of the cases, if the condition inside an if( ... ) statement is 
		/// statically evaluable, then the compiler may detect that and perform appropriate optimization.
		/// This static_if & static_if_else may not improve the code performance but even can be harmful 
		/// by preventing the compiler to catch the right semantics. 
		/// Furtheremore, callable may not be inlined even when appropriate.
		/// The purpose of this function is to overcome the incomplete support of constexpr.
		/// MSVC often calls a constexpr function even when it can be evaluated at compile-time, and this 
		/// prevents branching statement reduction. For example, for a function 
		///
		/// constexpr bool f(int arg) { ... }, 
		///
		/// the branching
		///
		/// if( f(arg) ) { do_something1(); }
		/// else { do_something2(); }
		///
		/// should turn into either
		///
		/// do_something1();
		///
		/// or 
		///
		/// do_something2();
		///
		/// without any branching statement whenever arg is a compile-time constant, but 
		/// MSVC often fails to do that because it mistakenly thinks f(arg) as a non-compile-time constant.
		/// This static_if will become completely useless after the support of "if constexpr" statements of C++17
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// static_for, breakable_static_for
		/// Unroll the loop "for( CounterType counter_ = counter; Test<counter_>::value; Next<counter_>::value ) { ... }"
		/// Test and Next should be template types having the member variable "value".
		/// Functor must have the function call operator with one template argument of type CounterType.
		/// Thus, ordinary functors (like lambdas) cannot be used as functors.
		/// Instead, such a functor can be converted to a usable form by passing it to the function make_loop_functor().
		/// Passing the loop counter as a template parameter rather than a usual function argument has an 
		/// advantage that it is guaranteed to be evaluated at compile-time.
		///
		/// Note that there is no guarantee that the generated code is perfectly the same as manual unrolling, 
		/// mainly due to inlining failure and other compiler details.
		/// "#pragma unroll" kind of things do the same (even sometimes better), 
		/// but this template-metaprogramming approach may harmonize better with other stuffs and is portable.
		///
		/// Be careful: loop unrolling not always result in better performance especially for simple and short loops.
		/// Compiler may often effectively understand the semantics inside the loop and perform various 
		/// optimization techniques including but not limited to loop unrolling, but complex template mechanisms 
		/// like static_for & breakable_static_for often prevent the compiler to catch the semantics. 
		/// For example, the following code may run slower if static_for is used instead of for:
		///
		/// int a[10], b[10];
		/// for( int i=0; i<10; i++ )
		///     a[i] = b[i];
		///
		/// Compiler may generate assembly codes performing vectorized mov if possible, and if not possible, 
		/// it may generate codes calling memcpy() or may unroll the loop depending on the situations.
		/// There is absolutely no benefit of static_for here.
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
		/// std::tuple version for std::for_each
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Runtime selection of functions with integer template parameter (from 0 to N)
		/// Build a constexpr table of function pointers, and choose one of them according to a 
		/// index variable given in runtime.
		/// For example, what we want to achieve is the following: we have a functor
		///
		/// struct functor {
		///   template <std::size_t index>
		///   void operator()() { // do something with index }
		/// };
		/// then the code 
		///
		/// call_by_index<10>(functor{}, i);
		///
		/// will effectively invoke the entry table[i] in the array
		///
		/// table[10] = { &functor::operator()<0>, ... , &functor::operator()<9> };
		///
		/// of member function pointers that is statically built.
		/// Note that all template specializations of operator() must have the same signature.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		// NVCC fails to understand this...
#if !defined(__CUDACC__)
		namespace detail {
			template <class Functor, class... Args, std::size_t... indices>
			auto call_by_index(Functor&& functor, std::size_t index, std::index_sequence<indices...>, Args&&... args) {
				static constexpr auto temp = &std::decay_t<Functor>::TEMPLATE_FUNCTION_CALL_OPERATOR<0>;
				static constexpr decltype(temp) table[] = { &std::decay_t<Functor>::TEMPLATE_FUNCTION_CALL_OPERATOR<indices>... };
				return (functor.*table[index])(std::forward<Args>(args)...);
			}

			template <std::size_t N, class Functor, class... Args>
			struct call_by_index_noexcept {
				static constexpr bool value = call_by_index_noexcept<N - 1, Functor, Args...>::value
					&& noexcept(std::declval<Functor>().TEMPLATE_FUNCTION_CALL_OPERATOR<N - 1>(std::declval<Args>()...));
			};

			template <class Functor, class... Args>
			struct call_by_index_noexcept<0, Functor, Args...> {
				static constexpr bool value = true;
			};
		}
		template <std::size_t N, class Functor, class... Args>
		FORCEINLINE auto call_by_index(Functor&& functor, std::size_t index, Args&&... args)
			noexcept(detail::call_by_index_noexcept<N, Functor, Args...>::value)
		{
			assert(index < N);
			return detail::call_by_index<Functor, Args...>(
				std::forward<Functor>(functor), index, std::make_index_sequence<N>{}, std::forward<Args>(args)...);
		}
#endif

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Check if a type supports std::tuple_element and std::tuple_size
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		struct is_tuple {
			// How std::get-like functionality is provided?
			// Note that even if there is not "get", is_tuple concludes that T is indeed tuple-like,
			// if std::tuple_element and std::tuple are both specialized.
			// If real tuple-like access is neccessary, the static member which_get should be inspected.
			enum which_get_t { none, member, adl };

		private:
			template <bool has_size, class dummy = void>
			struct impl {
				static constexpr std::size_t size = std::tuple_size<T>::value;

				

				template <bool length_zero, class dummy = void>
				struct check_get {
					template <class U, class = decltype(std::declval<U>().template get<0>())>
					static constexpr which_get_t check(int) { return member; }
					template <class U>
					static constexpr which_get_t check(float) { return false; }

					static constexpr bool value = check<T>(0);
				};

				template <class dummy>
				struct check_get<true, dummy> {
					static constexpr bool value = true;
				};

				template <bool length_zero, class dummy = void>
				struct check_tuple_element {
					static constexpr bool value = is_complete<std::tuple_element<0, T>>::value;
				};

				template <class dummy>
				struct check_tuple_element<true, dummy> {
					static constexpr bool value = true;
				};

				static constexpr bool value = check_get<size == 0>::value && check_tuple_element<size == 0>::value;
			};

			template <class dummy>
			struct impl<false, dummy> {
				static constexpr bool value = false;
			};

		public:
			static constexpr bool value = impl<is_complete<std::tuple_size<T>>::value>::value;
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// If the std::decay_t<Arg> is a tuple (that is, is_tuple evaluates to true), then unpack it and 
		/// apply the results to the functor. Otherwise, apply arg to the functor directly.
		/// Possibly much more simplified with std::apply of C++17
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
		/// If the argument is an lvalue reference, wrap it into std::reference_wrapper.
		/// Otherwise, do nothing. Main purpose is to be used in a generic context
		/// where an argument should be passed as a reference to functions such as std::async
		/// which deliberately remove reference qualifiers.
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
		/// To prevent "too perfect" behaviour of perfect forwarding;
		/// See Scott Meyers' Effective Modern C++, Item 27
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class Target, class Arg>
		using prevent_too_perfect_fwd = std::enable_if_t<
			!std::is_base_of<std::decay_t<Target>, std::decay_t<Arg>>::value
		>;
	}
}

namespace std {
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Specializations of std::tuple_element and std::tuple_length for jkl::tmp::typelist
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <std::size_t index, typename... Types>
	struct tuple_element<index, jkl::tmp::typelist<Types...>> {
		using type = tuple_element_t<index, typename jkl::tmp::typelist<Types...>::tuple_type>;
	};

	template <typename... Types>
	struct tuple_size<jkl::tmp::typelist<Types...>> : integral_constant<std::size_t, sizeof...(Types)> {};
}

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Swap two types in a typelist
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <std::size_t i, std::size_t j, class Typelist>
		struct swap_types_in_typelist {
		private:
			template <bool i_is_larger_than_j, bool i_is_same_as_j, class dummy>
			struct swap_types_impl;

			// When i == j
			template <class dummy>
			struct swap_types_impl<true, true, dummy> {
				using type = Typelist;
			};

			// When i > j
			template <class dummy>
			struct swap_types_impl<true, false, dummy> {
				using type = merge_typelist_t<
					typename Typelist::template sublist<0, j>, 
					typelist<typename Typelist::template type<i>>, 
					typename Typelist::template sublist<j + 1, i>,
					typelist<typename Typelist::template type<j>>,
					typename Typelist::template sublist<i + 1, Typelist::length>
				>;
			};

			// When i < j
			template <class dummy>
			struct swap_types_impl<false, false, dummy> {
				using type = merge_typelist_t<
					typename Typelist::template sublist<0, i>,
					typelist<typename Typelist::template type<j>>,
					typename Typelist::template sublist<i + 1, j>,
					typelist<typename Typelist::template type<i>>,
					typename Typelist::template sublist<j + 1, Typelist::length>
				>;
			};

		public:
			using type = typename swap_types_impl<i >= j, i == j, void>::type;
		};

		template <std::size_t i, std::size_t j, class Typelist>
		using swap_types_in_typelist_t = typename swap_types_in_typelist<i, j, Typelist>::type;

		template <std::size_t i, std::size_t j, class Tuple>
		struct swap_types_in_tuple {
			using type = typename swap_types_in_typelist_t<i, j, tuple_to_typelist_t<Tuple>>::tuple_type;
		};

		template <std::size_t i, std::size_t j, class Tuple>
		using swap_types_in_tuple_t = typename swap_types_in_tuple<i, j, Tuple>::type;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Sort the parameter pack and then apply them to a variadic template
		/// http://codereview.stackexchange.com/questions/131194/selection-sorting-a-type-list-compile-time
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <
			template<typename...> class VariadicTemplate, 
			template <typename, typename> class Comparator,
			typename... Types
		>
		struct sort_and_apply {
		private:
			// selection sort's "loop"
			template <std::size_t i, std::size_t j, std::size_t length, class ResultTuple>
			struct sort_and_apply_impl {
				// this is done until we have compared every element in the type list
				using after_swap = std::conditional_t<
					Comparator<
						std::tuple_element_t<i, ResultTuple>,
						std::tuple_element_t<j, ResultTuple>
					>::value,
					swap_types_in_tuple_t<i, j, ResultTuple>,	// true: swap(i, j)
					ResultTuple									// false: do nothing
				>;

				using type = typename sort_and_apply_impl<	// recurse until j == length
					i, j + 1, length, after_swap			// using the modified tuple
				>::type;
			};

			template <std::size_t i, std::size_t length, class ResultTuple>
			struct sort_and_apply_impl<i, length, length, ResultTuple> {
				// once j == length, we increment i and start j at i + 1 and recurse
				using type = typename sort_and_apply_impl<
					i + 1, i + 2, length, ResultTuple
				>::type;
			};

			template <std::size_t j, std::size_t length, class ResultTuple>
			struct sort_and_apply_impl<length, j, length, ResultTuple> {
				// once i == length, we know that every element has been compared
				using type = ResultTuple;
			};

			using result_tuple = typename sort_and_apply_impl<
				0, 1, sizeof...(Types), std::tuple<Types...>
			>::type;

			template <class SortedTuple>
			struct apply;

			template <typename... SortedTypes>
			struct apply<std::tuple<SortedTypes...>> {
				using type = VariadicTemplate<SortedTypes...>;
			};

		public:
			using type = typename apply<result_tuple>::type;
		};

		template <template<typename...> class VariadicTemplate, 
			template <typename, typename> class Comparator, typename... Types>
		using sort_and_apply_t = typename sort_and_apply<VariadicTemplate, Comparator, Types...>::type;

		template <template <typename, typename> class Comparator, class Typelist>
		struct sort_typelist;

		template <template <typename, typename> class Comparator, typename... Types>
		struct sort_typelist<Comparator, typelist<Types...>> {
			using type = sort_and_apply_t<typelist, Comparator, Types...>;
		};

		template <template <typename, typename> class Comparator, class Typelist>
		using sort_typelist_t = typename sort_typelist<Comparator, Typelist>::type;

		template <template <typename, typename> class Comparator, class Tuple>
		struct sort_tuple;

		template <template <typename, typename> class Comparator, typename... Types>
		struct sort_tuple<Comparator, std::tuple<Types...>> {
			using type = sort_and_apply_t<std::tuple, Comparator, Types...>;
		};

		template <template <typename, typename> class Comparator, class Tuple>
		using sort_tuple_t = typename sort_tuple<Comparator, Tuple>::type;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// The "default" wrapper
		/// When (partially or explicitly) specializing class templates, it is often desirable to 
		/// use a huge part of the default definition of the templates, while just adding some more 
		/// special features for the specialization. One way to do this is to make a separate base class 
		/// implementing all the "common" features, and then do not specialize the base class.
		/// A problem occurs when there are only "reusable parts" but no "common parts"; that is,  
		/// the parts of the default definition that can be reused are different for each specialization.
		///
		/// Here we provide an alternative method. Just define a template class as usual. If something 
		/// inside the class should be done or annotated with the template parameter T, then 
		/// do not use T directly and use the alias default_wrapper_t<T> defined below instead.
		/// When you are specializing the template, derive from your_template_name<default_wrapper<T>>.
		/// You only need to add or modify the things that are specially necessary for T, and everything else 
		/// will be automatically included in the specialization. Here is a use-case:
		///
		/// template <typename T> struct Sample {
		///   default_wrapper_t<T> f() { ... }
		///   default_wrapper_t<T> g() { ... }
		/// };
		/// // Modify g() when T is a pointer
		/// template <typename T> struct Sample<T*> : Sample<default_wrapper<T*>> {
		///   T* g() { ... }   // The return type of f() is also T*
		/// };
		/// // Modify f() when T is a reference
		/// template <typename T> struct Sample<T&> : Sample<default_wrapper<T&>> {
		///   T& f() { ... } // The return type of g() is also T&
		/// };
		///
		/// But be careful, if you override some method in the specialization, it will not be "really overriden", 
		/// because other methods in the default definition never call the new version provided for the 
		/// specialization. Also, default_wrapper only works with non-template type template parameters. 
		/// To work with non-type template parameters or template template parameters, 
		/// you need to write a similar template by yourself.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		struct default_wrapper {
			using type = T;
		};

		template <typename T>
		struct default_wrapper<default_wrapper<T>> {
			using type = typename default_wrapper<T>::type;
		};

		template <typename T>
		using default_wrapper_t = typename default_wrapper<T>::type;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Result holder
		/// Useful when one wishes to deduce the return type of a function by using the return value of 
		/// another function, especially when it is necessary to delay returning that result which prevent 
		/// you to just use auto to deduce the return type.
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Call chain
		/// Forward the result of a callable f to another callable g, according to the following rule:
		///  1. If f() is void, call g().
		///  2. If f() is a tuple, unpack it and apply to g.
		///  3. Otherwise, call g(f()).
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// A proxy class inheriting from a list of base classes
		/// Perhaps it's better to pre-eliminate any duplicated classes in the list?
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		template <class... Bases>
		struct inherits : Bases... {};
	}
}