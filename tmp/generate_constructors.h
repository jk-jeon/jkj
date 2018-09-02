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
#include <tuple>
#include <type_traits>
#include <utility>
#include "../portability.h"

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// When taking a constructor argument to initialize a member object of type T, the parameter type is
		//  1. T const&, when the argument is to be copied, or
		//  2. T&&, when the argument is to be moved.
		// This is the idiomatic way of taking such an argument.
		// However, if there are multiple such arguments, the complexity grows exponentially.
		// For example, to provide forwarding constructor for four such arguments, there should be
		// 2^4 = 16 constructors just to correctly forward arguments.
		// The perfect forwarding template constructor helps, but it has cost: we cannot no longer
		// pass a braced-initalizer-list at a place for an argument, because in that case
		// the compiler cannot figure out the parameter type to be deduced.
		// Another alternative is to use call-by-value instead. This will invoke one more move constructor
		// compared to the optimal case of providing all 16 constructors.
		// Hence, it is sometimes still necessary to provide all 16 constructors.
		//
		// To reduce boilerplates, here is a class that will automatically generate all constructors with
		// all combinations of given parameter types. A derived class can simply inherit those
		// automatically generated constructors.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		template <class BaseWithCatchAll, class ForwardTag, class... TuplesOfTuplesOfArgTypes>
		struct generate_constructors;

		namespace detail {
			template <std::size_t new_switch, class IndexSequence>
			struct add_to_index_sequence;

			template <std::size_t new_switch, std::size_t... switches>
			struct add_to_index_sequence<new_switch, std::index_sequence<switches...>> {
				using type = std::index_sequence<switches..., new_switch>;
			};

			template <std::size_t I, class TupleOfTuplesOfArgTypes, std::size_t depth>
			struct generate_switch_sequence {
				static constexpr std::size_t choices = std::tuple_size<std::tuple_element_t<I, TupleOfTuplesOfArgTypes>>::value;
				static_assert(choices != 0, "jkl::tmp: empty tuple is not allowed");
				using type = typename add_to_index_sequence<depth % choices,
					typename generate_switch_sequence<I - 1, TupleOfTuplesOfArgTypes, depth / choices>::type>::type;
			};

			template <class TupleOfTuplesOfArgTypes, std::size_t depth>
			struct generate_switch_sequence<0, TupleOfTuplesOfArgTypes, depth> {
				static constexpr std::size_t choices = std::tuple_size<std::tuple_element_t<0, TupleOfTuplesOfArgTypes>>::value;
				using type = std::index_sequence<depth>;
			};

			template <std::size_t I, class TupleOfTuplesOfArgTypes, std::size_t... switches>
			struct get_type {
				static constexpr std::size_t type_index =
					std::tuple_element_t<I, std::tuple<std::integral_constant<std::size_t, switches>...>>::value;
				using type = std::tuple_element_t<type_index, std::tuple_element_t<I, TupleOfTuplesOfArgTypes>>;
			};

			// If this is too small, the compiler may suffer from excessively many recursive type definitions
			static constexpr std::size_t generate_constructor_loop_step = 8;

			template <class BaseType, class TupleOfTuplesOfArgTypes,
				std::size_t depth, class IndexSequence, class... SwitchSequences>
			struct inheritance_chain;

			template <std::size_t I, class TupleOfTuplesOfArgTypes>
			struct calculate_max_depth : std::integral_constant<std::size_t,
				calculate_max_depth<I - 1, TupleOfTuplesOfArgTypes>::value *
				std::tuple_size<std::tuple_element_t<I, TupleOfTuplesOfArgTypes>>::value> {};

			template <class TupleOfTuplesOfArgTypes>
			struct calculate_max_depth<0, TupleOfTuplesOfArgTypes> : std::integral_constant<std::size_t,
				std::tuple_size<std::tuple_element_t<0, TupleOfTuplesOfArgTypes>>::value> {};

			template <class BaseType, class TupleOfTuplesOfArgTypes>
			struct inheritance_chain_helper {
				static constexpr std::size_t number_of_arguments = std::tuple_size<TupleOfTuplesOfArgTypes>::value;
				static constexpr std::size_t max_depth =
					calculate_max_depth<number_of_arguments - 1, TupleOfTuplesOfArgTypes>::value;


				template <std::size_t depth, bool is_tail_case = (depth <= generate_constructor_loop_step), class = void>
				struct get_base_type {
					using type = inheritance_chain<BaseType, TupleOfTuplesOfArgTypes,
						depth - generate_constructor_loop_step, std::make_index_sequence<number_of_arguments>,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 1>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 2>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 3>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 4>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 5>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 6>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 7>::type,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 8>::type
					>;
				};

				template <std::size_t depth, class dummy>
				struct get_base_type<depth, true, dummy> {
					using type = inheritance_chain<BaseType, TupleOfTuplesOfArgTypes,
						depth - 1, std::make_index_sequence<number_of_arguments>,
						typename generate_switch_sequence<number_of_arguments - 1, TupleOfTuplesOfArgTypes, depth - 1>::type>;
				};

				template <class dummy>
				struct get_base_type<0, true, dummy> {
					using type = BaseType;
				};

				template <std::size_t depth>
				using base_type = typename get_base_type<depth>::type;

				using initial_base_type = base_type<max_depth>;
			};

			template <class BaseType, class TupleOfTuplesOfArgTypes, std::size_t depth, std::size_t... I,
				std::size_t... switches1,
				std::size_t... switches2,
				std::size_t... switches3,
				std::size_t... switches4,
				std::size_t... switches5,
				std::size_t... switches6,
				std::size_t... switches7,
				std::size_t... switches8
			>
			struct inheritance_chain<BaseType, TupleOfTuplesOfArgTypes, depth, std::index_sequence<I...>,
				std::index_sequence<switches1...>,
				std::index_sequence<switches2...>,
				std::index_sequence<switches3...>,
				std::index_sequence<switches4...>,
				std::index_sequence<switches5...>,
				std::index_sequence<switches6...>,
				std::index_sequence<switches7...>,
				std::index_sequence<switches8...>
			> : inheritance_chain_helper<BaseType, TupleOfTuplesOfArgTypes>::template base_type<depth> {
			private:
				template <class, class, class...>
				friend struct generate_constructors_impl;

				template <class, class, std::size_t, class, class...>
				friend struct inheritance_chain;

				using base_type = typename inheritance_chain_helper<BaseType, TupleOfTuplesOfArgTypes>::
					template base_type<depth>;

				// Due to incomplete support of variable template of MSVC2015,
				// I had to used class template instead here...
				template <class... Args>
				using is_noexcept = typename base_type::template is_noexcept<Args...>;

				using forward_tag = typename base_type::forward_tag;

				template <std::size_t J>
				using get_type_t1 = typename get_type<J, TupleOfTuplesOfArgTypes, switches1...>::type;
				template <std::size_t J>
				using get_type_t2 = typename get_type<J, TupleOfTuplesOfArgTypes, switches2...>::type;
				template <std::size_t J>
				using get_type_t3 = typename get_type<J, TupleOfTuplesOfArgTypes, switches3...>::type;
				template <std::size_t J>
				using get_type_t4 = typename get_type<J, TupleOfTuplesOfArgTypes, switches4...>::type;
				template <std::size_t J>
				using get_type_t5 = typename get_type<J, TupleOfTuplesOfArgTypes, switches5...>::type;
				template <std::size_t J>
				using get_type_t6 = typename get_type<J, TupleOfTuplesOfArgTypes, switches6...>::type;
				template <std::size_t J>
				using get_type_t7 = typename get_type<J, TupleOfTuplesOfArgTypes, switches7...>::type;
				template <std::size_t J>
				using get_type_t8 = typename get_type<J, TupleOfTuplesOfArgTypes, switches8...>::type;

				// Another workaround for MSVC2015
				static constexpr bool is_noexcept_v1 = is_noexcept<get_type_t1<I>...>::value;
				static constexpr bool is_noexcept_v2 = is_noexcept<get_type_t2<I>...>::value;
				static constexpr bool is_noexcept_v3 = is_noexcept<get_type_t3<I>...>::value;
				static constexpr bool is_noexcept_v4 = is_noexcept<get_type_t4<I>...>::value;
				static constexpr bool is_noexcept_v5 = is_noexcept<get_type_t5<I>...>::value;
				static constexpr bool is_noexcept_v6 = is_noexcept<get_type_t6<I>...>::value;
				static constexpr bool is_noexcept_v7 = is_noexcept<get_type_t7<I>...>::value;
				static constexpr bool is_noexcept_v8 = is_noexcept<get_type_t8<I>...>::value;

			public:
				inheritance_chain() = default;
				using base_type::base_type;

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t1<I>... args)
					noexcept(is_noexcept_v1) :
					base_type(forward_tag{}, std::forward<get_type_t1<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t2<I>... args)
					noexcept(is_noexcept_v2) :
					base_type(forward_tag{}, std::forward<get_type_t2<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(
					typename get_type<I, TupleOfTuplesOfArgTypes, switches3...>::type... args)
					noexcept(is_noexcept_v3) :
					base_type(forward_tag{}, std::forward<get_type_t3<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t4<I>... args)
					noexcept(is_noexcept_v4) :
					base_type(forward_tag{}, std::forward<get_type_t4<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t5<I>... args)
					noexcept(is_noexcept_v5) :
					base_type(forward_tag{}, std::forward<get_type_t5<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t6<I>... args)
					noexcept(is_noexcept_v6) :
					base_type(forward_tag{}, std::forward<get_type_t6<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t7<I>... args)
					noexcept(is_noexcept_v7) :
					base_type(forward_tag{}, std::forward<get_type_t7<I>>(args)...) {}

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t8<I>... args)
					noexcept(is_noexcept_v8) :
					base_type(forward_tag{}, std::forward<get_type_t8<I>>(args)...) {}
			};

			template <class BaseType, class TupleOfTuplesOfArgTypes, std::size_t depth, std::size_t... I, std::size_t... switches>
			struct inheritance_chain<BaseType, TupleOfTuplesOfArgTypes, depth,
				std::index_sequence<I...>, std::index_sequence<switches...>> :
				inheritance_chain_helper<BaseType, TupleOfTuplesOfArgTypes>::template base_type<depth> {
			private:
				template <class, class, class...>
				friend struct generate_constructors_impl;

				template <class, class, std::size_t, class, class...>
				friend struct inheritance_chain;

				using base_type = typename inheritance_chain_helper<BaseType, TupleOfTuplesOfArgTypes>::
					template base_type<depth>;

				// Due to incomplete support of variable template of MSVC2015,
				// I had to used class template instead here...
				template <class... Args>
				using is_noexcept = typename base_type::template is_noexcept<Args...>;
				
				using forward_tag = typename base_type::forward_tag;

				template <std::size_t J>
				using get_type_t = typename get_type<J, TupleOfTuplesOfArgTypes, switches...>::type;

				// Another workaround for MSVC2015
				static constexpr bool is_noexcept_v = is_noexcept<get_type_t<I>...>::value;

			public:
				inheritance_chain() = default;
				using base_type::base_type;

				JKL_GPU_EXECUTABLE constexpr inheritance_chain(get_type_t<I>... args)
					noexcept(is_noexcept_v) :
					base_type(forward_tag{}, std::forward<get_type_t<I>>(args)...) {}
			};

			template <class BaseWithCatchAll, class ForwardTag, class... TuplesOfTuplesOfArgTypes>
			struct generate_constructors_impl;

			template <class BaseWithCatchAll, class ForwardTag,
				class FirstTupleOfTuplesOfArgTypes, class... RemainingTuplesOfTuplesOfArgTypes>
			struct generate_constructors_impl<BaseWithCatchAll, ForwardTag,
				FirstTupleOfTuplesOfArgTypes, RemainingTuplesOfTuplesOfArgTypes...> :
				detail::inheritance_chain_helper<
				generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
				FirstTupleOfTuplesOfArgTypes>::initial_base_type
			{
			private:
				template <class, class, class...>
				friend struct generate_constructors;

				template <class, class, class...>
				friend struct generate_constructors_impl;

				template <class, class, std::size_t, class, class...>
				friend struct inheritance_chain;

			#ifndef __NVCC__
				using base_type = typename detail::inheritance_chain_helper<
					generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
					FirstTupleOfTuplesOfArgTypes>::initial_base_type;

				// Due to incomplete support of variable template of MSVC2015,
				// I had to used class template instead here...
				template <class... Args>
				using is_noexcept = typename base_type::template is_noexcept<Args...>;
				
				using forward_tag = typename base_type::forward_tag;

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr generate_constructors_impl(forward_tag tag, Args&&... args)
					noexcept(is_noexcept<Args...>::value) :
					base_type(tag, std::forward<Args>(args)...) {}

				using base_type::base_type;
			#else
				// NVCC 9.2.148 fails to compile this file if base_type exists
				template <class... Args>
				using is_noexcept = typename detail::inheritance_chain_helper<
					generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
					FirstTupleOfTuplesOfArgTypes>::initial_base_type::template is_noexcept<Args...>;

				using forward_tag = typename detail::inheritance_chain_helper<
					generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
					FirstTupleOfTuplesOfArgTypes>::initial_base_type::forward_tag;

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr generate_constructors_impl(forward_tag tag, Args&&... args)
					noexcept(is_noexcept<Args...>::value) :
					detail::inheritance_chain_helper<
					generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
					FirstTupleOfTuplesOfArgTypes>::initial_base_type(tag, std::forward<Args>(args)...) {}

				using detail::inheritance_chain_helper<
					generate_constructors_impl<BaseWithCatchAll, ForwardTag, RemainingTuplesOfTuplesOfArgTypes...>,
					FirstTupleOfTuplesOfArgTypes>::initial_base_type::initial_base_type;
			#endif

			public:
				generate_constructors_impl() = default;
			};

			template <class BaseWithCatchAll, class ForwardTag>
			struct generate_constructors_impl<BaseWithCatchAll, ForwardTag> : 
				BaseWithCatchAll
			{
			private:
				template <class, class, class...>
				friend struct generate_constructors;

				template <class, class, class...>
				friend struct generate_constructors_impl;

				template <class, class, std::size_t, class, class...>
				friend struct inheritance_chain;

				// Workaround for prevented access to protected constructor when noexcept-ness is evaluated
				template <class... Args, class = typename BaseWithCatchAll::template is_nothrow_constructible<Args...>>
				static constexpr bool inspect_noexceptness(int) {
					return BaseWithCatchAll::template is_nothrow_constructible<Args...>::value;
				}
				template <class... Args>
				static constexpr bool inspect_noexceptness(...) {
					return std::is_nothrow_constructible<BaseWithCatchAll, Args...>::value;
				}

				// Due to incomplete support of variable template of MSVC2015,
				// I had to used class template instead here...
				template <class... Args>
				struct is_noexcept {
					static constexpr bool value = inspect_noexceptness<Args...>(0);
				};
				
				struct forward_tag {};
				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr generate_constructors_impl(forward_tag, Args&&... args)
					noexcept(is_noexcept<Args...>::value) :
					BaseWithCatchAll(ForwardTag{}, std::forward<Args>(args)...) {}

			public:
				generate_constructors_impl() = default;
				using BaseWithCatchAll::BaseWithCatchAll;
			};
		}

		// Inherit from this class
		//  - BaseWithCatchAll:
		//    a base class of the instantiated generate_constructors; it is supposed to have
		//    a protected (or public) perfect forwarding constructor that will be called by
		//    generated constructors. All constructors of BaseWithCatchAll will be also inherited.
		//  - ForwadTag:
		//    a type used for tag dispatching. Default-constructed ForwardTag object will
		//    be passed to BaseWithCatchAll constructors as the first argument, when the
		//    generated constructor forwards its parameters to BaseWithCatchAll.
		//  - TuplesOfTuplesOfArgTypes:
		//    something like std::tuple<std::tuple<T1, T2>, std::tuple<U1, U2, U3>>;
		//    if the above is given as one of TuplesOfTuplesOfArgTypes, then six constructors with the
		//    following paramter lists will be generated:
		//      1. (T1, U1),
		//      2. (T1, U2),
		//      3. (T1, U3),
		//      4. (T2, U1),
		//      5. (T2, U2), and
		//      6. (T2, U3).
		//    Typically, something like std::tuple<std::tuple<T const&, T&&>, std::tuple<U const&, U&&>>
		//    may be given.
		// NOTE: It seems it is simply impossible in the current language specification
		//       to correctly detect noexcept-ness of a protected constructors from a derived class.
		//       Hence, it is necessary for the base class
		//       to expose somehow calculated noexcept-ness of its constructors.
		//       In this library, we assume there would be an optional nested template class
		//       is_nothrow_constructible<Args...> in the base class, which has a
		//       static member variable 'value' indicating noexcept-ness of constructing
		//       the base class with the arguments of type Args&&'s. Too bad...
		
		template <class BaseWithCatchAll, class ForwardTag, class... TuplesOfTuplesOfArgTypes>
		struct generate_constructors :
			detail::generate_constructors_impl<BaseWithCatchAll, ForwardTag, TuplesOfTuplesOfArgTypes...>
		{
		private:
			using base_type = detail::generate_constructors_impl<BaseWithCatchAll,
				ForwardTag, TuplesOfTuplesOfArgTypes...>;
			using ultimate_base_type = BaseWithCatchAll;

		public:
			using base_type::base_type;
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// It is tedious to write TupleOfTuplesOfArgTypes; here are some convenient type generators
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		// (T, N) -> std::tuple<T, T, ... T> (N times repetition)
		namespace detail {
			template <class T, class IndexSequence>
			struct type_power_impl;

			template <class T, std::size_t... I>
			struct type_power_impl<T, std::index_sequence<I...>> {
				template <std::size_t I>
				using duplicate = T;

				using type = std::tuple<duplicate<I>...>;
			};
		}
		template <class T, std::size_t N>
		using type_power = typename detail::type_power_impl<T, std::make_index_sequence<N>>::type;

		// (T1, ..., Tn) -> std::tuple<std::tuple<T1 const&, T1&&>, ... , std::tuple<Tn const&, Tn&&>>
		namespace detail {
			template <class IndexSequence, class... T>
			struct copy_or_move_impl;

			template <std::size_t... I, class... T>
			struct copy_or_move_impl<std::index_sequence<I...>, T...> {
				template <std::size_t J>
				using get_type = std::tuple_element_t<J, std::tuple<T...>>;

				template <std::size_t J>
				using get_pair = std::tuple<get_type<J> const&, get_type<J>&&>;

				using type = std::tuple<get_pair<I>...>;
			};
		}
		template <class... T>
		using copy_or_move = typename detail::copy_or_move_impl<
			std::make_index_sequence<sizeof...(T)>, T...>::type;

		// (T, N) -> std::tuple<std::tuple<T const&, T&&>, ... std::tuple<T const&, T&&>> (N times repetition)
		template <class T, std::size_t N>
		using copy_or_move_n = type_power<std::tuple<T const&, T&&>, N>;

		// ((T11, ... , T1n), ... ,(Tm1, ... , Tmn)) -> (T11, ... , Tmn)
		template <class... Tuples>
		using concat_tuples = decltype(std::tuple_cat(std::declval<Tuples>()...));
	}
}