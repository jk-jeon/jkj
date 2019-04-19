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
#include "typelist.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Swap two types in a typelist
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
		// Sort the parameter pack and then apply them to a variadic template
		// http://codereview.stackexchange.com/questions/131194/selection-sorting-a-type-list-compile-time
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
	}
}