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

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// List of types
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename...>
		struct typelist;

		template <typename... Lists>
		struct merge_typelist;

		template <typename... FirstListTypes, typename... RemainingLists>
		struct merge_typelist<typelist<FirstListTypes...>, RemainingLists...> {
			using type = typename merge_typelist<typelist<FirstListTypes...>,
				typename merge_typelist<RemainingLists...>::type>::type;
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
	}
}

namespace std {
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Specializations of std::tuple_element and std::tuple_length for jkj::tmp::typelist
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <std::size_t index, typename... Types>
	struct tuple_element<index, jkj::tmp::typelist<Types...>> {
		using type = tuple_element_t<index, typename jkj::tmp::typelist<Types...>::tuple_type>;
	};

	template <typename... Types>
	struct tuple_size<jkj::tmp::typelist<Types...>> : integral_constant<std::size_t, sizeof...(Types)> {};
}