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

// This header file defines two classes and related functionalities: jkl::array and jkl::array_ref.
// jkl::array is a statically-sized multidimensional array class, and jkl::array_ref is a
// companion type. In many aspects, (statically-sized) arrays are just special cases of the more
// general construct: tuples. However, both the C-style arrays and std::array's behave somewhat
// differently from tuples in some contexts. One example is move semantics. The natural
// element access operator, the operator[], always returns lvalue references, though
// the natural element access operator, the std::get function, returns lvalue or rvalue references
// according to the argument passed. (Note, however, that std::get returns respectful references
// also for std::array; it is not defined for C-style arrays, though.)
// I think the reason behind this fact goes back to the awkward relation between
// arrays and pointers that has resided since the old days of C. Arrays in C do have very weird
// semantics: basically, they follow the value semantics, but when passed into/returned from
// functions, they suddenly become pointers,which provide (broken, partial) reference semantics.
// Or, this is perhaps also partially due to the fact that there is no "pointers-to-rvalues" in
// the language. Hence, in some sense, the operator[] "breaks" the proper value semantics by
// unexpectedly converting an rvalue into an lvalue. One aspect of jkl::array is to fix this.
// It provides operator[] which returns respectful references. Its member function data() also
// returns either usual pointer or jkl::rvalue_ptr according to the lvalue/rvalue-ness of
// "this" pointer. Provided iterators also respect that.
// Another purpose of jkl::array and jkl::array_ref is to provide "flattened view" of
// multidimensional array. It is quite useful if we can view, iterate over, or manipulate a
// multidimensional array using the pointer to the first element. This gives us multiple advantages.
// For example, we can easily make "subarray views" from an array. This is technically possible for
// C-style arrays or std::array's using wrapper classes, but with C-style arrays or std::array,
// we always face into one of the following possibilities:
//   1. "Subarray views" of the same size would become different types if originated from arrays
//      with different sizes. This is quite annoying, and sometimes (quite often in generic context)
//      makes the situation needlessly complicated. Or,
//   2. reinterpret_cast or similar are involved, which implies that operations cannot be constexpr,
//      and sometimes (indeed, quite often) they result in an UB. Note that, according to
//      C/C++ standards, it is forbidden (UB) to access an element in a multidimensional array using
//      the pointer to the first element. Or,
//   3. Requires needless copies.
// Also, when C-style arrays or std::array's are used, we can't reliably find the actual dimension
// of a given array. For example, if we have std::array<std::array<int, 2>, 2>, normally it would
// mean a 2-dimensional array of size 2x2. However, who knows if the intended semantics is
// 1-dimensional array of std::array<int, 2>?
// Distinguishing these two (which is required not very rarely) is fundamentally impossible
// if we construct multidimensional arrays as multiple iterates of 1-dimensional arrays.

// Implementing jkl::array_ref, which is a "reference-like" class, revealed that
// it is extremely difficult in current C++, to describe the proper behavior such a class
// should have, when cv-qualifiers, ref-qualifiers, "reference-like" and "regular" types
// are all involved together. Note that it is literally impossible to make a class
// perfectly "reference-like" because some very special features of C++ references,
// for example lifetime extension of temporaries, are never possible.
// Due to this complexity, I can't sure about the correctness of the code.
// It seems almost impossible to make a test that fully covers all the use-cases.

#pragma once
#include <array>
#include <cassert>
#include <utility>
#include "rvalue_ptr.h"
#include "pseudo_ptr.h"

namespace jkl {
	namespace array_detail {
		// std::array, with improved value semantics regarding rvalue ref-qualifier and multidimensional support
		template <class T, std::size_t N1, std::size_t... Nr>
		class array;

		// Proxy reference-like view to std::array
		// Main purpose of this class is to allow an array-like view into a contiguous memory chunk
		template <class Reference, std::size_t N1, std::size_t... Nr>
		class array_ref;

		template <std::size_t... N>
		struct select_array_type;

		template <>
		struct select_array_type<> {
			template <class T>
			using c_array_type = T;
			template <class T>
			using std_array_type = T;
			template <class T>
			using jkl_array_type = T;
			template <class Reference>
			using array_ref_type = Reference;

			static constexpr std::size_t total_size = 1;
		};

		template <std::size_t N1, std::size_t... Nr>
		struct select_array_type<N1, Nr...> {
			// Be aware of the order!
			// The resulting type is T[N1][N2]...[NLast], NOT T[NLast]...[N2][N1]!
			// Well, array syntax of C (thus of C++) is horribly broken...
			template <class T>
			using c_array_type = typename select_array_type<Nr...>::template c_array_type<T>[N1];

			template <class T>
			using std_array_type = std::array<typename select_array_type<Nr...>::template std_array_type<T>, N1>;

			template <class T>
			using jkl_array_type = array<T, N1, Nr...>;

			template <class Reference>
			using array_ref_type = array_ref<Reference, N1, Nr...>;

			static constexpr std::size_t total_size = N1 * select_array_type<Nr...>::total_size;
		};

		// Perhaps not needed with C++17 fold expression...
		template <std::size_t... N>
		static constexpr std::size_t calculate_product = select_array_type<N...>::total_size;

		template <class Reference, class T>
		using forward_qualification = std::conditional_t<std::is_lvalue_reference_v<Reference>,
			std::conditional_t<std::is_const_v<std::remove_reference_t<Reference>>, T const&, T&>,
			std::conditional_t<std::is_const_v<std::remove_reference_t<Reference>>, T const&&, T&&>>;

		// Helps preventing undesired artifacts related to implicit conversion
		template <class T>
		struct type_wrapper {};

		// Check if a given type is a specialization of one of fours above
		// Note that, for example, int[2][3][4] can be interpreted as a C-style array
		// of size [2][3] of int[4]'s.
		template <class Type, std::size_t... N>
		struct is_c_array {
			template <class T, bool value>
			struct helper : std::integral_constant<bool, value> {
				using type = T;
			};

			template <std::size_t remaining_dimensions, class T, std::size_t... Nr>
			static constexpr auto check(T const&, std::index_sequence<Nr...>) noexcept {
				return helper<void, false>{};
			}

			template <std::size_t remaining_dimensions, class T, std::size_t Nfinal,
				std::size_t... Nr, class = std::enable_if_t<remaining_dimensions == 0>
			>
			static constexpr auto check(T const(&)[Nfinal], std::index_sequence<Nr...>) noexcept {
				return helper<Type,
					std::is_same_v<std::index_sequence<Nr..., Nfinal>, std::index_sequence<N...>>>{};
			}

			template <std::size_t remaining_dimensions, class T, std::size_t Nfinal,
				std::size_t... Nr, class = std::enable_if_t<remaining_dimensions != 0>, class = void
			>
			static constexpr auto check(T const(&temp)[Nfinal], std::index_sequence<Nr...>) noexcept {
				return check<remaining_dimensions - 1>(temp[0], std::index_sequence<Nr..., Nfinal>{});
			}

			using check_return = decltype(check<sizeof...(N)-1>(std::declval<Type>(), std::index_sequence<>{}));

			using element = typename check_return::type;
			static constexpr bool value = check_return::value;
		};
		template <class Type>
		struct is_c_array<Type> {
			using element = Type;
			// There is no such thing as an array of references,
			// but any non-reference type might be treated as a 0-dimensional C-style array type.
			static constexpr bool value = !std::is_reference_v<Type>;
		};

		template <class Type, std::size_t... N>
		static constexpr bool is_c_array_v = is_c_array<Type, N...>::value;

		template <class Type, std::size_t... N>
		struct is_std_array {
			template <class T, bool value>
			struct helper : std::integral_constant<bool, value> {
				using type = T;
			};

			template <std::size_t remaining_dimensions, class T, std::size_t... Nr>
			static constexpr auto check(T const&, std::index_sequence<Nr...>) noexcept {
				return helper<void, false>{};
			}

			template <std::size_t remaining_dimensions, class T, std::size_t Nfinal,
				std::size_t... Nr, class = std::enable_if_t<remaining_dimensions == 0>
			>
			static constexpr auto check(std::array<T, Nfinal> const&, std::index_sequence<Nr...>) noexcept {
				return helper<Type,
					std::is_same_v<std::index_sequence<Nr..., Nfinal>, std::index_sequence<N...>>>{};
			}

			template <std::size_t remaining_dimensions, class T, std::size_t Nfinal,
				std::size_t... Nr, class = std::enable_if_t<remaining_dimensions != 0>, class = void
			>
			static constexpr auto check(std::array<T, Nfinal> const& temp, std::index_sequence<Nr...>) noexcept {
				return check<remaining_dimensions - 1>(temp[0], std::index_sequence<Nr..., Nfinal>{});
			}

			using check_return = decltype(check<sizeof...(N)-1>(std::declval<Type>(), std::index_sequence<>{}));

			using element = typename check_return::type;
			static constexpr bool value = check_return::value;
		};
		template <class Type>
		struct is_std_array<Type> {
			using element = void;
			static constexpr bool value = false;
		};

		template <class Type, std::size_t... N>
		static constexpr bool is_std_array_v = is_std_array<Type, N...>::value;

		template <class Type, std::size_t... N>
		struct is_jkl_array {
			template <class T>
			static constexpr auto check(type_wrapper<T>) noexcept {
				return std::false_type{};
			}

			template <class T>
			static constexpr auto check(type_wrapper<array<T, N...>>) noexcept {
				return std::true_type{};
			}

			static constexpr bool value = decltype(check(type_wrapper<Type>{}))::value;
		};
		template <class Type>
		struct is_jkl_array<Type> {
			static constexpr bool value = !std::is_reference_v<Type>;
		};

		template <class Type, std::size_t... N>
		static constexpr bool is_jkl_array_v = is_jkl_array<Type, N...>::value;

		template <class Type, std::size_t... N>
		struct is_array_ref {
			template <class T>
			static constexpr auto check(type_wrapper<T>) noexcept {
				return std::false_type{};
			}

			template <class T>
			static constexpr auto check(type_wrapper<array_ref<T, N...>>) noexcept {
				return std::true_type{};
			}

			static constexpr bool value = decltype(check(type_wrapper<Type>{}))::value;
		};
		template <class Type>
		struct is_array_ref<Type> {
			static constexpr bool value = std::is_reference_v<Type>;
		};

		template <class Type, std::size_t... N>
		static constexpr bool is_array_ref_v = is_array_ref<Type, N...>::value;


		template <std::size_t N, class Array>
		constexpr auto get_first_address(Array&& arr) noexcept;

		template <std::size_t N1, std::size_t N2, std::size_t... Nr, class Array>
		constexpr auto get_first_address(Array&& arr) noexcept {
			return get_first_address<N2, Nr...>(arr[0]);
		}

		template <std::size_t N, class Array>
		constexpr auto get_first_address(Array&& arr) noexcept {
			return &arr[0];
		}

		template <class ThisElmtRef, class ThatElmtRef, class = void>
		struct inspect_equality : std::false_type {};

		template <class ThisElmtRef, class ThatElmtRef>
		struct inspect_equality<ThisElmtRef, ThatElmtRef,
			std::void_t<decltype(std::declval<ThisElmtRef>() == std::declval<ThatElmtRef>())>>
			: std::true_type {};

		template <class ThisElmtRef, class ThatElmtRef, bool has_equality>
		struct equality_noexcept : std::false_type {};

		template <class ThisElmtRef, class ThatElmtRef>
		struct equality_noexcept<ThisElmtRef, ThatElmtRef, true> {
			static constexpr bool value =
				noexcept(std::declval<ThisElmtRef>() == std::declval<ThatElmtRef>());
		};

		template <class ThisElmtRef, class ThatElmtRef, class = void>
		struct inspect_inequality : std::false_type {};

		template <class ThisElmtRef, class ThatElmtRef>
		struct inspect_inequality<ThisElmtRef, ThatElmtRef,
			std::void_t<decltype(std::declval<ThisElmtRef>() != std::declval<ThatElmtRef>())>>
			: std::true_type {};


		template <class ThisElmtRef, class ThatElmtRef, bool has_inequality>
		struct inequality_noexcept : std::false_type {};

		template <class ThisElmtRef, class ThatElmtRef>
		struct inequality_noexcept<ThisElmtRef, ThatElmtRef, true> {
			static constexpr bool value =
				noexcept(std::declval<ThisElmtRef>() != std::declval<ThatElmtRef>());
		};

		// Provide a function that serves the role of the expression (elmt1 != elmt2).
		// Also, check if such an expression is noexcept.
		// - If operator!= is defined for the provided arguments, call it.
		// - If operator!= is not defined but operator== is defined for the provided arguments,
		//   use operator== to get the behavior of operator!=.
		// I'm not sure if this is the correct way though, because there may be several concerns:
		// - If operator== is defined but operator!= is not defined, this is an unusual situation.
		//   That possibly means that even if "a == b" makes sense, "a != b" on the other hand
		//   does not make sense by whatever reason. What we really need is "a != b" not "a == b",
		//   this perhaps means we are doing wrong.
		// - What if both operator== and operator!= are defined, but only operator== is noexcept?
		//   Should we use operator== instead of operator!= in this case?
		//   Again, this is an unusual situation, because "a != b" should be just equivalent to
		//   "!(a == b)" mostly. Asymmetry in noexcept-ness may say that something is terribly
		//   abnormal so any ordinary reasoning might not work. Or, such an asymmetry would
		//   be just caused by some implementation characteristics. Remember, the reason why
		//   we can separately overload == and != is that there may be additional optimization
		//   chances. Thus, depending on how to implement those, there might be the case that
		//   one is noexcept while the other is not.
		// - Or, disparity between operator== and operator!= might have been there
		//   just because of the type authors' carelessness. Thus, while it is possible to
		//   detect such disparity, it might not be a good idea to trigger an error there.
		//   It would be good if there was a standard way to trigger a compiler warning...
		template <class ThisElmtRef, class ThatElmtRef>
		struct element_equality_helper {
			static constexpr bool are_equality_comparable =
				inspect_equality<ThisElmtRef, ThatElmtRef>::value;
			static constexpr bool are_nothrow_equality_comparable =
				equality_noexcept<ThisElmtRef, ThatElmtRef, are_equality_comparable>::value;
			static constexpr bool are_inequality_comparable =
				inspect_inequality<ThisElmtRef, ThatElmtRef>::value;
			static constexpr bool are_nothrow_inequality_comparable =
				inequality_noexcept<ThisElmtRef, ThatElmtRef, are_inequality_comparable>::value;

			template <class dummy = void, class = std::enable_if_t<are_inequality_comparable, dummy>>
			static constexpr bool not_equal(ThisElmtRef&& a, ThatElmtRef&& b)
				noexcept(are_nothrow_inequality_comparable)
			{
				return std::forward<ThisElmtRef>(a) != std::forward<ThatElmtRef>(b);
			}

			template <class dummy = void,
				class = std::enable_if_t<are_equality_comparable && !are_inequality_comparable, dummy>, class = void>
			static constexpr bool not_equal(ThisElmtRef&& a, ThatElmtRef&& b)
				noexcept(are_nothrow_equality_comparable)
			{
				return !(std::forward<ThisElmtRef>(a) == std::forward<ThatElmtRef>(b));
			}
		};

		struct bad_type_tag {};
		template <>
		struct element_equality_helper<bad_type_tag, bad_type_tag> {
			static constexpr bool are_equality_comparable = false;
			static constexpr bool are_nothrow_equality_comparable = false;
			static constexpr bool are_inequality_comparable = false;
			static constexpr bool are_nothrow_inequality_comparable = false;
		};

		template <std::size_t... N>
		struct type_publish_helper {
			template <class T>
			struct publish_array {
				static_assert(!std::is_reference_v<T>,
					"jkl::array: jkl::array cannot be instantiated with a reference type");
				using type = typename select_array_type<N...>::template jkl_array_type<T>;
			};

			template <class Reference>
			struct publish_array_ref {
				static_assert(std::is_reference_v<Reference>,
					"jkl::array: jkl::array_ref should be instantiated with a reference type");
				using type = typename select_array_type<N...>::template array_ref_type<Reference>;
			};
		};
	}

	template <class T, std::size_t... N>
	using array = typename array_detail::type_publish_helper<N...>::template publish_array<T>::type;

	template <class Reference, std::size_t... N>
	using array_ref = typename array_detail::type_publish_helper<N...>::template publish_array_ref<Reference>::type;

	// Convert jkl::array<T, N...> to array_ref with appropriate ref & cv qualification
	template <class T, std::size_t... N>
	constexpr array_ref<T&, N...> make_array_ref(array<T, N...>& x) noexcept {
		return x;
	}
	template <class T, std::size_t... N>
	constexpr array_ref<T const&, N...> make_array_ref(array<T, N...> const& x) noexcept {
		return x;
	}
	template <class T, std::size_t... N>
	constexpr array_ref<T&&, N...> make_array_ref(array<T, N...>&& x) noexcept {
		return std::move(x);
	}
	template <class T, std::size_t... N>
	constexpr array_ref<T const&&, N...> make_array_ref(array<T, N...> const&& x) noexcept {
		return std::move(x);
	}

	// Convert std::array<T, N> to array_ref with appropriate ref & cv qualification
	template <class T, std::size_t N>
	constexpr array_ref<T&, N> make_array_ref(std::array<T, N>& x) noexcept {
		return x;
	}
	template <class T, std::size_t N>
	constexpr array_ref<T const&, N> make_array_ref(std::array<T, N> const& x) noexcept {
		return x;
	}
	template <class T, std::size_t N>
	constexpr array_ref<T&&, N> make_array_ref(std::array<T, N>&& x) noexcept {
		return std::move(x);
	}
	template <class T, std::size_t N>
	constexpr array_ref<T const&&, N> make_array_ref(std::array<T, N> const&& x) noexcept {
		return std::move(x);
	}

	// Convert pointer to array_ref with appropriate ref & cv qualification
	// Note that no bound-check is enforced
	template <std::size_t... N, class T>
	constexpr array_ref<T&, N...> make_array_ref(T* ptr) noexcept {
		return array_ref<T&, N...>{ ptr };
	}
	template <std::size_t... N, class T>
	constexpr array_ref<T const&, N...> make_array_ref(T const* ptr) noexcept {
		return array_ref<T const&, N...>{ ptr };
	}
	template <std::size_t... N, class T>
	constexpr array_ref<T&&, N...> make_array_ref(rvalue_ptr<T> ptr) noexcept {
		return array_ref<T&&, N...>{ ptr };
	}
	template <std::size_t... N, class T>
	constexpr array_ref<T const&&, N...> make_array_ref(rvalue_ptr<T const> ptr) noexcept {
		return array_ref<T const&&, N...>{ ptr };
	}

	namespace array_detail {
		enum class array_type { jkl_array, jkl_array_ref, other };
		
		template <class Array, array_type, std::size_t... N>
		struct get_element_reference;

		template <class JklArray, std::size_t... N>
		struct get_element_reference<JklArray, array_type::jkl_array, N...> {
			using unqualified = std::remove_cv_t<std::remove_reference_t<JklArray>>;
			using reference = decltype(std::declval<JklArray>()[0]);
			using type = forward_qualification<JklArray, typename unqualified::element>;

			static constexpr bool is_jkl = true;
			using dimension_spec = std::index_sequence<N...>;
		};

		template <class JklArrayRef, std::size_t... N>
		struct get_element_reference<JklArrayRef, array_type::jkl_array_ref, N...> {
			using unqualified = std::remove_cv_t<std::remove_reference_t<JklArrayRef>>;
			using reference = decltype(std::declval<JklArrayRef>()[0]);
			using type = std::conditional_t<std::is_lvalue_reference_v<JklArrayRef>,
				std::add_lvalue_reference_t<std::remove_reference_t<typename unqualified::element_rvalue_reference>>,
				typename unqualified::element_rvalue_reference
			>;

			static constexpr bool is_jkl = true;
			using dimension_spec = std::index_sequence<N...>;
		};

		template <class OtherArray, array_type, std::size_t... N>
		struct get_element_reference<OtherArray, array_type::other, N...> {
			using unqualified = std::remove_cv_t<std::remove_reference_t<OtherArray>>;

			static_assert(is_c_array_v<unqualified, N...> || is_std_array_v<unqualified, N...>,
				"jkl::array: jkl::array or jkl::array_ref are compatible only with each other, C-style arrays, and std::array; "
				"did you check dimensions of arrays are the same?");

			using element_type = std::remove_pointer_t<decltype(get_first_address<N...>(std::declval<OtherArray>()))>;
			using reference = std::conditional_t<std::is_lvalue_reference_v<OtherArray>,
				decltype(std::declval<OtherArray>()[0]), decltype(std::move(std::declval<OtherArray>()[0]))>;
			using type = std::conditional_t<std::is_lvalue_reference_v<OtherArray>,
				element_type&, element_type&&>;

			static constexpr bool is_jkl = false;
			using dimension_spec = std::index_sequence<N...>;
		};

		template <class Reference>
		struct bad_get_element_reference {
			using reference = bad_type_tag;
			using type = bad_type_tag;
			static constexpr bool is_jkl = false;
			using dimension_spec = bad_type_tag;
		};

		// Main dirty template meta-stuffs for dealing with pair of arrays.
		template <class ThisArray, class ThatArray>
		struct array_helper {
			template <class T, std::size_t... N, class Array,
				class = std::enable_if_t<!is_jkl_array_v<Array, N...> && !is_array_ref_v<Array, N...>>
			>
			static constexpr auto check(type_wrapper<array<T, N...>>, type_wrapper<Array>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array, N...>{},
					get_element_reference<ThatArray, array_type::other, N...>{});
			}

			template <class T, std::size_t... N, class Array,
				class = std::enable_if_t<!is_jkl_array_v<Array, N...> && !is_array_ref_v<Array, N...>>
			>
			static constexpr auto check(type_wrapper<array_ref<T, N...>>, type_wrapper<Array>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array_ref, N...>{},
					get_element_reference<ThatArray, array_type::other, N...>{});
			}

			template <class Array, class T, std::size_t... N,
				class = std::enable_if_t<!is_jkl_array_v<Array, N...> && !is_array_ref_v<Array, N...>>
			>
			static constexpr auto check(type_wrapper<Array>, type_wrapper<array<T, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::other, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array, N...>{});
			}

			template <class Array, class T, std::size_t... N,
				class = std::enable_if_t<!is_jkl_array_v<Array, N...> && !is_array_ref_v<Array, N...>>
			>
			static constexpr auto check(type_wrapper<Array>, type_wrapper<array_ref<T, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::other, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array_ref, N...>{});
			}

			template <class T, class U, std::size_t... N>
			static constexpr auto check(type_wrapper<array<T, N...>>, type_wrapper<array<U, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array, N...>{});
			}

			template <class T, class U, std::size_t... N>
			static constexpr auto check(type_wrapper<array<T, N...>>, type_wrapper<array_ref<U, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array_ref, N...>{});
			}

			template <class T, class U, std::size_t... N>
			static constexpr auto check(type_wrapper<array_ref<T, N...>>, type_wrapper<array<U, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array_ref, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array, N...>{});
			}

			template <class T, class U, std::size_t... N>
			static constexpr auto check(type_wrapper<array_ref<T, N...>>, type_wrapper<array_ref<U, N...>>) {
				return std::make_pair(
					get_element_reference<ThisArray, array_type::jkl_array_ref, N...>{},
					get_element_reference<ThatArray, array_type::jkl_array_ref, N...>{});
			}

			static constexpr auto check(...) {
				return std::make_pair(
					bad_get_element_reference<ThisArray>{},
					bad_get_element_reference<ThatArray>{});
			}

			using helper_pair = decltype(check(
				type_wrapper<std::remove_cv_t<std::remove_reference_t<ThisArray>>>{},
				type_wrapper<std::remove_cv_t<std::remove_reference_t<ThatArray>>>{}));

			using this_helper = typename helper_pair::first_type;
			using that_helper = typename helper_pair::second_type;

			using this_reference = typename this_helper::reference;
			using this_element_reference = typename this_helper::type;
			using this_element = std::remove_cv_t<std::remove_reference_t<this_element_reference>>;
			using that_reference = typename that_helper::reference;
			using that_element_reference = typename that_helper::type;
			using that_element = std::remove_cv_t<std::remove_reference_t<that_element_reference>>;

			static constexpr bool same_dimension =
				std::is_same_v<this_reference, bad_type_tag> ? false :
				std::is_same_v<that_reference, bad_type_tag> ? false :
				std::is_same_v<typename this_helper::dimension_spec, typename that_helper::dimension_spec>;

			// Assuming same_dimension
			using dimension_spec = typename this_helper::dimension_spec;

			static constexpr bool enable_switch_this_is_jkl = same_dimension && this_helper::is_jkl && !that_helper::is_jkl;
			static constexpr bool enable_switch_that_is_jkl = same_dimension && !this_helper::is_jkl && that_helper::is_jkl;
			static constexpr bool enable_switch_both_are_jkl = same_dimension && this_helper::is_jkl && that_helper::is_jkl;

			static constexpr bool element_equality_comparable =
				element_equality_helper<this_element_reference, that_element_reference>::are_equality_comparable;
			static constexpr bool element_nothrow_equality_comparable =
				element_equality_helper<this_element_reference, that_element_reference>::are_nothrow_equality_comparable;
			static constexpr bool element_inequality_comparable =
				element_equality_helper<this_element_reference, that_element_reference>::are_inequality_comparable;
			static constexpr bool element_nothrow_inequality_comparable =
				element_equality_helper<this_element_reference, that_element_reference>::are_nothrow_inequality_comparable;

			static constexpr bool are_comparable = same_dimension &&
				(element_equality_comparable || element_inequality_comparable);
			static constexpr bool are_nothrow_comparable = are_comparable &&
				((element_inequality_comparable && element_nothrow_inequality_comparable) ||
				(!element_inequality_comparable && element_nothrow_equality_comparable));

			template <class dummy = void, class = std::enable_if_t<are_comparable && dimension_spec::size() >= 2, dummy>>
			static constexpr bool not_equal(this_reference a, that_reference b)
				noexcept(are_nothrow_comparable)
			{
				return std::forward<this_reference>(a) != std::forward<that_reference>(b);
			}

			template <class dummy = void, class = std::enable_if_t<are_comparable, dummy>, class = void>
			static constexpr bool not_equal(this_element_reference a, that_element_reference b)
				noexcept(are_nothrow_comparable)
			{
				return element_equality_helper<this_element_reference, that_element_reference>::not_equal(
					std::forward<this_element_reference>(a),
					std::forward<that_element_reference>(b));
			}

			static constexpr bool are_swappable = same_dimension &&
				std::is_same_v<std::remove_reference_t<this_element_reference>,
				std::remove_reference_t<that_element_reference>> &&
				std::is_swappable_v<std::remove_reference_t<this_element_reference>>;
			static constexpr bool are_nothrow_swappable = are_swappable
				&& std::is_nothrow_swappable_v<std::remove_reference_t<this_element_reference>>;

			static constexpr bool are_assignable = same_dimension &&
				std::is_assignable_v<this_element_reference, that_element_reference>;
			static constexpr bool are_nothrow_assignable = are_assignable
				&& std::is_nothrow_assignable_v<this_element_reference, that_element_reference>;

			static constexpr bool are_constructible = same_dimension &&
				std::is_constructible_v<this_element, that_element_reference>;
			static constexpr bool are_nothrow_constructible = are_assignable
				&& std::is_nothrow_constructible_v<this_element, that_element_reference>;
		};

		// Relations
		// NOTE: lexicographic comparisons are not provided

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>>
		constexpr bool operator==(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < a.size(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					std::forward<ThisArray>(a)[i],
					static_cast<typename array_helper<ThisArray, ThatArray>::that_reference>(b[i])) )
				{
					return false;
				}
			}
			return true;
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>>
		constexpr bool operator!=(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < a.size(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					std::forward<ThisArray>(a)[i],
					static_cast<typename array_helper<ThisArray, ThatArray>::that_reference>(b[i])) )
				{
					return true;
				}
			}
			return false;
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>, class = void>
		constexpr bool operator==(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < b.size(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					static_cast<typename array_helper<ThisArray, ThatArray>::this_reference>(a[i]),
					std::forward<ThatArray>(b)[i]) )
				{
					return false;
				}
			}
			return true;
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>, class = void>
		constexpr bool operator!=(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < b.size(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					static_cast<typename array_helper<ThisArray, ThatArray>::this_reference>(a[i]),
					std::forward<ThatArray>(b)[i]) )
				{
					return true;
				}
			}
			return false;
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_both_are_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>, class = void, class = void>
		constexpr bool operator==(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < a.num_elements(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					std::forward<ThisArray>(a).data()[i],
					std::forward<ThatArray>(b).data()[i]) )
				{
					return false;
				}
			}
			return true;
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_both_are_jkl &&
			array_helper<ThisArray, ThatArray>::are_comparable>, class = void, class = void>
		constexpr bool operator!=(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_comparable)
		{
			for( std::size_t i = 0; i < a.num_elements(); ++i ) {
				if( array_helper<ThisArray, ThatArray>::not_equal(
					std::forward<ThisArray>(a).data()[i],
					std::forward<ThatArray>(b).data()[i]) )
				{
					return true;
				}
			}
			return false;
		}

		// Swap
		// To make the ADL to properly select this swap functions, we can't rely on
		// universal references and SFINAE. We should define all possible overloads of swap
		// manually, although it is extremely tedious. Note that we also overload for array<T, N...>&&,
		// as swapping rvalues is occasionally useful. Of course, this overload will
		// generate errors when picked, if the element type itself does not support swap for rvalues.

		/* When left-hand side is jkl */

		template <class ThisArray, class ThatArray>
		constexpr void swap_impl_lhs(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_swappable)
		{
			using std::swap;
			for( std::size_t i = 0; i < a.size(); ++i ) {
				swap(std::forward<ThisArray>(a)[i],
					static_cast<typename array_helper<ThisArray, ThatArray>::that_reference>(b[i]));
			}
		}

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array<T, N...>&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array<T, N...>&, ThatArray>::are_swappable>>
		constexpr void swap(array<T, N...>& a, ThatArray&& b)
			noexcept(array_helper<array<T, N...>&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(a, std::forward<ThatArray>(b));
		}

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array<T, N...>&&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array<T, N...>&&, ThatArray>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, ThatArray&& b)
			noexcept(array_helper<array<T, N...>&&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(std::move(a), std::forward<ThatArray>(b));
		}

		// NOTE: const overloads are insufficient, as std::swap will be considered
		//       a better match than const overloads for non-const arguments

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array_ref<T, N...>&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array_ref<T, N...>&, ThatArray>::are_swappable>>
		constexpr void swap(array_ref<T, N...>& a, ThatArray&& b)
			noexcept(array_helper<array_ref<T, N...>&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(a, std::forward<ThatArray>(b));
		}

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array_ref<T, N...> const&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array_ref<T, N...> const&, ThatArray>::are_swappable>>
		constexpr void swap(array_ref<T, N...> const& a, ThatArray&& b)
			noexcept(array_helper<array_ref<T, N...> const&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(a, std::forward<ThatArray>(b));
		}

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array_ref<T, N...>&&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array_ref<T, N...>&&, ThatArray>::are_swappable>>
		constexpr void swap(array_ref<T, N...>&& a, ThatArray&& b)
			noexcept(array_helper<array_ref<T, N...>&&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(std::move(a), std::forward<ThatArray>(b));
		}

		template <class T, std::size_t... N, class ThatArray, class = std::enable_if_t<
			array_helper<array_ref<T, N...> const&&, ThatArray>::enable_switch_this_is_jkl &&
			array_helper<array_ref<T, N...> const&&, ThatArray>::are_swappable>>
		constexpr void swap(array_ref<T, N...> const&& a, ThatArray&& b)
			noexcept(array_helper<array_ref<T, N...> const&&, ThatArray>::are_nothrow_swappable)
		{
			swap_impl_lhs(std::move(a), std::forward<ThatArray>(b));
		}

		/* When right-hand side is jkl */

		template <class ThisArray, class ThatArray>
		constexpr void swap_impl_rhs(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_swappable)
		{
			using std::swap;
			for( std::size_t i = 0; i < b.size(); ++i ) {
				swap(static_cast<typename array_helper<ThisArray, ThatArray>::this_reference>(a[i]),
					std::forward<ThatArray>(b)[i]);
			}
		}

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array<T, N...>&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array<T, N...>&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array<T, N...>& b)
			noexcept(array_helper<ThisArray, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), b);
		}

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array<T, N...>&&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array<T, N...>&&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array<T, N...>&& b)
			noexcept(array_helper<ThisArray, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), std::move(b));
		}

		// NOTE: const overloads are insufficient, as std::swap will be considered better
		//       than const overloads for non-const arguments

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array_ref<T, N...>&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array_ref<T, N...>&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array_ref<T, N...>& b)
			noexcept(array_helper<ThisArray, array_ref<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), b);
		}

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array_ref<T, N...> const&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array_ref<T, N...> const&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array_ref<T, N...> const& b)
			noexcept(array_helper<ThisArray, array_ref<T, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), b);
		}

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array_ref<T, N...>&&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array_ref<T, N...>&&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array_ref<T, N...>&& b)
			noexcept(array_helper<ThisArray, array_ref<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), std::move(b));
		}

		template <class ThisArray, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<ThisArray, array_ref<T, N...> const&&>::enable_switch_that_is_jkl &&
			array_helper<ThisArray, array_ref<T, N...> const&&>::are_swappable>>
		constexpr void swap(ThisArray&& a, array_ref<T, N...> const&& b)
			noexcept(array_helper<ThisArray, array_ref<T, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_rhs(std::forward<ThisArray>(a), std::move(b));
		}

		/* When both sides are jkl */

		template <class ThisArray, class ThatArray>
		constexpr void swap_impl_both(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_swappable)
		{
			using std::swap;
			for( std::size_t i = 0; i < a.num_elements(); ++i ) {
				swap(std::forward<ThisArray>(a).data()[i], std::forward<ThatArray>(b).data()[i]);
			}
		}

		// jkl::array vs jkl::array

		template <class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array<T, N...>& b)
			noexcept(array_helper<array<T, N...>&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array<T, N...>&& b)
			noexcept(array_helper<array<T, N...>&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array<T, N...>& b)
			noexcept(array_helper<array<T, N...>&&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array<T, N...>&& b)
			noexcept(array_helper<array<T, N...>&&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		// jkl::array vs jkl::array_ref

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array_ref<Ref, N...>&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array_ref<Ref, N...>& b)
			noexcept(array_helper<array<T, N...>&, array_ref<Ref, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array_ref<Ref, N...> const&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array_ref<Ref, N...> const& b)
			noexcept(array_helper<array<T, N...>&, array_ref<Ref, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array_ref<Ref, N...>&&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array_ref<Ref, N...>&& b)
			noexcept(array_helper<array<T, N...>&, array_ref<Ref, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&, array_ref<Ref, N...> const&&>::are_swappable>>
		constexpr void swap(array<T, N...>& a, array_ref<Ref, N...> const&& b)
			noexcept(array_helper<array<T, N...>&, array_ref<Ref, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array_ref<Ref, N...>&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array_ref<Ref, N...>& b)
			noexcept(array_helper<array<T, N...>&&, array_ref<Ref, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array_ref<Ref, N...> const&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array_ref<Ref, N...> const& b)
			noexcept(array_helper<array<T, N...>&&, array_ref<Ref, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array_ref<Ref, N...>&&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array_ref<Ref, N...>&& b)
			noexcept(array_helper<array<T, N...>&&, array_ref<Ref, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		template <class T, class Ref, std::size_t... N, class = std::enable_if_t<
			array_helper<array<T, N...>&&, array_ref<Ref, N...> const&&>::are_swappable>>
		constexpr void swap(array<T, N...>&& a, array_ref<Ref, N...> const&& b)
			noexcept(array_helper<array<T, N...>&&, array_ref<Ref, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		// jkl::array_ref vs jkl::array

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...>&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...>& a, array<T, N...>& b)
			noexcept(array_helper<array_ref<Ref, N...>&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...> const&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...> const& a, array<T, N...>& b)
			noexcept(array_helper<array_ref<Ref, N...> const&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...>&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...>& a, array<T, N...>&& b)
			noexcept(array_helper<array_ref<Ref, N...>&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...> const&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...> const& a, array<T, N...>&& b)
			noexcept(array_helper<array_ref<Ref, N...> const&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...>&&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...>&& a, array<T, N...>& b)
			noexcept(array_helper<array_ref<Ref, N...>&&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...> const&&, array<T, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...> const&& a, array<T, N...>& b)
			noexcept(array_helper<array_ref<Ref, N...> const&&, array<T, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...>&&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...>&& a, array<T, N...>&& b)
			noexcept(array_helper<array_ref<Ref, N...>&&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		template <class Ref, class T, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref, N...> const&&, array<T, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref, N...> const&& a, array<T, N...>&& b)
			noexcept(array_helper<array_ref<Ref, N...> const&&, array<T, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		// jkl::array_ref vs jkl::array_ref

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>& a, array_ref<Ref2, N...>& b)
			noexcept(array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...> const&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>& a, array_ref<Ref2, N...> const& b)
			noexcept(array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>& a, array_ref<Ref2, N...>&& b)
			noexcept(array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...> const&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>& a, array_ref<Ref2, N...> const&& b)
			noexcept(array_helper<array_ref<Ref1, N...>&, array_ref<Ref2, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const& a, array_ref<Ref2, N...>& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...> const&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const& a, array_ref<Ref2, N...> const& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(a, b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const& a, array_ref<Ref2, N...>&& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...> const&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const& a, array_ref<Ref2, N...> const&& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&, array_ref<Ref2, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(a, std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>&& a, array_ref<Ref2, N...>& b)
			noexcept(array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...> const&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>&& a, array_ref<Ref2, N...> const& b)
			noexcept(array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>&& a, array_ref<Ref2, N...>&& b)
			noexcept(array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...> const&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...>&& a, array_ref<Ref2, N...> const&& b)
			noexcept(array_helper<array_ref<Ref1, N...>&&, array_ref<Ref2, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...>&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const&& a, array_ref<Ref2, N...>& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...>&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...> const&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const&& a, array_ref<Ref2, N...> const& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...> const&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), b);
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...>&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const&& a, array_ref<Ref2, N...>&& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...>&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}

		template <class Ref1, class Ref2, std::size_t... N, class = std::enable_if_t<
			array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...> const&&>::are_swappable>>
		constexpr void swap(array_ref<Ref1, N...> const&& a, array_ref<Ref2, N...> const&& b)
			noexcept(array_helper<array_ref<Ref1, N...> const&&, array_ref<Ref2, N...> const&&>::are_nothrow_swappable)
		{
			swap_impl_both(std::move(a), std::move(b));
		}


		// Assignment
		// Erasing overloads (using SFINAE) based on assignability of elements is not a good idea,
		// so we use static_assert on that instead.
		// The basic reason is that const lvalue references can bind to rvalues.
		// For example, a temporary of type array_ref<int&&, N...> can't be in the left-hand side of assignment,
		// as assignment into rvalues is prevented for primitive types. However, an lvalue reference into
		// the type array_ref<int&&, N...>, regardless of constness, is treated as an lvalue reference into an int array.
		// If assign_impl(array_ref<int&&, N...>&&, ThatArray&&) overload is erased, therefore,
		// a call to assign_impl with a temporary of type array_ref<int&&, NB...> will
		// resolved to assign_impl(array_ref<int&&, N...> const&, ThatArray&&), which is wrong.
		// This wrong call will end up with something like int& = int&,
		// although we would expect something like int&& = int&, which should result in a compile error.
		
		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_this_is_jkl>>
		constexpr void assign_impl(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_assignable)
		{
			static_assert(array_helper<ThisArray, ThatArray>::are_assignable,
				"jkl::array: elements of the source array are not assignable to "
				"elements of the destination array");
			for( std::size_t i = 0; i < a.size(); ++i ) {
				std::forward<ThisArray>(a)[i]
					= static_cast<typename array_helper<ThisArray, ThatArray>::that_reference>(b[i]);
			}
		}

		template <class ThisArray, class ThatArray, class = std::enable_if_t<
			array_helper<ThisArray, ThatArray>::enable_switch_both_are_jkl>, class = void>
		constexpr void assign_impl(ThisArray&& a, ThatArray&& b)
			noexcept(array_helper<ThisArray, ThatArray>::are_nothrow_assignable)
		{
			static_assert(array_helper<ThisArray, ThatArray>::are_assignable,
				"jkl::array: elements of the source array are not assignable to "
				"elements of the destination array");
			for( std::size_t i = 0; i < a.num_elements(); ++i ) {
				std::forward<ThisArray>(a).data()[i] = std::forward<ThatArray>(b).data()[i];
			}
		}


		// get
		template <std::size_t I, class T, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array<T, N1, Nr...>& arr) noexcept
		{
			return arr[I];
		}
		template <std::size_t I, class T, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array<T, N1, Nr...> const& arr) noexcept
		{
			return arr[I];
		}
		template <std::size_t I, class T, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array<T, N1, Nr...>&& arr) noexcept
		{
			return std::move(arr)[I];
		}
		template <std::size_t I, class T, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array<T, N1, Nr...> const&& arr) noexcept
		{
			return std::move(arr)[I];
		}
		template <std::size_t I, class Reference, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array_ref<Reference, N1, Nr...>& arr) noexcept
		{
			return arr[I];
		}
		template <std::size_t I, class Reference, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array_ref<Reference, N1, Nr...> const& arr) noexcept
		{
			return arr[I];
		}
		template <std::size_t I, class Reference, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array_ref<Reference, N1, Nr...>&& arr) noexcept
		{
			return std::move(arr)[I];
		}
		template <std::size_t I, class Reference, std::size_t N1, std::size_t... Nr>
		decltype(auto) get(array_ref<Reference, N1, Nr...> const&& arr) noexcept
		{
			return std::move(arr)[I];
		}


		template <std::size_t rank>
		class stride_base {
		public:
			// strides_[0] = (*this)[1] - (*this)[0]
			// strides_[1] = (*this)[0][1] - (*this)[0][0]
			// strides_[2] = (*this)[0][0][1] - (*this)[0][0][0]
			// ...
			using strides_type = array<std::size_t, rank>;

			constexpr std::size_t stride() const noexcept {
				return strides_[0];
			}

			constexpr auto& strides() const noexcept {
				return strides_;
			}

			constexpr auto substrides() const noexcept {
				return strides_.template subarray<rank - 1>(1);
			}

		private:
			strides_type strides_;
		};

		template <>
		class stride_base<0> {
		public:
			using strides_type = array<std::size_t, 0>;

			constexpr stride_base() noexcept {}
			constexpr stride_base(strides_type const&) noexcept {}

			constexpr std::size_t stride() const noexcept {
				return 1;
			}

			constexpr strides_type strides() const noexcept {
				return{};
			}
		};
				

		// Calculate (N1 x ... x Nn, N2 x ... x Nn, ... , Nn) from (N1, ... , Nn)
		template <std::size_t... Nr>
		struct calculate_strides;
		
		template <std::size_t N1, std::size_t... Nr>
		struct calculate_strides {
			using strides_type = array<std::size_t, sizeof...(Nr)+1>;

			template <std::size_t... I>
			constexpr strides_type calculate(std::index_sequence<I...>) {
				return{ N1 * calculate_strides<Nr...>::value[0],
					calculate_strides<Nr...>::value[I]... };
			}

			constexpr strides_type calculate(std::index_sequence<>) {
				return{ N1 };
			}

			static constexpr strides_type value =
				calculate(std::make_index_sequence<sizeof...(Nr)>{});
		};

		template <>
		struct calculate_strides<> {
			using strides_type = array<std::size_t, 0>;
			static constexpr strides_type value = {};
		};

		// Array iterator
		template <class Reference, std::size_t N1, std::size_t... Nr>
		class iterator : private stride_base<sizeof...(Nr)> {
			using stride_base = stride_base<sizeof...(Nr)>;
		public:
			// As this iterator dereferences to a proxy, rigorously speaking,
			// this iterator is not a RandomAccessIterator; it is only an InputIterator.
			using iterator_category = std::random_access_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using element = std::remove_cv_t<std::remove_reference_t<Reference>>;
			using value_type = jkl::array<element, Nr...>;
			using reference = jkl::array_ref<Reference, Nr...>;
			using pointer = std::conditional_t<sizeof...(Nr) == 0,
				std::remove_reference_t<Reference>*,
				pseudo_ptr<reference>>;

			using strides_type = typename stride_base::strides_type;

			template <class, std::size_t, std::size_t...>
			friend class iterator;

			using element_pointer = std::conditional_t<std::is_lvalue_reference_v<Reference>,
				std::remove_reference_t<Reference>*,
				rvalue_ptr<std::remove_reference_t<Reference>>
			>;

			iterator() = default;

			constexpr explicit iterator(
				element_pointer ptr,
				strides_type const& strides) noexcept
				: stride_base{ strides }, ptr{ ptr } {}

			template <class NonConstReference, class = std::enable_if_t<
				std::is_const_v<std::remove_reference_t<Reference>>, NonConstReference>
			> constexpr iterator(iterator<NonConstReference, N1, Nr...> itr) noexcept
				: stride_base{ itr.strides() }, ptr{ itr.ptr } {}

			// Dereference
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) == 0, dummy>>
			constexpr reference operator*() const noexcept {
				return *ptr;
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) != 0, dummy>, class = void>
			constexpr reference operator*() const noexcept {
				return reference{ ptr, stride_base::substrides() };
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) == 0, dummy>>
			constexpr reference operator[](difference_type n) const noexcept {
				return *(ptr + n);
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) != 0, dummy>, class = void>
			constexpr reference operator[](difference_type n) const noexcept {
				return reference{ ptr + n * stride_base::stride(), stride_base::substrides() };
			}

			// Member access (NOTE: this produces lvalues even from rvalues)
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) == 0, dummy>>
			constexpr pointer operator->() const noexcept {
				auto&& ref = *ptr;
				return &ref;
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) != 0, dummy>, class = void>
			constexpr pointer operator->() const noexcept {
				return pointer{ **this };
			}

			// Increment
			NONCONST_CONSTEXPR iterator& operator++() noexcept {
				ptr += stride_base::stride();
				return *this;
			}
			NONCONST_CONSTEXPR iterator operator++(int) noexcept {
				auto prev_ptr = ptr;
				ptr += stride_base::stride();
				return iterator{ prev_ptr, stride_base::strides() };
			}
			NONCONST_CONSTEXPR iterator& operator+=(difference_type n) noexcept {
				ptr += n * stride_base::stride();
				return *this;
			}
			constexpr iterator operator+(difference_type n) const noexcept {
				return iterator{ ptr + n * stride_base::stride(), stride_base::strides() };
			}
			friend constexpr iterator operator+(difference_type n, iterator itr) noexcept {
				return itr + n;
			}

			// Decrement
			NONCONST_CONSTEXPR iterator& operator--() noexcept {
				ptr -= stride_base::stride();
				return *this;
			}
			NONCONST_CONSTEXPR iterator operator--(int) noexcept {
				auto prev_ptr = ptr;
				ptr -= stride_base::stride();
				return iterator{ prev_ptr, stride_base::strides() };
			}
			NONCONST_CONSTEXPR iterator& operator-=(difference_type n) noexcept {
				ptr -= n * stride_base::stride();
				return *this;
			}
			constexpr iterator operator-(difference_type n) const noexcept {
				return iterator{ ptr - n * stride_base::stride(), stride_base::strides() };
			}

			// Distance
			template <class OtherReference>
			constexpr difference_type operator-(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return difference_type((ptr - itr.ptr) / stride_base::stride());
			}

			// Relations
			template <class OtherReference>
			constexpr bool operator==(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr == itr.ptr);
			}
			template <class OtherReference>
			constexpr bool operator!=(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr != itr.ptr);
			}
			template <class OtherReference>
			constexpr bool operator<(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr < itr.ptr);
			}
			template <class OtherReference>
			constexpr bool operator<=(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr <= itr.ptr);
			}
			template <class OtherReference>
			constexpr bool operator>(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr > itr.ptr);
			}
			template <class OtherReference>
			constexpr bool operator>=(iterator<OtherReference, N1, Nr...> itr) const noexcept {
				return relation_impl(itr, ptr >= itr.ptr);
			}

		private:
			element_pointer ptr;

			template <class Iterator>
			static constexpr bool relation_impl(Iterator, bool result) noexcept {
				static_assert(std::is_same_v<value_type, typename Iterator::value_type>,
					"jkl::array: iterators are not comparable");
				return result;
			}
		};

		// Common things for all 4 combinations
		template <class Reference, std::size_t N1, std::size_t... Nr>
		class array_ref_base : private stride_base<sizeof...(Nr)> {
			using stride_base = stride_base<sizeof...(Nr)>;

		public:
			using element = std::remove_cv_t<std::remove_reference_t<Reference>>;

			using element_reference = std::add_lvalue_reference_t<Reference>;
			using element_const_reference = std::add_const_t<std::remove_reference_t<Reference>>&;
			using element_rvalue_reference = Reference;
			using element_const_rvalue_reference = std::conditional_t<std::is_lvalue_reference_v<Reference>,
				element_const_reference,
				std::add_const_t<std::remove_reference_t<Reference>>&&
			>;

			using iterator = jkl::array_detail::iterator<element_reference, N1, Nr...>;
			using const_iterator = jkl::array_detail::iterator<element_const_reference, N1, Nr...>;
			using rvalue_iterator = jkl::array_detail::iterator<element_rvalue_reference, N1, Nr...>;
			using const_rvalue_iterator = jkl::array_detail::iterator<element_const_rvalue_reference, N1, Nr...>;

			using reverse_iterator = std::reverse_iterator<iterator>;
			using const_reverse_iterator = std::reverse_iterator<const_iterator>;			
			using rvalue_reverse_iterator = std::reverse_iterator<rvalue_iterator>;
			using const_rvalue_reverse_iterator = std::reverse_iterator<const_rvalue_iterator>;

			using element_pointer = typename iterator::element_pointer;
			using element_const_pointer = typename const_iterator::element_pointer;
			using element_rvalue_pointer = typename rvalue_iterator::element_pointer;
			using element_const_rvalue_pointer = typename const_rvalue_iterator::element_pointer;
			
			using reference = typename iterator::reference;
			using const_reference = typename const_iterator::reference;
			using rvalue_reference = typename rvalue_iterator::reference;
			using const_rvalue_reference = typename const_rvalue_iterator::reference;

			using pointer = typename iterator::pointer;
			using const_pointer = typename const_iterator::pointer;

			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using value_type = typename iterator::value_type;

			static constexpr size_type rank = sizeof...(Nr)+1;
			using shape_type = array<size_type, rank>;
			using strides_type = typename stride_base::strides_type;

		private:
			template <class T>
			using c_array = typename select_array_type<N1, Nr...>::template c_array_type<T>;
			template <class T>
			using std_array = typename select_array_type<N1, Nr...>::template std_array_type<T>;

			using qualified_c_array = forward_qualification<Reference, c_array<element>>;
			using qualified_std_array = forward_qualification<Reference, std_array<element>>;

			static constexpr bool std_array_contiguous = sizeof(std_array<element>) == sizeof(element) * calculate_product<N1, Nr...>;

		public:
			// Construct from C-style array
			// Rigorously speaking, iterating over a multi-dimensional C-style array with a single pointer to the
			// element type can lead to an UB. That is, we can't really do something like (&arr[0][0])[n] if n >= M
			// where arr is of type int[N][M]. The C standard (and the C++ standard) doesn't allow "flattened view"
			// of multi-dimensional array; see http://c-faq.com/aryptr/ary2dfunc2.html for more information.
			// However, accessing a multi-dimensional array using the pointer to the 1st element is somewhat an established style,
			// and I think accepting a multi-dimensional array and convert it to a pointer is virtually of no harm.
			constexpr array_ref_base(qualified_c_array arr) noexcept
				: stride_base{ calculate_strides<Nr...>::value },
				ptr{ static_cast<element_rvalue_pointer>(get_first_address<N1, Nr...>(arr)) } {}

			// Construct from std::array
			// std::array<T, N> is not guaranteed to have the size sizeof(T) * N.
			// That means, there may be padding between two consecutive std::array<T, N2>'s inside
			// std::array<std::array<T, N2>, N1>. Hence, in this case array_ref can't be compatible with std::array.
			// Also, note that by the same reason above, using the pointer to the 1st element is not strictly conforming,
			// so it can be potentially wrong. Though I think such a chance is extremely rare.
			template <class dummy = void, class = std::enable_if_t<std_array_contiguous, dummy>>
			constexpr explicit array_ref_base(qualified_std_array arr) noexcept
				: stride_base{ calculate_strides<Nr...>::value },
				ptr{ static_cast<element_rvalue_pointer>(get_first_address<N1, Nr...>(arr)) } {}
			
			// Construct from pointer
			constexpr explicit array_ref_base(element_rvalue_pointer ptr) noexcept
				: stride_base{ calculate_strides<Nr...>::value }, ptr{ ptr } {}

			constexpr explicit array_ref_base(element_rvalue_pointer ptr,
				strides_type const& strides) noexcept
				: strides_base{ strides }, ptr{ ptr } {}

			// Copy constructor
			constexpr array_ref_base(array_ref_base const&) = delete;

			// Copy assignment
			array_ref_base& operator=(array_ref_base const&) = delete;

			// Get pointer
			constexpr auto data() const& noexcept {
				// Possibly an rvalue-to-lvalue conversion
				auto&& ref = *ptr;
				return &ref;
			}
			constexpr auto data() const&& noexcept {
				return ptr;
			}

			// Access elements
		private:
			template <class ArrayRef>
			constexpr static decltype(auto) at_impl(ArrayRef&& arr, size_type pos) {
				if( pos >= N1 )
					throw std::out_of_range{ "jkl::array: out of range" };
				return std::forward<ArrayRef>(arr)[pos];
			}
			template <class ArrayRef>
			constexpr static decltype(auto) front_impl(ArrayRef&& arr) noexcept {
				assert(N1 > 0);
				return std::forward<ArrayRef>(arr)[0];
			}
			template <class ArrayRef>
			constexpr static decltype(auto) back_impl(ArrayRef&& arr) noexcept {
				assert(N1 > 0);
				return std::forward<ArrayRef>(arr)[N1 - 1];
			}
		public:
			constexpr decltype(auto) at(size_type pos) const& {
				return at_impl(*this, pos);
			}
			constexpr decltype(auto) at(size_type pos) const&& {
				return at_impl(std::move(*this), pos);
			}
			constexpr decltype(auto) operator[](size_type pos) const& noexcept {
				assert(pos < N1);
				return begin()[pos];
			}
			constexpr decltype(auto) operator[](size_type pos) const&& noexcept {
				assert(pos < N1);
				return std::move(*this).begin()[pos];
			}
			constexpr decltype(auto) front() const& noexcept {
				return front_impl(*this);
			}
			constexpr decltype(auto) front() const&& noexcept {
				return front_impl(std::move(*this));
			}
			constexpr decltype(auto) back() const& noexcept {
				return back_impl(*this);
			}
			constexpr decltype(auto) back() const&& noexcept {
				return back_impl(std::move(*this));
			}

			// Iterators
			constexpr iterator begin() const& noexcept {
				return iterator{ data(), stride_base::strides() };
			}
			constexpr const_iterator cbegin() const& noexcept {
				return begin();
			}
			constexpr rvalue_iterator begin() const&& noexcept {
				return rvalue_iterator{ std::move(*this).data(), stride_base::strides() };
			}
			constexpr const_rvalue_iterator cbegin() const&& noexcept {
				return std::move(*this).begin();
			}

			constexpr iterator end() const& noexcept {
				return iterator{ data() + size(), stride_base::strides() };
			}
			constexpr const_iterator cend() const& noexcept {
				return end();
			}
			constexpr rvalue_iterator end() const&& noexcept {
				return rvalue_iterator{ std::move(*this).data() + size(), stride_base::strides() };
			}
			constexpr const_rvalue_iterator cend() const&& noexcept {
				return std::move(*this).end();
			}

			constexpr reverse_iterator rbegin() const& noexcept {
				return std::make_reverse_iterator(end());
			}
			constexpr const_reverse_iterator crbegin() const& noexcept {
				return rbegin();
			}
			constexpr rvalue_reverse_iterator rbegin() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).end());
			}
			constexpr const_rvalue_reverse_iterator crbegin() const&& noexcept {
				return std::move(*this).rbegin();
			}

			constexpr reverse_iterator rend() const& noexcept {
				return std::make_reverse_iterator(begin());
			}
			constexpr const_reverse_iterator crend() const& noexcept {
				return rend();
			}
			constexpr rvalue_reverse_iterator rend() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).begin());
			}
			constexpr const_rvalue_reverse_iterator crend() const&& noexcept {
				return std::move(*this).rend();
			}

			// Capacity
			constexpr bool empty() const noexcept {
				return size() == 0;
			}
			constexpr size_type size() const noexcept {
				return N1;
			}
			constexpr size_type max_size() const noexcept {
				return size();
			}
			constexpr size_type num_elements() const noexcept {
				return calculate_product<N1, Nr...>;
			}
			constexpr shape_type shape() const noexcept {
				return{ N1, Nr... };
			}
			constexpr size_type stride() const noexcept {
				return stride_base::stride();
			}
			constexpr strides_type strides() const noexcept {
				return stride_base::strides();
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) != 0, dummy>>
			constexpr auto substrides() const noexcept {
				return stride_base::substrides();
			}
			constexpr bool is_contiguous() const noexcept {
				return stride() == calculate_product<Nr...>;
			}

		private:
			element_rvalue_pointer ptr;
		};

		// Common things for mutable references
		template <class Reference, std::size_t... N>
		class array_ref_mutable_base : public array_ref_base<Reference, N...> {
			using base_type = array_ref_base<Reference, N...>;

		public:
			using base_type::base_type;
			using base_type::data;
			using base_type::num_elements;
			using element = typename base_type::element;

			constexpr void fill(element const& x) const&
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				// std::fill is constexpr since C++20; before that, we role our own
				for( std::size_t i = 0; i < num_elements(); ++i )
					data()[i] = x;
			}

			constexpr void fill(element const& x) const&&
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				// std::fill is constexpr since C++20; before that, we role our own
				for( std::size_t i = 0; i < num_elements(); ++i )
					std::move(*this).data()[i] = x;
			}

			// Swap
			template <class ThatArray>
			constexpr void swap(ThatArray&& that) const&
				noexcept(std::is_nothrow_swappable_v<element>)
			{
				jkl::swap(static_cast<array_ref<Reference, N...> const&>(*this),
					std::forward<ThatArray>(that));
			}

			template <class ThatArray>
			constexpr void swap(ThatArray&& that) const&&
				noexcept(std::is_nothrow_swappable_v<element>)
			{
				jkl::swap(static_cast<array_ref<Reference, N...> const&&>(*this),
					std::forward<ThatArray>(that));
			}
		};



		// array_ref for lvalue reference
		template <class T, std::size_t N1, std::size_t... Nr>
		class array_ref<T&, N1, Nr...> : public array_ref_mutable_base<T&, N1, Nr...> {
			using base_type = array_ref_mutable_base<T&, N1, Nr...>;

		public:
			using base_type::base_type;
			using base_type::data;
			using base_type::strides;
			using element = typename base_type::element;

			// Copy constructor
			constexpr array_ref(array_ref const& that) noexcept
				: base_type{ that.data(), that.strides() } {}

			// T const& can bind to T&
			constexpr operator array_ref<T const&, N1, Nr...>() const noexcept {
				return array_ref<T const&, N1, Nr...>{ data(), strides() };
			}
			// T&& can bind to T& by explicit casting
			constexpr explicit operator array_ref<T&&, N1, Nr...>() const noexcept {
				return array_ref<T&&, N1, Nr...>{ rvalue_ptr<T>{ data(), strides() } };
			}
			// T const&& can bind to T& by explicit casting
			constexpr explicit operator array_ref<T const&&, N1, Nr...>() const noexcept {
				return array_ref<T const&&, N1, Nr...>{ rvalue_ptr<T const>{ data(), strides() } };
			}

			// Overwrite elements
			constexpr array_ref& operator=(array_ref const& that) &
				noexcept(std::is_nothrow_copy_assignable_v<element>)	// Copy assignment
			{
				assign_impl(*this, that);
				return *this;
			}

			template <class ThatArray, class =
				decltype(assign_impl(std::declval<array_ref const&>(), std::declval<ThatArray>()))>
			constexpr array_ref const& operator=(ThatArray&& that) const
				noexcept(array_helper<array_ref const&, ThatArray>::are_nothrow_assignable)
			{
				assign_impl(*this, std::forward<ThatArray>(that));
				return *this;
			}
		};

		// array_ref for const lvalue reference
		template <class T, std::size_t N1, std::size_t... Nr>
		class array_ref<T const&, N1, Nr...> : public array_ref_base<T const&, N1, Nr...> {
			using base_type = array_ref_base<T const&, N1, Nr...>;

		public:
			using base_type::base_type;

			// Copy constructor
			constexpr array_ref(array_ref const& that) noexcept : base_type{ that.data() } {}
		};

		// array_ref for rvalue reference
		template <class T, std::size_t N1, std::size_t... Nr>
		class array_ref<T&&, N1, Nr...> : public array_ref_mutable_base<T&&, N1, Nr...> {
			using base_type = array_ref_mutable_base<T&&, N1, Nr...>;

		public:
			using base_type::base_type;
			using base_type::data;
			using element = typename base_type::element;

			// NOTE:
			// Note that the expression "x", where x is a variable of type T&&, is of type T&.
			// Hence, an lvalue reference to array_ref<T&&, N> should be treated as something like
			// array_ref<T&, N>. Thus, here we need to care about ref-qualifiers.

			// Move constructor
			constexpr array_ref(array_ref&& that) noexcept : base_type{ that.data() } {}

			// T& can bind to T&
			constexpr operator array_ref<T&, N1, Nr...>() const& noexcept {
				return array_ref<T&, N1, Nr...>{ data() };
			}
			// T const& can bind to both T& and T&&
			constexpr operator array_ref<T const&, N1, Nr...>() const noexcept {
				return array_ref<T const&, N1, Nr...>{ data() };
			}
			// T&& can bind to T& by explicit casting
			constexpr explicit operator array_ref<T&&, N1, Nr...>() const& noexcept {
				return{ rvalue_ptr<T>{ data() } };
			}
			// T const&& can bind to T& by explicit casting
			constexpr explicit operator array_ref<T const&&, N1, Nr...>() const& noexcept {
				return array_ref<T const&&, N1, Nr...>{ rvalue_ptr<T const>{ data() } };
			}
			// T const&& can bind to T&&
			constexpr operator array_ref<T const&&, N1, Nr...>() const&& noexcept {
				return array_ref<T const&&, N1, Nr...>{ rvalue_ptr<T const>{ data() } };
			}

			// Overwrite elements
			constexpr array_ref& operator=(array_ref const& that) &
				noexcept(std::is_nothrow_copy_assignable_v<element>)	// Copy assignment
			{
				assign_impl(*this, that);
				return *this;
			}			

			template <class ThatArray, class =
				decltype(assign_impl(std::declval<array_ref const&>(), std::declval<ThatArray>()))>
			constexpr array_ref const& operator=(ThatArray&& that) const&
				noexcept(array_helper<array_ref const&, ThatArray>::are_nothrow_assignable)
			{
				assign_impl(*this, std::forward<ThatArray>(that));
				return *this;
			}

			template <class ThatArray, class =
				decltype(assign_impl(std::declval<array_ref const&&>(), std::declval<ThatArray>()))>
			constexpr array_ref const&& operator=(ThatArray&& that) const&&
				noexcept(array_helper<array_ref const&&, ThatArray>::are_nothrow_assignable)
			{
				assign_impl(std::move(*this), std::forward<ThatArray>(that));
				return std::move(*this);
			}
		};

		// array_ref for const rvalue reference
		template <class T, std::size_t N1, std::size_t... Nr>
		class array_ref<T const&&, N1, Nr...> : public array_ref_base<T const&&, N1, Nr...> {
			using base_type = array_ref_base<T const&&, N1, Nr...>;

		public:
			using base_type::base_type;
			using base_type::data;

			// Move constructor
			constexpr array_ref(array_ref&& that) noexcept : base_type{ that.data() } {}

			// T const& can bind to both T const& and T const&&
			constexpr operator array_ref<T const&, N1, Nr...>() const noexcept {
				return{ data() };
			}
		};


		
		// array class
		template <class T, std::size_t N1, std::size_t... Nr>
		class array {
		public:
			static_assert(!std::is_reference_v<T>, "jkl::array: array of references is not allowed; "
				"try array_ref<Reference, N...> instead");

			using element = std::remove_cv_t<T>;

			using element_reference = T&;
			using element_const_reference = std::add_const_t<T>&;
			using element_rvalue_reference = T&&;
			using element_const_rvalue_reference = std::add_const_t<T>&&;

			using iterator = jkl::array_detail::iterator<element_reference, N1, Nr...>;
			using const_iterator = jkl::array_detail::iterator<element_const_reference, N1, Nr...>;
			using rvalue_iterator = jkl::array_detail::iterator<element_rvalue_reference, N1, Nr...>;
			using const_rvalue_iterator = jkl::array_detail::iterator<element_const_rvalue_reference, N1, Nr...>;

			using reverse_iterator = std::reverse_iterator<iterator>;
			using const_reverse_iterator = std::reverse_iterator<const_iterator>;
			using rvalue_reverse_iterator = std::reverse_iterator<rvalue_iterator>;
			using const_rvalue_reverse_iterator = std::reverse_iterator<const_rvalue_iterator>;

			using element_pointer = typename iterator::element_pointer;
			using element_const_pointer = typename const_iterator::element_pointer;
			using element_rvalue_pointer = typename rvalue_iterator::element_pointer;
			using element_const_rvalue_pointer = typename const_rvalue_iterator::element_pointer;

			using reference = typename iterator::reference;
			using const_reference = typename const_iterator::reference;
			using rvalue_reference = typename rvalue_iterator::reference;
			using const_rvalue_reference = typename const_rvalue_iterator::reference;

			using pointer = typename iterator::pointer;
			using const_pointer = typename const_iterator::pointer;

			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using value_type = typename iterator::value_type;

			static constexpr size_type rank = sizeof...(Nr)+1;
			using shape_type = array<size_type, rank>;

		private:
			template <class T>
			using c_array = typename select_array_type<N1, Nr...>::template c_array_type<T>;
			template <class T>
			using std_array = typename select_array_type<N1, Nr...>::template std_array_type<T>;

			// It is not easy to construct an array member without triggering any default construction.
			// These are the things used for that purpose.
			
			template <class U>
			auto&& access_element(U&& x, std::index_sequence<>) {
				return std::forward<U>(x);
			}
			template <class Array, std::size_t I1, std::size_t... Ir>
			auto&& access_element(Array&& arr, std::index_sequence<I1, Ir...>) {
				using reference = std::conditional_t<std::is_lvalue_reference_v<Array>,
					decltype(arr[I1]),
					std::add_rvalue_reference_t<std::remove_reference_t<decltype(arr[I1])>>
				>;
				return access_element(static_cast<reference>(arr[I1]), std::index_sequence<Ir...>{});
			}

			template <class Indices, class UnexpandedDims>
			struct index_temporary {};

			template <std::size_t index, class IndexTemporary>
			struct decay_index_temporary;

			template <std::size_t index, std::size_t... previous_indices,
				std::size_t first_unexpanded_dim, std::size_t... remaining_unexpanded_dims>
			struct decay_index_temporary<index,
				index_temporary<
				std::index_sequence<previous_indices...>,
				std::index_sequence<first_unexpanded_dim, remaining_unexpanded_dims...>
				>>
			{
				using type = index_temporary<
					std::index_sequence<previous_indices..., index>,
					std::index_sequence<remaining_unexpanded_dims...>
				>;
			};

			template <std::size_t index, class IndexTemporary>
			using decay_index_temporary_t = typename decay_index_temporary<index, IndexTemporary>::type;

			template <class IndexTemporary>
			struct get_indices;

			template <std::size_t... indices>
			struct get_indices<index_temporary<std::index_sequence<indices...>, std::index_sequence<>>>
			{
				using type = std::index_sequence<indices...>;
			};

			template <class IndexTemporary>
			using get_indices_t = typename get_indices<IndexTemporary>::type;

			struct expand_tag {};
			struct expanding_tag {};

			// Expand the first unexpanded dimension of the first index temporary object
			template <class Array, std::size_t... indices,
				std::size_t first_unexpanded_dim, std::size_t... remaining_unexpanded_dims,
				class... RemainingIndexTemporaries
			> constexpr array(expand_tag, Array&& arr,
				index_temporary<
				std::index_sequence<indices...>,
				std::index_sequence<first_unexpanded_dim, remaining_unexpanded_dims...>
				> first_it,
				RemainingIndexTemporaries... remaining_its)
				: array{ expanding_tag{}, std::forward<Array>(arr),
				std::make_index_sequence<first_unexpanded_dim>{},
				first_it, remaining_its... } {}

			// Put expanded index temporaries after the previous index temporary object list
			template <class Array, std::size_t... indices, class FirstIndexTemporary,
				class... RemainingIndexTemporaries
			> constexpr array(expanding_tag, Array&& arr,
				std::index_sequence<indices...>,
				FirstIndexTemporary,
				RemainingIndexTemporaries... remaining_its)
				: array{ expand_tag{}, std::forward<Array>(arr), remaining_its...,
				decay_index_temporary_t<indices, FirstIndexTemporary>{}... } {}

			// If the first index temporary has no unexpanded dimension,
			// get elements of arr and forward them to the standard constructor
			template <class Array, std::size_t... first_indices,
				class... RemainingIndexTemporaries
			> constexpr array(expand_tag, Array&& arr,
				index_temporary<std::index_sequence<first_indices...>, std::index_sequence<>>,
				RemainingIndexTemporaries...)
				: array{ access_element(std::forward<Array>(arr), std::index_sequence<first_indices...>{}),
				access_element(std::forward<Array>(arr), get_indices_t<RemainingIndexTemporaries>{})... } {}

			// To reduce boilerplate further
			struct start_expansion_tag {};
			template <class Array>
			constexpr array(start_expansion_tag, Array&& arr)
				: array{ expand_tag{}, std::forward<Array>(arr),
				index_temporary<std::index_sequence<>, std::index_sequence<N1, Nr...>>{} } {}
			
			template <class dummy, class... U>
			struct check_constructor_arg_validity;

			template <class dummy>
			struct check_constructor_arg_validity<dummy> {
				static constexpr bool value = true;
				static constexpr bool is_noexcept = true;
			};

			template <class dummy, class First, class... Remainings>
			struct check_constructor_arg_validity<dummy, First, Remainings...>
			{
				static constexpr bool value =
					std::is_constructible_v<element, First> &&
					check_constructor_arg_validity<dummy, Remainings...>::value;
				static constexpr bool is_noexcept =
					std::is_nothrow_constructible_v<element, First> &&
					check_constructor_arg_validity<dummy, Remainings...>::is_noexcept;
			};

			// For generic construction from 4 types of arrays
			template <class Array, bool is_jkl>
			struct construct_from_array_enable_switch {
				using unqualified = std::remove_cv_t<std::remove_reference_t<Array>>;

				static constexpr bool is_proper_array = is_jkl ?
					(is_jkl_array_v<unqualified, N1, Nr...> ||
						is_array_ref_v<unqualified, N1, Nr...>) :
					(is_c_array_v<unqualified, N1, Nr...> ||
						is_std_array_v<unqualified, N1, Nr...>);

				template <bool is_proper, class dummy = void>
				struct check_constructibility : std::false_type {};

				template <class dummy>
				struct check_constructibility<true, dummy> {
					static constexpr bool value =
						array_helper<array&, Array>::are_constructible;
				};

				static constexpr bool value =
					!std::is_base_of_v<array, unqualified> &&	// prevent too perfect forwarding
					check_constructibility<is_proper_array>::value;
			};

		public:
			// Standard constructor
			// Directly initialize elements
			template <class... U, class = std::enable_if_t<
				check_constructor_arg_validity<void, U...>::value &&
				sizeof...(U) != 1 &&
				sizeof...(U) == calculate_product<N1, Nr...>>
			> constexpr array(U&&... elmts)
				noexcept(check_constructor_arg_validity<void, U...>::is_noexcept)
				: data_{ std::forward<U>(elmts)... } {}

			template <class... U, class = std::enable_if_t<
				check_constructor_arg_validity<void, U...>::value &&
				std::is_default_constructible_v<element> &&
				sizeof...(U) != 1 &&
				sizeof...(U) + 1 <= calculate_product<N1, Nr...>>, class = void
			> constexpr array(U&&... elmts)
				noexcept(check_constructor_arg_validity<void, U...>::is_noexcept &&
					std::is_nothrow_default_constructible_v<element>)
				: data_{ std::forward<U>(elmts)... } {}

			// The case of only 1 argument should be treated separately
			template <class U, class = std::enable_if_t<
				std::is_constructible_v<element, U> &&
				calculate_product<N1, Nr...> == 1 &&
				!std::is_base_of_v<array, std::decay_t<U>>>
			> constexpr array(U&& elmt)
				//noexcept(std::is_nothrow_constructible_v<element, U>)
				: data_{ std::forward<U>(elmt) } {}

			template <class U, class = std::enable_if_t<
				std::is_constructible_v<element, U> &&
				std::is_default_constructible_v<element> &&
				calculate_product<N1, Nr...> >= 2 &&
				!std::is_base_of_v<array, std::decay_t<U>>>, class = void
			> constexpr array(U&& elmt)
				noexcept(std::is_nothrow_constructible_v<element, U>)
				: data_{ std::forward<U>(elmt) } {}

			// Construct from a C-style array of same type by copy
			template <class dummy = void, class = std::enable_if_t<
				std::is_copy_constructible_v<element>, dummy>
			> constexpr array(c_array<element> const& arr)
				noexcept(std::is_nothrow_copy_constructible_v<element>)
				: array{ start_expansion_tag{}, arr } {}

			// Construct from a C-style array of same type by move
			template <class dummy = void, class = std::enable_if_t<
				std::is_move_constructible_v<element>, dummy>
			> constexpr array(c_array<element>&& arr)
				noexcept(std::is_nothrow_move_constructible_v<element>)
				: array{ start_expansion_tag{}, std::move(arr) } {}

			// Construct from a C-style array or a std::array possibly of different type
			template <class Array, class = std::enable_if_t<
				construct_from_array_enable_switch<Array, false>::value>, class = void, class = void
			> explicit constexpr array(Array&& arr)
				noexcept(array_helper<array&, Array>::are_nothrow_constructible)
				: array{ start_expansion_tag{}, std::forward<Array>(arr) } {}
			
			// Construct from jkl::array or jkl::array_ref possibly of different type
			template <class Array, class = std::enable_if_t<
				construct_from_array_enable_switch<Array, true>::value>, class = void, class = void, class = void
			> constexpr array(Array&& arr)
				noexcept(array_helper<array&, Array>::are_nothrow_constructible)
				: array{ start_expansion_tag{}, std::forward<Array>(arr) } {}

			// Default copy/move constructors
			constexpr array(array const&) = default;
			constexpr array(array&&) = default;

			// Get jkl::array_ref
			constexpr operator array_ref<element&, N1, Nr...>() & noexcept {
				return array_ref<element&, N1, Nr...>{ data() };
			}
			constexpr operator array_ref<element const&, N1, Nr...>() const& noexcept {
				return array_ref<element const&, N1, Nr...>{ data() };
			}
			constexpr operator array_ref<element&&, N1, Nr...>() && noexcept {
				return array_ref<element&&, N1, Nr...>{ std::move(*this).data() };
			}
			constexpr operator array_ref<element const&&, N1, Nr...>() const&& noexcept {
				return array_ref<element const&&, N1, Nr...>{ std::move(*this).data() };
			}

			// Assignments
			constexpr array& operator=(array const&) & = default;
			constexpr array& operator=(array&&) & = default;
			template <class ThatArray, class =
				decltype(assign_impl(std::declval<array&>(), std::declval<ThatArray>())),
				class = std::enable_if_t<!std::is_base_of_v<array, std::decay_t<ThatArray>>>
			> constexpr array& operator=(ThatArray&& that) &
				noexcept(array_helper<array&, ThatArray>::are_nothrow_assignable)
			{
				assign_impl(*this, std::forward<ThatArray>(that));
				return *this;
			}
			template <class ThatArray, class =
				decltype(assign_impl(std::declval<array&&>(), std::declval<ThatArray>()))
			> constexpr array&& operator=(ThatArray&& that) &&
				noexcept(array_helper<array&&, ThatArray>::are_nothrow_assignable)
			{
				assign_impl(std::move(*this), std::forward<ThatArray>(that));
				return std::move(*this);
			}


			// Get pointer
			constexpr auto data() & noexcept {
				return data_.data();
			}
			constexpr auto data() const& noexcept {
				return data_.data();
			}
			constexpr auto data() && noexcept {
				return make_rvalue_ptr(data_.data());
			}
			constexpr auto data() const&& noexcept {
				return make_rvalue_ptr(data_.data());
			}

			// Access elements
		private:
			template <class Array>
			constexpr static decltype(auto) at_impl(Array&& arr, size_type pos) {
				if( pos >= N1 )
					throw std::out_of_range{ "jkl::array: out of range" };
				return std::forward<Array>(arr)[pos];
			}
			template <class Array>
			constexpr static decltype(auto) front_impl(Array&& arr) noexcept {
				assert(N1 > 0);
				return std::forward<Array>(arr)[0];
			}
			template <class Array>
			constexpr static decltype(auto) back_impl(Array&& arr) noexcept {
				assert(N1 > 0);
				return std::forward<Array>(arr)[N1 - 1];
			}
		public:
			constexpr decltype(auto) at(size_type pos) & {
				return at_impl(*this, pos);
			}
			constexpr decltype(auto) at(size_type pos) const& {
				return at_impl(*this, pos);
			}
			constexpr decltype(auto) at(size_type pos) && {
				return at_impl(std::move(*this), pos);
			}
			constexpr decltype(auto) at(size_type pos) const&& {
				return at_impl(std::move(*this), pos);
			}
			constexpr decltype(auto) operator[](size_type pos) & noexcept {
				assert(pos < N1);
				return begin()[pos];
			}
			constexpr decltype(auto) operator[](size_type pos) const& noexcept {
				assert(pos < N1);
				return begin()[pos];
			}
			constexpr decltype(auto) operator[](size_type pos) && noexcept {
				assert(pos < N1);
				return std::move(*this).begin()[pos];
			}
			constexpr decltype(auto) operator[](size_type pos) const&& noexcept {
				assert(pos < N1);
				return std::move(*this).begin()[pos];
			}
			constexpr decltype(auto) front() & noexcept {
				return front_impl(*this);
			}
			constexpr decltype(auto) front() const& noexcept {
				return front_impl(*this);
			}
			constexpr decltype(auto) front() && noexcept {
				return front_impl(std::move(*this));
			}
			constexpr decltype(auto) front() const&& noexcept {
				return front_impl(std::move(*this));
			}
			constexpr decltype(auto) back() & noexcept {
				return back_impl(*this);
			}
			constexpr decltype(auto) back() const& noexcept {
				return back_impl(*this);
			}
			constexpr decltype(auto) back() && noexcept {
				return back_impl(std::move(*this));
			}
			constexpr decltype(auto) back() const&& noexcept {
				return back_impl(std::move(*this));
			}

			// Iterators
			constexpr iterator begin() & noexcept {
				return iterator{ data() };
			}
			constexpr const_iterator begin() const& noexcept {
				return const_iterator{ data() };
			}
			constexpr const_iterator cbegin() const& noexcept {
				return begin();
			}
			constexpr rvalue_iterator begin() && noexcept {
				return rvalue_iterator{ std::move(*this).data() };
			}
			constexpr const_rvalue_iterator begin() const&& noexcept {
				return const_rvalue_iterator{ std::move(*this).data() };
			}
			constexpr const_rvalue_iterator cbegin() const&& noexcept {
				return std::move(*this).begin();
			}

			constexpr iterator end() & noexcept {
				return iterator{ data() + size() };
			}
			constexpr const_iterator end() const& noexcept {
				return const_iterator{ data() + size() };
			}
			constexpr const_iterator cend() const& noexcept {
				return end();
			}
			constexpr rvalue_iterator end() && noexcept {
				return rvalue_iterator{ std::move(*this).data() + size() };
			}
			constexpr const_rvalue_iterator end() const&& noexcept {
				return const_rvalue_iterator{ std::move(*this).data() + size() };
			}
			constexpr const_rvalue_iterator cend() const&& noexcept {
				return std::move(*this).end();
			}

			constexpr reverse_iterator rbegin() & noexcept {
				return std::make_reverse_iterator(end());
			}
			constexpr const_reverse_iterator rbegin() const& noexcept {
				return std::make_reverse_iterator(end());
			}
			constexpr const_reverse_iterator crbegin() const& noexcept {
				return rbegin();
			}
			constexpr rvalue_reverse_iterator rbegin() && noexcept {
				return std::make_reverse_iterator(std::move(*this).end());
			}
			constexpr const_rvalue_reverse_iterator rbegin() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).end());
			}
			constexpr const_rvalue_reverse_iterator crbegin() const&& noexcept {
				return std::move(*this).rbegin();
			}

			constexpr reverse_iterator rend() & noexcept {
				return std::make_reverse_iterator(begin());
			}
			constexpr const_reverse_iterator rend() const& noexcept {
				return std::make_reverse_iterator(begin());
			}
			constexpr const_reverse_iterator crend() const& noexcept {
				return rend();
			}
			constexpr rvalue_reverse_iterator rend() && noexcept {
				return std::make_reverse_iterator(std::move(*this).begin());
			}
			constexpr const_rvalue_reverse_iterator rend() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).begin());
			}
			constexpr const_rvalue_reverse_iterator crend() const&& noexcept {
				return std::move(*this).rend();
			}

			// Capacity
			constexpr bool empty() const noexcept {
				return size() == 0;
			}
			constexpr size_type size() const noexcept {
				return N1;
			}
			constexpr size_type max_size() const noexcept {
				return size();
			}
			constexpr size_type num_elements() const noexcept {
				return calculate_product<N1, Nr...>;
			}
			constexpr shape_type shape() const noexcept {
				return{ N1, Nr... };
			}

			// fill
			constexpr void fill(element const& x) &
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				// std::fill is constexpr since C++20; before that, we role our own
				for( std::size_t i = 0; i < num_elements(); ++i )
					data()[i] = x;
			}
			constexpr void fill(element const& x) &&
				noexcept(std::is_nothrow_assignable_v<element&&, element const&>)
			{
				// std::fill is constexpr since C++20; before that, we role our own
				for( std::size_t i = 0; i < num_elements(); ++i )
					std::move(*this).data()[i] = x;
			}

			// swap
			template <class ThatArray>
			constexpr void swap(ThatArray&& that) &
				noexcept(std::is_nothrow_swappable_v<element>)
			{
				jkl::swap(*this, std::forward<ThatArray>(that));
			}

			template <class ThatArray>
			constexpr void swap(ThatArray&& that) &&
				noexcept(std::is_nothrow_swappable_v<element>)
			{
				jkl::swap(std::move(*this), std::forward<ThatArray>(that));
			}
			
		private:
			std::array<element, calculate_product<N1, Nr...>> data_;
		};
	}

	using array_detail::operator==;
	using array_detail::operator!=;
	using array_detail::swap;
	using array_detail::get;
}