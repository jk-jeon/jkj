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
#include <complex>
#include <type_traits>
#include <utility>
#include "../portability.h"
#include "../tmp/assert_helper.h"
#include "../tmp/is_tuple.h"

namespace jkl {
	namespace math {
		struct default_storage_traits {
		//private:
			// 1. If the storage class is std::complex, treat real part as the x-component
			//    and imaginary part as the y-component; otherwise,
			// 2. If the storage class provides a member called x, y, z, or w,
			//   - if that member is a member function, call it; or,
			//   - if that member is a member object, use its value
			//    (even if that member object is of callable type.).
			// 3. Otherwise, if there is no member ftn/obj called x, y, z, or w,
			//   - if the storage type is tuple-like, call an appropriate
			//     tuple access function (jkl::tmp::get);
			//   - otherwise, try to call operator[] instead; if there is no operator[] neither,
			//     generate a compile error.

			template <class Storage>
			struct is_tuple_like : std::integral_constant<bool,
				jkl::tmp::is_tuple<tmp::remove_cvref_t<Storage>>::value> {};

			template <class Storage, class Index = std::size_t, class = void>
			struct has_array_operator : std::false_type {};

			template <class Storage, class Index>
			struct has_array_operator<Storage, Index,
				VOID_T<decltype(std::declval<Storage>()[std::declval<Index>()])>> : std::true_type
			{
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) call(Storage&& s, Index&& idx)
					noexcept(noexcept(std::forward<Storage>(s)[std::forward<Index>(idx)]))
				{
					return std::forward<Storage>(s)[std::forward<Index>(idx)];
				}

				template <std::size_t I>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) call(Storage&& s)
					noexcept(noexcept(std::forward<Storage>(s)[I]))
				{
					return std::forward<Storage>(s)[I];
				}
			};

#define JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR(x) struct x##_inspector {\
			template <class Storage, class = decltype(&Storage::x)>\
			static constexpr bool is_member_object(int) {\
				return std::is_member_object_pointer<decltype(&Storage::x)>::value;\
			}\
			template <class Storage>\
			static constexpr bool is_member_object(float) {\
				return false;\
			}\
			template <class Storage, class = decltype(std::declval<Storage>().x())>\
			static constexpr bool is_callable_member(int) {\
				return true;\
			}\
			template <class Storage>\
			static constexpr bool is_callable_member(float) {\
				return false;\
			}\
			template <class Storage>\
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto&& get(Storage&& s) noexcept {\
				return std::forward<Storage>(s).x;\
			}\
			template <class Storage>\
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) call(Storage&& s)\
				noexcept(noexcept(std::forward<Storage>(s).x()))\
			{\
				return std::forward<Storage>(s).x();\
			}\
			template <class Storage>\
			static constexpr decltype(auto) inspect_member_type(Storage&& s) noexcept {\
				return s.x;\
			}\
		}

			JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR(x);
			JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR(y);
			JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR(z);
			JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR(w);

		#undef JKL_DEFAULT_STORAGE_TRAITS_DEFINE_INSPECTOR
			
			struct choose_between_tuple_and_array_tag {
				template <class Storage>
				static constexpr bool is_member_object(int) {
					return false;
				}
				template <class Storage>
				static constexpr bool is_callable_member(int) {
					return false;
				}
			};

			template <std::size_t I>
			struct choose_inspector {
				using type = choose_between_tuple_and_array_tag;
			};
			template <>
			struct choose_inspector<0> {
				using type = x_inspector;
			};
			template <>
			struct choose_inspector<1> {
				using type = y_inspector;
			};
			template <>
			struct choose_inspector<2> {
				using type = z_inspector;
			};
			template <>
			struct choose_inspector<3> {
				using type = w_inspector;
			};
			template <std::size_t I>
			using choose_inspector_t = typename choose_inspector<I>::type;

			enum method_category { mem_obj, mem_ftn, tuple_like, array_op, no_way };

			template <class Storage, std::size_t I>
			static constexpr method_category choose_appropriate() {
				return choose_inspector_t<I>::template is_member_object<
					tmp::remove_cvref_t<Storage>>(0) ? method_category::mem_obj :
					choose_inspector_t<I>::template is_callable_member<
					tmp::remove_cvref_t<Storage>>(0) ? method_category::mem_ftn :
					is_tuple_like<Storage>::value ? method_category::tuple_like :
					has_array_operator<Storage>::value ? method_category::array_op :
					method_category::no_way;
			}

			template <class Storage, std::size_t I, method_category mc = choose_appropriate<Storage, I>()>
			struct call_appropriate {};
			
			template <class Storage, std::size_t I>
			struct call_appropriate<Storage, I, method_category::mem_obj> {
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto&& get(Storage&& s) noexcept {
					return choose_inspector_t<I>::get(std::forward<Storage>(s));
				}

				using type = decltype(choose_inspector_t<I>::inspect_member_type(
					std::declval<Storage&>()));
			};

			template <class Storage, std::size_t I>
			struct call_appropriate<Storage, I, method_category::mem_ftn> {
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
					noexcept(noexcept(choose_inspector_t<I>::call(std::forward<Storage>(s))))
				{
					return choose_inspector_t<I>::call(std::forward<Storage>(s));
				}

				using type = std::remove_reference_t<decltype(choose_inspector_t<I>::call(
					std::declval<Storage&>()))>;
			};

			template <class Storage, std::size_t I>
			struct call_appropriate<Storage, I, method_category::tuple_like> {
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
					noexcept(noexcept(jkl::tmp::get<I>(std::forward<Storage>(s))))
				{
					return jkl::tmp::get<I>(std::forward<Storage>(s));
				}

				// This type should be "needed" only when Storage is not a reference type,
				// but regardless of that the class template call_appropriate should be
				// specialized for reference types also, so this std::remove_reference_t
				// is necessary to suppress compile errors.
				using type = std::tuple_element_t<I, std::remove_reference_t<Storage>>;
			};

			template <class Storage, std::size_t I>
			struct call_appropriate<Storage, I, method_category::array_op> {
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
					noexcept(noexcept(has_array_operator<Storage>::template call<I>(std::forward<Storage>(s))))
				{
					return has_array_operator<Storage>::template call<I>(std::forward<Storage>(s));
				}
				
				using type = std::remove_reference_t<decltype(has_array_operator<Storage&>::template call<I>(
					std::declval<Storage&>()))>;
			};

		public:
			template <std::size_t I, class Storage>
			struct tuple_element;

			//// Special treatments for std::complex; named access

			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& x(std::complex<T>& s) noexcept {
				return reinterpret_cast<T(&)[2]>(s)[0];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& x(std::complex<T> const& s) noexcept {
				return reinterpret_cast<T const(&)[2]>(s)[0];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& x(std::complex<T>&& s) noexcept {
				return std::move(reinterpret_cast<T(&)[2]>(s)[0]);
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& x(std::complex<T> const&& s) noexcept {
				return std::move(reinterpret_cast<T const(&)[2]>(s)[0]);
			}

			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& y(std::complex<T>& s) noexcept {
				return reinterpret_cast<T(&)[2]>(s)[1];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& y(std::complex<T> const& s) noexcept {
				return reinterpret_cast<T const(&)[2]>(s)[1];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& y(std::complex<T>&& s) noexcept {
				return std::move(reinterpret_cast<T(&)[2]>(s)[1]);
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&& y(std::complex<T> const&& s) noexcept {
				return std::move(reinterpret_cast<T const(&)[2]>(s)[1]);
			}

			//// Special treatments for std::complex; tuple-like access

			template <std::size_t I, class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR
				std::enable_if_t<I == 0 || I == 1, T&> get(std::complex<T>& s) noexcept
			{
				return reinterpret_cast<T(&)[2]>(s)[I];
			}
			template <std::size_t I, class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR
				std::enable_if_t<I == 0 || I == 1, T const&> get(std::complex<T> const& s) noexcept
			{
				return reinterpret_cast<T const(&)[2]>(s)[I];
			}
			template <std::size_t I, class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR
				std::enable_if_t<I == 0 || I == 1, T&&> get(std::complex<T>&& s) noexcept
			{
				return std::move(reinterpret_cast<T(&)[2]>(s)[I]);
			}
			template <std::size_t I, class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR
				std::enable_if_t<I == 0 || I == 1, T const&&> get(std::complex<T> const&& s) noexcept
			{
				return std::move(reinterpret_cast<T const(&)[2]>(s)[I]);
			}

			//// Special treatments for std::complex; array operator access

			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&&
				array_operator(std::complex<T>& s, std::size_t idx) noexcept
			{
				return reinterpret_cast<T(&)[2]>(s)[idx];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&&
				array_operator(std::complex<T> const& s, std::size_t idx) noexcept
			{
				return reinterpret_cast<T const(&)[2]>(s)[idx];
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&&
				array_operator(std::complex<T>&& s, std::size_t idx) noexcept
			{
				return std::move(reinterpret_cast<T(&)[2]>(s)[idx]);
			}
			template <class T>
			FORCEINLINE JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR auto&&
				array_operator(std::complex<T> const&& s, std::size_t idx) noexcept
			{
				return std::move(reinterpret_cast<T const(&&)[2]>(s)[idx]);
			}

			//// Special treatments for std::complex; tuple element type

			template <class T>
			struct tuple_element<0, std::complex<T>> {
				using type = T;
			};
			template <class T>
			struct tuple_element<1, std::complex<T>> {
				using type = T;
			};


			//// General case; named access

			template <class Storage>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto x(Storage&& s)
				noexcept(noexcept(call_appropriate<Storage, 0>::get(std::forward<Storage>(s))))
				-> decltype(call_appropriate<Storage, 0>::get(std::forward<Storage>(s)))
			{
				return call_appropriate<Storage, 0>::get(std::forward<Storage>(s));
			}
			template <class Storage>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto y(Storage&& s)
				noexcept(noexcept(call_appropriate<Storage, 1>::get(std::forward<Storage>(s))))
				-> decltype(call_appropriate<Storage, 1>::get(std::forward<Storage>(s)))
			{
				return call_appropriate<Storage, 1>::get(std::forward<Storage>(s));
			}
			template <class Storage>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto z(Storage&& s)
				noexcept(noexcept(call_appropriate<Storage, 2>::get(std::forward<Storage>(s))))
				-> decltype(call_appropriate<Storage, 2>::get(std::forward<Storage>(s)))
			{
				return call_appropriate<Storage, 2>::get(std::forward<Storage>(s));
			}
			template <class Storage>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto w(Storage&& s)
				noexcept(noexcept(call_appropriate<Storage, 3>::get(std::forward<Storage>(s))))
				-> decltype(call_appropriate<Storage, 3>::get(std::forward<Storage>(s)))
			{
				return call_appropriate<Storage, 3>::get(std::forward<Storage>(s));
			}

			//// General case; tuple-like access

			template <std::size_t I, class Storage>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto get(Storage&& s)
				noexcept(noexcept(call_appropriate<Storage, I>::get(std::forward<Storage>(s))))
				-> decltype(call_appropriate<Storage, I>::get(std::forward<Storage>(s)))
			{
				return call_appropriate<Storage, I>::get(std::forward<Storage>(s));
			}

			//// General case; array operator access

			template <class Storage, class Index, class = std::enable_if_t<has_array_operator<Storage, Index>::value>>
			FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) array_operator(Storage&& s, Index&& idx)
				noexcept(noexcept(has_array_operator<Storage, Index>::call(std::forward<Storage>(s), std::forward<Index>(idx))))
			{
				return has_array_operator<Storage, Index>::call(std::forward<Storage>(s), std::forward<Index>(idx));
			}

			//// General case; tuple element type

			template <std::size_t I, class Storage>
			struct tuple_element {
				using type = typename call_appropriate<Storage, I>::type;
			};

			//// To support reference storage type
			//// NOTE: reference storage is only partially supported
			
			template <std::size_t I, class Storage>
			struct tuple_element<I, Storage&> {
				using type = typename tuple_element<I, Storage>::type&;
			};
			template <std::size_t I, class Storage>
			struct tuple_element<I, Storage&&> {
				using type = typename tuple_element<I, Storage>::type&&;
			};


			// Define the type that will actually be used as a member object of TargetType.
			// Different traits may use different way of defining storage_type.
			// The only role of this type is to provide indirection of constructor parameters.
			template <class Storage, class TargetType>
			using storage_wrapper = Storage;

			// How to get the actual storage from storage_wrapper
			template <class StorageWrapper>
			JKL_GPU_EXECUTABLE static constexpr auto&& get_storage(StorageWrapper&& s) noexcept {
				return std::forward<StorageWrapper>(s);
			}
		};


		// Since storage traits is a customization point, another level of indirection is necessary.
		// Implementations of Rn_elmt's will use this class instead.
		namespace detail {
			// I want to put this enum inside storage_traits_inspector, but VC++ 15.7.5 refuse it...
			enum class storage_access_method { named, tuple_like, array_operator, no_way };

			template <class StorageTraits>
			struct storage_traits_inspector {
				// void_t_workaround_XX is a workaround for a bug in VC++ 15.7.5 related to std::void_t

				struct void_t_workaround_x {};
				template <class Storage, class = void>
				struct has_x : std::false_type {};
				template <class Storage>
				struct has_x<Storage, VOID_T<decltype(StorageTraits::x(std::declval<Storage>())), void_t_workaround_x>>
					: std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::x(std::declval<Storage>()));
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s) noexcept(is_noexcept) {
						return StorageTraits::x(std::forward<Storage>(s));
					}
				};

				struct void_t_workaround_y {};
				template <class Storage, class = void>
				struct has_y : std::false_type {};
				template <class Storage>
				struct has_y<Storage, VOID_T<decltype(StorageTraits::y(std::declval<Storage>())), void_t_workaround_y>>
					: std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::y(std::declval<Storage>()));
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s) noexcept(is_noexcept) {
						return StorageTraits::y(std::forward<Storage>(s));
					}
				};

				struct void_t_workaround_z {};
				template <class Storage, class = void>
				struct has_z : std::false_type {};
				template <class Storage>
				struct has_z<Storage, VOID_T<decltype(StorageTraits::z(std::declval<Storage>())), void_t_workaround_z>>
					: std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::z(std::declval<Storage>()));
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s) noexcept(is_noexcept) {
						return StorageTraits::z(std::forward<Storage>(s));
					}
				};

				struct void_t_workaround_w {};
				template <class Storage, class = void>
				struct has_w : std::false_type {};
				template <class Storage>
				struct has_w<Storage, VOID_T<decltype(StorageTraits::w(std::declval<Storage>())), void_t_workaround_w>>
					: std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::w(std::declval<Storage>()));
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s) noexcept(is_noexcept) {
						return StorageTraits::w(std::forward<Storage>(s));
					}
				};

				struct void_t_workaround_get {};
				template <std::size_t I, class Storage, class = void>
				struct has_get : std::false_type {};
				template <std::size_t I, class Storage>
				struct has_get<I, Storage, VOID_T<decltype(StorageTraits::template get<I>(std::declval<Storage>())), void_t_workaround_get>>
					: std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::template get<I>(std::declval<Storage>()));
				};

				struct void_t_workaround_array_operator {};
				template <class Storage, class Index, class = void>
				struct has_array_operator : std::false_type {};
				template <class Storage, class Index>
				struct has_array_operator<Storage, Index,
					VOID_T<decltype(StorageTraits::array_operator(std::declval<Storage>(), std::declval<Index>())),
					void_t_workaround_array_operator>> : std::true_type
				{
					static constexpr bool is_noexcept = noexcept(StorageTraits::array_operator(
						std::declval<Storage>(), std::declval<Index>()));
				};

				template <std::size_t I, class Storage>
				struct choose_inspector {
					using type = std::false_type;
				};
				template <class Storage>
				struct choose_inspector<0, Storage> {
					using type = has_x<Storage>;
				};
				template <class Storage>
				struct choose_inspector<1, Storage> {
					using type = has_y<Storage>;
				};
				template <class Storage>
				struct choose_inspector<2, Storage> {
					using type = has_z<Storage>;
				};
				template <class Storage>
				struct choose_inspector<3, Storage> {
					using type = has_w<Storage>;
				};
				template <std::size_t I, class Storage>
				using choose_inspector_t = typename choose_inspector<I, Storage>::type;

				template <std::size_t I, class Storage>
				static constexpr storage_access_method choose_appropriate() {
					return choose_inspector_t<I, Storage>::value ? storage_access_method::named :
						has_get<I, Storage>::value ? storage_access_method::tuple_like :
						has_array_operator<Storage, std::size_t>::value ? storage_access_method::array_operator :
						storage_access_method::no_way;
				}

				template <std::size_t I, class Storage, storage_access_method mc = choose_appropriate<I, Storage>()>
				struct call_appropriate {
					static_assert(jkl::tmp::assert_helper<Storage>::value,
						"jkl::math: can't find any way to access to the storage");
				};

				template <std::size_t I, class Storage>
				struct call_appropriate<I, Storage, storage_access_method::named> {
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
						noexcept(choose_inspector_t<I, Storage>::is_noexcept)
					{
						return choose_inspector_t<I, Storage>::get(std::forward<Storage>(s));
					}
				};

				template <std::size_t I, class Storage>
				struct call_appropriate<I, Storage, storage_access_method::tuple_like> {
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
						noexcept(has_get<I, Storage>::is_noexcept)
					{
						return StorageTraits::template get<I>(std::forward<Storage>(s));
					}
				};

				template <std::size_t I, class Storage>
				struct call_appropriate<I, Storage, storage_access_method::array_operator> {
					FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
						noexcept(has_array_operator<Storage, std::size_t>::is_noexcept)
					{
						return StorageTraits::array_operator(std::forward<Storage>(s), I);
					}
				};

				// Tuple-like access
				template <std::size_t I, class Storage>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Storage&& s)
					noexcept(noexcept(call_appropriate<I, Storage>::get(std::forward<Storage>(s))))
				{
					return call_appropriate<I, Storage>::get(std::forward<Storage>(s));
				}

				// Array operator
				template <class Storage, class Index, class = std::enable_if_t<has_array_operator<Storage, Index>::value>>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr decltype(auto) array_operator(Storage&& s, Index&& idx)
					noexcept(has_array_operator<Storage, Index>::is_noexcept)
				{
					return StorageTraits::array_operator(std::forward<Storage>(s), std::forward<Index>(idx));
				}

				// Inspect if tuple-like access should be successful
				template <std::size_t I, class Storage>
				struct can_get : std::integral_constant<bool,
					choose_appropriate<I, Storage>() != storage_access_method::no_way> {};

				// Inspect if array operator access should be successful
				template <class Storage, class Index>
				struct can_index : std::integral_constant<bool,
					has_array_operator<Storage, Index>::value> {};
			};
		}

		

		// Compose two storage traits classes to make another traits class.
		// It provides tuple-like accessor (get() function) and array_operator() by first looking at
		// the first traits class, and use the second traits class if first traits class cannot handle
		// the passed parameter. The strategy how this composite traits class inspect
		// a traits class is just same as the storage_inspector. For example, though a traits class
		// may not explicitly define get<0>(), it is treated as if it has one if it has array_operator()
		// instead. Note that this composite traits class does not try to "optimize"
		// when choosing between first and second traits classes. It just check if the first one is feasible
		// or not, and use the second one if that is not the case.
		// This class template is not intended to be specialized by users, but specialization is
		// technically not banned.
		// tuple_element and storage_type are also brought from the first traits, and
		// if that fails then brought them from the second traits.

		template <class FirstTraits, class SecondTraits>
		struct compose_storage_traits {
			struct type {
			private:
				template <std::size_t I, class Storage, class StorageTraits>
				using can_get = typename detail::storage_traits_inspector<StorageTraits>::template can_get<I, Storage>;

				template <std::size_t I, class Storage, class StorageTraits, bool = can_get<I, Storage, StorageTraits>::value>
				struct get_return_type {
					using type = void;
				};

				template <std::size_t I, class Storage, class StorageTraits>
				struct get_return_type<I, Storage, StorageTraits, true> {
					using type = decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::declval<Storage>()));
				};

				template <std::size_t I, class Storage>
				using first_get_return_type = std::enable_if_t<
					can_get<I, Storage, FirstTraits>::value,
					typename get_return_type<I, Storage, FirstTraits>::type>;

				template <std::size_t I, class Storage>
				using second_get_return_type = std::enable_if_t<
					!can_get<I, Storage, FirstTraits>::value && can_get<I, Storage, SecondTraits>::value,
					typename get_return_type<I, Storage, SecondTraits>::type>;
								

				template <class Storage, class Index, class StorageTraits>
				using can_index = typename detail::storage_traits_inspector<StorageTraits>::template can_index<Storage, Index>;

				template <class Storage, class Index, class StorageTraits, bool = can_index<Storage, Index, StorageTraits>::value>
				struct array_operator_return_type {
					using type = void;
				};

				template <class Storage, class Index, class StorageTraits>
				struct array_operator_return_type<Storage, Index, StorageTraits, true> {
					using type = decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(
						std::declval<Storage>(), std::declval<Index>()));
				};

				template <class Storage, class Index>
				using first_array_operator_return_type = std::enable_if_t<
					can_index<Storage, Index, FirstTraits>::value,
					typename array_operator_return_type<Storage, Index, FirstTraits>::type>;

				template <class Storage, class Index>
				using second_array_operator_return_type = std::enable_if_t<
					!can_index<Storage, Index, FirstTraits>::value && can_index<Storage, Index, SecondTraits>::value,
					typename array_operator_return_type<Storage, Index, SecondTraits>::type>;


				template <std::size_t I, class Storage, class = void>
				struct bring_tuple_element {
					using type = typename SecondTraits::template tuple_element<I, Storage>;
				};

				struct void_t_workaround_has_tuple_element {};	// Workaround for std::void_t bug
				template <std::size_t I, class Storage>
				struct bring_tuple_element<I, Storage,
					VOID_T<typename FirstTraits::template tuple_element<I, Storage>, void_t_workaround_has_tuple_element>>
				{
					using type = typename FirstTraits::template tuple_element<I, Storage>;
				};


				template <class StorageParameter, class TargetType, class = void>
				struct bring_storage_type {
					using type = typename SecondTraits::template storage_type<StorageParameter, TargetType>;
				};

				struct void_t_workaround_has_storage_type {};	// Workaround for std::void_t bug
				template <class StorageParameter, class TargetType>
				struct bring_storage_type<StorageParameter, TargetType,
					VOID_T<typename FirstTraits::template storage_type<StorageParameter, TargetType>, void_t_workaround_has_storage_type>>
				{
					using type = typename FirstTraits::template storage_type<StorageParameter, TargetType>;
				};

			public:
				// Tuple-like access
				template <std::size_t I, class Storage>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr first_get_return_type<I, Storage> get(Storage&& s)
					noexcept(noexcept(detail::storage_traits_inspector<FirstTraits>::template get<I>(std::forward<Storage>(s))))
				{
					return detail::storage_traits_inspector<FirstTraits>::template get<I>(std::forward<Storage>(s));
				}
				template <std::size_t I, class Storage, class = void>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr second_get_return_type<I, Storage> get(Storage&& s)
					noexcept(noexcept(detail::storage_traits_inspector<SecondTraits>::template get<I>(std::forward<Storage>(s))))
				{
					return detail::storage_traits_inspector<SecondTraits>::template get<I>(std::forward<Storage>(s));
				}

				// Array operator
				template <class Storage, class Index>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr first_array_operator_return_type<Storage, Index>
					array_operator(Storage&& s, Index&& idx)
					noexcept(noexcept(detail::storage_traits_inspector<FirstTraits>::array_operator(
						std::forward<Storage>(s), std::forward<Index>(idx))))
				{
					return detail::storage_traits_inspector<FirstTraits>::array_operator(
						std::forward<Storage>(s), std::forward<Index>(idx));
				}
				template <class Storage, class Index, class = void>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr second_array_operator_return_type<Storage, Index>
					array_operator(Storage&& s, Index&& idx)
					noexcept(noexcept(detail::storage_traits_inspector<SecondTraits>::array_operator(
						std::forward<Storage>(s), std::forward<Index>(idx))))
				{
					return detail::storage_traits_inspector<SecondTraits>::array_operator(
						std::forward<Storage>(s), std::forward<Index>(idx));
				}

				// Type definitions
				template <std::size_t I, class Storage>
				using tuple_element = typename bring_tuple_element<I, Storage>::type;
				template <class StorageParameter, class TargetType>
				using storage_type = typename bring_storage_type<StorageParameter, TargetType>::type;
			};
		};

		template <class FirstTraits, class SecondTraits>
		using composite_storage_traits = typename compose_storage_traits<FirstTraits, SecondTraits>::type;

		template <>
		struct compose_storage_traits<default_storage_traits, default_storage_traits> {
			using type = default_storage_traits;
		};



		// A tag type for make functions to request type deduction
		// The resulting component type is the std::common_type_t of all given component types.
		// Be careful that std::common_type_t might not give the correct common type.
		struct deduce_type_tag {};

		namespace detail {
			template <class ComponentType, class... Args>
			struct deduce_common_type {
				using type = ComponentType;
			};

			template <class... Args>
			struct deduce_common_type<deduce_type_tag, Args...> {
				using type = std::common_type_t<Args...>;
			};

			template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
			struct deduce_1d_component_type {
				using type = ComponentType;
			};

			template <std::size_t N, class Storage, class StorageTraits>
			struct deduce_1d_component_type<N, deduce_type_tag, Storage, StorageTraits> {
			private:
				template <std::size_t I>
				using component_type = tmp::remove_cvref_t<decltype(
					detail::storage_traits_inspector<StorageTraits>::template get<I>(
						std::declval<Storage>()))>;

				template <std::size_t... I>
				static constexpr auto deduce_common_type(std::index_sequence<I...>) {
					return std::declval<std::common_type_t<component_type<I>...>>();
				}

			public:
				using type = decltype(deduce_common_type(std::make_index_sequence<N>{}));
			};



			template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
			using deduced_1d_component_type =
				typename deduce_1d_component_type<N, ComponentType, Storage, StorageTraits>::type;

			template <class ComponentType, class... Args>
			using deduced_component_type_from_args =
				typename deduce_common_type<ComponentType, Args...>::type;
		}


		// Specialize this template to make binary operations between two classes with
		// different storage traits. The type member of this template class will be the
		// resulting object's storage traits.

		template <class StorageTraits1, class StorageTraits2>
		struct select_storage_traits {};

		template <class StorageTraits1, class StorageTraits2>
		using select_storage_traits_t = typename select_storage_traits<StorageTraits1, StorageTraits2>::type;

		template <class StorageTraits>
		struct select_storage_traits<StorageTraits, StorageTraits> {
			using type = StorageTraits;
		};

		// Prevent the specialization of default_storage_traits to be overridden
		template <>
		struct select_storage_traits<default_storage_traits, default_storage_traits> {
			using type = default_storage_traits;
		};
	}
}
