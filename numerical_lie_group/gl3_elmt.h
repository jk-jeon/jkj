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
#include <cassert>
#include "Rn_elmt.h"

namespace jkl {
	namespace math {
		// 3x3 matrix
		// Since 2^9 = 512 is too large, we don't try to delegate to constructor_provider.
		template <class ComponentType, class Storage, class StorageTraits>
		class gl3_elmt
		{
		public:
			using component_type = ComponentType;
			static constexpr std::size_t components = 9;
			using storage_type = Storage;
			using storage_traits = StorageTraits;

		private:
			using storage_wrapper = typename StorageTraits::template storage_wrapper<Storage, gl3_elmt>;
			storage_wrapper r_;

		public:
			// Direct access to the internal storage
#if defined(_MSC_VER) && _MSC_VER <= 1900
			// MSVC2015 has a bug that it is not possible to move a built-in array
			// So for MSVC2015, we don't define rvalue ref-qualified storage()'s...
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type& storage() noexcept {
				return storage_traits::get_storage(r_);
			}
			JKL_GPU_EXECUTABLE constexpr storage_type const& storage() const noexcept {
				return storage_traits::get_storage(r_);
			}
#else
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type& storage() & noexcept {
				return storage_traits::get_storage(r_);
			}
			JKL_GPU_EXECUTABLE constexpr storage_type const& storage() const& noexcept {
				return storage_traits::get_storage(r_);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type&& storage() && noexcept {
				return storage_traits::get_storage(std::move(*this).r_);
			}
			JKL_GPU_EXECUTABLE constexpr storage_type const&& storage() const&& noexcept {
				return storage_traits::get_storage(std::move(*this).r_);
			}
#endif

			// Row access requirements
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkl::math: gl3_elmt requires access to the first row from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkl::math: gl3_elmt requires access to the second row from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, Storage const&>::value,
				"jkl::math: gl3_elmt requires access to the third row from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// Tuple-style row accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}

			// Component access requirements
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the first row of gl3_elmt requires access to the first column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the first row of gl3_elmt requires access to the second column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the first row of gl3_elmt requires access to the third column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the second row of gl3_elmt requires access to the first column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the second row of gl3_elmt requires access to the second column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the second row of gl3_elmt requires access to the third column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the third row of gl3_elmt requires access to the first column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the third row of gl3_elmt requires access to the second column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(
					std::declval<Storage const&>()))>::value,
				"jkl::math: the third row of gl3_elmt requires access to the third column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// Tuple-style component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<J>(get<I>())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<J>(get<I>());
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<J>(get<I>())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<J>(get<I>());
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<J>(std::move(*this).get<I>())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<J>(std::move(*this).get<I>());
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<J>(std::move(*this).get<I>())))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<J>(std::move(*this).get<I>());
			}

			// Array-style row accessors
		private:
			template <class StorageType, class dummy = void, bool can_index =
				detail::storage_traits_inspector<StorageTraits>::template can_index<StorageType, std::size_t>::value>
			struct has_array_operator : std::false_type {
				using type = void;
			};

			template <class StorageType, class dummy>
			struct has_array_operator<StorageType, dummy, true> {
				using type = decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(
					std::declval<StorageType>(), std::size_t(0)));

				// Can access to a component of a row?
				template <std::size_t I>
				struct has_component : std::integral_constant<bool,
					detail::storage_traits_inspector<StorageTraits>::template can_get<I, type>::value> {};

				template <std::size_t I, bool has_component_ = has_component<I>::value>
				struct is_component_convertible : std::false_type {};

				template <std::size_t I>
				struct is_component_convertible<I, true> {
					using component_type = decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(
						std::declval<type>()));
					static constexpr bool value = std::is_convertible<component_type, ComponentType const&>::value;
				};

				static constexpr bool value =
					is_component_convertible<0>::value &&
					is_component_convertible<1>::value &&
					is_component_convertible<2>::value;
			};

			template <class StorageType, class dummy>
			using array_operator_return_t = std::enable_if_t<has_array_operator<StorageType>::value,
				typename has_array_operator<StorageType, dummy>::type>;

		public:
			template <class dummy = void>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage&, dummy>
				operator[](std::size_t idx) & noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx);
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr array_operator_return_t<Storage const&, dummy>
				operator[](std::size_t idx) const& noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx);
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage&&, dummy>
				operator[](std::size_t idx) && noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr array_operator_return_t<Storage const&&, dummy>
				operator[](std::size_t idx) const&& noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}


			// Default constructor; components might be filled with garbages
			gl3_elmt() = default;

			// Construct the storage directly
			template <class... Args>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(direct_construction, Args&&... args)
				noexcept(std::is_nothrow_constructible<storage_wrapper, Args...>::value)
				: r_(std::forward<Args>(args)...) {}

			// Component-wise constructor
			template <class Arg00, class Arg01, class Arg02,
				class Arg10, class Arg11, class Arg12,
				class Arg20, class Arg21, class Arg22>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				Arg00&& arg00, Arg01&& arg01, Arg02&& arg02,
				Arg10&& arg10, Arg11&& arg11, Arg12&& arg12,
				Arg20&& arg20, Arg21&& arg21, Arg22&& arg22)
				noexcept(noexcept(storage_wrapper{
					{ std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02) },
					{ std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12) },
					{ std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22) } }))
				: r_{ { std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02) },
			{ std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12) },
			{ std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22) } } {}
			
			// Call-by-value component-wise constructor
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				ComponentType arg00, ComponentType arg01, ComponentType arg02,
				ComponentType arg10, ComponentType arg11, ComponentType arg12,
				ComponentType arg20, ComponentType arg21, ComponentType arg22)
				noexcept(noexcept(storage_wrapper{
					{ std::move(arg00), std::move(arg01), std::move(arg02) },
					{ std::move(arg10), std::move(arg11), std::move(arg12) },
					{ std::move(arg20), std::move(arg21), std::move(arg22) } })) :
					r_{ { std::move(arg00), std::move(arg01), std::move(arg02) },
			{ std::move(arg10), std::move(arg11), std::move(arg12) },
			{ std::move(arg20), std::move(arg21), std::move(arg22) } } {}

			// Row-wise constructor
			// MSVC 15.7 has a bug that it evaluates the noexcept specifier "too early";
			// that is, it evaluates noexcept specifier before the overload resolution
			// comes into play. This occassioanly causes this overload to erroneously
			// instantiate gl3_elmt with a wrong storage type, causing a hard-error.
			// The std::enable_if_t switch is a workaround for that.
			template <class Row0, class Row1, class Row2, class = std::enable_if_t<
				detail::row_ref_tuple_traits<StorageTraits, Row0, Row1, Row2>::value>>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(Row0&& r0, Row1&& r1, Row2&& r2)
				noexcept(noexcept(gl3_elmt{ gl3_elmt<ComponentType,
					detail::tuple<Row0&&, Row1&&, Row2&&>,
					detail::row_ref_tuple_traits<StorageTraits, Row0, Row1, Row2>>{ direct_construction{},
					std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row0>(r2) } })) :
				gl3_elmt{ gl3_elmt<ComponentType, detail::tuple<Row0&&, Row1&&, Row2&&>,
				detail::row_ref_tuple_traits<StorageTraits, Row0, Row1, Row2>>{ direct_construction{},
				std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row0>(r2) } } {}

			// Call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				R3_elmt<ComponentType> r0, R3_elmt<ComponentType> r1, R3_elmt<ComponentType> r2)
				noexcept(noexcept(storage_wrapper{
					{ std::move(r0).x(), std::move(r0).y(), std::move(r0).z() },
					{ std::move(r1).x(), std::move(r1).y(), std::move(r1).z() },
					{ std::move(r2).x(), std::move(r2).y(), std::move(r2).z() } })) :
					r_{ { std::move(r0).x(), std::move(r0).y(), std::move(r0).z() },
			{ std::move(r1).x(), std::move(r1).y(), std::move(r1).z() },
			{ std::move(r2).x(), std::move(r2).y(), std::move(r2).z() } } {}
			
			// Convert from gl3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl3_elmt,
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(storage_wrapper{
					{ that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>() },
					{ that.template get<1, 0>(), that.template get<1, 1>(), that.template get<1, 2>() },
					{ that.template get<2, 0>(), that.template get<2, 1>(), that.template get<2, 2>() } })) :
				r_{ { that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>() },
			{ that.template get<1, 0>(), that.template get<1, 1>(), that.template get<1, 2>() },
			{ that.template get<2, 0>(), that.template get<2, 1>(), that.template get<2, 2>() } } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl3_elmt,
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(storage_wrapper{
					{ std::move(that).template get<0, 0>(),
					std::move(that).template get<0, 1>(),
					std::move(that).template get<0, 2>() },
					{ std::move(that).template get<1, 0>(),
					std::move(that).template get<1, 1>(),
					std::move(that).template get<1, 2>() },
					{ std::move(that).template get<2, 0>(),
					std::move(that).template get<2, 1>(),
					std::move(that).template get<2, 2>() } })) : r_{
						{ std::move(that).template get<0, 0>(),
						std::move(that).template get<0, 1>(),
						std::move(that).template get<0, 2>() },
			{ std::move(that).template get<1, 0>(),
			std::move(that).template get<1, 1>(),
			std::move(that).template get<1, 2>() },
			{ std::move(that).template get<2, 0>(),
			std::move(that).template get<2, 1>(),
			std::move(that).template get<2, 2>() } } {}

			// Convert from sym3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(storage_wrapper{
					{ that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>() },
					{ that.template get<1, 0>(), that.template get<1, 1>(), that.template get<1, 2>() },
					{ that.template get<2, 0>(), that.template get<2, 1>(), that.template get<2, 2>() } })) :
					r_{ { that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>() },
			{ that.template get<1, 0>(), that.template get<1, 1>(), that.template get<1, 2>() },
			{ that.template get<2, 0>(), that.template get<2, 1>(), that.template get<2, 2>() } } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(storage_wrapper{
					{ std::move(that).template get<0, 0>(),
					that.template get<0, 1>(),
					that.template get<0, 2>() },
					{ that.template get<1, 0>(),
					std::move(that).template get<1, 1>(),
					that.template get<1, 2>() },
					{ that.template get<2, 0>(),
					that.template get<2, 1>(),
					std::move(that).template get<2, 2>() } })) : r_{
						{ std::move(that).template get<0, 0>(),
						that.template get<0, 1>(),
						that.template get<0, 2>() },
			{ that.template get<1, 0>(),
			std::move(that).template get<1, 1>(),
			that.template get<1, 2>() },
			{ that.template get<2, 0>(),
			that.template get<2, 1>(),
			std::move(that).template get<2, 2>() } } {}


			// Copy and move
			gl3_elmt(gl3_elmt const&) = default;
			gl3_elmt(gl3_elmt&&) = default;
			gl3_elmt& operator=(gl3_elmt const&) & = default;
			gl3_elmt& operator=(gl3_elmt&&) & = default;

			// Assignment from gl3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl3_elmt,
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(
					noexcept(get<0, 0>() = that.template get<0, 0>()) &&
					noexcept(get<0, 1>() = that.template get<0, 1>()) &&
					noexcept(get<0, 2>() = that.template get<0, 2>()) &&
					noexcept(get<1, 0>() = that.template get<1, 0>()) &&
					noexcept(get<1, 1>() = that.template get<1, 1>()) &&
					noexcept(get<1, 2>() = that.template get<1, 2>()) &&
					noexcept(get<2, 0>() = that.template get<2, 0>()) &&
					noexcept(get<2, 1>() = that.template get<2, 1>()) &&
					noexcept(get<2, 2>() = that.template get<2, 2>()))
			{
				get<0, 0>() = that.template get<0, 0>();
				get<0, 1>() = that.template get<0, 1>();
				get<0, 2>() = that.template get<0, 2>();
				get<1, 0>() = that.template get<1, 0>();
				get<1, 1>() = that.template get<1, 1>();
				get<1, 2>() = that.template get<1, 2>();
				get<2, 0>() = that.template get<2, 0>();
				get<2, 1>() = that.template get<2, 1>();
				get<2, 2>() = that.template get<2, 2>();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl3_elmt,
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(
					noexcept(get<0, 0>() = std::move(that).template get<0, 0>()) &&
					noexcept(get<0, 1>() = std::move(that).template get<0, 1>()) &&
					noexcept(get<0, 2>() = std::move(that).template get<0, 2>()) &&
					noexcept(get<1, 0>() = std::move(that).template get<1, 0>()) &&
					noexcept(get<1, 1>() = std::move(that).template get<1, 1>()) &&
					noexcept(get<1, 2>() = std::move(that).template get<1, 2>()) &&
					noexcept(get<2, 0>() = std::move(that).template get<2, 0>()) &&
					noexcept(get<2, 1>() = std::move(that).template get<2, 1>()) &&
					noexcept(get<2, 2>() = std::move(that).template get<2, 2>()))
			{
				get<0, 0>() = std::move(that).template get<0, 0>();
				get<0, 1>() = std::move(that).template get<0, 1>();
				get<0, 2>() = std::move(that).template get<0, 2>();
				get<1, 0>() = std::move(that).template get<1, 0>();
				get<1, 1>() = std::move(that).template get<1, 1>();
				get<1, 2>() = std::move(that).template get<1, 2>();
				get<2, 0>() = std::move(that).template get<2, 0>();
				get<2, 1>() = std::move(that).template get<2, 1>();
				get<2, 2>() = std::move(that).template get<2, 2>();
				return *this;
			}
			
			// Assignment from sym3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(
					noexcept(get<0, 0>() = that.template get<0, 0>()) &&
					noexcept(get<0, 1>() = that.template get<0, 1>()) &&
					noexcept(get<0, 2>() = that.template get<0, 2>()) &&
					noexcept(get<1, 0>() = that.template get<1, 0>()) &&
					noexcept(get<1, 1>() = that.template get<1, 1>()) &&
					noexcept(get<1, 2>() = that.template get<1, 2>()) &&
					noexcept(get<2, 0>() = that.template get<2, 0>()) &&
					noexcept(get<2, 1>() = that.template get<2, 1>()) &&
					noexcept(get<2, 2>() = that.template get<2, 2>()))
			{
				get<0, 0>() = that.template get<0, 0>();
				get<0, 1>() = that.template get<0, 1>();
				get<0, 2>() = that.template get<0, 2>();
				get<1, 0>() = that.template get<1, 0>();
				get<1, 1>() = that.template get<1, 1>();
				get<1, 2>() = that.template get<1, 2>();
				get<2, 0>() = that.template get<2, 0>();
				get<2, 1>() = that.template get<2, 1>();
				get<2, 2>() = that.template get<2, 2>();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(
					noexcept(get<0, 0>() = std::move(that).template get<0, 0>()) &&
					noexcept(get<0, 1>() = that.template get<0, 1>()) &&
					noexcept(get<0, 2>() = that.template get<0, 2>()) &&
					noexcept(get<1, 0>() = std::move(that).template get<1, 0>()) &&
					noexcept(get<1, 1>() = std::move(that).template get<1, 1>()) &&
					noexcept(get<1, 2>() = that.template get<1, 2>()) &&
					noexcept(get<2, 0>() = std::move(that).template get<2, 0>()) &&
					noexcept(get<2, 1>() = std::move(that).template get<2, 1>()) &&
					noexcept(get<2, 2>() = std::move(that).template get<2, 2>()))
			{
				get<0, 0>() = std::move(that).template get<0, 0>();
				get<0, 1>() = that.template get<0, 1>();
				get<0, 2>() = that.template get<0, 2>();
				get<1, 0>() = std::move(that).template get<1, 0>();
				get<1, 1>() = std::move(that).template get<1, 1>();
				get<1, 2>() = that.template get<1, 2>();
				get<2, 0>() = std::move(that).template get<2, 0>();
				get<2, 1>() = std::move(that).template get<2, 1>();
				get<2, 2>() = std::move(that).template get<2, 2>();
				return *this;
			}

		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr decltype(auto) det_impl(Matrix&& m)
				noexcept(noexcept(
					std::forward<Matrix>(m).template get<0, 0>() * (
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 2>() -
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 1>()) +
					std::forward<Matrix>(m).template get<0, 1>() * (
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 0>() -
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 2>()) +
					std::forward<Matrix>(m).template get<0, 2>() * (
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 1>() -
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 0>())))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (
					std::forward<Matrix>(m).template get<0, 0>() * (
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 2>() -
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 1>()) +
					std::forward<Matrix>(m).template get<0, 1>() * (
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 0>() -
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 2>()) +
					std::forward<Matrix>(m).template get<0, 2>() * (
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 1>() -
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 0>()));
			}

			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr decltype(auto) trace_impl(Matrix&& m)
				noexcept(noexcept(
					std::forward<Matrix>(m).template get<0, 0>() +
					std::forward<Matrix>(m).template get<1, 1>() +
					std::forward<Matrix>(m).template get<2, 2>()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (
					std::forward<Matrix>(m).template get<0, 0>() +
					std::forward<Matrix>(m).template get<1, 1>() +
					std::forward<Matrix>(m).template get<2, 2>());
			}

		public:
			JKL_GPU_EXECUTABLE constexpr decltype(auto) det() const&
				noexcept(noexcept(det_impl(*this)))
			{
				return det_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) det() &&
				noexcept(noexcept(det_impl(std::move(*this))))
			{
				return det_impl(std::move(*this));
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) trace() const&
				noexcept(noexcept(trace_impl(*this)))
			{
				return trace_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) trace() &&
				noexcept(noexcept(trace_impl(std::move(*this))))
			{
				return trace_impl(std::move(*this));
			}

			JKL_GPU_EXECUTABLE constexpr gl3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<gl3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<gl3_elmt>::value)
			{
				return std::move(*this);
			}

		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr gl3_elmt minus_impl(Matrix&& m)
				noexcept(noexcept(gl3_elmt{
					-std::forward<Matrix>(m).get<0, 0>(),
					-std::forward<Matrix>(m).get<0, 1>(),
					-std::forward<Matrix>(m).get<0, 2>(),
					-std::forward<Matrix>(m).get<1, 0>(),
					-std::forward<Matrix>(m).get<1, 1>(),
					-std::forward<Matrix>(m).get<1, 2>(),
					-std::forward<Matrix>(m).get<2, 0>(),
					-std::forward<Matrix>(m).get<2, 1>(),
					-std::forward<Matrix>(m).get<2, 2>() }))
			{
				return{
					-std::forward<Matrix>(m).get<0, 0>(),
					-std::forward<Matrix>(m).get<0, 1>(),
					-std::forward<Matrix>(m).get<0, 2>(),
					-std::forward<Matrix>(m).get<1, 0>(),
					-std::forward<Matrix>(m).get<1, 1>(),
					-std::forward<Matrix>(m).get<1, 2>(),
					-std::forward<Matrix>(m).get<2, 0>(),
					-std::forward<Matrix>(m).get<2, 1>(),
					-std::forward<Matrix>(m).get<2, 2>()
				};
			}

		public:
			JKL_GPU_EXECUTABLE constexpr gl3_elmt operator-() const&
				noexcept(noexcept(minus_impl(*this)))
			{
				return minus_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl3_elmt operator-() &&
				noexcept(noexcept(minus_impl(std::move(*this))))
			{
				return minus_impl(std::move(*this));
			}

		private:
			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& inplace_add_impl(OtherMatrix&& m) noexcept(
				noexcept(get<0, 0>() += std::forward<OtherMatrix>(m).template get<0, 0>()) &&
				noexcept(get<0, 1>() += std::forward<OtherMatrix>(m).template get<0, 1>()) &&
				noexcept(get<0, 2>() += std::forward<OtherMatrix>(m).template get<0, 2>()) &&
				noexcept(get<1, 0>() += std::forward<OtherMatrix>(m).template get<1, 0>()) &&
				noexcept(get<1, 1>() += std::forward<OtherMatrix>(m).template get<1, 1>()) &&
				noexcept(get<1, 2>() += std::forward<OtherMatrix>(m).template get<1, 2>()) &&
				noexcept(get<2, 0>() += std::forward<OtherMatrix>(m).template get<2, 0>()) &&
				noexcept(get<2, 1>() += std::forward<OtherMatrix>(m).template get<2, 1>()) &&
				noexcept(get<2, 2>() += std::forward<OtherMatrix>(m).template get<2, 2>()))
			{
				get<0, 0>() += std::forward<OtherMatrix>(m).template get<0, 0>();
				get<0, 1>() += std::forward<OtherMatrix>(m).template get<0, 1>();
				get<0, 2>() += std::forward<OtherMatrix>(m).template get<0, 2>();
				get<1, 0>() += std::forward<OtherMatrix>(m).template get<1, 0>();
				get<1, 1>() += std::forward<OtherMatrix>(m).template get<1, 1>();
				get<1, 2>() += std::forward<OtherMatrix>(m).template get<1, 2>();
				get<2, 0>() += std::forward<OtherMatrix>(m).template get<2, 0>();
				get<2, 1>() += std::forward<OtherMatrix>(m).template get<2, 1>();
				get<2, 2>() += std::forward<OtherMatrix>(m).template get<2, 2>();
				return *this;
			}

			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& inplace_sub_impl(OtherMatrix&& m) noexcept(
				noexcept(get<0, 0>() -= std::forward<OtherMatrix>(m).template get<0, 0>()) &&
				noexcept(get<0, 1>() -= std::forward<OtherMatrix>(m).template get<0, 1>()) &&
				noexcept(get<0, 2>() -= std::forward<OtherMatrix>(m).template get<0, 2>()) &&
				noexcept(get<1, 0>() -= std::forward<OtherMatrix>(m).template get<1, 0>()) &&
				noexcept(get<1, 1>() -= std::forward<OtherMatrix>(m).template get<1, 1>()) &&
				noexcept(get<1, 2>() -= std::forward<OtherMatrix>(m).template get<1, 2>()) &&
				noexcept(get<2, 0>() -= std::forward<OtherMatrix>(m).template get<2, 0>()) &&
				noexcept(get<2, 1>() -= std::forward<OtherMatrix>(m).template get<2, 1>()) &&
				noexcept(get<2, 2>() -= std::forward<OtherMatrix>(m).template get<2, 2>()))
			{
				get<0, 0>() -= std::forward<OtherMatrix>(m).template get<0, 0>();
				get<0, 1>() -= std::forward<OtherMatrix>(m).template get<0, 1>();
				get<0, 2>() -= std::forward<OtherMatrix>(m).template get<0, 2>();
				get<1, 0>() -= std::forward<OtherMatrix>(m).template get<1, 0>();
				get<1, 1>() -= std::forward<OtherMatrix>(m).template get<1, 1>();
				get<1, 2>() -= std::forward<OtherMatrix>(m).template get<1, 2>();
				get<2, 0>() -= std::forward<OtherMatrix>(m).template get<2, 0>();
				get<2, 1>() -= std::forward<OtherMatrix>(m).template get<2, 1>();
				get<2, 2>() -= std::forward<OtherMatrix>(m).template get<2, 2>();
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator+=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_add_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator+=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_add_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator-=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_sub_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator-=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_sub_impl(std::move(that));
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator*=(OtherComponentType const& k) noexcept(
				noexcept(get<0, 0>() *= k) &&
				noexcept(get<0, 1>() *= k) &&
				noexcept(get<0, 2>() *= k) &&
				noexcept(get<1, 0>() *= k) &&
				noexcept(get<1, 1>() *= k) &&
				noexcept(get<1, 2>() *= k) &&
				noexcept(get<2, 0>() *= k) &&
				noexcept(get<2, 1>() *= k) &&
				noexcept(get<2, 2>() *= k))
			{
				get<0, 0>() *= k;
				get<0, 1>() *= k;
				get<0, 2>() *= k;
				get<1, 0>() *= k;
				get<1, 1>() *= k;
				get<1, 2>() *= k;
				get<2, 0>() *= k;
				get<2, 1>() *= k;
				get<2, 2>() *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator/=(OtherComponentType const& k) noexcept(
				noexcept(get<0, 0>() /= k) &&
				noexcept(get<0, 1>() /= k) &&
				noexcept(get<0, 2>() /= k) &&
				noexcept(get<1, 0>() /= k) &&
				noexcept(get<1, 1>() /= k) &&
				noexcept(get<1, 2>() /= k) &&
				noexcept(get<2, 0>() /= k) &&
				noexcept(get<2, 1>() /= k) &&
				noexcept(get<2, 2>() /= k))
			{
				get<0, 0>() /= k;
				get<0, 1>() /= k;
				get<0, 2>() /= k;
				get<1, 0>() /= k;
				get<1, 1>() /= k;
				get<1, 2>() /= k;
				get<2, 0>() /= k;
				get<2, 1>() /= k;
				get<2, 2>() /= k;
				return *this;
			}

		private:
			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& inplace_mul_impl(OtherMatrix&& m)
				noexcept(noexcept(*this = *this * std::forward<OtherMatrix>(m)))
			{
				*this = *this * std::forward<OtherMatrix>(m);
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator*=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_mul_impl(that)))
			{
				return inplace_mul_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator*=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_mul_impl(std::move(that))))
			{
				return inplace_mul_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator/=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(*this * that.inv()))
			{
				return *this * that.inv();
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt& operator/=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(*this * std::move(that).inv()))
			{
				return *this * std::move(that).inv();
			}

		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr gl3_elmt transpose_impl(Matrix&& m)
				noexcept(noexcept(gl3_elmt{
				std::forward<Matrix>(m).template get<0, 0>(),
				std::forward<Matrix>(m).template get<0, 1>(),
				std::forward<Matrix>(m).template get<0, 2>(),
				std::forward<Matrix>(m).template get<1, 0>(),
				std::forward<Matrix>(m).template get<1, 1>(),
				std::forward<Matrix>(m).template get<1, 2>(),
				std::forward<Matrix>(m).template get<2, 0>(),
				std::forward<Matrix>(m).template get<2, 1>(),
				std::forward<Matrix>(m).template get<2, 2>() }))
			{
				return{
					std::forward<Matrix>(m).template get<0, 0>(),
					std::forward<Matrix>(m).template get<0, 1>(),
					std::forward<Matrix>(m).template get<0, 2>(),
					std::forward<Matrix>(m).template get<1, 0>(),
					std::forward<Matrix>(m).template get<1, 1>(),
					std::forward<Matrix>(m).template get<1, 2>(),
					std::forward<Matrix>(m).template get<2, 0>(),
					std::forward<Matrix>(m).template get<2, 1>(),
					std::forward<Matrix>(m).template get<2, 2>()
				};
			}

		public:
			JKL_GPU_EXECUTABLE constexpr gl3_elmt t() const&
				noexcept(noexcept(transpose_impl(*this)))
			{
				return transpose_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl3_elmt t() &&
				noexcept(noexcept(transpose_impl(std::move(*this))))
			{
				return transpose_impl(std::move(*this));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const
				noexcept(
					noexcept(get<0, 0>() == that.template get<0, 0>()) &&
					noexcept(get<0, 1>() == that.template get<0, 1>()) &&
					noexcept(get<0, 2>() == that.template get<0, 2>()) &&
					noexcept(get<1, 0>() == that.template get<1, 0>()) &&
					noexcept(get<1, 1>() == that.template get<1, 1>()) &&
					noexcept(get<1, 2>() == that.template get<1, 2>()) &&
					noexcept(get<2, 0>() == that.template get<2, 0>()) &&
					noexcept(get<2, 1>() == that.template get<2, 1>()) &&
					noexcept(get<2, 2>() == that.template get<2, 2>()))
			{
				return get<0, 0>() == that.template get<0, 0>()
					&& get<0, 1>() == that.template get<0, 1>()
					&& get<0, 2>() == that.template get<0, 2>()
					&& get<1, 0>() == that.template get<1, 0>()
					&& get<1, 1>() == that.template get<1, 1>()
					&& get<1, 2>() == that.template get<1, 2>()
					&& get<2, 0>() == that.template get<2, 0>()
					&& get<2, 1>() == that.template get<2, 1>()
					&& get<2, 2>() == that.template get<2, 2>();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const
				noexcept(noexcept((*this) == that))
			{
				return !((*this) == that);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const&
				noexcept(noexcept(jkl::math::is_invertible(det())))
			{
				return jkl::math::is_invertible(det());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool is_invertible() &&
				noexcept(noexcept(jkl::math::is_invertible(std::move(*this).det())))
			{
				return jkl::math::is_invertible(std::move(*this).det());
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept(
				noexcept(std::is_nothrow_copy_constructible<decltype(get<0>())>::value) &&
				noexcept(std::is_nothrow_copy_constructible<decltype(get<1>())>::value) &&
				noexcept(std::is_nothrow_copy_constructible<decltype(get<2>())>::value) &&
				noexcept(close_to_one(std::declval<R3_elmt<ComponentType, decltype(get<0>()), StorageTraits>>().normsq())) &&
				noexcept(close_to_one(std::declval<R3_elmt<ComponentType, decltype(get<1>()), StorageTraits>>().normsq())) &&
				noexcept(close_to_one(std::declval<R3_elmt<ComponentType, decltype(get<2>()), StorageTraits>>().normsq())) &&
				noexcept(close_to_zero(dot(
					std::declval<R3_elmt<ComponentType, decltype(get<0>()), StorageTraits>>(),
					std::declval<R3_elmt<ComponentType, decltype(get<1>()), StorageTraits>>()))) &&
				noexcept(close_to_zero(dot(
					std::declval<R3_elmt<ComponentType, decltype(get<1>()), StorageTraits>>(),
					std::declval<R3_elmt<ComponentType, decltype(get<2>()), StorageTraits>>()))) &&
				noexcept(close_to_zero(dot(
					std::declval<R3_elmt<ComponentType, decltype(get<2>()), StorageTraits>>(),
					std::declval<R3_elmt<ComponentType, decltype(get<0>()), StorageTraits>>()))))
			{
				using temp_vec0 = R3_elmt<ComponentType, decltype(get<0>()), StorageTraits>;
				using temp_vec1 = R3_elmt<ComponentType, decltype(get<1>()), StorageTraits>;
				using temp_vec2 = R3_elmt<ComponentType, decltype(get<2>()), StorageTraits>;

				return
					close_to_one(temp_vec0{ direct_construction{}, get<0>() }.normsq()) &&
					close_to_one(temp_vec1{ direct_construction{}, get<1>() }.normsq()) &&
					close_to_one(temp_vec2{ direct_construction{}, get<2>() }.normsq()) &&
					close_to_zero(dot(
						temp_vec0{ direct_construction{}, get<0>() },
						temp_vec1{ direct_construction{}, get<1>() })) &&
					close_to_zero(dot(
						temp_vec1{ direct_construction{}, get<1>() },
						temp_vec2{ direct_construction{}, get<2>() })) &&
					close_to_zero(dot(
						temp_vec2{ direct_construction{}, get<2>() },
						temp_vec0{ direct_construction{}, get<0>() }));
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const
				noexcept(noexcept(det() > jkl::math::zero<ComponentType>() && is_orthogonal()))
			{
				return det() > jkl::math::zero<ComponentType>() && is_orthogonal();
			}

			JKL_GPU_EXECUTABLE constexpr bool is_symmetric() const noexcept(noexcept(
				close_to(get<0, 1>(), get<1, 0>()) &&
				close_to(get<1, 2>(), get<2, 1>()) &&
				close_to(get<2, 0>(), get<0, 2>())))
			{
				return
					close_to(get<0, 1>(), get<1, 0>()) &&
					close_to(get<1, 2>(), get<2, 1>()) &&
					close_to(get<2, 0>(), get<0, 2>());
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const
				noexcept(noexcept(is_symmetric() &&
					get<0, 0>() > jkl::math::zero<ComponentType>() &&
					get<0, 0>() * get<1, 1>() > get<0, 1>() * get<1, 0>() &&
					det() > jkl::math::zero<ComponentType>()))
			{
				return is_symmetric() &&
					get<0, 0>() > jkl::math::zero<ComponentType>() &&
					get<0, 0>() * get<1, 1>() > get<0, 1>() * get<1, 0>() &&
					det() > jkl::math::zero<ComponentType>();
			}

			JKL_GPU_EXECUTABLE static constexpr gl3_elmt zero()
				noexcept(std::is_nothrow_constructible<gl3_elmt,
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>())>::value)
			{
				return{
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>()
				};
			}

			JKL_GPU_EXECUTABLE constexpr static GL3_elmt<ComponentType, Storage, StorageTraits> unity()
				noexcept(std::is_nothrow_default_constructible<GL3_elmt<ComponentType, Storage, StorageTraits>>::value)
			{
				return{};
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.det()))
		{
			return m.det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(gl3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).det()))
		{
			return std::move(m).det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.trace()))
		{
			return m.trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(gl3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).trace()))
		{
			return std::move(m).trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(gl3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_orthogonal(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_orthogonal()))
		{
			return m.is_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_special_orthogonal()))
		{
			return m.is_special_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_symmetric(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_symmetric()))
		{
			return m.is_symmetric();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_positive_definite(gl3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_positive_definite()))
		{
			return m.is_positive_definite();
		}

		// 3x3 invertible matrix
		template <class ComponentType, class Storage, class StorageTraits>
		class GL3_elmt : public gl3_elmt<ComponentType, Storage, StorageTraits>
		{
			using gl3_elmt_type = gl3_elmt<ComponentType, Storage, StorageTraits>;

			template <class Matrix>
			static constexpr gl3_elmt_type check_and_forward(Matrix&& m) {
				return m.is_invertible() ? std::forward<Matrix>(m) :
					throw input_validity_error<GL3_elmt>{ "jkl::math: the matrix is not invertible" };
			}

		public:
			// Initialize to the unity
			JKL_GPU_EXECUTABLE constexpr GL3_elmt() noexcept(noexcept(gl3_elmt_type{
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>() })) :
				gl3_elmt_type{
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>() } {}

			// No check component-wise constructor
			template <class Arg00, class Arg01, class Arg02,
				class Arg10, class Arg11, class Arg12,
				class Arg20, class Arg21, class Arg22>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				Arg00&& arg00, Arg01&& arg01, Arg02&& arg02,
				Arg10&& arg10, Arg11&& arg11, Arg12&& arg12,
				Arg20&& arg20, Arg21&& arg21, Arg22&& arg22, no_validity_check)
				noexcept(noexcept(gl3_elmt_type{
					std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02),
					std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12),
					std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22) })) :
					gl3_elmt_type{
				std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02),
				std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12),
				std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22) } {}

			// Checking component-wise constructor
			template <class Arg00, class Arg01, class Arg02,
				class Arg10, class Arg11, class Arg12,
				class Arg20, class Arg21, class Arg22>
			constexpr GL3_elmt(
				Arg00&& arg00, Arg01&& arg01, Arg02&& arg02,
				Arg10&& arg10, Arg11&& arg11, Arg12&& arg12,
				Arg20&& arg20, Arg21&& arg21, Arg22&& arg22) :
				gl3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
					detail::tuple<Arg00&&, Arg01&&, Arg02&&>,
					detail::tuple<Arg10&&, Arg11&&, Arg12&&>,
					detail::tuple<Arg20&&, Arg21&&, Arg22&&>>>{ direct_construction{},
					detail::tuple<Arg00&&, Arg01&&, Arg02&&>(
						std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02)),
					detail::tuple<Arg00&&, Arg01&&, Arg02&&>(
						std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12)),
					detail::tuple<Arg00&&, Arg01&&, Arg02&&>(
						std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22)) }) } {}

			// No check call-by-value component-wise constructor
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				ComponentType arg00, ComponentType arg01, ComponentType arg02,
				ComponentType arg10, ComponentType arg11, ComponentType arg12,
				ComponentType arg20, ComponentType arg21, ComponentType arg22, no_validity_check)
				noexcept(noexcept(gl3_elmt_type{
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg00), std::move(arg01), std::move(arg02)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg10), std::move(arg11), std::move(arg12)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg20), std::move(arg21), std::move(arg22)) })) :
					gl3_elmt_type{
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg00), std::move(arg01), std::move(arg02)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg10), std::move(arg11), std::move(arg12)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg20), std::move(arg21), std::move(arg22)) } {}

			// Checking call-by-value component-wise constructor
			constexpr GL3_elmt(
				ComponentType arg00, ComponentType arg01, ComponentType arg02,
				ComponentType arg10, ComponentType arg11, ComponentType arg12,
				ComponentType arg20, ComponentType arg21, ComponentType arg22) :
				gl3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>,
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>,
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>>>{
				direct_construction{},
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
						std::move(arg00), std::move(arg01), std::move(arg02)),
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
						std::move(arg10), std::move(arg11), std::move(arg12)),
					detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
						std::move(arg20), std::move(arg21), std::move(arg22)) }) } {}

			// No check row-wise constructor
			template <class Row0, class Row1, class Row2>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(Row0&& r0, Row1&& r1, Row2&& r2, no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type, Row0, Row1, Row2>::value) :
				gl3_elmt_type{ std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row2>(r2) } {}

			// Checking row-wise constructor
			template <class Row0, class Row1, class Row2>
			constexpr GL3_elmt(Row0&& r0, Row1&& r1, Row2&& r2) :
				gl3_elmt_type{ check_and_forward(gl3_elmt<ComponentType,
					detail::tuple<Row0&&, Row1&&, Row2&&>,
					detail::row_ref_tuple_traits<StorageTraits, Row0, Row1, Row2>>{ direct_construction{},
				std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row2>(r2) }) } {}

			// No check call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				R3_elmt<ComponentType> r0, R3_elmt<ComponentType> r1, R3_elmt<ComponentType> r2,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					R3_elmt<ComponentType>, R3_elmt<ComponentType>, R3_elmt<ComponentType>>::value) :
				gl3_elmt_type{ gl3_elmt<ComponentType, detail::tuple<
					R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&>>{
				direct_construction{}, std::move(r0), std::move(r1), std::move(r2) } } {}

			// Checking call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			constexpr GL3_elmt(
				R3_elmt<ComponentType> r0, R3_elmt<ComponentType> r1, R3_elmt<ComponentType> r2) :
				gl3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
					R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&>>{
				direct_construction{}, std::move(r0), std::move(r1), std::move(r2) }) } {}

			// Convert from GL3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL3_elmt,
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				gl3_elmt_type{ that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL3_elmt,
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				gl3_elmt_type{ std::move(that) } {}

			// Convert from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				gl3_elmt_type{ that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				gl3_elmt_type{ std::move(that) } {}

			// Convert from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				gl3_elmt_type{ check_and_forward(that) } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				gl3_elmt_type{ check_and_forward(std::move(that)) } {}

			// Convert from sym3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				gl3_elmt_type{ that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				gl3_elmt_type{ std::move(that) } {}

			// Convert from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL3_elmt(sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				gl3_elmt_type{ check_and_forward(that) } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL3_elmt(sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				gl3_elmt_type{ check_and_forward(std::move(that)) } {}

			// Convert from posdef3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				gl3_elmt_type{ that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<gl3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				gl3_elmt_type{ std::move(that) } {}


			// Copy and move
			GL3_elmt(GL3_elmt const&) = default;
			GL3_elmt(GL3_elmt&&) = default;
			GL3_elmt& operator=(GL3_elmt const&) & = default;
			GL3_elmt& operator=(GL3_elmt&&) & = default;

		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& assign_no_check_impl(Matrix&& m)
				noexcept(noexcept(static_cast<gl3_elmt_type&>(*this) = std::forward<Matrix>(m)))
			{
				static_cast<gl3_elmt_type&>(*this) = std::forward<Matrix>(m);
				return *this;
			}

		public:
			// Assignment from GL3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL3_elmt,
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL3_elmt,
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL3_elmt>{ "jkl::math: the matrix is not invertible" };
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL3_elmt>{ "jkl::math: the matrix is not invertible" };
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from sym3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& assign_no_check(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& assign_no_check(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from sym3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL3_elmt>{ "jkl::math: the matrix is not invertible" };
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL3_elmt>{ "jkl::math: the matrix is not invertible" };
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from posdef3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}


			// Remove mutable lvalue element accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<gl3_elmt_type const&>().template get<I>()))
			{
				return static_cast<gl3_elmt_type const&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<gl3_elmt_type&&>().template get<I>()))
			{
				return static_cast<gl3_elmt_type&&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<gl3_elmt_type const&&>().template get<I>()))
			{
				return static_cast<gl3_elmt_type const&&>(*this).template get<I>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<gl3_elmt_type const&>().template get<I, J>()))
			{
				return static_cast<gl3_elmt_type const&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<gl3_elmt_type&&>().template get<I, J>()))
			{
				return static_cast<gl3_elmt_type&&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<gl3_elmt_type const&&>().template get<I, J>()))
			{
				return static_cast<gl3_elmt_type const&&>(*this).template get<I, J>();
			}

			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const&
				noexcept(noexcept(std::declval<gl3_elmt_type const&>()[idx]))
				-> decltype(std::declval<gl3_elmt_type const&>()[idx])
			{
				return static_cast<gl3_elmt_type const&>(*this)[idx];
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) &&
				noexcept(noexcept(std::declval<gl3_elmt_type&&>()[idx]))
				-> decltype(std::declval<gl3_elmt_type&&>()[idx])
			{
				return static_cast<gl3_elmt_type&&>(*this)[idx];
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const&&
				noexcept(noexcept(std::declval<gl3_elmt_type const&&>()[idx]))
				-> decltype(std::declval<gl3_elmt_type const&&>()[idx])
			{
				return static_cast<gl3_elmt_type const&&>(*this)[idx];
			}


			JKL_GPU_EXECUTABLE constexpr GL3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<GL3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<GL3_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr GL3_elmt operator-() const&
				noexcept(noexcept(gl3_elmt_type{ -static_cast<gl3_elmt_type const&>(*this) }))
			{
				return{ -static_cast<gl3_elmt_type const&>(*this), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL3_elmt operator-() &&
				noexcept(noexcept(gl3_elmt_type{ -static_cast<gl3_elmt_type&&>(*this) }))
			{
				return{ -static_cast<gl3_elmt_type&&>(*this), no_validity_check{} };
			}

			// Remove += and -= operators
		private:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt_type& operator+=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that);
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt_type& operator+=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that);

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt_type& operator-=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that);
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl3_elmt_type& operator-=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that);

		public:
			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<gl3_elmt_type&>(*this) *= k))
			{
				static_cast<gl3_elmt_type&>(*this) *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<gl3_elmt_type&>(*this) /= k))
			{
				static_cast<gl3_elmt_type&>(*this) /= k;
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator*=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(static_cast<gl3_elmt_type&>(*this) *= that))
			{
				static_cast<gl3_elmt_type&>(*this) *= that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator*=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(static_cast<gl3_elmt_type&>(*this) *= std::move(that)))
			{
				static_cast<gl3_elmt_type&>(*this) *= std::move(that);
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr GL3_elmt t() const&
				noexcept(noexcept(static_cast<gl3_elmt_type const&>(*this).t()))
			{
				return{ static_cast<gl3_elmt_type const&>(*this).t(), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL3_elmt t() &&
				noexcept(noexcept(static_cast<gl3_elmt_type&&>(*this).t()))
			{
				return{ static_cast<gl3_elmt_type&&>(*this).t(), no_validity_check{} };
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const noexcept
			{
				return true;
			}


			// Division
		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR GL3_elmt inv_impl(Matrix&& m) noexcept(
				noexcept(gl3_elmt_type{
					m.template get<1, 1>() * m.template get<2, 2>() -
					m.template get<1, 2>() * m.template get<2, 1>(),
					m.template get<0, 2>() * m.template get<2, 1>() -
					m.template get<0, 1>() * m.template get<2, 2>(),
					m.template get<0, 1>() * m.template get<1, 2>() -
					m.template get<0, 2>() * m.template get<1, 1>(),
					m.template get<1, 2>() * m.template get<2, 0>() -
					m.template get<1, 0>() * m.template get<2, 2>(),
					m.template get<0, 0>() * m.template get<2, 2>() -
					m.template get<0, 2>() * m.template get<2, 0>(),
					m.template get<0, 2>() * m.template get<1, 0>() -
					m.template get<0, 0>() * m.template get<1, 2>(),
					m.template get<1, 0>() * m.template get<2, 1>() -
					m.template get<1, 1>() * m.template get<2, 0>(),
					m.template get<0, 1>() * m.template get<2, 0>() -
					m.template get<0, 0>() * m.template get<2, 1>(),
					m.template get<0, 0>() * m.template get<1, 1>() -
					m.template get<0, 1>() * m.template get<1, 0>() }) &&
					noexcept(std::declval<GL3_elmt&>() /= std::forward<Matrix>(m).det()))
			{
				GL3_elmt ret_value{
					std::forward<Matrix>(m).template get<1, 1>() * std::forward<Matrix>(m).template get<2, 2>() -
					std::forward<Matrix>(m).template get<1, 2>() * std::forward<Matrix>(m).template get<2, 1>(),
					std::forward<Matrix>(m).template get<0, 2>() * std::forward<Matrix>(m).template get<2, 1>() -
					std::forward<Matrix>(m).template get<0, 1>() * std::forward<Matrix>(m).template get<2, 2>(),
					std::forward<Matrix>(m).template get<0, 1>() * std::forward<Matrix>(m).template get<1, 2>() -
					std::forward<Matrix>(m).template get<0, 2>() * std::forward<Matrix>(m).template get<1, 1>(),
					std::forward<Matrix>(m).template get<1, 2>() * std::forward<Matrix>(m).template get<2, 0>() -
					std::forward<Matrix>(m).template get<1, 0>() * std::forward<Matrix>(m).template get<2, 2>(),
					std::forward<Matrix>(m).template get<0, 0>() * std::forward<Matrix>(m).template get<2, 2>() -
					std::forward<Matrix>(m).template get<0, 2>() * std::forward<Matrix>(m).template get<2, 0>(),
					std::forward<Matrix>(m).template get<0, 2>() * std::forward<Matrix>(m).template get<1, 0>() -
					std::forward<Matrix>(m).template get<0, 0>() * std::forward<Matrix>(m).template get<1, 2>(),
					std::forward<Matrix>(m).template get<1, 0>() * std::forward<Matrix>(m).template get<2, 1>() -
					std::forward<Matrix>(m).template get<1, 1>() * std::forward<Matrix>(m).template get<2, 0>(),
					std::forward<Matrix>(m).template get<0, 1>() * std::forward<Matrix>(m).template get<2, 0>() -
					std::forward<Matrix>(m).template get<0, 0>() * std::forward<Matrix>(m).template get<2, 1>(),
					std::forward<Matrix>(m).template get<0, 0>() * std::forward<Matrix>(m).template get<1, 1>() -
					std::forward<Matrix>(m).template get<0, 1>() * std::forward<Matrix>(m).template get<1, 0>(),
					no_validity_check{} };
				ret_value /= std::forward<Matrix>(m).det();
				return ret_value;
			}

		public:
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt inv() const&
				noexcept(noexcept(inv_impl(*this)))
			{
				return inv_impl(*this);
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt inv() &&
				noexcept(noexcept(inv_impl(std::move(*this))))
			{
				return inv_impl(std::move(*this));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator/=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(*this *= that.inv()))
			{
				return *this *= that.inv();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL3_elmt& operator/=(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(*this *= std::move(that).inv()))
			{
				return *this *= std::move(that).inv();
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(GL3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(GL3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(GL3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.inv()))
		{
			return m.inv();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(GL3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).inv()))
		{
			return std::move(m).inv();
		}


		//// Binary operations for 2D matrices

		namespace detail {
			template <class T, template <class, class, class> class Template, class = void>
			struct get_gl3_elmt_impl : std::false_type {
				using type = void;
			};

			template <class T, template <class, class, class> class Template>
			struct get_gl3_elmt_impl<T, Template, VOID_T<
				typename T::component_type,
				typename T::storage_type,
				typename T::storage_traits>> {
			private:
				using target_type = Template<
					typename T::component_type,
					typename T::storage_type,
					typename T::storage_traits>;

			public:
				static constexpr bool value = std::is_base_of<target_type, T>::value;
				using type = std::conditional_t<value, target_type, void>;
			};

			template <class T>
			using get_gl3_elmt = get_gl3_elmt_impl<T, gl3_elmt>;

			template <class T>
			using get_GL3_elmt = get_gl3_elmt_impl<T, GL3_elmt>;
			

			template <class LeftOperand, class RightOperand>
			struct get_gl3_elmt_binary_result_impl : std::conditional_t<
				get_gl3_elmt<LeftOperand>::value, std::conditional_t<
				get_gl3_elmt<RightOperand>::value, get_gl3_elmt_binary_result_impl<
				typename get_gl3_elmt<LeftOperand>::type,
				typename get_gl3_elmt<RightOperand>::type>,
				empty_type>, empty_type> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_binary_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				gl3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = gl3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl3_elmt_binary_result = typename get_gl3_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;

			
			template <class LeftOperand, class RightOperand>
			struct get_gl3_elmt_mult_result_impl :
				get_gl3_elmt_binary_result_impl<LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_mult_result_impl<
				GL3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_mult_result_impl<
				SO3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_mult_result_impl<
				GL3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SO3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_mult_result_impl<
				SO3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SO3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = SO3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl3_elmt_mult_result = typename get_gl3_elmt_mult_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class LeftOperand, class RightOperand>
			struct get_gl3_elmt_div_result_impl {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_div_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = gl3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_div_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SO3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = gl3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_div_result_impl<
				GL3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_div_result_impl<
				GL3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SO3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl3_elmt_div_result_impl<
				SO3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SO3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = SO3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl3_elmt_div_result = typename get_gl3_elmt_div_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;
			

			template <class Scalar, class Matrix, bool from_left>
			struct get_gl3_elmt_scalar_mult_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_gl3_elmt_scalar_mult_result_impl_impl<Scalar,
				gl3_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = gl3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_gl3_elmt_scalar_mult_result_impl_impl<Scalar,
				GL3_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = GL3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_gl3_elmt_scalar_mult_result_impl_impl<Scalar,
				SO3_elmt<ComponentType, Storage, StorageTraits>, from_left> :
				get_gl3_elmt_scalar_mult_result_impl_impl<Scalar,
				GL3_elmt<ComponentType, Storage, StorageTraits>, from_left> {};

			template <class Scalar, class Matrix, bool from_left>
			struct get_gl3_elmt_scalar_mult_result_impl : std::conditional_t<
				get_gl3_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>::value,
				get_gl3_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>,
				get_gl3_elmt_scalar_mult_result_impl_impl<void, void, false>> {};

			template <class Scalar, class Matrix, bool from_left>
			using get_gl3_elmt_scalar_mult_result = typename get_gl3_elmt_scalar_mult_result_impl<
				tmp::remove_cvref_t<Scalar>,
				tmp::remove_cvref_t<Matrix>, from_left>::type;


			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<gl3_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = gl3_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args...>::value)
				{
					return{ std::forward<Args>(args)... };
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<GL3_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = GL3_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args..., no_validity_check>::value)
				{
					return{ std::forward<Args>(args)..., no_validity_check{} };
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<SO3_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = SO3_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args..., no_validity_check>::value)
				{
					return{ std::forward<Args>(args)..., no_validity_check{} };
				}
			};
		}

		// Binary addition of gl3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator+(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(a).template get<0, 0>() +
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() +
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<0, 2>() +
				std::forward<RightOperand>(b).template get<0, 2>(),
				std::forward<LeftOperand>(a).template get<1, 0>() +
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() +
				std::forward<RightOperand>(b).template get<1, 1>(),
				std::forward<LeftOperand>(a).template get<1, 2>() +
				std::forward<RightOperand>(b).template get<1, 2>(),
				std::forward<LeftOperand>(a).template get<2, 0>() +
				std::forward<RightOperand>(b).template get<2, 0>(),
				std::forward<LeftOperand>(a).template get<2, 1>() +
				std::forward<RightOperand>(b).template get<2, 1>(),
				std::forward<LeftOperand>(a).template get<2, 2>() +
				std::forward<RightOperand>(b).template get<2, 2>() }))
			-> detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot add two gl3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot add two gl3_elmt's; failed to deduce the resulting storage type");

			return{
				std::forward<LeftOperand>(a).template get<0, 0>() +
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() +
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<0, 2>() +
				std::forward<RightOperand>(b).template get<0, 2>(),
				std::forward<LeftOperand>(a).template get<1, 0>() +
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() +
				std::forward<RightOperand>(b).template get<1, 1>(),
				std::forward<LeftOperand>(a).template get<1, 2>() +
				std::forward<RightOperand>(b).template get<1, 2>(),
				std::forward<LeftOperand>(a).template get<2, 0>() +
				std::forward<RightOperand>(b).template get<2, 0>(),
				std::forward<LeftOperand>(a).template get<2, 1>() +
				std::forward<RightOperand>(b).template get<2, 1>(),
				std::forward<LeftOperand>(a).template get<2, 2>() +
				std::forward<RightOperand>(b).template get<2, 2>()
			};
		}

		// Binary subtraction of gl3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator-(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(a).template get<0, 0>() -
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() -
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<0, 2>() -
				std::forward<RightOperand>(b).template get<0, 2>(),
				std::forward<LeftOperand>(a).template get<1, 0>() -
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() -
				std::forward<RightOperand>(b).template get<1, 1>(),
				std::forward<LeftOperand>(a).template get<1, 2>() -
				std::forward<RightOperand>(b).template get<1, 2>(),
				std::forward<LeftOperand>(a).template get<2, 0>() -
				std::forward<RightOperand>(b).template get<2, 0>(),
				std::forward<LeftOperand>(a).template get<2, 1>() -
				std::forward<RightOperand>(b).template get<2, 1>(),
				std::forward<LeftOperand>(a).template get<2, 2>() -
				std::forward<RightOperand>(b).template get<2, 2>() }))
			-> detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot subtract two gl3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot subtract two gl3_elmt's; failed to deduce the resulting storage type");

			return{
				std::forward<LeftOperand>(a).template get<0, 0>() -
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() -
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<0, 2>() -
				std::forward<RightOperand>(b).template get<0, 2>(),
				std::forward<LeftOperand>(a).template get<1, 0>() -
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() -
				std::forward<RightOperand>(b).template get<1, 1>(),
				std::forward<LeftOperand>(a).template get<1, 2>() -
				std::forward<RightOperand>(b).template get<1, 2>(),
				std::forward<LeftOperand>(a).template get<2, 0>() -
				std::forward<RightOperand>(b).template get<2, 0>(),
				std::forward<LeftOperand>(a).template get<2, 1>() -
				std::forward<RightOperand>(b).template get<2, 1>(),
				std::forward<LeftOperand>(a).template get<2, 2>() -
				std::forward<RightOperand>(b).template get<2, 2>()
			};
		}

		// Binary multiplication of gl3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator*(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl3_elmt_mult_result<LeftOperand, RightOperand>>::make(
					// First row
					a.template get<0, 0>() * b.template get<0, 0>() +
					a.template get<0, 1>() * b.template get<1, 0>() +
					a.template get<0, 2>() * b.template get<2, 0>(),
					a.template get<0, 0>() * b.template get<0, 1>() +
					a.template get<0, 1>() * b.template get<1, 1>() +
					a.template get<0, 2>() * b.template get<2, 1>(),
					a.template get<0, 0>() * b.template get<0, 2>() +
					a.template get<0, 1>() * b.template get<1, 2>() +
					a.template get<0, 2>() * b.template get<2, 2>(),
					// Second row
					a.template get<1, 0>() * b.template get<0, 0>() +
					a.template get<1, 1>() * b.template get<1, 0>() +
					a.template get<1, 2>() * b.template get<2, 0>(),
					a.template get<1, 0>() * b.template get<0, 1>() +
					a.template get<1, 1>() * b.template get<1, 1>() +
					a.template get<1, 2>() * b.template get<2, 1>(),
					a.template get<1, 0>() * b.template get<0, 2>() +
					a.template get<1, 1>() * b.template get<1, 2>() +
					a.template get<1, 2>() * b.template get<2, 2>(),
					// Third row
					a.template get<2, 0>() * b.template get<0, 0>() +
					a.template get<2, 1>() * b.template get<1, 0>() +
					a.template get<2, 2>() * b.template get<2, 0>(),
					a.template get<2, 0>() * b.template get<0, 1>() +
					a.template get<2, 1>() * b.template get<1, 1>() +
					a.template get<2, 2>() * b.template get<2, 1>(),
					a.template get<2, 0>() * b.template get<0, 2>() +
					a.template get<2, 1>() * b.template get<1, 2>() +
					a.template get<2, 2>() * b.template get<2, 2>())))
			-> detail::get_gl3_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl3_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot multiply two gl3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply two gl3_elmt's; failed to deduce the resulting storage type");
			
			return detail::call_unchecking<result_type>::make(
				// First row
				a.template get<0, 0>() * b.template get<0, 0>() +
				a.template get<0, 1>() * b.template get<1, 0>() +
				a.template get<0, 2>() * b.template get<2, 0>(),
				a.template get<0, 0>() * b.template get<0, 1>() +
				a.template get<0, 1>() * b.template get<1, 1>() +
				a.template get<0, 2>() * b.template get<2, 1>(),
				a.template get<0, 0>() * b.template get<0, 2>() +
				a.template get<0, 1>() * b.template get<1, 2>() +
				a.template get<0, 2>() * b.template get<2, 2>(),
				// Second row
				a.template get<1, 0>() * b.template get<0, 0>() +
				a.template get<1, 1>() * b.template get<1, 0>() +
				a.template get<1, 2>() * b.template get<2, 0>(),
				a.template get<1, 0>() * b.template get<0, 1>() +
				a.template get<1, 1>() * b.template get<1, 1>() +
				a.template get<1, 2>() * b.template get<2, 1>(),
				a.template get<1, 0>() * b.template get<0, 2>() +
				a.template get<1, 1>() * b.template get<1, 2>() +
				a.template get<1, 2>() * b.template get<2, 2>(),
				// Third row
				a.template get<2, 0>() * b.template get<0, 0>() +
				a.template get<2, 1>() * b.template get<1, 0>() +
				a.template get<2, 2>() * b.template get<2, 0>(),
				a.template get<2, 0>() * b.template get<0, 1>() +
				a.template get<2, 1>() * b.template get<1, 1>() +
				a.template get<2, 2>() * b.template get<2, 1>(),
				a.template get<2, 0>() * b.template get<0, 2>() +
				a.template get<2, 1>() * b.template get<1, 2>() +
				a.template get<2, 2>() * b.template get<2, 2>());
		}


		// Binary division of gl3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator/(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv()))
			-> detail::get_gl3_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl3_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot divide two gl3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot divide two gl3_elmt's; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv();
		}
		

		// Scalar multiplication of gl3_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
					std::forward<Matrix>(m).template get<0, 0>() * k,
					std::forward<Matrix>(m).template get<0, 1>() * k,
					std::forward<Matrix>(m).template get<0, 2>() * k,
					std::forward<Matrix>(m).template get<1, 0>() * k,
					std::forward<Matrix>(m).template get<1, 1>() * k,
					std::forward<Matrix>(m).template get<1, 2>() * k,
					std::forward<Matrix>(m).template get<2, 0>() * k,
					std::forward<Matrix>(m).template get<2, 1>() * k,
					std::forward<Matrix>(m).template get<2, 2>() * k)))
			-> detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply gl3_elmt with a scalar; failed to deduce the resulting storage type");
			
			using component_type = typename result_type::component_type;
			assert(!detail::get_GL3_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).template get<0, 0>() * k,
				std::forward<Matrix>(m).template get<0, 1>() * k,
				std::forward<Matrix>(m).template get<0, 2>() * k,
				std::forward<Matrix>(m).template get<1, 0>() * k,
				std::forward<Matrix>(m).template get<1, 1>() * k,
				std::forward<Matrix>(m).template get<1, 2>() * k,
				std::forward<Matrix>(m).template get<2, 0>() * k,
				std::forward<Matrix>(m).template get<2, 1>() * k,
				std::forward<Matrix>(m).template get<2, 2>() * k);
		}

		// Scalar multiplication of gl3_elmt's from left
		template <class Scalar, class Matrix>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Scalar const& k, Matrix&& m)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, true>>::make(
					k * std::forward<Matrix>(m).template get<0, 0>(),
					k * std::forward<Matrix>(m).template get<0, 1>(),
					k * std::forward<Matrix>(m).template get<0, 2>(),
					k * std::forward<Matrix>(m).template get<1, 0>(), 
					k * std::forward<Matrix>(m).template get<1, 1>(),
					k * std::forward<Matrix>(m).template get<1, 2>(),
					k * std::forward<Matrix>(m).template get<2, 0>(),
					k * std::forward<Matrix>(m).template get<2, 1>(),
					k * std::forward<Matrix>(m).template get<2, 2>())))
			-> detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, true>
		{
			using result_type = detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot multiply gl3_elmt with a scalar; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply gl3_elmt with a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_GL3_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				k * std::forward<Matrix>(m).template get<0, 0>(),
				k * std::forward<Matrix>(m).template get<0, 1>(),
				k * std::forward<Matrix>(m).template get<0, 2>(),
				k * std::forward<Matrix>(m).template get<1, 0>(),
				k * std::forward<Matrix>(m).template get<1, 1>(),
				k * std::forward<Matrix>(m).template get<1, 2>(),
				k * std::forward<Matrix>(m).template get<2, 0>(),
				k * std::forward<Matrix>(m).template get<2, 1>(),
				k * std::forward<Matrix>(m).template get<2, 2>());
		}

		// Scalar division of gl3_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
					std::forward<Matrix>(m).template get<0, 0>() / k,
					std::forward<Matrix>(m).template get<0, 1>() / k,
					std::forward<Matrix>(m).template get<0, 2>() / k,
					std::forward<Matrix>(m).template get<1, 0>() / k,
					std::forward<Matrix>(m).template get<1, 1>() / k,
					std::forward<Matrix>(m).template get<1, 2>() / k,
					std::forward<Matrix>(m).template get<2, 0>() / k,
					std::forward<Matrix>(m).template get<2, 1>() / k,
					std::forward<Matrix>(m).template get<2, 2>() / k)))
			-> detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_gl3_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot divide gl3_elmt by a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_GL3_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).template get<0, 0>() / k,
				std::forward<Matrix>(m).template get<0, 1>() / k,
				std::forward<Matrix>(m).template get<0, 2>() / k,
				std::forward<Matrix>(m).template get<1, 0>() / k,
				std::forward<Matrix>(m).template get<1, 1>() / k,
				std::forward<Matrix>(m).template get<1, 2>() / k,
				std::forward<Matrix>(m).template get<2, 0>() / k,
				std::forward<Matrix>(m).template get<2, 1>() / k,
				std::forward<Matrix>(m).template get<2, 2>() / k);
		}
	}
}
