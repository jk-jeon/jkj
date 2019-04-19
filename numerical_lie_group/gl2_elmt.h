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

namespace jkj {
	namespace math {
		namespace detail {
			template <class ComponentType, class Storage, class StorageTraits>
			struct gl2_elmt_base {
				using storage_type = Storage;
				using storage_traits = StorageTraits;

			protected:
				using storage_wrapper = typename StorageTraits::template storage_wrapper<
					Storage, gl2_elmt<ComponentType, Storage, StorageTraits>>;
				storage_wrapper r_;

			public:
				gl2_elmt_base() = default;

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr gl2_elmt_base(direct_construction, Args&&... args)
					noexcept(std::is_nothrow_constructible<storage_wrapper, Args...>::value) :
					r_(std::forward<Args>(args)...) {}

			protected:
				// See: https://stackoverflow.com/questions/51501156/evaluating-noexcept-specifier-before-template-type-deduction/51504042
				// noexcept specifier should be evaluated after the overload resolution, but
				// it seems MSVC evaluates it too early. That sometimes causes some obscure compile errors, so
				// we should separate the cases when Arg00 ~ Arg11 can indeed construct storage_wrapper
				// to workaround this bug.
				template <class Arg00, class Arg01, class Arg10, class Arg11, class = void>
				struct is_nothrow_constructible : std::false_type {};

				template <class Arg00, class Arg01, class Arg10, class Arg11>
				struct is_nothrow_constructible<Arg00, Arg01, Arg10, Arg11,
					VOID_T<decltype(storage_wrapper{ { std::declval<Arg00>(), std::declval<Arg01>() },
						{ std::declval<Arg10>(), std::declval<Arg11>() } })>>
				{
					static constexpr bool value = noexcept(storage_wrapper{
						{ std::declval<Arg00>(), std::declval<Arg01>() },
						{ std::declval<Arg10>(), std::declval<Arg11>() } });
				};

				// Component-wise constructor
				template <class Arg00, class Arg01, class Arg10, class Arg11>
				JKL_GPU_EXECUTABLE constexpr gl2_elmt_base(forward_to_storage_tag,
					Arg00&& arg00, Arg01&& arg01,
					Arg10&& arg10, Arg11&& arg11)
					noexcept(is_nothrow_constructible<Arg00, Arg01, Arg10, Arg11>::value) :
					r_{ { std::forward<Arg00>(arg00), std::forward<Arg01>(arg01) },
				{ std::forward<Arg10>(arg10), std::forward<Arg11>(arg11) } } {}
			};
		}

		// 2x2 matrix
		template <class ComponentType, class Storage, class StorageTraits>
		class gl2_elmt :
			public detail::constructor_provider<4, ComponentType,
			detail::gl2_elmt_base<ComponentType, Storage, StorageTraits>>
		{
			using constructor_provider = detail::constructor_provider<4, ComponentType,
				detail::gl2_elmt_base<ComponentType, Storage, StorageTraits>>;
			using storage_wrapper = typename constructor_provider::storage_wrapper;

		public:
			using constructor_provider::storage;
			using constructor_provider::constructor_provider;

			// Row access requirements
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkj::math: gl2_elmt requires access to the first row from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkj::math: gl2_elmt requires access to the second row from the storage; "
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
				"jkj::math: the first row of gl2_elmt requires access to the first column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(
					std::declval<Storage const&>()))>::value,
				"jkj::math: the first row of gl2_elmt requires access to the second column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(
					std::declval<Storage const&>()))>::value,
				"jkj::math: the second row of gl2_elmt requires access to the first column from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(
					std::declval<Storage const&>()))>::value,
				"jkj::math: the second row of gl2_elmt requires access to the second column from the storage; "
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
					is_component_convertible<1>::value;
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
			gl2_elmt() = default;
			
			// Row-wise constructor
			template <class Row0, class Row1>
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(Row0&& r0, Row1&& r1)
				noexcept(noexcept(gl2_elmt{ gl2_elmt<ComponentType,
					detail::tuple<Row0&&, Row1&&>,
					detail::row_ref_tuple_traits<StorageTraits, Row0, Row1>>{ direct_construction{},
					std::forward<Row0>(r0), std::forward<Row1>(r1) } })) :
				gl2_elmt{ gl2_elmt<ComponentType, detail::tuple<Row0&&, Row1&&>,
				detail::row_ref_tuple_traits<StorageTraits, Row0, Row1>>{ direct_construction{},
				std::forward<Row0>(r0), std::forward<Row1>(r1) } } {}

			// Call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(R2_elmt<ComponentType> r0, R2_elmt<ComponentType> r1)
				noexcept(noexcept(constructor_provider(std::move(r0).x(), std::move(r0).y(),
					std::move(r1).x(), std::move(r1).y()))) :
				constructor_provider(std::move(r0).x(), std::move(r0).y(),
					std::move(r1).x(), std::move(r1).y()) {}
			
			// Convert from gl2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl2_elmt,
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.template get<0, 0>(), that.template get<0, 1>(),
					that.template get<1, 0>(), that.template get<1, 1>()))) :
				constructor_provider(that.template get<0, 0>(), that.template get<0, 1>(),
				that.template get<1, 0>(), that.template get<1, 1>()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl2_elmt,
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(std::move(that).template get<0, 0>(),
					std::move(that).template get<0, 1>(),
					std::move(that).template get<1, 0>(), std::move(that).template get<1, 1>()))) :
				constructor_provider(std::move(that).template get<0, 0>(), std::move(that).template get<0, 1>(),
				std::move(that).template get<1, 0>(), std::move(that).template get<1, 1>()) {}
			
			// Convert from sym2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType const&, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.xx(), that.xy(), that.yx(), that.yy()))) :
				constructor_provider(that.xx(), that.xy(), that.yx(), that.yy()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr gl2_elmt(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(std::move(that).xx(), that.xy(),
					that.yx(), std::move(that).yy()))) :
				constructor_provider(std::move(that).xx(), that.xy(), that.yx(), std::move(that).yy()) {}


			// Copy and move
			gl2_elmt(gl2_elmt const&) = default;
			gl2_elmt(gl2_elmt&&) = default;
			gl2_elmt& operator=(gl2_elmt const&) & = default;
			gl2_elmt& operator=(gl2_elmt&&) & = default;

			// Assignment from gl2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl2_elmt,
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(get<0, 0>() = that.template get<0, 0>()) &&
					noexcept(get<0, 1>() = that.template get<0, 1>()) &&
					noexcept(get<1, 0>() = that.template get<1, 0>()) &&
					noexcept(get<1, 1>() = that.template get<1, 1>()))
			{
				get<0, 0>() = that.template get<0, 0>();
				get<0, 1>() = that.template get<0, 1>();
				get<1, 0>() = that.template get<1, 0>();
				get<1, 1>() = that.template get<1, 1>();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<gl2_elmt,
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(get<0, 0>() = std::move(that).template get<0, 0>()) &&
					noexcept(get<0, 1>() = std::move(that).template get<0, 1>()) &&
					noexcept(get<1, 0>() = std::move(that).template get<1, 0>()) &&
					noexcept(get<1, 1>() = std::move(that).template get<1, 1>()))
			{
				get<0, 0>() = std::move(that).template get<0, 0>();
				get<0, 1>() = std::move(that).template get<0, 1>();
				get<1, 0>() = std::move(that).template get<1, 0>();
				get<1, 1>() = std::move(that).template get<1, 1>();
				return *this;
			}


			// Assignment from sym2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator=(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(get<0, 0>() = that.xx()) &&
					noexcept(get<0, 1>() = that.xy()) &&
					noexcept(get<1, 0>() = that.yx()) &&
					noexcept(get<1, 1>() = that.yy()))
			{
				get<0, 0>() = that.xx();
				get<0, 1>() = that.xy();
				get<1, 0>() = that.yx();
				get<1, 1>() = that.yy();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator=(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(get<0, 0>() = std::move(that).xx()) &&
					noexcept(get<0, 1>() = that.xy()) &&
					noexcept(get<1, 0>() = std::move(that).yx()) &&
					noexcept(get<1, 1>() = std::move(that).yy()))
			{
				get<0, 0>() = std::move(that).xx();
				get<0, 1>() = that.xy();
				get<1, 0>() = std::move(that).yx();
				get<1, 1>() = std::move(that).yy();
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr decltype(auto) det() const& noexcept(noexcept(
				get<0, 0>() * get<1, 1>() - get<0, 1>() * get<1, 0>()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (get<0, 0>() * get<1, 1>() - get<0, 1>() * get<1, 0>());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) det() && noexcept(noexcept(
				std::move(*this).template get<0, 0>() * std::move(*this).template get<1, 1>()
					- std::move(*this).template get<0, 1>() * std::move(*this).template get<1, 0>()))
			{
				return std::move(*this).template get<0, 0>() * std::move(*this).template get<1, 1>()
					- std::move(*this).template get<0, 1>() * std::move(*this).template get<1, 0>();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) trace() const& noexcept(noexcept(
				get<0, 0>() + get<1, 1>()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (get<0, 0>() + get<1, 1>());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) trace() && noexcept(noexcept(
				std::move(*this).template get<0, 0>() + std::move(*this).template get<1, 1>()))
			{
				return std::move(*this).template get<0, 0>() + std::move(*this).template get<1, 1>();
			}

			JKL_GPU_EXECUTABLE constexpr gl2_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<gl2_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl2_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<gl2_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr gl2_elmt operator-() const&
				noexcept(noexcept(gl2_elmt{ -get<0, 0>(), -get<0, 1>(), -get<1, 0>(), -get<1, 1>() }))
			{
				return{ -get<0, 0>(), -get<0, 1>(), -get<1, 0>(), -get<1, 1>() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl2_elmt operator-() &&
				noexcept(noexcept(gl2_elmt{
				-std::move(*this).template get<0, 0>(),
				-std::move(*this).template get<0, 1>(),
				-std::move(*this).template get<1, 0>(),
				-std::move(*this).template get<1, 1>() }))
			{
				return{ -std::move(*this).template get<0, 0>(), -std::move(*this).template get<0, 1>(),
					-std::move(*this).template get<1, 0>(), -std::move(*this).template get<1, 1>() };
			}

		private:
			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& inplace_add_impl(OtherMatrix&& m) noexcept(
				noexcept(get<0, 0>() += std::forward<OtherMatrix>(m).template get<0, 0>()) &&
				noexcept(get<0, 1>() += std::forward<OtherMatrix>(m).template get<0, 1>()) &&
				noexcept(get<1, 0>() += std::forward<OtherMatrix>(m).template get<1, 0>()) &&
				noexcept(get<1, 1>() += std::forward<OtherMatrix>(m).template get<1, 1>()))
			{
				get<0, 0>() += std::forward<OtherMatrix>(m).template get<0, 0>();
				get<0, 1>() += std::forward<OtherMatrix>(m).template get<0, 1>();
				get<1, 0>() += std::forward<OtherMatrix>(m).template get<1, 0>();
				get<1, 1>() += std::forward<OtherMatrix>(m).template get<1, 1>();
				return *this;
			}

			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& inplace_sub_impl(OtherMatrix&& m) noexcept(
				noexcept(get<0, 0>() -= std::forward<OtherMatrix>(m).template get<0, 0>()) &&
				noexcept(get<0, 1>() -= std::forward<OtherMatrix>(m).template get<0, 1>()) &&
				noexcept(get<1, 0>() -= std::forward<OtherMatrix>(m).template get<1, 0>()) &&
				noexcept(get<1, 1>() -= std::forward<OtherMatrix>(m).template get<1, 1>()))
			{
				get<0, 0>() -= std::forward<OtherMatrix>(m).template get<0, 0>();
				get<0, 1>() -= std::forward<OtherMatrix>(m).template get<0, 1>();
				get<1, 0>() -= std::forward<OtherMatrix>(m).template get<1, 0>();
				get<1, 1>() -= std::forward<OtherMatrix>(m).template get<1, 1>();
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator+=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_add_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator+=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_add_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator-=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_sub_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator-=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_sub_impl(std::move(that));
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator*=(OtherComponentType const& k) noexcept(
				noexcept(get<0, 0>() *= k) &&
				noexcept(get<0, 1>() *= k) &&
				noexcept(get<1, 0>() *= k) &&
				noexcept(get<1, 1>() *= k))
			{
				get<0, 0>() *= k;
				get<0, 1>() *= k;
				get<1, 0>() *= k;
				get<1, 1>() *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator/=(OtherComponentType const& k) noexcept(
					noexcept(get<0, 0>() /= k) &&
					noexcept(get<0, 1>() /= k) &&
					noexcept(get<1, 0>() /= k) &&
					noexcept(get<1, 1>() /= k))
			{
				get<0, 0>() /= k;
				get<0, 1>() /= k;
				get<1, 0>() /= k;
				get<1, 1>() /= k;
				return *this;
			}

		private:
			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& inplace_mul_impl(OtherMatrix&& m)
				noexcept(noexcept(*this = *this * std::forward<OtherMatrix>(m)))
			{
				*this = *this * std::forward<OtherMatrix>(m);
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator*=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_mul_impl(that)))
			{
				return inplace_mul_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator*=(
					gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_mul_impl(std::move(that))))
			{
				return inplace_mul_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator/=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(*this * that.inv()))
			{
				return *this * that.inv();
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt& operator/=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(*this * std::move(that).inv()))
			{
				return *this * std::move(that).inv();
			}

			JKL_GPU_EXECUTABLE constexpr gl2_elmt t() const& noexcept(noexcept(gl2_elmt{
				get<0, 0>(), get<1, 0>(),
				get<0, 1>(), get<1, 1>() }))
			{
				return{ get<0, 0>(), get<1, 0>(),
					get<0, 1>(), get<1, 1>() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR gl2_elmt t() && noexcept(noexcept(gl2_elmt{
				std::move(*this).template get<0, 0>(),
				std::move(*this).template get<1, 0>(),
				std::move(*this).template get<0, 1>(),
				std::move(*this).template get<1, 1>() }))
			{
				return{ std::move(*this).template get<0, 0>(),
					std::move(*this).template get<1, 0>(),
					std::move(*this).template get<0, 1>(),
					std::move(*this).template get<1, 1>() };
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const noexcept(
					noexcept(get<0, 0>() == that.template get<0, 0>()) &&
					noexcept(get<0, 1>() == that.template get<0, 1>()) &&
					noexcept(get<1, 0>() == that.template get<1, 0>()) &&
					noexcept(get<1, 1>() == that.template get<1, 1>()))
			{
				return get<0, 0>() == that.template get<0, 0>()
					&& get<0, 1>() == that.template get<0, 1>()
					&& get<1, 0>() == that.template get<1, 0>()
					&& get<1, 1>() == that.template get<1, 1>();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const
				noexcept(noexcept((*this) == that))
			{
				return !((*this) == that);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const&
				noexcept(noexcept(jkj::math::is_invertible(det())))
			{
				return jkj::math::is_invertible(det());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool is_invertible() &&
				noexcept(noexcept(jkj::math::is_invertible(std::move(*this).det())))
			{
				return jkj::math::is_invertible(std::move(*this).det());
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept(
				noexcept(std::is_nothrow_copy_constructible<decltype(get<0>())>::value) &&
				noexcept(std::is_nothrow_copy_constructible<decltype(get<1>())>::value) &&
				noexcept(close_to_one(std::declval<R2_elmt<ComponentType, decltype(get<0>()), StorageTraits>>().normsq())) &&
				noexcept(close_to_one(std::declval<R2_elmt<ComponentType, decltype(get<1>()), StorageTraits>>().normsq())) &&
				noexcept(close_to_zero(dot(std::declval<R2_elmt<ComponentType, decltype(get<0>()), StorageTraits>>(),
					std::declval<R2_elmt<ComponentType, decltype(get<1>()), StorageTraits>>()))))
			{
				using temp_vec0 = R2_elmt<ComponentType, decltype(get<0>()), StorageTraits>;
				using temp_vec1 = R2_elmt<ComponentType, decltype(get<1>()), StorageTraits>;

				return
					close_to_one(temp_vec0{ direct_construction{}, get<0>() }.normsq()) &&
					close_to_one(temp_vec1{ direct_construction{}, get<1>() }.normsq()) &&
					close_to_zero(dot(temp_vec0{ direct_construction{}, get<0>() },
						temp_vec1{ direct_construction{}, get<1>() }));
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const
				noexcept(noexcept(det() > jkj::math::zero<ComponentType>() && is_orthogonal()))
			{
				return det() > jkj::math::zero<ComponentType>() && is_orthogonal();
			}

			JKL_GPU_EXECUTABLE constexpr bool is_symmetric() const
				noexcept(noexcept(close_to(get<0, 1>(), get<1, 0>())))
			{
				return close_to(get<0, 1>(), get<1, 0>());
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const
				noexcept(noexcept(is_symmetric() &&
					get<0, 0>() > jkj::math::zero<ComponentType>() &&
					det() > jkj::math::zero<ComponentType>()))
			{
				return is_symmetric() &&
					get<0, 0>() > jkj::math::zero<ComponentType>() &&
					det() > jkj::math::zero<ComponentType>();
			}

			JKL_GPU_EXECUTABLE static constexpr gl2_elmt zero()
				noexcept(noexcept(gl2_elmt{
				jkj::math::zero<ComponentType>(),
				jkj::math::zero<ComponentType>(),
				jkj::math::zero<ComponentType>(),
				jkj::math::zero<ComponentType>() }))
			{
				return{ jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>(),
					jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>() };
			}

			JKL_GPU_EXECUTABLE static constexpr GL2_elmt<ComponentType, Storage, StorageTraits> unity()
				noexcept(noexcept(gl2_elmt{
				jkj::math::unity<ComponentType>(),
				jkj::math::zero<ComponentType>(),
				jkj::math::zero<ComponentType>(),
				jkj::math::unity<ComponentType>() }))
			{
				return{ jkj::math::unity<ComponentType>(), jkj::math::zero<ComponentType>(),
					jkj::math::zero<ComponentType>(), jkj::math::unity<ComponentType>(),
					no_validity_check{} };
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.det()))
		{
			return m.det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(gl2_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).det()))
		{
			return std::move(m).det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.trace()))
		{
			return m.trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(gl2_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).trace()))
		{
			return std::move(m).trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(gl2_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_orthogonal(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_orthogonal()))
		{
			return m.is_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_special_orthogonal()))
		{
			return m.is_special_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_symmetric(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_symmetric()))
		{
			return m.is_symmetric();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_positive_definite(gl2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_positive_definite()))
		{
			return m.is_positive_definite();
		}

		// 2x2 invertible matrix
		namespace detail {
			// To suppress generation of inherited constructors
			template <class ComponentType, class Storage, class StorageTraits>
			struct GL2_elmt_base : gl2_elmt<ComponentType, Storage, StorageTraits> {
			private:
				using base_type = gl2_elmt<ComponentType, Storage, StorageTraits>;
				using target_type = GL2_elmt<ComponentType, Storage, StorageTraits>;

				template <class Matrix>
				static constexpr base_type check_and_forward(Matrix&& m) {
					return m.is_invertible() ? std::move(m) :
						throw input_validity_error<target_type>{ "jkj::math: the matrix is not invertible" };
				}

			protected:
				// [NOTE]
				// I think it would be a bad idea to add assert() inside no-check constructors,
				// because the main cause of the break of invertibility is the accumulation of
				// floating-point operation errors. It would be frequently OK if the invertibility
				// condition is just "broken slightly" by such an error, and
				// those spurious failures indeed occur quite occasionally.
				// Or even "not-very-slight-violations" might sometimes be OK as well.
				// Adding assert() may make the class too strict.

				// Default constructor; components might be filled with garbages
				GL2_elmt_base() = default;

				// No check matrix constructor
				template <class Matrix>
				JKL_GPU_EXECUTABLE constexpr GL2_elmt_base(forward_to_storage_tag,
					Matrix&& m, no_validity_check)
					noexcept(noexcept(base_type{ std::forward<Matrix>(m) })) :
					base_type{ std::forward<Matrix>(m) } {}

				// Checking matrix constructor
				template <class Matrix>
				constexpr GL2_elmt_base(forward_to_storage_tag, Matrix&& m) :
					base_type{ check_and_forward(std::forward<Matrix>(m)) } {}

				// No check row-wise constructor
				template <class Row0, class Row1>
				JKL_GPU_EXECUTABLE constexpr GL2_elmt_base(forward_to_storage_tag,
					Row0&& r0, Row1&& r1, no_validity_check)
					noexcept(noexcept(base_type{ std::forward<Row0>(r0), std::forward<Row1>(r1) })) :
					base_type{ std::forward<Row0>(r0), std::forward<Row1>(r1) } {}

				// Checking row-wise constructor
				template <class Row0, class Row1>
				constexpr GL2_elmt_base(forward_to_storage_tag,
					Row0&& r0, Row1&& r1) :
					base_type{ check_and_forward(gl2_elmt<ComponentType, detail::tuple<Row0&&, Row1&&>,
						detail::row_ref_tuple_traits<StorageTraits, Row0, Row1>>{
					direct_construction{}, std::forward<Row0>(r0), std::forward<Row1>(r1) }) } {}

				template <class... Args>
				struct is_nothrow_constructible : std::false_type{};

				template <class Arg00, class Arg01, class Arg10, class Arg11>
				struct is_nothrow_constructible<Arg00, Arg01, Arg10, Arg11, no_validity_check>
				{
					static constexpr bool value = base_type::template is_nothrow_constructible<
						Arg00, Arg01, Arg10, Arg11>::value;
				};

				// No check component-wise constructor
				template <class Arg00, class Arg01, class Arg10, class Arg11>
				JKL_GPU_EXECUTABLE constexpr GL2_elmt_base(forward_to_storage_tag,
					Arg00&& arg00, Arg01&& arg01,
					Arg10&& arg10, Arg11&& arg11, no_validity_check)
					noexcept(noexcept(base_type{ std::forward<Arg00>(arg00), std::forward<Arg01>(arg01),
						std::forward<Arg10>(arg10), std::forward<Arg11>(arg11) })) :
					base_type{ std::forward<Arg00>(arg00), std::forward<Arg01>(arg01),
					std::forward<Arg10>(arg10), std::forward<Arg11>(arg11) } {}

				// Checking component-wise constructor
				template <class Arg00, class Arg01, class Arg10, class Arg11>
				constexpr GL2_elmt_base(forward_to_storage_tag,
					Arg00&& arg00, Arg01&& arg01,
					Arg10&& arg10, Arg11&& arg11) :
					base_type{ check_and_forward(gl2_elmt<ComponentType,
						detail::tuple<detail::tuple<Arg00&&, Arg01&&>, detail::tuple<Arg10&&, Arg11&&>>> {
					std::forward<Arg00>(arg00), std::forward<Arg01>(arg01),
					std::forward<Arg10>(arg10), std::forward<Arg11>(arg11) }) } {}
			};
		}

		template <class ComponentType, class Storage, class StorageTraits>
		class GL2_elmt : public tmp::generate_constructors<
			detail::GL2_elmt_base<ComponentType, Storage, StorageTraits>,
			detail::forward_to_storage_tag,
			tmp::copy_or_move_n<ComponentType, 4>,
			tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
			std::tuple<std::tuple<no_validity_check>>>>
		{
			using base_type = tmp::generate_constructors<
				detail::GL2_elmt_base<ComponentType, Storage, StorageTraits>,
				detail::forward_to_storage_tag,
				tmp::copy_or_move_n<ComponentType, 4>,
				tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
				std::tuple<std::tuple<no_validity_check>>>>;

			using gl2_elmt_type = gl2_elmt<ComponentType, Storage, StorageTraits>;

		public:
			using base_type::base_type;

			// No check row-wise constructor
			template <class Row0, class Row1>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(Row0&& r0, Row1&& r1, no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type, Row0, Row1>::value) :
				base_type{ detail::forward_to_storage_tag{},
				std::forward<Row0>(r0), std::forward<Row1>(r1), no_validity_check{} } {}

			// Checking row-wise constructor
			template <class Row0, class Row1>
			constexpr GL2_elmt(Row0&& r0, Row1&& r1) :
				base_type{ detail::forward_to_storage_tag{},
				std::forward<Row0>(r0), std::forward<Row1>(r1) } {}

			// No check call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(R2_elmt<ComponentType> r0,
				R2_elmt<ComponentType> r1, no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					R2_elmt<ComponentType>, R2_elmt<ComponentType>>::value) :
				base_type{ detail::forward_to_storage_tag{},
				std::move(r0), std::move(r1), no_validity_check{} } {}

			// Checking call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			constexpr GL2_elmt(R2_elmt<ComponentType> r0, R2_elmt<ComponentType> r1) :
				base_type{ detail::forward_to_storage_tag{}, std::move(r0), std::move(r1) } {}

			// Convert from GL2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL2_elmt,
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL2_elmt,
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), no_validity_check{} } {}

			// Convert from gl2_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), no_validity_check{} } {}

			// Convert from gl2_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL2_elmt(gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				base_type{ detail::forward_to_storage_tag{}, that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL2_elmt(gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that) } {}
			
			// Convert from sym2_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), no_validity_check{} } {}

			// Convert from sym2_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL2_elmt(sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				base_type{ detail::forward_to_storage_tag{}, that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr GL2_elmt(sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that) } {}

			// Convert from posdef2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				posdef2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(
				posdef2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<gl2_elmt_type,
					sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), no_validity_check{} } {}


			// Copy and move
			GL2_elmt(GL2_elmt const&) = default;
			GL2_elmt(GL2_elmt&&) = default;
			GL2_elmt& operator=(GL2_elmt const&) & = default;
			GL2_elmt& operator=(GL2_elmt&&) & = default;

		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& assign_no_check_impl(Matrix&& m)
				noexcept(noexcept(static_cast<gl2_elmt_type&>(*this) = std::forward<Matrix>(m)))
			{
				static_cast<gl2_elmt_type&>(*this) = std::forward<Matrix>(m);
				return *this;
			}

		public:
			// Assignment from GL2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL2_elmt,
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<GL2_elmt,
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from gl2_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& assign_no_check(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& assign_no_check(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from gl2_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL2_elmt>{ "jkj::math: the matrix is not invertible" };
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL2_elmt>{ "jkj::math: the matrix is not invertible" };
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from sym2_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& assign_no_check(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& assign_no_check(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from sym2_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL2_elmt>{ "jkj::math: the matrix is not invertible" };
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				sym2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_invertible() )
					throw input_validity_error<GL2_elmt>{ "jkj::math: the matrix is not invertible" };
				return assign_no_check_impl(std::move(that));
			}

			// Assignment from posdef2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				posdef2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check_impl(that)))
			{
				return assign_no_check_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator=(
				posdef2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check_impl(std::move(that))))
			{
				return assign_no_check_impl(std::move(that));
			}


			// Remove mutable lvalue element accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<base_type const&>().template get<I>()))
			{
				return static_cast<base_type const&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<base_type&&>().template get<I>()))
			{
				return static_cast<base_type&&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<base_type const&&>().template get<I>()))
			{
				return static_cast<base_type const&&>(*this).template get<I>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<base_type const&>().template get<I, J>()))
			{
				return static_cast<base_type const&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<base_type&&>().template get<I, J>()))
			{
				return static_cast<base_type&&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<base_type const&&>().template get<I, J>()))
			{
				return static_cast<base_type const&&>(*this).template get<I, J>();
			}

			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const&
				noexcept(noexcept(std::declval<base_type const&>()[idx]))
				-> decltype(std::declval<base_type const&>()[idx])
			{
				return static_cast<base_type const&>(*this)[idx];
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) &&
				noexcept(noexcept(std::declval<base_type&&>()[idx]))
				-> decltype(std::declval<base_type&&>()[idx])
			{
				return static_cast<base_type&&>(*this)[idx];
			}
			template <class dummy = void>
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const&&
				noexcept(noexcept(std::declval<base_type const&&>()[idx]))
				-> decltype(std::declval<base_type const&&>()[idx])
			{
				return static_cast<base_type const&&>(*this)[idx];
			}


			JKL_GPU_EXECUTABLE constexpr GL2_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<GL2_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL2_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<GL2_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr GL2_elmt operator-() const&
				noexcept(noexcept(gl2_elmt_type{ -static_cast<gl2_elmt_type const&>(*this) }))
			{
				return{ -static_cast<gl2_elmt_type const&>(*this), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL2_elmt operator-() &&
				noexcept(noexcept(gl2_elmt_type{ -static_cast<gl2_elmt_type&&>(*this) }))
			{
				return{ -static_cast<gl2_elmt_type&&>(*this), no_validity_check{} };
			}

			// Remove += and -= operators
		private:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt_type& operator+=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that);
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt_type& operator+=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that);

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt_type& operator-=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that);
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR gl2_elmt_type& operator-=(
				gl2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that);

		public:
			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<gl2_elmt_type&>(*this) *= k))
			{
				static_cast<gl2_elmt_type&>(*this) *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<gl2_elmt_type&>(*this) /= k))
			{
				static_cast<gl2_elmt_type&>(*this) /= k;
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator*=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(static_cast<gl2_elmt_type&>(*this) *= that))
			{
				static_cast<gl2_elmt_type&>(*this) *= that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator*=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(static_cast<gl2_elmt_type&>(*this) *= std::move(that)))
			{
				static_cast<gl2_elmt_type&>(*this) *= std::move(that);
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr GL2_elmt t() const&
				noexcept(noexcept(static_cast<gl2_elmt_type const&>(*this).t()))
			{
				return{ static_cast<gl2_elmt_type const&>(*this).t(), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR GL2_elmt t() &&
				noexcept(noexcept(static_cast<gl2_elmt_type&&>(*this).t()))
			{
				return{ static_cast<gl2_elmt_type&&>(*this).t(), no_validity_check{} };
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const noexcept
			{
				return true;
			}


			// Division
		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR GL2_elmt inv_impl(Matrix&& m) noexcept(
				noexcept(gl2_elmt_type{
					std::forward<Matrix>(m).template get<1, 1>(),
					-std::forward<Matrix>(m).template get<0, 1>(),
					-std::forward<Matrix>(m).template get<1, 0>(),
					std::forward<Matrix>(m).template get<0, 0>() }) &&
					noexcept(std::declval<GL2_elmt&>() /= std::declval<GL2_elmt&>().det()))
			{
				GL2_elmt ret_value{
					std::forward<Matrix>(m).template get<1, 1>(),
					-std::forward<Matrix>(m).template get<0, 1>(),
					-std::forward<Matrix>(m).template get<1, 0>(),
					std::forward<Matrix>(m).template get<0, 0>(),
					no_validity_check{} };
				ret_value /= ret_value.det();
				return ret_value;
			}

		public:
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt inv() const&
				noexcept(noexcept(inv_impl(*this)))
			{
				return inv_impl(*this);
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt inv() &&
				noexcept(noexcept(inv_impl(std::move(*this))))
			{
				return inv_impl(std::move(*this));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator/=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(*this *= that.inv()))
			{
				return *this *= that.inv();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR GL2_elmt& operator/=(
				GL2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(*this *= std::move(that)))
			{
				return *this *= std::move(that).inv();
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(GL2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(GL2_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(GL2_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.inv()))
		{
			return m.inv();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(GL2_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).inv()))
		{
			return std::move(m).inv();
		}


		//// Binary operations for 2D matrices

		namespace detail {
			template <class T, template <class, class, class> class Template, class = void>
			struct get_gl2_elmt_impl : std::false_type {
				using type = void;
			};

			template <class T, template <class, class, class> class Template>
			struct get_gl2_elmt_impl<T, Template, VOID_T<
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
			using get_gl2_elmt = get_gl2_elmt_impl<T, gl2_elmt>;

			template <class T>
			using get_GL2_elmt = get_gl2_elmt_impl<T, GL2_elmt>;


			template <class LeftOperand, class RightOperand>
			struct get_gl2_elmt_binary_result_impl : std::conditional_t<
				get_gl2_elmt<LeftOperand>::value, std::conditional_t<
				get_gl2_elmt<RightOperand>::value, get_gl2_elmt_binary_result_impl<
				typename get_gl2_elmt<LeftOperand>::type,
				typename get_gl2_elmt<RightOperand>::type>,
				empty_type>, empty_type> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl2_elmt_binary_result_impl<
				gl2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				gl2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = gl2_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl2_elmt_binary_result = typename get_gl2_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;

			
			template <class LeftOperand, class RightOperand>
			struct get_gl2_elmt_mult_result_impl :
				get_gl2_elmt_binary_result_impl<LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl2_elmt_mult_result_impl<
				GL2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL2_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl2_elmt_mult_result = typename get_gl2_elmt_mult_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class LeftOperand, class RightOperand>
			struct get_gl2_elmt_div_result_impl {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl2_elmt_div_result_impl<
				gl2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = gl2_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_gl2_elmt_div_result_impl<
				GL2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = GL2_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_gl2_elmt_div_result = typename get_gl2_elmt_div_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;
			

			template <class Scalar, class Matrix, bool from_left>
			struct get_gl2_elmt_scalar_mult_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_gl2_elmt_scalar_mult_result_impl_impl<Scalar,
				gl2_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = gl2_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_gl2_elmt_scalar_mult_result_impl_impl<Scalar,
				GL2_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = GL2_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, class Matrix, bool from_left>
			struct get_gl2_elmt_scalar_mult_result_impl : std::conditional_t<
				get_gl2_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>::value,
				get_gl2_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>,
				get_gl2_elmt_scalar_mult_result_impl_impl<void, void, false>> {};

			template <class Scalar, class Matrix, bool from_left>
			using get_gl2_elmt_scalar_mult_result = typename get_gl2_elmt_scalar_mult_result_impl<
				tmp::remove_cvref_t<Scalar>,
				tmp::remove_cvref_t<Matrix>, from_left>::type;


			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<gl2_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = gl2_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args...>::value)
				{
					return{ std::forward<Args>(args)... };
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<GL2_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = GL2_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args..., no_validity_check>::value)
				{
					return{ std::forward<Args>(args)..., no_validity_check{} };
				}
			};
		}

		// Binary addition of gl2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator+(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(a).template get<0, 0>() +
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() +
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<1, 0>() +
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() +
				std::forward<RightOperand>(b).template get<1, 1>() }))
			-> detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot add two gl2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot add two gl2_elmt's; failed to deduce the resulting storage type");

			return{ std::forward<LeftOperand>(a).template get<0, 0>() +
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() +
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<1, 0>() +
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() +
				std::forward<RightOperand>(b).template get<1, 1>()
			};
		}

		// Binary subtraction of gl2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator-(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(a).template get<0, 0>() -
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() -
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<1, 0>() -
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() -
				std::forward<RightOperand>(b).template get<1, 1>() }))
			-> detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl2_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot subtract two gl2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot subtract two gl2_elmt's; failed to deduce the resulting storage type");

			return{ std::forward<LeftOperand>(a).template get<0, 0>() -
				std::forward<RightOperand>(b).template get<0, 0>(),
				std::forward<LeftOperand>(a).template get<0, 1>() -
				std::forward<RightOperand>(b).template get<0, 1>(),
				std::forward<LeftOperand>(a).template get<1, 0>() -
				std::forward<RightOperand>(b).template get<1, 0>(),
				std::forward<LeftOperand>(a).template get<1, 1>() -
				std::forward<RightOperand>(b).template get<1, 1>()
			};
		}

		// Binary multiplication of gl2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator*(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl2_elmt_mult_result<LeftOperand, RightOperand>>::make(
				a.template get<0, 0>() * b.template get<0, 0>() +
				a.template get<0, 1>() * b.template get<1, 0>(),
				a.template get<0, 0>() * b.template get<0, 1>() +
				a.template get<0, 1>() * b.template get<1, 1>(),
				a.template get<1, 0>() * b.template get<0, 0>() +
				a.template get<1, 1>() * b.template get<1, 0>(),
				a.template get<1, 0>() * b.template get<0, 1>() +
				a.template get<1, 1>() * b.template get<1, 1>())))
			-> detail::get_gl2_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl2_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot multiply two gl2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply two gl2_elmt's; failed to deduce the resulting storage type");
			
			return detail::call_unchecking<result_type>::make(
				a.template get<0, 0>() * b.template get<0, 0>() +
				a.template get<0, 1>() * b.template get<1, 0>(),
				a.template get<0, 0>() * b.template get<0, 1>() +
				a.template get<0, 1>() * b.template get<1, 1>(),
				a.template get<1, 0>() * b.template get<0, 0>() +
				a.template get<1, 1>() * b.template get<1, 0>(),
				a.template get<1, 0>() * b.template get<0, 1>() +
				a.template get<1, 1>() * b.template get<1, 1>());
		}


		// Binary division of gl2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator/(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv()))
			-> detail::get_gl2_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_gl2_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot divide two gl2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot divide two gl2_elmt's; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv();
		}
		

		// Scalar multiplication of gl2_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
			std::forward<Matrix>(m).template get<0, 0>() * k,
				std::forward<Matrix>(m).template get<0, 1>() * k,
				std::forward<Matrix>(m).template get<1, 0>() * k,
				std::forward<Matrix>(m).template get<1, 1>() * k)))
			-> detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply gl2_elmt with a scalar; failed to deduce the resulting storage type");
			
			using component_type = typename result_type::component_type;
			assert(!detail::get_GL2_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).template get<0, 0>() * k,
				std::forward<Matrix>(m).template get<0, 1>() * k,
				std::forward<Matrix>(m).template get<1, 0>() * k,
				std::forward<Matrix>(m).template get<1, 1>() * k);
		}

		// Scalar multiplication of gl2_elmt's from left
		template <class Scalar, class Matrix>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Scalar const& k, Matrix&& m)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, true>>::make(
			k * std::forward<Matrix>(m).template get<0, 0>(),
				k * std::forward<Matrix>(m).template get<0, 1>(),
				k * std::forward<Matrix>(m).template get<1, 0>(), 
				k * std::forward<Matrix>(m).template get<1, 1>())))
			-> detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, true>
		{
			using result_type = detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot multiply gl2_elmt with a scalar; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply gl2_elmt with a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_GL2_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				k * std::forward<Matrix>(m).template get<0, 0>(),
				k * std::forward<Matrix>(m).template get<0, 1>(),
				k * std::forward<Matrix>(m).template get<1, 0>(),
				k * std::forward<Matrix>(m).template get<1, 1>());
		}

		// Scalar division of gl2_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
			std::forward<Matrix>(m).template get<0, 0>() / k,
				std::forward<Matrix>(m).template get<0, 1>() / k,
				std::forward<Matrix>(m).template get<1, 0>() / k,
				std::forward<Matrix>(m).template get<1, 1>() / k)))
			-> detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_gl2_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot divide gl2_elmt by a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_GL2_elmt<result_type>::value || is_invertible(component_type(k)));

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).template get<0, 0>() / k,
				std::forward<Matrix>(m).template get<0, 1>() / k,
				std::forward<Matrix>(m).template get<1, 0>() / k,
				std::forward<Matrix>(m).template get<1, 1>() / k);
		}
	}
}
