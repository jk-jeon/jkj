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
#include <cmath>
#include <initializer_list>
#include "../tmp/is_braces_constructible.h"
#include "general.h"

namespace jkj {
	namespace math {
		namespace detail {
			template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
			struct Rn_elmt_base {
				using storage_type = Storage;
				using storage_traits = StorageTraits;

			protected:
				using storage_wrapper = typename StorageTraits::template storage_wrapper<
					Storage, Rn_elmt<N, ComponentType, Storage, StorageTraits>>;
				storage_wrapper r_;

			public:
				Rn_elmt_base() = default;

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr Rn_elmt_base(direct_construction, Args&&... args)
					noexcept(std::is_nothrow_constructible<storage_wrapper, Args...>::value) :
					r_(std::forward<Args>(args)...) {}

			protected:
				template <class... Args>
				struct is_nothrow_constructible {
					static constexpr bool value =
						tmp::is_nothrow_braces_constructible<storage_wrapper, Args...>::value;
				};

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr Rn_elmt_base(forward_to_storage_tag, Args&&... args)
					noexcept(is_nothrow_constructible<Args...>::value) :
					r_{ std::forward<Args>(args)... } {}
			};
		}
		
		// N-dimensional Eulclidean space
		template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
		class Rn_elmt {
		public:
			using component_type = ComponentType;
			static constexpr std::size_t components = N;
			using storage_type = Storage;
			using storage_traits = StorageTraits;

		private:
			using storage_wrapper = typename StorageTraits::template storage_wrapper<Storage, Rn_elmt>;
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

			// Rn_elmt requires array operator access
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_index<storage_type const&, std::size_t>::value,
				"jkj::math: Rn_elmt requires array-like access to the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// Tuple-style accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}

			// Array-style accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx)))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx);
			}
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx)))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(storage(), idx);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}
			JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(std::declval<Storage const&>(), 0)),
				ComponentType const&>::value, "jkj::math: Rn_elmt requires array-like access to the storage; "
				"the array-like access deduced from the given storage traits cannot be converted to the component type");

			// Default constructor; components might be filled with garbages
			Rn_elmt() = default;

			// Construct the storage directly
			template <class... Args>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(direct_construction, Args&&... args)
				noexcept(std::is_nothrow_constructible<storage_wrapper, Args...>::value)
				: r_(std::forward<Args>(args)...) {}

			// Construction from initilizer-list
			// - Lists with length longer than N are trimmed
			// - Undefined behaviour if the list has shorter length
			GENERALIZED_CONSTEXPR Rn_elmt(std::initializer_list<ComponentType> list)
				noexcept(std::is_nothrow_default_constructible<storage_wrapper>::value &&
					noexcept((*this)[0] = std::declval<ComponentType&>()))
			{
				auto itr = list.begin();
				for( std::size_t i = 0; i < N; ++i, ++itr )
					(*this)[i] = *itr;
			}

			// Convert from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt(
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_default_constructible<storage_wrapper>::value &&
					noexcept((*this)[0] = that[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] = that[i];
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt(
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_default_constructible<storage_wrapper>::value &&
					noexcept((*this)[0] = std::move(that)[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] = std::move(that)[i];
			}

			// Copy and move
			Rn_elmt(Rn_elmt const&) = default;
			Rn_elmt(Rn_elmt&&) = default;
			Rn_elmt& operator=(Rn_elmt const&) & = default;
			Rn_elmt& operator=(Rn_elmt&&) & = default;

			// Assignment from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept((*this)[0] = that[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] = that[i];
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept((*this)[0] = std::move(that)[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] = std::move(that)[i];
				return *this;
			}


			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR ComponentType normsq() const
				noexcept(noexcept(jkj::math::zero<ComponentType>()) &&
					noexcept(std::declval<ComponentType&>() += (*this)[0] * (*this)[0]))
			{
				auto sum = jkj::math::zero<ComponentType>();
				for( std::size_t i = 0; i < N; ++i )
					sum += (*this)[i] * (*this)[i];
				return sum;
			}

		private:
			static constexpr bool check_norm_noexcept() {
				using std::sqrt;
				return noexcept(sqrt(std::declval<Rn_elmt>().normsq()));
			}
		public:
			JKL_GPU_EXECUTABLE decltype(auto) norm() const
				noexcept(check_norm_noexcept())
			{
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<Rn_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<Rn_elmt>::value)
			{
				return std::move(*this);
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt operator-() const&
				noexcept(noexcept(zero()) && noexcept(std::declval<Rn_elmt&>() = -(*this)[0]))
			{
				auto ret_value = zero();
				for( std::size_t i = 0; i < N; ++i )
					ret_value[i] = -(*this)[i];
				return ret_value;
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt operator-() &&
				noexcept(noexcept(zero()) && noexcept(std::declval<Rn_elmt&>() = -std::move(*this)[0]))
			{
				auto ret_value = zero();
				for( std::size_t i = 0; i < N; ++i )
					ret_value[i] = -std::move(*this)[i];
				return ret_value;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt&
				operator+=(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept((*this)[0] += that[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] += that[i];
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt&
				operator+=(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept((*this)[0] += std::move(that)[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] += std::move(that)[i];
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt&
				operator-=(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept((*this)[0] -= that[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] -= that[i];
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt&
				operator-=(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept((*this)[0] -= std::move(that)[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] -= std::move(that)[i];
				return *this;
			}

			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept((*this)[0] = k))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] *= k;
				return *this;
			}
			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept((*this)[0] /= k))
			{
				for( std::size_t i = 0; i < N; ++i )
					(*this)[i] /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE Rn_elmt& normalize() noexcept(noexcept(operator/=(norm())))
			{
				return operator/=(norm());
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR bool
				operator==(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& v) const
				noexcept(noexcept((*this)[0] != v[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					if( (*this)[i] != v[i] )
						return false;
				return true;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR bool
				operator!=(Rn_elmt<N, OtherComponentType, OtherStorage, OtherStorageTraits> const& v) const
				noexcept(noexcept((*this)[0] != v[0]))
			{
				for( std::size_t i = 0; i < N; ++i )
					if( (*this)[i] != v[i] )
						return true;
				return false;
			}

			JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR Rn_elmt zero()
				noexcept(noexcept(std::declval<Rn_elmt&>()[0] = jkj::math::zero<ComponentType>()))
			{
				Rn_elmt ret;
				for( std::size_t i = 0; i < N; ++i )
					ret[i] = jkj::math::zero<ComponentType>();
				return ret;
			}
		};
		
		// Copy or move a storage into an Rn_elmt
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <std::size_t N, class ComponentType = deduce_type_tag,
			class StorageTraits = default_storage_traits, class Storage>
		Rn_elmt<N, detail::deduced_1d_component_type<N, ComponentType, Storage, StorageTraits>,
			tmp::remove_cvref_t<Storage>, StorageTraits>
			make_Rn_elmt(Storage&& rn_like_storage)
			noexcept(std::is_nothrow_constructible<Rn_elmt<N,
				detail::deduced_1d_component_type<N, ComponentType, Storage, StorageTraits>,
				tmp::remove_cvref_t<Storage>, StorageTraits>, direct_construction, Storage>::value)
		{
			return{ direct_construction{}, std::forward<Storage>(rn_like_storage) };
		}

		// Create an Rn_elmt with the standard (built-in array) storage
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag, class... GivenTypes>
		Rn_elmt<sizeof...(GivenTypes), detail::deduced_component_type_from_args<ComponentType, GivenTypes...>>
			make_Rn_elmt(GivenTypes&&... components)
			noexcept(std::is_nothrow_constructible<Rn_elmt<sizeof...(GivenTypes),
				detail::deduced_component_type_from_args<ComponentType, GivenTypes...>>, GivenTypes...>::value)
		{
			return{ std::forward<GivenTypes>(components)... };
		}


		// Specialization for N = 2
		template <class ComponentType, class Storage, class StorageTraits>
		class Rn_elmt<2, ComponentType, Storage, StorageTraits> :
			public detail::constructor_provider<2, ComponentType,
			detail::Rn_elmt_base<2, ComponentType, Storage, StorageTraits>>
		{
			using constructor_provider = detail::constructor_provider<2, ComponentType,
				detail::Rn_elmt_base<2, ComponentType, Storage, StorageTraits>>;

		public:
			using constructor_provider::storage;
			using constructor_provider::constructor_provider;

			// R2_elmt requires x & y component access
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkj::math: R2_elmt requires access to the x-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkj::math: R2_elmt requires access to the y-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// x-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R2_elmt requires access to the x-component from the storage; "
				"the x-component deduced from the given storage traits cannot be converted to the given component type");

			// y-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R2_elmt requires access to the y-component from the storage; "
				"the y-component deduced from the given storage traits cannot be converted to the given component type");

			// Tuple-style accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}

			// Array-style accessors
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
				
				static constexpr bool value = std::is_convertible<type, ComponentType const&>::value;
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&, dummy>
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&&, dummy>
				operator[](std::size_t idx) const&& noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}


			// Default constructor; components might be filled with garbages
			Rn_elmt() = default;
			
			// Convert from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.x(), that.y()))) :
				constructor_provider(that.x(), that.y()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(std::move(that).x(), std::move(that).y()))) :
				constructor_provider(std::move(that).x(), std::move(that).y()) {}


			// Copy and move
			Rn_elmt(Rn_elmt const&) = default;
			Rn_elmt(Rn_elmt&&) = default;
			Rn_elmt& operator=(Rn_elmt const&) & = default;
			Rn_elmt& operator=(Rn_elmt&&) & = default;

			// Assignment from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(x() = that.x()) && noexcept(y() = that.y()))
			{
				x() = that.x();
				y() = that.y();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(x() = std::move(that).x()) && noexcept(y() = std::move(that).y()))
			{
				x() = std::move(that).x();
				y() = std::move(that).y();
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr decltype(auto) normsq() const
				noexcept(noexcept(x() * x() + y() * y()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (x() * x() + y() * y());
			}

		private:
			static constexpr bool check_norm_noexcept() {
				using std::sqrt;
				return noexcept(sqrt(std::declval<Rn_elmt>().normsq()));
			}
		public:
			JKL_GPU_EXECUTABLE decltype(auto) norm() const noexcept(check_norm_noexcept()) {
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<Rn_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<Rn_elmt>::value)
			{
				return std::move(*this);
			}
			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator-() const&
				noexcept(std::is_nothrow_constructible<Rn_elmt, decltype(-x()), decltype(-y())>::value)
			{
				return{ -x(), -y() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator-() &&
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(-std::move(*this).x()), decltype(-std::move(*this).y())>::value)
			{
				return{ -std::move(*this).x(), -std::move(*this).y() };
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() += that.x()) && noexcept(y() += that.y()))
			{
				x() += that.x();
				y() += that.y();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() += std::move(that).x()) && noexcept(y() += std::move(that).y()))
			{
				x() += std::move(that).x();
				y() += std::move(that).y();
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() -= that.x()) && noexcept(y() -= that.y()))
			{
				x() -= that.x();
				y() -= that.y();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() -= std::move(that).x()) && noexcept(y() -= std::move(that).y()))
			{
				x() -= std::move(that).x();
				y() -= std::move(that).y();
				return *this;
			}

			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(x() *= k) && noexcept(y() *= k))
			{
				x() *= k;
				y() *= k;
				return *this;
			}
			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(x() /= k) && noexcept(y() /= k))
			{
				x() /= k;
				y() /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE Rn_elmt& normalize() noexcept(noexcept(operator/=(norm())))
			{
				return operator/=(norm());
			}

			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				R2_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(x() == v.x()) && noexcept(y() == v.y()))
			{
				return x() == v.x() && y() == v.y();
			}
			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				R2_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(!(*this == v)))
			{
				return !(*this == v);
			}

			JKL_GPU_EXECUTABLE static constexpr Rn_elmt zero()
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>())>::value)
			{
				return{ jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>() };
			}
		};

		//// Convenient make functions for R2_elmt

		// Copy or move a storage into an R2_elmt
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag,
			class StorageTraits = default_storage_traits, class Storage>
			R2_elmt<detail::deduced_1d_component_type<2, ComponentType, Storage, StorageTraits>, Storage, StorageTraits>
			make_R2_elmt(Storage&& r2_like_storage)
			noexcept(std::is_nothrow_constructible<R2_elmt<
				detail::deduced_1d_component_type<2, ComponentType, Storage, StorageTraits>,
				tmp::remove_cvref_t<Storage>, StorageTraits>, direct_construction, Storage>::value)
		{
			return{ direct_construction{}, std::forward<Storage>(r2_like_storage) };
		}

		// Create an R2_elmt with the standard (built-in array) storage
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag, class ArgX, class ArgY>
		R2_elmt<detail::deduced_component_type_from_args<ComponentType, ArgX, ArgY>>
			make_R2_elmt(ArgX&& x, ArgY&& y)
			noexcept(std::is_nothrow_constructible<R2_elmt<detail::deduced_component_type_from_args<ComponentType, ArgX, ArgY>>,
				ArgX, ArgY>::value)
		{
			return{ std::forward<ArgX>(x), std::forward<ArgY>(y) };
		}


		// Specialization for N = 3
		template <class ComponentType, class Storage, class StorageTraits>
		class Rn_elmt<3, ComponentType, Storage, StorageTraits> :
			public detail::constructor_provider<3, ComponentType,
			detail::Rn_elmt_base<3, ComponentType, Storage, StorageTraits>>
		{
			using constructor_provider = detail::constructor_provider<3, ComponentType,
				detail::Rn_elmt_base<3, ComponentType, Storage, StorageTraits>>;

		public:
			using constructor_provider::storage;
			using constructor_provider::constructor_provider;

			// R3_elmt requires x & y & z component access
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkj::math: R3_elmt requires access to the x-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkj::math: R3_elmt requires access to the y-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, Storage const&>::value,
				"jkj::math: R3_elmt requires access to the z-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// x-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R3_elmt requires access to the x-component from the storage; "
				"the x-component deduced from the given storage traits cannot be converted to the component type");

			// y-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R3_elmt requires access to the y-component from the storage; "
				"the y-component deduced from the given storage traits cannot be converted to the component type");

			// z-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto z() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto z() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto z() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto z() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R3_elmt requires access to the z-component from the storage; "
				"the z-component deduced from the given storage traits cannot be converted to the component type");

			// Tuple-style accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}

			// Array-style accessors
		private:
			template <class StorageType, class = void, bool can_index =
				detail::storage_traits_inspector<StorageTraits>::template can_index<StorageType, std::size_t>::value>
			struct has_array_operator : std::false_type {
				using type = void;
			};

			template <class StorageType, class dummy>
			struct has_array_operator<StorageType, dummy, true> {
				using type = decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(
					std::declval<StorageType>(), std::size_t(0)));

				static constexpr bool value = std::is_convertible<type, ComponentType const&>::value;
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&, dummy>
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&&, dummy>
				operator[](std::size_t idx) const&& noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}


			// Default constructor; components might be filled with garbages
			Rn_elmt() = default;
			
			// Convert from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
					R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.x(), that.y(), that.z()))) :
				constructor_provider(that.x(), that.y(), that.z()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(std::move(that).x(), std::move(that).y(), std::move(that).z()))) :
				constructor_provider(std::move(that).x(), std::move(that).y(), std::move(that).z()) {}


			// Copy and move
			Rn_elmt(Rn_elmt const&) = default;
			Rn_elmt(Rn_elmt&&) = default;
			Rn_elmt& operator=(Rn_elmt const&) & = default;
			Rn_elmt& operator=(Rn_elmt&&) & = default;

			// Assignment from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(x() = that.x()) && noexcept(y() = that.y()) && noexcept(z() = that.z()))
			{
				x() = that.x();
				y() = that.y();
				z() = that.z();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(x() = std::move(that).x()) &&
					noexcept(y() = std::move(that).y()) &&
					noexcept(z() = std::move(that).z()))
			{
				x() = std::move(that).x();
				y() = std::move(that).y();
				z() = std::move(that).z();
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr decltype(auto) normsq() const
				noexcept(noexcept(x() * x() + y() * y() + z() * z()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (x() * x() + y() * y() + z() * z());
			}

		private:
			static constexpr bool check_norm_noexcept() {
				using std::sqrt;
				return noexcept(sqrt(std::declval<Rn_elmt>().normsq()));
			}
		public:
			JKL_GPU_EXECUTABLE decltype(auto) norm() const noexcept(check_norm_noexcept()) {
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<Rn_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<Rn_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator-() const&
				noexcept(std::is_nothrow_constructible<Rn_elmt, decltype(-x()), decltype(-y()), decltype(-z())>::value)
			{
				return{ -x(), -y(), -z() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator-() &&
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(-std::move(*this).x()),
					decltype(-std::move(*this).y()),
					decltype(-std::move(*this).z())>::value)
			{
				return{ -std::move(*this).x(), -std::move(*this).y(), -std::move(*this).z() };
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() += that.x())
					&& noexcept(y() += that.y())
					&& noexcept(z() += that.z()))
			{
				x() += that.x();
				y() += that.y();
				z() += that.z();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() += std::move(that).x())
					&& noexcept(y() += std::move(that).y())
					&& noexcept(z() += std::move(that).z()))
			{
				x() += std::move(that).x();
				y() += std::move(that).y();
				z() += std::move(that).z();
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() -= that.x())
					&& noexcept(y() -= that.y())
					&& noexcept(z() -= that.z()))
			{
				x() -= that.x();
				y() -= that.y();
				z() -= that.z();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() -= std::move(that).x())
					&& noexcept(y() -= std::move(that).y())
					&& noexcept(z() -= std::move(that).z()))
			{
				x() -= std::move(that).x();
				y() -= std::move(that).y();
				z() -= std::move(that).z();
				return *this;
			}

			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(x() *= k) && noexcept(y() *= k) && noexcept(z() *= k))
			{
				x() *= k;
				y() *= k;
				z() *= k;
				return *this;
			}
			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(x() /= k) && noexcept(y() /= k) && noexcept(z() /= k))
			{
				x() /= k;
				y() /= k;
				z() /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE Rn_elmt& normalize() noexcept(noexcept(operator/=(norm())))
			{
				return operator/=(norm());
			}

			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				R3_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(x() == v.x()) && noexcept(y() == v.y()) && noexcept(z() == v.z()))
			{
				return x() == v.x() && y() == v.y() && z() == v.z();
			}
			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				R3_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(!(*this == v)))
			{
				return !(*this == v);
			}

			JKL_GPU_EXECUTABLE static constexpr Rn_elmt zero()
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>())>::value)
			{
				return{ jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>() };
			}
		};

		//// Convenient make functions for R3_elmt

		// Copy or move a storage into an R3_elmt
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag,
			class StorageTraits = default_storage_traits, class Storage>
			R3_elmt<detail::deduced_1d_component_type<3, ComponentType, Storage, StorageTraits>, Storage, StorageTraits>
			make_R3_elmt(Storage&& r3_like_storage)
			noexcept(std::is_nothrow_constructible<R3_elmt<
				detail::deduced_1d_component_type<3, ComponentType, Storage, StorageTraits>,
				tmp::remove_cvref_t<Storage>, StorageTraits>, direct_construction, Storage>::value)
		{
			return{ direct_construction{}, std::forward<Storage>(r3_like_storage) };
		}

		// Create an R3_elmt with the standard (built-in array) storage
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag, class ArgX, class ArgY, class ArgZ>
		R3_elmt<detail::deduced_component_type_from_args<ComponentType, ArgX, ArgY, ArgZ>>
			make_R3_elmt(ArgX&& x, ArgY&& y, ArgZ&& z)
			noexcept(std::is_nothrow_constructible<R3_elmt<detail::deduced_component_type_from_args<ComponentType, ArgX, ArgY, ArgZ>>,
				ArgX, ArgY, ArgZ>::value)
		{
			return{ std::forward<ArgX>(x), std::forward<ArgY>(y), std::forward<ArgZ>(z) };
		}


		// Specialization for N = 4
		template <class ComponentType, class Storage, class StorageTraits>
		class Rn_elmt<4, ComponentType, Storage, StorageTraits> :
			public detail::constructor_provider<4, ComponentType,
			detail::Rn_elmt_base<4, ComponentType, Storage, StorageTraits>>
		{
			using constructor_provider = detail::constructor_provider<4, ComponentType,
				detail::Rn_elmt_base<4, ComponentType, Storage, StorageTraits>>;

		public:
			using constructor_provider::storage;
			using constructor_provider::constructor_provider;

			// R4_elmt requires x & y & z & w component access
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkj::math: R4_elmt requires access to the x-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkj::math: R4_elmt requires access to the y-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, Storage const&>::value,
				"jkj::math: R4_elmt requires access to the z-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<3, Storage const&>::value,
				"jkj::math: R4_elmt requires access to the w-component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// x-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto x() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto x() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R4_elmt requires access to the x-component from the storage; "
				"the x-component deduced from the given storage traits cannot be converted to the component type");

			// y-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto y() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto y() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R4_elmt requires access to the y-component from the storage; "
				"the y-component deduced from the given storage traits cannot be converted to the component type");

			// z-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto z() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto z() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto z() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto z() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R4_elmt requires access to the z-component from the storage; "
				"the z-component deduced from the given storage traits cannot be converted to the component type");

			// w-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto w() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto w() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto w() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto w() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkj::math: R4_elmt requires access to the w-component from the storage; "
				"the w-component deduced from the given storage traits cannot be converted to the component type");

			// Tuple-style accessors
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<I>(std::move(*this).storage());
			}

			// Array-style accessors
		private:
			template <class StorageType, class = void, bool can_index =
				detail::storage_traits_inspector<StorageTraits>::template can_index<StorageType, std::size_t>::value>
				struct has_array_operator : std::false_type {
				using type = void;
			};

			template <class StorageType, class dummy>
			struct has_array_operator<StorageType, dummy, true> {
				using type = decltype(detail::storage_traits_inspector<StorageTraits>::array_operator(
					std::declval<StorageType>(), std::size_t(0)));

				static constexpr bool value = std::is_convertible<type, ComponentType const&>::value;
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&, dummy>
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR array_operator_return_t<Storage const&&, dummy>
				operator[](std::size_t idx) const&& noexcept(noexcept(
					detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx)))
			{
				return detail::storage_traits_inspector<StorageTraits>::array_operator(std::move(*this).storage(), idx);
			}


			// Default constructor; components might be filled with garbages
			Rn_elmt() = default;

			// Convert from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.x(), that.y(), that.z(), that.w()))) :
				constructor_provider(that.x(), that.y(), that.z(), that.w()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr Rn_elmt(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(std::move(that).x(), std::move(that).y(),
					std::move(that).z(), std::move(that).w()))) :
				constructor_provider(std::move(that).x(), std::move(that).y(),
					std::move(that).z(), std::move(that).w()) {}


			// Copy and move
			Rn_elmt(Rn_elmt const&) = default;
			Rn_elmt(Rn_elmt&&) = default;
			Rn_elmt& operator=(Rn_elmt const&) & = default;
			Rn_elmt& operator=(Rn_elmt&&) & = default;

			// Assignment from vector of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(x() = that.x()) && noexcept(y() = that.y()) &&
					noexcept(z() = that.z()) && noexcept(w() = that.w()))
			{
				x() = that.x();
				y() = that.y();
				z() = that.z();
				w() = that.w();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<Rn_elmt,
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(x() = std::move(that).x()) &&
					noexcept(y() = std::move(that).y()) &&
					noexcept(z() = std::move(that).z()) &&
					noexcept(w() = std::move(that).w()))
			{
				x() = std::move(that).x();
				y() = std::move(that).y();
				z() = std::move(that).z();
				w() = std::move(that).w();
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr decltype(auto) normsq() const
				noexcept(noexcept(x() * x() + y() * y() + z() * z() + w() * w()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (x() * x() + y() * y() + z() * z() + w() * w());
			}

		private:
			static constexpr bool check_norm_noexcept() {
				using std::sqrt;
				return noexcept(sqrt(std::declval<Rn_elmt>().normsq()));
			}
		public:
			JKL_GPU_EXECUTABLE decltype(auto) norm() const noexcept(check_norm_noexcept()) {
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<Rn_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<Rn_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator-() const&
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(-x()), decltype(-y()), decltype(-z()), decltype(-w())>::value)
			{
				return{ -x(), -y(), -z(), -w() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR Rn_elmt operator-() &&
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(-std::move(*this).x()),
					decltype(-std::move(*this).y()),
					decltype(-std::move(*this).z()),
					decltype(-std::move(*this).w())>::value)
			{
				return{ -std::move(*this).x(), -std::move(*this).y(),
					-std::move(*this).z(), -std::move(*this).w() };
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() += that.x())
					&& noexcept(y() += that.y())
					&& noexcept(z() += that.z())
					&& noexcept(w() += that.w()))
			{
				x() += that.x();
				y() += that.y();
				z() += that.z();
				w() += that.w();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator+=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() += std::move(that).x())
					&& noexcept(y() += std::move(that).y())
					&& noexcept(z() += std::move(that).z())
					&& noexcept(w() += std::move(that).w()))
			{
				x() += std::move(that).x();
				y() += std::move(that).y();
				z() += std::move(that).z();
				w() += std::move(that).w();
				return *this;
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(x() -= that.x())
					&& noexcept(y() -= that.y())
					&& noexcept(z() -= that.z())
					&& noexcept(w() -= that.w()))
			{
				x() -= that.x();
				y() -= that.y();
				z() -= that.z();
				w() -= that.w();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator-=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(x() -= std::move(that).x())
					&& noexcept(y() -= std::move(that).y())
					&& noexcept(z() -= std::move(that).z())
					&& noexcept(w() -= std::move(that).w()))
			{
				x() -= std::move(that).x();
				y() -= std::move(that).y();
				z() -= std::move(that).z();
				w() -= std::move(that).w();
				return *this;
			}

			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(x() *= k) && noexcept(y() *= k) && noexcept(z() *= k) && noexcept(w() *= k))
			{
				x() *= k;
				y() *= k;
				z() *= k;
				w() *= k;
				return *this;
			}
			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR Rn_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(x() /= k) && noexcept(y() /= k) && noexcept(z() /= k) && noexcept(w() /= k))
			{
				x() /= k;
				y() /= k;
				z() /= k;
				w() /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE Rn_elmt& normalize() noexcept(noexcept(operator/=(norm())))
			{
				return operator/=(norm());
			}

			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				R3_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(x() == v.x()) && noexcept(y() == v.y()) && noexcept(z() == v.z()) && noexcept(w() == v.w()))
			{
				return x() == v.x() && y() == v.y() && z() == v.z() && w() == v.w();
			}
			template <class OtherComponentType, class Storage, class StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				R3_elmt<OtherComponentType, Storage, StorageTraits> const& v) const
				noexcept(noexcept(!(*this == v)))
			{
				return !(*this == v);
			}

			JKL_GPU_EXECUTABLE static constexpr Rn_elmt zero()
				noexcept(std::is_nothrow_constructible<Rn_elmt,
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>()),
					decltype(jkj::math::zero<ComponentType>())>::value)
			{
				return{ jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>(),
					jkj::math::zero<ComponentType>(), jkj::math::zero<ComponentType>() };
			}
		};

		//// Convenient make functions for R4_elmt

		// Copy or move a storage into an R4_elmt
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag,
			class StorageTraits = default_storage_traits, class Storage>
			R4_elmt<detail::deduced_1d_component_type<4, ComponentType, Storage, StorageTraits>, Storage, StorageTraits>
			make_R4_elmt(Storage&& r4_like_storage)
			noexcept(std::is_nothrow_constructible<R3_elmt<
				detail::deduced_1d_component_type<4, ComponentType, Storage, StorageTraits>,
				tmp::remove_cvref_t<Storage>, StorageTraits>, direct_construction, Storage>::value)
		{
			return{ direct_construction{}, std::forward<Storage>(r4_like_storage) };
		}

		// Create an R4_elmt with the standard (built-in array) storage
		// When ComponentType is not explicitly specified, 
		// deduce the argument ComponentType as the std::common_type_t of all component types
		template <class ComponentType = deduce_type_tag, class ArgX, class ArgY, class ArgZ, class ArgW>
		R4_elmt<detail::deduced_component_type_from_args<ComponentType, ArgX, ArgY, ArgZ, ArgW>>
			make_R4_elmt(ArgX&& x, ArgY&& y, ArgZ&& z, ArgW&& w)
			noexcept(std::is_nothrow_constructible<R4_elmt<detail::deduced_component_type_from_args<
				ComponentType, ArgX, ArgY, ArgZ, ArgW>>,
				ArgX, ArgY, ArgZ, ArgW>::value)
		{
			return{ std::forward<ArgX>(x), std::forward<ArgY>(y), std::forward<ArgZ>(z), std::forward<ArgW>(w) };
		}


		// Unary operations on Rn_elmt
		template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) normsq(Rn_elmt<N, ComponentType, Storage, StorageTraits> const& v)
			noexcept(noexcept(v.normsq()))
		{
			return v.normsq();
		}

		template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) norm(Rn_elmt<N, ComponentType, Storage, StorageTraits> const& v)
			noexcept(noexcept(v.norm()))
		{
			return v.norm();
		}

		// Member function normalize() normalizes the instance; on the other hand,
		// free function normalize() don't touch the given object. It makes a new object.
		template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE Rn_elmt<N, ComponentType, Storage, StorageTraits>
			normalize(Rn_elmt<N, ComponentType, Storage, StorageTraits> const& v)
			noexcept(noexcept(v / v.norm()))
		{
			return v / v.norm();
		}


		// tuple-like access to Rn_elmt
		template <std::size_t I, std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) get(Rn_elmt<N, ComponentType, Storage, StorageTraits>& v)
			noexcept(noexcept(v.template get<I>()))
		{
			return v.template get<I>();
		}
		template <std::size_t I, std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) get(Rn_elmt<N, ComponentType, Storage, StorageTraits> const& v)
			noexcept(noexcept(v.template get<I>()))
		{
			return v.template get<I>();
		}
		template <std::size_t I, std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) get(Rn_elmt<N, ComponentType, Storage, StorageTraits>&& v)
			noexcept(noexcept(std::move(v).template get<I>()))
		{
			return std::move(v).template get<I>();
		}
		template <std::size_t I, std::size_t N, class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) get(Rn_elmt<N, ComponentType, Storage, StorageTraits> const&& v)
			noexcept(noexcept(std::move(v).template get<I>()))
		{
			return std::move(v).template get<I>();
		}
	}
}

namespace std {
	template <std::size_t I, std::size_t N, class ComponentType, class Storage, class StorageTraits>
	struct tuple_element<I, jkj::math::Rn_elmt<N, ComponentType, Storage, StorageTraits>> {
		using type = typename StorageTraits::template tuple_element<I, Storage>::type;
	};

	template <std::size_t N, class ComponentType, class Storage, class StorageTraits>
	struct tuple_size<jkj::math::Rn_elmt<N, ComponentType, Storage, StorageTraits>> :
		std::integral_constant<std::size_t, N> {};
}

namespace jkj {
	namespace math {
		//// Binary operations between Rn_elmt's, or between an Rn_elmt and a scalar

		namespace detail {
			template <class ResultType>
			struct binary_op_impl {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE auto operator()(Kernel&& k, First&& u, Second&& v) const
					noexcept(noexcept(std::declval<ResultType&>()[0] =
						k(std::forward<First>(u)[0], std::forward<Second>(v)[0])))
				{
					ResultType result;
					for( std::size_t i = 0; i < ResultType::components; ++i )
						result[i] = k(std::forward<First>(u)[i], std::forward<Second>(v)[i]);
					return result;
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct binary_op_impl<R2_elmt<ComponentType, Storage, StorageTraits>> {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(Kernel&& k, First&& u, Second&& v) const
					noexcept(noexcept(R2_elmt<ComponentType, Storage, StorageTraits>{
					k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
						k(std::forward<First>(u).y(), std::forward<Second>(v).y()) }))
				{
					return R2_elmt<ComponentType, Storage, StorageTraits>{
						k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
						k(std::forward<First>(u).y(), std::forward<Second>(v).y())
					};
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct binary_op_impl<R3_elmt<ComponentType, Storage, StorageTraits>> {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(Kernel&& k, First&& u, Second&& v) const
					noexcept(noexcept(R3_elmt<ComponentType, Storage, StorageTraits>{
					k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
						k(std::forward<First>(u).y(), std::forward<Second>(v).y()),
						k(std::forward<First>(u).z(), std::forward<Second>(v).z()) }))
				{
					return R3_elmt<ComponentType, Storage, StorageTraits>{
						k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
						k(std::forward<First>(u).y(), std::forward<Second>(v).y()),
						k(std::forward<First>(u).z(), std::forward<Second>(v).z())
					};
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct binary_op_impl<R4_elmt<ComponentType, Storage, StorageTraits>> {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(Kernel&& k, First&& u, Second&& v) const
					noexcept(noexcept(R4_elmt<ComponentType, Storage, StorageTraits>{
					k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
						k(std::forward<First>(u).y(), std::forward<Second>(v).y()),
						k(std::forward<First>(u).z(), std::forward<Second>(v).z()),
						k(std::forward<First>(u).w(), std::forward<Second>(v).w()) }))
				{
					return R4_elmt<ComponentType, Storage, StorageTraits>{
						k(std::forward<First>(u).x(), std::forward<Second>(v).x()),
							k(std::forward<First>(u).y(), std::forward<Second>(v).y()),
							k(std::forward<First>(u).z(), std::forward<Second>(v).z()),
							k(std::forward<First>(u).w(), std::forward<Second>(v).w())
					};
				}
			};

			template <no_operation_reason reason>
			struct binary_op_impl<no_operation_tag<reason>> {
				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr no_operation_tag<reason> operator()(Args...) const noexcept
				{
					return{};
				}
			};

			template <class ComponentType>
			struct scalar_wrapper {
				ComponentType const& k;
				JKL_GPU_EXECUTABLE constexpr auto& operator[](std::size_t idx) const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto& x() const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto& y() const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto& z() const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto& w() const noexcept { return k; }
			};
			template <class ComponentType>
			JKL_GPU_EXECUTABLE constexpr auto wrap_scalar(ComponentType const& k) noexcept {
				return scalar_wrapper<ComponentType>{ k };
			}

			// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
			// To workaround this bug, the expressions are wrapped with parantheses

			struct sum_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) operator()(First&& a, Second&& b) const
					noexcept(noexcept(std::forward<First>(a) + std::forward<Second>(b)))
				{
					return (std::forward<First>(a) + std::forward<Second>(b));
				}
			};

			struct diff_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) operator()(First&& a, Second&& b) const
					noexcept(noexcept(std::forward<First>(a) - std::forward<Second>(b)))
				{
					return (std::forward<First>(a) - std::forward<Second>(b));
				}
			};

			struct mult_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) operator()(First&& a, Second&& b) const
					noexcept(noexcept(std::forward<First>(a) * std::forward<Second>(b)))
				{
					return (std::forward<First>(a) * std::forward<Second>(b));
				}
			};

			struct div_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) operator()(First&& a, Second&& b) const
					noexcept(noexcept(std::forward<First>(a) / std::forward<Second>(b)))
				{
					return (std::forward<First>(a) / std::forward<Second>(b));
				}
			};

			// For SFINAE trick

			template <class LeftOperand, class RightOperand>
			struct get_Rn_elmt_binary_result_impl {};

			template <std::size_t N,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_Rn_elmt_binary_result_impl<
				Rn_elmt<N, LeftComponentType, LeftStorage, LeftStorageTraits>,
				Rn_elmt<N, RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = Rn_elmt_binary_result<N,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_Rn_elmt_binary_result = typename get_Rn_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class Scalar, class Vector, bool from_left>
			struct get_Rn_elmt_scalar_mult_result_impl_impl {
				static constexpr bool value = false;
			};

			template <std::size_t N, class Scalar, bool from_left,
				class ComponentType, class Storage, class StorageTraits>
			struct get_Rn_elmt_scalar_mult_result_impl_impl<Scalar,
				Rn_elmt<N, ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = Rn_elmt_scalar_mult_result<N, Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, class Vector, bool from_left>
			struct get_Rn_elmt_scalar_mult_result_impl : std::conditional_t<
				get_Rn_elmt_scalar_mult_result_impl_impl<Scalar, Vector, from_left>::value,
				get_Rn_elmt_scalar_mult_result_impl_impl<Scalar, Vector, from_left>,
				get_Rn_elmt_scalar_mult_result_impl_impl<void, void, false>> {};

			template <class Scalar, class Vector, bool from_left>
			using get_Rn_elmt_scalar_mult_result = typename get_Rn_elmt_scalar_mult_result_impl<
				tmp::remove_cvref_t<Scalar>,
				tmp::remove_cvref_t<Vector>, from_left>::type;


			template <class LeftOperand, class RightOperand>
			struct get_R2_elmt_outer_result_impl {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_R2_elmt_outer_result_impl<
				R2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				R2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = R2_elmt_outer_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_R2_elmt_outer_result = typename get_R2_elmt_outer_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class LeftOperand, class RightOperand>
			struct get_R3_elmt_outer_result_impl {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
				struct get_R3_elmt_outer_result_impl<
				R3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				R3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = R3_elmt_outer_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_R3_elmt_outer_result = typename get_R3_elmt_outer_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;

			template <class Matrix, class Vector, bool from_left>
			struct get_gl2_elmt_action_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl2_elmt_action_result_impl_impl<
				gl2_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left>
			{
				using type = Rn_elmt_scalar_mult_result<2, MatrixComponentType, from_left,
					VectorComponentType, VectorStorage, VectorStorageTraits>;

				// Remove this function from the overload set if Matrix has not compatible component type
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Matrix, class Vector, bool from_left>
			struct get_gl2_elmt_action_result_impl : std::conditional_t<
				get_gl2_elmt_action_result_impl_impl<Matrix, Vector, from_left>::value,
				get_gl2_elmt_action_result_impl_impl<Matrix, Vector, from_left>,
				get_gl2_elmt_action_result_impl_impl<void, void, false>> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl2_elmt_action_result_impl<
				GL2_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl2_elmt_action_result_impl<
				gl2_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl2_elmt_action_result_impl<
				sym2_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl2_elmt_action_result_impl<
				gl2_elmt<MatrixComponentType, R2_elmt<MatrixComponentType>[2], default_storage_traits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl2_elmt_action_result_impl<
				posdef2_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl2_elmt_action_result_impl<
				gl2_elmt<MatrixComponentType, R2_elmt<MatrixComponentType>[2], default_storage_traits>,
				R2_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class Matrix, class Vector, bool from_left>
			using get_gl2_elmt_action_result = typename get_gl2_elmt_action_result_impl<
				std::remove_cv_t<std::remove_reference_t<Matrix>>,
				std::remove_cv_t<std::remove_reference_t<Vector>>, from_left>::type;


			template <class Matrix, class Vector, bool from_left>
			struct get_gl3_elmt_action_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl3_elmt_action_result_impl_impl<
				gl3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left>
			{
				using type = Rn_elmt_scalar_mult_result<3, MatrixComponentType, from_left,
					VectorComponentType, VectorStorage, VectorStorageTraits>;

				// Remove this function from the overload set if Matrix has not compatible component type
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Matrix, class Vector, bool from_left>
			struct get_gl3_elmt_action_result_impl : std::conditional_t<
				get_gl3_elmt_action_result_impl_impl<Matrix, Vector, from_left>::value,
				get_gl3_elmt_action_result_impl_impl<Matrix, Vector, from_left>,
				get_gl3_elmt_action_result_impl_impl<void, void, false>> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl3_elmt_action_result_impl<
				GL3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl3_elmt_action_result_impl<
				gl3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl3_elmt_action_result_impl<
				sym3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl3_elmt_action_result_impl<
				gl3_elmt<MatrixComponentType, R3_elmt<MatrixComponentType>[3], default_storage_traits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl3_elmt_action_result_impl<
				posdef3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl3_elmt_action_result_impl<
				gl3_elmt<MatrixComponentType, R3_elmt<MatrixComponentType>[3], default_storage_traits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class MatrixComponentType, class MatrixStorage, class MatrixStorageTraits,
				class VectorComponentType, class VectorStorage, class VectorStorageTraits,
				bool from_left>
			struct get_gl3_elmt_action_result_impl<
				SO3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> :
				get_gl3_elmt_action_result_impl<
				gl3_elmt<MatrixComponentType, MatrixStorage, MatrixStorageTraits>,
				R3_elmt<VectorComponentType, VectorStorage, VectorStorageTraits>, from_left> {};

			template <class Matrix, class Vector, bool from_left>
			using get_gl3_elmt_action_result = typename get_gl3_elmt_action_result_impl<
				std::remove_cv_t<std::remove_reference_t<Matrix>>,
				std::remove_cv_t<std::remove_reference_t<Vector>>, from_left>::type;
		}

		// Binary addition of Rn_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator+(LeftOperand&& v, RightOperand&& w)
			noexcept(noexcept(detail::binary_op_impl<
				detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>>{}(
				detail::sum_kernel{}, std::forward<LeftOperand>(v), std::forward<RightOperand>(w))))
			-> detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>;
			
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot add two Rn_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot add two Rn_elmt's; failed to deduce the resulting storage type");

			return detail::binary_op_impl<result_type>{}(detail::sum_kernel{},
				std::forward<LeftOperand>(v), std::forward<RightOperand>(w));
		}

		// Binary subtraction of Rn_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator-(LeftOperand&& v, RightOperand&& w)
			noexcept(noexcept(detail::binary_op_impl<
				detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>>{}(
					detail::diff_kernel{}, std::forward<LeftOperand>(v), std::forward<RightOperand>(w))))
			-> detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot subtract two Rn_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot subtract two Rn_elmt's; failed to deduce the resulting storage type");

			return detail::binary_op_impl<result_type>{}(detail::diff_kernel{},
				std::forward<LeftOperand>(v), std::forward<RightOperand>(w));
		}

		// Scalar multiplication of Rn_elmt's from right
		template <class Vector, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Vector&& v, Scalar const& k)
			noexcept(noexcept(detail::binary_op_impl<
				detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>>{}(
					detail::mult_kernel{}, std::forward<Vector>(v), detail::wrap_scalar(k))))
			-> detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>
		{
			using result_type = detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply Rn_elmt with a scalar; failed to deduce the resulting storage type");

			return detail::binary_op_impl<result_type>{}(detail::mult_kernel{},
				std::forward<Vector>(v), detail::wrap_scalar(k));
		}

		// Scalar multiplication of Rn_elmt's from left
		template <class Scalar, class Vector>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Scalar const& k, Vector&& v)
			noexcept(noexcept(detail::binary_op_impl<
				detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, true>>{}(
					detail::mult_kernel{}, detail::wrap_scalar(k), std::forward<Vector>(v))))
			-> detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, true>
		{
			using result_type = detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply Rn_elmt with a scalar; failed to deduce the resulting storage type");

			return detail::binary_op_impl<result_type>{}(detail::mult_kernel{},
				detail::wrap_scalar(k), std::forward<Vector>(v));
		}

		// Scalar division of Rn_elmt's from right
		template <class Vector, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Vector&& v, Scalar const& k)
			noexcept(noexcept(detail::binary_op_impl<
				detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>>{}(
					detail::div_kernel{}, std::forward<Vector>(v), detail::wrap_scalar(k))))
			-> detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>
		{
			using result_type = detail::get_Rn_elmt_scalar_mult_result<Scalar, Vector, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot divide Rn_elmt by a scalar; failed to deduce the resulting storage type");

			return detail::binary_op_impl<result_type>{}(detail::div_kernel{},
				std::forward<Vector>(v), detail::wrap_scalar(k));
		}

		// Dot product
		template <std::size_t N,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		JKL_GPU_EXECUTABLE constexpr auto dot(
			Rn_elmt<N, LeftComponentType, LeftStorage, LeftStorageTraits> const& v,
			Rn_elmt<N, RightComponentType, RightStorage, RightStorageTraits> const& w)
			noexcept(noexcept(zero<LeftComponentType>() * zero<RightComponentType>()) &&
				noexcept(std::declval<std::add_lvalue_reference_t<
					decltype(zero<LeftComponentType>() * zero<RightComponentType>())>>()
					+= v[0] * w[0]))
			-> decltype(zero<LeftComponentType>() * zero<RightComponentType>())
		{
			auto sum = zero<LeftComponentType>() * zero<RightComponentType>();
			for( std::size_t i = 0; i < N; ++i )
				sum += v[i] * w[i];
			return sum;
		}

		// Dot product 2D
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		JKL_GPU_EXECUTABLE constexpr auto dot(
			R2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits> const& v,
			R2_elmt<RightComponentType, RightStorage, RightStorageTraits> const& w)
			noexcept(noexcept(v.x() * w.x() + v.y() * w.y()))
			-> decltype(v.x() * w.x() + v.y() * w.y())
		{
			return v.x() * w.x() + v.y() * w.y();
		}

		// Dot product 3D
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		JKL_GPU_EXECUTABLE constexpr auto dot(
			R3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits> const& v,
			R3_elmt<RightComponentType, RightStorage, RightStorageTraits> const& w)
			noexcept(noexcept(v.x() * w.x() + v.y() * w.y() + v.z() * w.z()))
			-> decltype(v.x() * w.x() + v.y() * w.y() + v.z() * w.z())
		{
			return v.x() * w.x() + v.y() * w.y() + v.z() * w.z();
		}

		// Dot product 4D
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			JKL_GPU_EXECUTABLE constexpr auto dot(
				R4_elmt<LeftComponentType, LeftStorage, LeftStorageTraits> const& v,
				R4_elmt<RightComponentType, RightStorage, RightStorageTraits> const& w)
			noexcept(noexcept(v.x() * w.x() + v.y() * w.y() + v.z() * w.z() + v.w() * w.w()))
			-> decltype(v.x() * w.x() + v.y() * w.y() + v.z() * w.z() + v.w() * w.w())
		{
			return v.x() * w.x() + v.y() * w.y() + v.z() * w.z() + v.w() * w.w();
		}

		// Signed area of R2_elmt's
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		JKL_GPU_EXECUTABLE constexpr auto signed_area(
			R2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits> const& v,
			R2_elmt<RightComponentType, RightStorage, RightStorageTraits> const& w)
			noexcept(noexcept(v.x() * w.y() - v.y() * w.x()))
			-> decltype(v.x() * w.y() - v.y() * w.x())
		{
			return v.x() * w.y() - v.y() * w.x();
		}

		// Cross product of R3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto cross(LeftOperand&& v, RightOperand&& w)
			noexcept(noexcept(detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>
			{ v.y() * w.z() - v.z() * w.y(),
			v.z() * w.x() - v.x() * w.z(),
			v.x() * w.y() - v.y() * w.x() }))
			-> detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_Rn_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot compute cross product; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot compute cross product; failed to deduce the resulting storage type");
			
			return{
				v.y() * w.z() - v.z() * w.y(),
				v.z() * w.x() - v.x() * w.z(),
				v.x() * w.y() - v.y() * w.x()
			};
		}

		// Outer product of R2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto outer(LeftOperand&& v, RightOperand&& w)
			noexcept(noexcept(detail::get_R2_elmt_outer_result<LeftOperand, RightOperand>{
			v.x() * w.x(), v.x() * w.y(), v.y() * w.x(), v.y() * w.y() }))
			-> detail::get_R2_elmt_outer_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_R2_elmt_outer_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot compute outer product; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot compute outer product; failed to deduce the resulting storage type");

			return{
				v.x() * w.x(), v.x() * w.y(),
				v.y() * w.x(), v.y() * w.y()
			};
		}

		// Outer product of R3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto outer(LeftOperand&& v, RightOperand&& w)
			noexcept(noexcept(detail::get_R3_elmt_outer_result<LeftOperand, RightOperand>{
				v.x() * w.x(), v.x() * w.y(), v.x() * w.z(),
				v.y() * w.x(), v.y() * w.y(), v.y() * w.z(),
				v.z() * w.x(), v.z() * w.y(), v.z() * w.z() }))
			-> detail::get_R3_elmt_outer_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_R3_elmt_outer_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot compute outer product; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot compute outer product; failed to deduce the resulting storage type");

			return{
				v.x() * w.x(), v.x() * w.y(), v.x() * w.z(),
				v.y() * w.x(), v.y() * w.y(), v.y() * w.z(),
				v.z() * w.x(), v.z() * w.y(), v.z() * w.z()
			};
		}

		// Left action of gl2 on R2
		template <class Matrix, class Vector>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Matrix&& m, Vector&& v)
			noexcept(noexcept(detail::get_gl2_elmt_action_result<Matrix, Vector, true>{
			std::forward<Matrix>(m).template get<0, 0>() * v.x() +
				std::forward<Matrix>(m).template get<0, 1>() * v.y(),
				std::forward<Matrix>(m).template get<1, 0>() * v.x() +
				std::forward<Matrix>(m).template get<1, 1>() * v.y()
		}))
			-> detail::get_gl2_elmt_action_result<Matrix, Vector, true>
		{
			using result_type = detail::get_gl2_elmt_action_result<Matrix, Vector, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot act gl2_elmt on R2_elmt; failed to deduce the resulting storage type");

			return {
				std::forward<Matrix>(m).template get<0, 0>() * v.x() +
				std::forward<Matrix>(m).template get<0, 1>() * v.y(),
				std::forward<Matrix>(m).template get<1, 0>() * v.x() +
				std::forward<Matrix>(m).template get<1, 1>() * v.y()
			};
		}

		// Left action of gl3 on R3
		template <class Matrix, class Vector>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Matrix&& m, Vector&& v)
			noexcept(noexcept(detail::get_gl3_elmt_action_result<Matrix, Vector, true>{
				std::forward<Matrix>(m).template get<0, 0>() * v.x() +
				std::forward<Matrix>(m).template get<0, 1>() * v.y() +
				std::forward<Matrix>(m).template get<0, 2>() * v.z(),
				std::forward<Matrix>(m).template get<1, 0>() * v.x() +
				std::forward<Matrix>(m).template get<1, 1>() * v.y() +
				std::forward<Matrix>(m).template get<1, 2>() * v.z(),
				std::forward<Matrix>(m).template get<2, 0>() * v.x() +
				std::forward<Matrix>(m).template get<2, 1>() * v.y() +
				std::forward<Matrix>(m).template get<2, 2>() * v.z()
		}))
			-> detail::get_gl3_elmt_action_result<Matrix, Vector, true>
		{
			using result_type = detail::get_gl3_elmt_action_result<Matrix, Vector, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot act gl3_elmt on R3_elmt; failed to deduce the resulting storage type");

			return {
				std::forward<Matrix>(m).template get<0, 0>() * v.x() +
				std::forward<Matrix>(m).template get<0, 1>() * v.y() +
				std::forward<Matrix>(m).template get<0, 2>() * v.z(),
				std::forward<Matrix>(m).template get<1, 0>() * v.x() +
				std::forward<Matrix>(m).template get<1, 1>() * v.y() +
				std::forward<Matrix>(m).template get<1, 2>() * v.z(),
				std::forward<Matrix>(m).template get<2, 0>() * v.x() +
				std::forward<Matrix>(m).template get<2, 1>() * v.y() +
				std::forward<Matrix>(m).template get<2, 2>() * v.z()
			};
		}
	}
}
