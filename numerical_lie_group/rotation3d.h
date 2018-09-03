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
#include "polar_decomposition.h"

namespace jkl {
	namespace math {
		// Unit quaternion
		namespace detail {
			// To suppress generation of inherited constructors
			template <class ComponentType, class Storage, class StorageTraits>
			struct SU2_elmt_base : R4_elmt<ComponentType, Storage, StorageTraits> {
			private:
				using base_type = R4_elmt<ComponentType, Storage, StorageTraits>;
				using target_type = SU2_elmt<ComponentType, Storage, StorageTraits>;

				template <class Vector>
				static constexpr base_type check_and_forward(Vector&& v) {
					return close_to_one(v.normsq()) ? std::move(v) :
						throw input_validity_error<target_type>{ "jkl::math: the vector is not normalized" };
				}

				template <class Vector>
				static base_type normalize_and_forward(Vector&& v) {
					auto norm = v.norm();
					return is_invertible(norm) ? std::move(v /= norm) :
						throw input_validity_error<target_type>{ "jkl::math: the vector cannot be normalized" };
				}

			protected:
				// Default constructor; components might be filled with garbages
				SU2_elmt_base() = default;

				// [NOTE]
				// I think it would be a bad idea to add assert() inside no-check constructors,
				// because the main cause of the break of invertibility is the accumulation of
				// floating-point operation errors. It would be frequently OK if the invertibility
				// condition is just "broken slightly" by such an error, and
				// those spurious failures indeed occur quite occasionally.
				// Or even "not-very-slight-violations" might sometimes be OK as well.
				// Adding assert() may make the class too strict.

				template <class... Args>
				struct is_nothrow_constructible : std::false_type {};

				template <class ArgX, class ArgY, class ArgZ, class ArgW>
				struct is_nothrow_constructible<ArgX, ArgY, ArgZ, ArgW, no_validity_check>
				{
					static constexpr bool value = std::is_nothrow_constructible<base_type,
						ArgX, ArgY, ArgZ, ArgW>::value;
				};

				// No check component-wise constructor
				template <class ArgX, class ArgY, class ArgZ, class ArgW>
				JKL_GPU_EXECUTABLE constexpr SU2_elmt_base(forward_to_storage_tag,
					ArgX&& x, ArgY&& y,
					ArgZ&& z, ArgW&& w, no_validity_check)
					noexcept(noexcept(base_type{ std::forward<ArgX>(x), std::forward<ArgY>(y),
						std::forward<ArgZ>(z), std::forward<ArgW>(w) })) :
					base_type{ std::forward<ArgX>(x), std::forward<ArgY>(y),
					std::forward<ArgZ>(z), std::forward<ArgW>(w) } {}

				// Checking component-wise constructor
				template <class ArgX, class ArgY, class ArgZ, class ArgW>
				constexpr SU2_elmt_base(forward_to_storage_tag,
					ArgX&& x, ArgY&& y,
					ArgZ&& z, ArgW&& w) :
					base_type{ check_and_forward(R4_elmt<ComponentType,
						detail::tuple<ArgX&&, ArgY&&, ArgZ&&, ArgW&&>>{
					direct_construction{},
						std::forward<ArgX>(x), std::forward<ArgY>(y),
						std::forward<ArgZ>(z), std::forward<ArgW>(w) }) } {}

				// Auto normalizing component-wise constructor
				template <class ArgX, class ArgY, class ArgZ, class ArgW>
				constexpr SU2_elmt_base(forward_to_storage_tag,
					ArgX&& x, ArgY&& y,
					ArgZ&& z, ArgW&& w, auto_reform) :
					base_type{ normalize_and_forward(R4_elmt<ComponentType,
						detail::tuple<ArgX&&, ArgY&&, ArgZ&&, ArgW&&>>{
					direct_construction{},
						std::forward<ArgX>(x), std::forward<ArgY>(y),
						std::forward<ArgZ>(z), std::forward<ArgW>(w) }) } {}
			};
		}

		template <class ComponentType, class Storage, class StorageTraits>
		class SU2_elmt : private tmp::generate_constructors<
			detail::SU2_elmt_base<ComponentType, Storage, StorageTraits>,
			detail::forward_to_storage_tag,
			tmp::copy_or_move_n<ComponentType, 4>,
			tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
			std::tuple<std::tuple<no_validity_check>>>,
			tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
			std::tuple<std::tuple<auto_reform>>>>
		{
			using base_type = tmp::generate_constructors<
				detail::SU2_elmt_base<ComponentType, Storage, StorageTraits>,
				detail::forward_to_storage_tag,
				tmp::copy_or_move_n<ComponentType, 4>,
				tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
				std::tuple<std::tuple<no_validity_check>>>,
				tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 4>,
				std::tuple<std::tuple<auto_reform>>>>;

			using R4_elmt_type = R4_elmt<ComponentType, Storage, StorageTraits>;

		public:
			using component_type = ComponentType;
			static constexpr std::size_t components = 4;

			using storage_type = typename base_type::storage_type;
			using storage_traits = typename base_type::storage_traits;

			using base_type::x;
			using base_type::y;
			using base_type::z;
			using base_type::w;

			using base_type::base_type;

			// Take R3_elmt and a scalar (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits, class Scalar,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value &&
				std::is_convertible<Scalar, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& v, Scalar&& w,
				no_validity_check)
				noexcept(noexcept(base_type{ v.x(), v.y(), v.z(),
					std::forward<Scalar>(w), no_validity_check{} })) :
				base_type{ v.x(), v.y(), v.z(), std::forward<Scalar>(w), no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits, class Scalar,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value &&
				std::is_convertible<Scalar, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& v, Scalar&& w,
				no_validity_check)
				noexcept(noexcept(base_type{ std::move(v).x(), std::move(v).y(), std::move(v).z(),
					std::forward<Scalar>(w), no_validity_check{} })) :
				base_type{ std::move(v).x(), std::move(v).y(), std::move(v).z(),
				std::forward<Scalar>(w), no_validity_check{} } {}

			// Take an R3_elmt and a scalar (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits, class Scalar,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value &&
				std::is_convertible<Scalar, ComponentType>::value>>
			constexpr SU2_elmt(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& v, Scalar&& w) :
				base_type{ v.x(), v.y(), v.z(), std::forward<Scalar>(w) } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits, class Scalar,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value &&
				std::is_convertible<Scalar, ComponentType>::value>>
			constexpr SU2_elmt(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& v, Scalar&& w) :
				base_type { std::move(v).x(), std::move(v).y(), std::move(v).z(), std::forward<Scalar>(w) } {}

			// Convert from SU2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SU2_elmt,
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<R4_elmt_type,
					R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ that.x(), that.y(), that.z(), that.w(), no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SU2_elmt,
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<R4_elmt_type,
					R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ std::move(that).x(), std::move(that).y(),
				std::move(that).z(), std::move(that).w(), no_validity_check{} } {}

			// Convert from R4_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<R4_elmt_type,
					R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ that.x(), that.y(), that.z(), that.w(), no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SU2_elmt(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<R4_elmt_type,
					R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ std::move(that).x(), std::move(that).y(),
				std::move(that).z(), std::move(that).w(), no_validity_check{} } {}

			// Convert from R4_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr SU2_elmt(R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				base_type{ that.x(), that.y(), that.z(), that.w() } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr SU2_elmt(R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				base_type{ std::move(that).x(), std::move(that).y(),
					std::move(that).z(), std::move(that).w() } {}

			// Convert from R4_elmt of other component type (normalizing)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			SU2_elmt(R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that, auto_reform) :
				base_type{ that.x(), that.y(), that.z(), that.w(), auto_reform{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			SU2_elmt(R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that, auto_reform) :
				base_type{ std::move(that).x(), std::move(that).y(),
					std::move(that).z(), std::move(that).w(), auto_reform{} } {}


			// Copy and move
			SU2_elmt(SU2_elmt const&) = default;
			SU2_elmt(SU2_elmt&&) = default;
			SU2_elmt& operator=(SU2_elmt const&) & = default;
			SU2_elmt& operator=(SU2_elmt&&) & = default;

			// Assignment from SU2_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SU2_elmt,
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check(that)))
			{
				return assign_no_check(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SU2_elmt,
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check(std::move(that))))
			{
				return assign_no_check(std::move(that));
			}

			// Assignment from R4_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& assign_no_check(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(static_cast<R4_elmt_type&>(*this) = that))
			{
				static_cast<R4_elmt_type&>(*this) = that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& assign_no_check(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(static_cast<R4_elmt_type&>(*this) = std::move(that)))
			{
				static_cast<R4_elmt_type&>(*this) = std::move(that);
				return *this;
			}

			// Assignment from R4_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR SU2_elmt& operator=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !close_to_one(that.normsq()) )
					throw input_validity_error<SU2_elmt>{ "jkl::math: the vectir is not normalized" };
				static_cast<R4_elmt_type&>(*this) = that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR SU2_elmt& operator=(
				R4_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !close_to_one(that.normsq()) )
					throw input_validity_error<SU2_elmt>{ "jkl::math: the vectir is not normalized" };
				static_cast<R4_elmt_type&>(*this) = std::move(that);
				return *this;
			}
			

			// Remove mutable lvalue element accessors
			JKL_GPU_EXECUTABLE constexpr decltype(auto) x() const&
				noexcept(noexcept(std::declval<base_type const&>().x()))
			{
				return static_cast<base_type const&>(*this).x();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) x() &&
				noexcept(noexcept(std::declval<base_type&&>().x()))
			{
				return static_cast<base_type&&>(*this).x();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) x() const&&
				noexcept(noexcept(std::declval<base_type const&&>().x()))
			{
				return static_cast<base_type const&&>(*this).x();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) y() const&
				noexcept(noexcept(std::declval<base_type const&>().y()))
			{
				return static_cast<base_type const&>(*this).y();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) y() &&
				noexcept(noexcept(std::declval<base_type&&>().y()))
			{
				return static_cast<base_type&&>(*this).y();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) y() const&&
				noexcept(noexcept(std::declval<base_type const&&>().y()))
			{
				return static_cast<base_type const&&>(*this).y();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) z() const&
				noexcept(noexcept(std::declval<base_type const&>().z()))
			{
				return static_cast<base_type const&>(*this).z();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) z() &&
				noexcept(noexcept(std::declval<base_type&&>().z()))
			{
				return static_cast<base_type&&>(*this).z();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) z() const&&
				noexcept(noexcept(std::declval<base_type const&&>().z()))
			{
				return static_cast<base_type const&&>(*this).z();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) w() const&
				noexcept(noexcept(std::declval<base_type const&>().w()))
			{
				return static_cast<base_type const&>(*this).w();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) w() &&
				noexcept(noexcept(std::declval<base_type&&>().w()))
			{
				return static_cast<base_type&&>(*this).w();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) w() const&&
				noexcept(noexcept(std::declval<base_type const&&>().w()))
			{
				return static_cast<base_type const&&>(*this).w();
			}


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

			template <class dummy = void>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) const&
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
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto operator[](std::size_t idx) const&&
				noexcept(noexcept(std::declval<base_type const&&>()[idx]))
				-> decltype(std::declval<base_type const&&>()[idx])
			{
				return static_cast<base_type const&&>(*this)[idx];
			}


			JKL_GPU_EXECUTABLE constexpr decltype(auto) scalar_part() const noexcept(noexcept(w()))
			{
				return w();
			}

			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[3],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE constexpr
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> vector_part() const
				noexcept(noexcept(R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>{
				x(), y(), z() }))
			{
				return{ x(), y(), z() };
			}

		private:
			template <class LieAlgElmt>
			JKL_GPU_EXECUTABLE static SU2_elmt exp_impl(LieAlgElmt&& exponent)
			{
				using std::sin;
				using std::cos;

				auto angle = exponent.norm();
				R4_elmt_type ret_value{
					std::forward<LieAlgElmt>(exponent).x(),
					std::forward<LieAlgElmt>(exponent).y(),
					std::forward<LieAlgElmt>(exponent).z(),
					cos(angle / 2) };

				if( almost_smaller(angle, 0) ) {
					ret_value.x() /= 2;
					ret_value.y() /= 2;
					ret_value.z() /= 2;
				}
				else {
					auto k = sin(angle / 2) / angle;
					ret_value.x() *= k;
					ret_value.y() *= k;
					ret_value.z() *= k;
				}
				return{ std::move(ret_value), no_validity_check{} };
			}
			
		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SU2_elmt exp(
				su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& exponent)
			{
				return exp_impl(exponent);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SU2_elmt exp(
				su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& exponent)
			{
				return exp_impl(std::move(exponent));
			}
			JKL_GPU_EXECUTABLE static SU2_elmt exp(su2_elmt<ComponentType> const& exponent)
			{
				return exp_impl(exponent);
			}
			JKL_GPU_EXECUTABLE static SU2_elmt exp(su2_elmt<ComponentType>&& exponent)
			{
				return exp_impl(std::move(exponent));
			}

		private:
			template <class ReturnType, class ThisType>
			JKL_GPU_EXECUTABLE static ReturnType log_impl(ThisType&& q)
			{
				using std::sin;

				auto ang = std::forward<ThisType>(q).angle();
				ReturnType v{
					std::forward<ThisType>(q).x(),
					std::forward<ThisType>(q).y(),
					std::forward<ThisType>(q).z() };

				if( ang > constants<ComponentType>::pi ) {
					ang = 2 * constants<ComponentType>::pi - std::move(ang);
					v = -std::move(v);
				}
				auto s = sin(ang / 2);
				if( close_to_zero(s) )
					return 2 * std::move(v);
				else
					return (std::move(ang) / std::move(s)) * std::move(v);
			}

		public:
			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[3],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> log() const&
			{
				return log_impl<su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>,
					SU2_elmt const&>(*this);
			}
			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[3],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> log() &&
			{
				return log_impl<su2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>,
					SU2_elmt>(std::move(*this));
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt inv() const&
				noexcept(noexcept(SU2_elmt{ -x(), -y(), -z(), w(), no_validity_check{} }))
			{
				return{ -x(), -y(), -z(), w(), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SU2_elmt inv() &&
				noexcept(noexcept(SU2_elmt{ -std::move(*this).x(), -std::move(*this).y(), -std::move(*this).z(),
					std::move(*this).w(), no_validity_check{} }))
			{
				return{ -std::move(*this).x(), -std::move(*this).y(), -std::move(*this).z(),
					std::move(*this).w(), no_validity_check{} };
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt operator+() const&
				noexcept(noexcept(std::is_nothrow_copy_constructible<SU2_elmt>::value))
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SU2_elmt operator+() &&
				noexcept(noexcept(std::is_nothrow_move_constructible<SU2_elmt>::value))
			{
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt operator-() const&
				noexcept(noexcept(SU2_elmt{ -x(), -y(), -z(), -w(), no_validity_check{} }))
			{
				return{ -x(), -y(), -z(), -w(), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SU2_elmt operator-() &&
				noexcept(noexcept(SU2_elmt{ -std::move(*this).x(), -std::move(*this).y(), -std::move(*this).z(),
					-std::move(*this).w(), no_validity_check{} }))
			{
				return{ -std::move(*this).x(), -std::move(*this).y(), -std::move(*this).z(),
					-std::move(*this).w(), no_validity_check{} };
			}

		private:
			template <class OtherSU2>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& inplace_mul_impl(OtherSU2&& q)
				noexcept(noexcept(*this = *this * std::forward<OtherSU2>(q)))
			{
				*this = *this * std::forward<OtherSU2>(q);
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator*=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& q)
				noexcept(noexcept(inplace_mul_impl(q)))
			{
				return inplace_mul_impl(q);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator*=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& q)
				noexcept(noexcept(inplace_mul_impl(std::move(q))))
			{
				return inplace_mul_impl(std::move(q));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator/=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& q)
				noexcept(noexcept(*this *= q.inv()))
			{
				return *this *= q.inv();
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SU2_elmt& operator/=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& q)
				noexcept(noexcept(*this *= std::move(q).inv()))
			{
				return *this *= std::move(q).inv();
			}

		private:
			template <class ThisType>
			JKL_GPU_EXECUTABLE static decltype(auto) angle_impl(ThisType&& q)
			{
				using std::acos;
				auto ww = std::forward<ThisType>(q).w();

				// Clamp the range to avoid NaN caused by numerical errors
				if( ww > jkl::math::unity<ComponentType>() )
					ww = jkl::math::unity<ComponentType>();
				else if( ww < -jkl::math::unity<ComponentType>() )
					ww = -jkl::math::unity<ComponentType>();

				return 2 * acos(std::move(ww));
			}

			template <class ThisType>
			JKL_GPU_EXECUTABLE static decltype(auto) abs_angle_impl(ThisType&& q)
			{
				auto ang = std::forward<ThisType>(q).angle();
				if( ang > constants<ComponentType>::pi )
					return 2 * constants<ComponentType>::pi - std::move(ang);
				else
					return std::move(ang);
			}

		public:
			// from 0 to 2pi
			JKL_GPU_EXECUTABLE decltype(auto) angle() const& noexcept {
				return angle_impl(*this);
			}
			JKL_GPU_EXECUTABLE decltype(auto) angle() && noexcept {
				return angle_impl(std::move(*this));
			}

			// convert [0,2pi) to (-pi,pi] and take abs()
			JKL_GPU_EXECUTABLE auto abs_angle() const& noexcept {
				return abs_angle_impl(*this);
			}
			JKL_GPU_EXECUTABLE auto abs_angle() && noexcept {
				return abs_angle_impl(std::move(*this));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& q) const
				noexcept(noexcept(w() == q.w() && x() == q.x() && y() == q.y() && z() == q.z()))
			{
				return w() == q.w() && x() == q.x() && y() == q.y() && z() == q.z();
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& q) const
				noexcept(noexcept(!(*this == q)))
			{
				return !(*this == q);
			}

			JKL_GPU_EXECUTABLE static constexpr SU2_elmt unity()
				noexcept(noexcept(R4_elmt_type{
				jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(), jkl::math::unity<ComponentType>() }))
			{
				return{ jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(), jkl::math::unity<ComponentType>(),
					no_validity_check{} };
			}

			// Action on R3
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr
				detail::get_Rn_elmt_scalar_mult_result<ComponentType,
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>, true>
				rotate(R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& v) const
				noexcept(noexcept(detail::get_Rn_elmt_scalar_mult_result<ComponentType,
					R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>, true>{
				(1 - 2*y()*y() - 2*z()*z()) * v.x() + 2*(x()*y() - z()*w()) * v.y() + 2*(x()*z() + y()*w()) * v.z(),
					2*(y()*x() + z()*w()) * v.x() + (1 - 2*z()*z() - 2*x()*x()) * v.y() + 2*(y()*z() - x()*w()) * v.z(),
					2*(z()*x() - y()*w()) * v.x() + 2*(z()*y() + x()*w()) * v.y() + (1 - 2*x()*x() - 2*y()*y()) * v.z()
			}))
			{
				return{
					(1 - 2*y()*y() - 2*z()*z()) * v.x() + 2*(x()*y() - z()*w()) * v.y() + 2*(x()*z() + y()*w()) * v.z(),
					2*(y()*x() + z()*w()) * v.x() + (1 - 2*z()*z() - 2*x()*x()) * v.y() + 2*(y()*z() - x()*w()) * v.z(),
					2*(z()*x() - y()*w()) * v.x() + 2*(z()*y() + x()*w()) * v.y() + (1 - 2*x()*x() - 2*y()*y()) * v.z() };
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) inv(SU2_elmt<ComponentType, Storage, StorageTraits> const& q)
			noexcept(noexcept(q.inv()))
		{
			return q.inv();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) inv(SU2_elmt<ComponentType, Storage, StorageTraits>&& q)
			noexcept(noexcept(std::move(q).inv()))
		{
			return std::move(q).inv();
		}
		template <class ComponentType, class ReturnStorage = ComponentType[3],
			class ReturnStorageTraits = default_storage_traits,
			class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) log(SU2_elmt<ComponentType, Storage, StorageTraits> const& q)
			noexcept(noexcept(q.template log<ComponentType, ReturnStorage, ReturnStorageTraits>()))
		{
			return q.template log<ComponentType, ReturnStorage, ReturnStorageTraits>();
		}
		template <class ComponentType, class ReturnStorage = ComponentType[3],
			class ReturnStorageTraits = default_storage_traits,
			class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) log(SU2_elmt<ComponentType, Storage, StorageTraits>&& q)
			noexcept(noexcept(std::move(q).template log<ComponentType, ReturnStorage, ReturnStorageTraits>()))
		{
			return std::move(q).template log<ComponentType, ReturnStorage, ReturnStorageTraits>();
		}


		// 3x3 rotation matrix
		template <class ComponentType, class Storage, class StorageTraits>
		class SO3_elmt : public GL3_elmt<ComponentType, Storage, StorageTraits>
		{
			using GL3_elmt_type = GL3_elmt<ComponentType, Storage, StorageTraits>;

			template <class Matrix>
			static constexpr GL3_elmt_type check_and_forward(Matrix&& m) {
				return m.is_special_orthogonal() ? GL3_elmt_type{ std::forward<Matrix>(m), no_validity_check{} } :
					throw input_validity_error<SO3_elmt>{ "jkl::math: the matrix is not special orthogonal" };
			}

		public:
			// Default constructor; components might be filled with garbages
			SO3_elmt() = default;

			// No check component-wise constructor
			template <class Arg00, class Arg01, class Arg02,
				class Arg10, class Arg11, class Arg12,
				class Arg20, class Arg21, class Arg22>
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				Arg00&& arg00, Arg01&& arg01, Arg02&& arg02,
				Arg10&& arg10, Arg11&& arg11, Arg12&& arg12,
				Arg20&& arg20, Arg21&& arg21, Arg22&& arg22, no_validity_check)
				noexcept(noexcept(GL3_elmt_type{
				std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02),
				std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12),
				std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22),
				no_validity_check{} })) :
				GL3_elmt_type{
				std::forward<Arg00>(arg00), std::forward<Arg01>(arg01), std::forward<Arg02>(arg02),
				std::forward<Arg10>(arg10), std::forward<Arg11>(arg11), std::forward<Arg12>(arg12),
				std::forward<Arg20>(arg20), std::forward<Arg21>(arg21), std::forward<Arg22>(arg22),
				no_validity_check{} } {}

			// Checking component-wise constructor
			template <class Arg00, class Arg01, class Arg02,
				class Arg10, class Arg11, class Arg12,
				class Arg20, class Arg21, class Arg22>
			constexpr SO3_elmt(
				Arg00&& arg00, Arg01&& arg01, Arg02&& arg02,
				Arg10&& arg10, Arg11&& arg11, Arg12&& arg12,
				Arg20&& arg20, Arg21&& arg21, Arg22&& arg22) :
				GL3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
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
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				ComponentType arg00, ComponentType arg01, ComponentType arg02,
				ComponentType arg10, ComponentType arg11, ComponentType arg12,
				ComponentType arg20, ComponentType arg21, ComponentType arg22, no_validity_check)
				noexcept(noexcept(GL3_elmt_type{
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg00), std::move(arg01), std::move(arg02)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg10), std::move(arg11), std::move(arg12)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg20), std::move(arg21), std::move(arg22)),
				no_validity_check{} })) :
				GL3_elmt_type{
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg00), std::move(arg01), std::move(arg02)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg10), std::move(arg11), std::move(arg12)),
				detail::tuple<ComponentType&&, ComponentType&&, ComponentType&&>(
					std::move(arg20), std::move(arg21), std::move(arg22)),
				no_validity_check{} } {}

			// Checking call-by-value component-wise constructor
			constexpr SO3_elmt(
				ComponentType arg00, ComponentType arg01, ComponentType arg02,
				ComponentType arg10, ComponentType arg11, ComponentType arg12,
				ComponentType arg20, ComponentType arg21, ComponentType arg22) :
				GL3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
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
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(Row0&& r0, Row1&& r1, Row2&& r2, no_validity_check)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type, Row0, Row1, Row2, no_validity_check>::value) :
				GL3_elmt_type{ std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row2>(r2),
				no_validity_check{} } {}

			// Checking row-wise constructor
			template <class Row0, class Row1, class Row2>
			constexpr SO3_elmt(Row0&& r0, Row1&& r1, Row2&& r2) :
				GL3_elmt_type{ check_and_forward(gl3_elmt<ComponentType,
					detail::tuple<Row0&&, Row1&&, Row2&&>,
					detail::row_ref_tuple_traits<StorageTraits, Row0, Row1, Row2>>{ direct_construction{},
					std::forward<Row0>(r0), std::forward<Row1>(r1), std::forward<Row2>(r2) }) } {}

			// No check call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				R3_elmt<ComponentType> r0, R3_elmt<ComponentType> r1, R3_elmt<ComponentType> r2,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type,
					R3_elmt<ComponentType>, R3_elmt<ComponentType>, R3_elmt<ComponentType>, no_validity_check>::value) :
				GL3_elmt_type{ gl3_elmt<ComponentType, detail::tuple<
				R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&>>{
				direct_construction{}, std::move(r0), std::move(r1), std::move(r2) }, no_validity_check{} } {}

			// Checking call-by-value row-wise constructor
			// The role of this constructor is to enable braces without explicit mention of ComponentType
			constexpr SO3_elmt(
				R3_elmt<ComponentType> r0, R3_elmt<ComponentType> r1, R3_elmt<ComponentType> r2) :
				GL3_elmt_type{ check_and_forward(gl3_elmt<ComponentType, detail::tuple<
					R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&, R3_elmt<ComponentType>&&>>{
				direct_construction{}, std::move(r0), std::move(r1), std::move(r2) }) } {}

			// Convert from SO3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SO3_elmt,
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type,
					GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				GL3_elmt_type{ that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SO3_elmt,
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type,
					GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				GL3_elmt_type{ std::move(that) } {}

			// Convert from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&, no_validity_check>::value) :
				GL3_elmt_type{ that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SO3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<GL3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>, no_validity_check>::value) :
				GL3_elmt_type{ std::move(that), no_validity_check{} } {}

			// Convert from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr SO3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				GL3_elmt_type{ check_and_forward(that) } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr SO3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				GL3_elmt_type{ check_and_forward(std::move(that)) } {}

			// Universal covering map
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE explicit constexpr SO3_elmt(
				SU2_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& q)
				noexcept(noexcept(SO3_elmt{
				1 - 2 * (q.y()*q.y() + q.z()*q.z()), 2 * (q.x()*q.y() - q.z()*q.w()), 2 * (q.z()*q.x() + q.y()*q.w()),
				2 * (q.x()*q.y() + q.z()*q.w()), 1 - 2 * (q.z()*q.z() + q.x()*q.x()), 2 * (q.y()*q.z() - q.x()*q.w()),
				2 * (q.z()*q.x() - q.y()*q.w()), 2 * (q.y()*q.z() + q.x()*q.w()), 1 - 2 * (q.x()*q.x() + q.y()*q.y()),
				no_validity_check{} })) : SO3_elmt{
				1 - 2 * (q.y()*q.y() + q.z()*q.z()), 2 * (q.x()*q.y() - q.z()*q.w()), 2 * (q.z()*q.x() + q.y()*q.w()),
				2 * (q.x()*q.y() + q.z()*q.w()), 1 - 2 * (q.z()*q.z() + q.x()*q.x()), 2 * (q.y()*q.z() - q.x()*q.w()),
				2 * (q.z()*q.x() - q.y()*q.w()), 2 * (q.y()*q.z() + q.x()*q.w()), 1 - 2 * (q.x()*q.x() + q.y()*q.y()),
				no_validity_check{} } {}


			// Copy and move
			SO3_elmt(SO3_elmt const&) = default;
			SO3_elmt(SO3_elmt&&) = default;
			SO3_elmt& operator=(SO3_elmt const&) & = default;
			SO3_elmt& operator=(SO3_elmt&&) & = default;

			// Assignment from SO3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SO3_elmt,
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator=(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(assign_no_check(that)))
			{
				return assign_no_check(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<SO3_elmt,
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator=(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(assign_no_check(std::move(that))))
			{
				return assign_no_check(std::move(that));
			}

			// Assignment from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(static_cast<GL3_elmt_type&>(*this).assign_no_check(that)))
			{
				static_cast<GL3_elmt_type&>(*this).assign_no_check(that);
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(static_cast<GL3_elmt_type&>(*this).assign_no_check(std::move(that))))
			{
				static_cast<GL3_elmt_type&>(*this).assign_no_check(std::move(that));
				return *this;
			}

			// Assignment from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR SO3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_special_othogonal() )
					throw input_validity_error<SO3_elmt>{ "jkl::math: the matrix is not special orthogonal" };
				static_cast<GL3_elmt_type&>(*this).assign_no_check(that);
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR SO3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_special_othogonal() )
					throw input_validity_error<SO3_elmt>{ "jkl::math: the matrix is not special orthogonal" };
				static_cast<GL3_elmt_type&>(*this).assign_no_check(std::move(that));
				return *this;
			}


			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<GL3_elmt_type const&>().template get<I>()))
			{
				return static_cast<GL3_elmt_type const&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<GL3_elmt_type&&>().template get<I>()))
			{
				return static_cast<GL3_elmt_type&&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<GL3_elmt_type const&&>().template get<I>()))
			{
				return static_cast<GL3_elmt_type const&&>(*this).template get<I>();
			}

			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<GL3_elmt_type const&>().template get<I, J>()))
			{
				return static_cast<GL3_elmt_type const&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<GL3_elmt_type&&>().template get<I, J>()))
			{
				return static_cast<GL3_elmt_type&&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<GL3_elmt_type const&&>().template get<I, J>()))
			{
				return static_cast<GL3_elmt_type const&&>(*this).template get<I, J>();
			}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<SO3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SO3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<SO3_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt operator-() const&
				noexcept(noexcept(GL3_elmt_type{ -static_cast<GL3_elmt_type const&>(*this) }))
			{
				return{ -static_cast<GL3_elmt_type const&>(*this), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SO3_elmt operator-() &&
				noexcept(noexcept(GL3_elmt_type{ -static_cast<GL3_elmt_type&&>(*this) }))
			{
				return{ -static_cast<GL3_elmt_type&&>(*this), no_validity_check{} };
			}

			
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
				JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator*=(
					SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(static_cast<GL3_elmt_type&>(*this) *= that))
			{
				static_cast<GL3_elmt_type&>(*this) *= that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
				JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator*=(
					SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(static_cast<GL3_elmt_type&>(*this) *= std::move(that)))
			{
				static_cast<GL3_elmt_type&>(*this) *= std::move(that);
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt t() const&
				noexcept(noexcept(static_cast<GL3_elmt_type const&>(*this).t()))
			{
				return{ static_cast<GL3_elmt_type const&>(*this).t(), no_validity_check{} };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SO3_elmt t() &&
				noexcept(noexcept(static_cast<GL3_elmt_type&&>(*this).t()))
			{
				return{ static_cast<GL3_elmt_type&&>(*this).t(), no_validity_check{} };
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const noexcept
			{
				return true;
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept
			{
				return true;
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const noexcept
			{
				return true;
			}


			// Division
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt inv() const&
				noexcept(noexcept(t()))
			{
				return t();
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt inv() &&
				noexcept(noexcept(std::move(*this).t()))
			{
				return std::move(*this).t();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator/=(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(*this *= that.inv()))
			{
				return *this *= that.inv();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SO3_elmt& operator/=(
				SO3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(*this *= std::move(that).inv()))
			{
				return *this *= std::move(that).inv();
			}

			JKL_GPU_EXECUTABLE static constexpr SO3_elmt unity()
				noexcept(noexcept(GL3_elmt_type::unity()))
			{
				return{
					jkl::math::unity<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::unity<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::unity<ComponentType>(),
					no_validity_check{}
				};
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SO3_elmt exp(
				so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& exponent)
			{
				return SO3_elmt(SU2_elmt<OtherComponentType>::exp(exponent));
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SO3_elmt exp(
				so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& exponent)
			{
				return SO3_elmt(SU2_elmt<OtherComponentType>::exp(std::move(exponent)));
			}

			// Action on R3
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) rotate(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& v) const
				noexcept(noexcept(*this * v))
			{
				return *this * v;
			}

			// Logarithmic map
		private:
			template <class ReturnType, class ThisType>
			JKL_GPU_EXECUTABLE static ReturnType log_impl(ThisType&& m)
			{
				// Take antisymmetric part
				auto x = (std::forward<ThisType>(m).template get<2, 1>()
					- std::forward<ThisType>(m).template get<1, 2>()) / 2;
				auto y = (m.template get<0, 2>()
					- std::forward<ThisType>(m).template get<2, 0>()) / 2;
				auto z = (std::forward<ThisType>(m).template get<1, 0>()
					- m.template get<0, 1>()) / 2;
				// Compute angle
				auto angle_cos = (m.trace() - 1) / 2;
				auto angle = acos(angle_cos);
				// arcsin(norm) / norm
				if( almost_smaller(angle, 0) )
					return{ x, y, z };
				else if( almost_larger(angle, constants<ComponentType>::pi) ) {
					x = sqrt((std::forward<ThisType>(m).template get<0, 0>() + 1) / 2);
					y = sqrt((std::forward<ThisType>(m).template get<1, 1>() + 1) / 2);
					z = sqrt((std::forward<ThisType>(m).template get<2, 2>() + 1) / 2);
					if( m.template get<0, 1>() < 0 && m.template get<0, 2>() < 0 )
						return{ -x, y, z };
					else if( m.template get<0, 1>() < 0 && m.template get<0, 2>() >= 0 )
						return{ x, -y, z };
					else if( m.template get<0, 1>() >= 0 && m.template get<0, 2>() < 0 )
						return{ x, y, -z };
					else
						return{ x, y, z };
				}
				auto d = angle / sin(angle);
				return{ x*d, y*d, z*d };
			}

		public:
			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[3],
				class OtherStorageTraits = default_storage_traits>
				JKL_GPU_EXECUTABLE so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> log() const&
			{
				return log_impl<so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>,
					SO3_elmt const&>(*this);
			}
			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[3],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> log() &&
			{
				return log_impl<so3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>,
					SO3_elmt>(*this);
			}

			// Inverse covering map
			template <class OtherComponentType = ComponentType,
				class OtherStorage = OtherComponentType[4],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE auto find_quaternion() const {
				return SU2_elmt<ComponentType, OtherStorage, OtherStorageTraits>::exp(log());
			}

			// Get Euler angles; find (theta_x, theta_y, theta_z) with R = R_z(theta_z)R_y(theta_y)R_x(theta_x)
			template <class OtherStorage = ComponentType[3],
				class OtherStorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE R3_elmt<ComponentType, OtherStorage, OtherStorageTraits> euler_angles() const {
				ComponentType theta_x, theta_y, theta_z;
				if( close_to(get<2, 0>(), jkl::math::unity<ComponentType>()) ) {
					theta_z = 0;
					theta_y = -constants<ComponentType>::pi / 2;
					theta_x = atan2(-get<0, 1>(), -get<0, 2>());
				}
				else if( close_to(get<2, 0>(), -jkl::math::unity<ComponentType>()) ) {
					theta_z = 0;
					theta_y = constants<ComponentType>::pi / 2;
					theta_x = atan2(get<0, 1>(), get<0, 2>());
				}
				else {
					theta_y = -asin(get<2, 0>());
					theta_x = atan2(get<2, 1>() / cos(theta_y), get<2, 2>() / cos(theta_y));
					theta_z = atan2(get<1, 0>() / cos(theta_y), get<0, 0>() / cos(theta_y));
				}
				return{ theta_x, theta_y, theta_z };
			}

			JKL_GPU_EXECUTABLE static SO3_elmt rotx(ComponentType theta)
				noexcept(noexcept(exp({ std::move(theta), jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>() })))
			{
				return exp({ std::move(theta), jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>() });
			}
			JKL_GPU_EXECUTABLE static SO3_elmt roty(ComponentType theta)
				noexcept(noexcept(exp({ jkl::math::zero<ComponentType>(), std::move(theta), jkl::math::zero<ComponentType>() })))
			{
				return exp({ jkl::math::zero<ComponentType>(), std::move(theta), jkl::math::zero<ComponentType>() });
			}
			JKL_GPU_EXECUTABLE static SO3_elmt rotz(ComponentType theta)
				noexcept(noexcept(exp({ jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>(), std::move(theta) })))
			{
				return exp({ jkl::math::zero<ComponentType>(), jkl::math::zero<ComponentType>(), std::move(theta) });
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SO3_elmt euler_to_SO3(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& euler_angles)
				noexcept(noexcept(rotz(euler_angles.z()) * roty(euler_angles.y()) * rotx(euler_angles.x())))
			{
				return rotz(euler_angles.z()) * roty(euler_angles.y()) * rotx(euler_angles.x());
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE static SO3_elmt euler_to_SO3(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& euler_angles)
				noexcept(noexcept(rotz(std::move(euler_angles).z()) *
					roty(std::move(euler_angles).y()) * rotx(std::move(euler_angles).x())))
			{
				return rotz(std::move(euler_angles).z()) *
					roty(std::move(euler_angles).y()) * rotx(std::move(euler_angles).x());
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(SO3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(SO3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(SO3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.inv()))
		{
			return m.inv();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(SO3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).inv()))
		{
			return std::move(m).inv();
		}
		template <class ComponentType, class ReturnStorage = ComponentType[3],
			class ReturnStorageTraits = default_storage_traits,
			class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) log(SO3_elmt<ComponentType, Storage, StorageTraits> const& q)
			noexcept(noexcept(q.template log<ComponentType, ReturnStorage, ReturnStorageTraits>()))
		{
			return q.template log<ComponentType, ReturnStorage, ReturnStorageTraits>();
		}
		template <class ComponentType, class ReturnStorage = ComponentType[3],
			class ReturnStorageTraits = default_storage_traits,
			class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE decltype(auto) log(SO3_elmt<ComponentType, Storage, StorageTraits>&& q)
			noexcept(noexcept(std::move(q).template log<ComponentType, ReturnStorage, ReturnStorageTraits>()))
		{
			return std::move(q).template log<ComponentType, ReturnStorage, ReturnStorageTraits>();
		}


		//// Binary operations between SU2_elmt

		namespace detail {
			template <class LeftOperand, class RightOperand>
			struct get_SU2_elmt_binary_result_impl {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_SU2_elmt_binary_result_impl<
				SU2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				SU2_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = SU2_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_SU2_elmt_binary_result = typename get_SU2_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;
		}

		// Binary multiplication of SU2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator*(LeftOperand&& p, RightOperand&& q)
			noexcept(noexcept(detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>{
			p.w() * q.vector_part() + q.w() * p.vector_part() + cross(p.vector_part(), q.vector_part()),
				p.w() * q.w() - dot(p.vector_part(), q.vector_part()),
				no_validity_check{} }))
			-> detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot multiply two SU2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply two SU2_elmt's; failed to deduce the resulting storage type");
			
			return{ p.w() * q.vector_part() + p.vector_part() * q.w() +
				cross(p.vector_part(), q.vector_part()),
				p.w() * q.w() - dot(p.vector_part(), q.vector_part()),
				no_validity_check{}
			};
		}

		// Binary division of SU2_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator/(LeftOperand&& p, RightOperand&& q)
			noexcept(noexcept(std::forward<LeftOperand>(p) * std::forward<RightOperand>(q).inv()))
			-> detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot divide two SU2_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot divide two SU2_elmt's; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(p) * std::forward<RightOperand>(q).inv();
		}

		// Interpolation
		template <class LeftOperand, class RightOperand, class Parameter>
		JKL_GPU_EXECUTABLE auto SU2_interpolation(LeftOperand&& p, RightOperand&& q, Parameter&& t)
			noexcept(noexcept(std::forward<LeftOperand>(p) *
				detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>::exp(
			(p.inv() * std::forward<RightOperand>(q)).log() * std::forward<Parameter>(t))))
			-> detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SU2_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot compute SU2 interpolation; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot compute SU2 interpolation; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(p) * result_type::exp(
				(p.inv() * std::forward<RightOperand>(q)).log() * std::forward<Parameter>(t));
		}
	}
}