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
#include "../tmp/assert_helper.h"
#include "rotation3d.h"

namespace jkj {
	namespace math {
		// 3-dimensional rigid transform
		namespace detail {
			struct use_SO3_tag {};

			template <class ComponentType, class SU2Storage, class SU2StorageTraits,
				class R3Storage, class R3StorageTraits>
			struct SE3_elmt_base
			{
				using rotation_type = SU2_elmt<ComponentType, SU2Storage, SU2StorageTraits>;
				using translation_type = R3_elmt<ComponentType, R3Storage, R3StorageTraits>;

			protected:
				rotation_type rot_;
				translation_type trans_;

				SE3_elmt_base() = default;

				template <class...>
				struct is_nothrow_constructible : std::false_type {};

				template <class RotArg, class TransArg>
				struct is_nothrow_constructible<RotArg, TransArg> {
					static constexpr bool value =
						std::is_nothrow_constructible<rotation_type, RotArg>::value &&
						std::is_nothrow_constructible<translation_type, TransArg>::value;
				};

				template <class RotArg, class TransArg>
				struct is_nothrow_constructible<use_SO3_tag, RotArg, TransArg> {
					static constexpr bool value =
						noexcept(std::declval<RotArg>().template find_quaternion<typename rotation_type::component_type,
							typename rotation_type::storage_type, typename rotation_type::storage_traits>()) &&
						std::is_nothrow_move_constructible<rotation_type>::value &&
						std::is_nothrow_constructible<translation_type, TransArg>::value;
				};

				template <class RotArg, class TransArg>
				JKL_GPU_EXECUTABLE constexpr SE3_elmt_base(forward_to_storage_tag, RotArg&& r, TransArg&& t)
					noexcept(is_nothrow_constructible<RotArg, TransArg>::value) :
					rot_(std::forward<RotArg>(r)), trans_(std::forward<TransArg>(t)) {}

				template <class RotArg, class TransArg>
				JKL_GPU_EXECUTABLE constexpr SE3_elmt_base(forward_to_storage_tag, use_SO3_tag, RotArg&& r, TransArg&& t)
					noexcept(is_nothrow_constructible<use_SO3_tag, RotArg, TransArg>::value) :
					rot_(std::forward<RotArg>(r).template find_quaternion<typename rotation_type::component_type,
					typename rotation_type::storage_type, typename rotation_type::storage_traits>()),
					trans_(std::forward<TransArg>(t)) {}
			};
		}

		template <class ComponentType, class SU2Storage, class SU2StorageTraits,
			class R3Storage, class R3StorageTraits>
		class SE3_elmt : public tmp::generate_constructors<
			detail::SE3_elmt_base<ComponentType, SU2Storage, SU2StorageTraits,
			R3Storage, R3StorageTraits>, detail::forward_to_storage_tag,
			tmp::copy_or_move<SU2_elmt<ComponentType, SU2Storage, SU2StorageTraits>,
			R3_elmt<ComponentType, R3Storage, R3StorageTraits>>>
		{
			using base_type = tmp::generate_constructors<
				detail::SE3_elmt_base<ComponentType, SU2Storage, SU2StorageTraits,
				R3Storage, R3StorageTraits>, detail::forward_to_storage_tag,
				tmp::copy_or_move<SU2_elmt<ComponentType, SU2Storage, SU2StorageTraits>,
				R3_elmt<ComponentType, R3Storage, R3StorageTraits>>>;

		public:
			using rotation_type = SU2_elmt<ComponentType, SU2Storage, SU2StorageTraits>;
			using translation_type = R3_elmt<ComponentType, R3Storage, R3StorageTraits>;

		public:
			using component_type = ComponentType;
			static constexpr std::size_t components = 7;

			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR rotation_type& rotation_q() & noexcept {
				return base_type::rot_;
			}
			JKL_GPU_EXECUTABLE constexpr rotation_type const& rotation_q() const& noexcept {
				return base_type::rot_;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR rotation_type&& rotation_q() && noexcept {
				return std::move(base_type::rot_);
			}
			JKL_GPU_EXECUTABLE constexpr rotation_type const&& rotation_q() const&& noexcept {
				return std::move(base_type::rot_);
			}

			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR translation_type& translation() & noexcept {
				return base_type::trans_;
			}
			JKL_GPU_EXECUTABLE constexpr translation_type const& translation() const& noexcept {
				return base_type::trans_;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR translation_type&& translation() && noexcept {
				return std::move(base_type::trans_);
			}
			JKL_GPU_EXECUTABLE constexpr translation_type const&& translation() const&& noexcept {
				return std::move(base_type::trans_);
			}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt<ComponentType> rotation() const&
				noexcept(noexcept(SO3_elmt<ComponentType>(base_type::rot_)))
			{
				return SO3_elmt<ComponentType>(base_type::rot_);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR SO3_elmt<ComponentType> rotation() &&
				noexcept(noexcept(SO3_elmt<ComponentType>(std::move(base_type::rot_))))
			{
				return SO3_elmt<ComponentType>(std::move(base_type::rot_));
			}

			using base_type::base_type;

			// Default constructor; components might be filled with garbages
			SE3_elmt() = default;

			// Take SO3_elmt and R3_elmt (both are template)
			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const& rot,
				R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits> const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const&,
					R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, trans } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const& rot,
				R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits>&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const&,
					R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, std::move(trans) } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>&& rot,
				R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits> const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>,
					R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits> const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), trans } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>&& rot,
				R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits>&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>,
					R3_elmt<OtherComponentType, OtherR3Storage, OtherR3StorageTraits>>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), std::move(trans) } {}

			// Take SO3_elmt and R3_elmt (R3_elmt is a concrete type)
			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const& rot,
				translation_type const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const&,
					translation_type const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, trans } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const& rot,
				translation_type&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits> const&,
					translation_type>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, std::move(trans) } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>&& rot,
				translation_type const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>,
					translation_type const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), trans } {}

			template <class OtherComponentType, class OtherSO3Storage, class OtherSO3StorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>&& rot,
				translation_type&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<OtherComponentType, OtherSO3Storage, OtherSO3StorageTraits>,
					translation_type>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), std::move(trans) } {}

			// Take SO3_elmt and R3_elmt (both are concrete types)
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<ComponentType> const& rot, translation_type const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<ComponentType> const&, translation_type const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, trans } {}

			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<ComponentType> const& rot, translation_type&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<ComponentType> const&, translation_type>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				rot, std::move(trans) } {}

			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<ComponentType>&& rot, translation_type const& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<ComponentType>, translation_type const&>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), trans } {}

			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SO3_elmt<ComponentType>&& rot, translation_type&& trans)
				noexcept(base_type::template is_nothrow_constructible<detail::use_SO3_tag,
					SO3_elmt<ComponentType>, translation_type>::value) :
				base_type{ detail::forward_to_storage_tag{}, detail::use_SO3_tag{},
				std::move(rot), std::move(trans) } {}


			// Convert from SE3_elmt of other component type
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<SE3_elmt,
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(base_type{ detail::forward_to_storage_tag{},
					that.rotation_q(), that.translation() })) :
				base_type{ detail::forward_to_storage_tag{},
				that.rotation_q(), that.translation() } {}

			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<SE3_elmt,
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr SE3_elmt(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(base_type{ detail::forward_to_storage_tag{},
					std::move(that).rotation_q(), std::move(that).translation() })) :
				base_type{ detail::forward_to_storage_tag{},
				std::move(that).rotation_q(), std::move(that).translation() } {}
			

			// Copy and move
			SE3_elmt(SE3_elmt const&) = default;
			SE3_elmt(SE3_elmt&&) = default;
			SE3_elmt& operator=(SE3_elmt const&) & = default;
			SE3_elmt& operator=(SE3_elmt&&) & = default;

			// Assignment from SE3_elmt of other component type
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<SE3_elmt,
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) &
				noexcept(noexcept(rotation_q() = that.rotation_q()) &&
					noexcept(translation() = that.translation()))
			{
				rotation_q() = that.rotation_q();
				translation() = that.translation();
				return *this;
			}
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<SE3_elmt,
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that) &
				noexcept(noexcept(rotation_q() = std::move(that).rotation_q()) &&
					noexcept(translation() = std::move(that).translation()))
			{
				rotation_q() = std::move(that).rotation_q();
				translation() = std::move(that).translation();
				return *this;
			}

			// Exponential map
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE static SE3_elmt exp(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& exponent)
			{
				using std::sin;
				using std::cos;

				auto rot = rotation_type::exp(exponent.rotation_part());
				auto const angle = exponent.rotation_part().norm();
				ComponentType alpha, beta;
				if( almost_smaller(angle, jkj::math::zero<ComponentType>()) ) {
					alpha = ComponentType(1) / 2 - angle*angle / 24;
					beta = ComponentType(1) / 6 - angle*angle / 120;
				}
				else {
					alpha = (ComponentType(1) - cos(angle)) / (angle * angle);
					beta = (angle - sin(angle)) / (angle * angle * angle);
				}
				auto cross_prod = cross(exponent.rotation_part(), exponent.translation_part());
				auto trans = exponent.translation_part() + alpha * cross_prod +
					beta * cross(exponent.rotation_part(), cross_prod);

				return{ rot, trans };
			}

		private:
			template <class ThisType>
			JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR SE3_elmt inv_impl(ThisType&& t)
				noexcept(noexcept(std::forward<ThisType>(t).rotation_q().inv()) &&
					noexcept(SE3_elmt{ std::declval<rotation_type>(),
						-std::declval<rotation_type&>().rotate(std::forward<ThisType>(t).translation()) }))
			{
				auto r = std::forward<ThisType>(t).rotation_q().inv();
				return{ std::move(r), -r.rotate(std::forward<ThisType>(t).translation()) };
			}

		public:
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt inv() const&
				noexcept(noexcept(inv_impl(*this)))
			{
				return inv_impl(*this);
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt inv() &&
				noexcept(noexcept(inv_impl(std::move(*this))))
			{
				return inv_impl(std::move(*this));
			}

		private:
			template <class OtherSE3_elmt>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& inplace_mul_impl(OtherSE3_elmt&& that) noexcept(
				noexcept(translation() += rotation_q().rotate(std::forward<OtherSE3_elmt>(that).translation()))
				&& noexcept(rotation_q() *= std::forward<OtherSE3_elmt>(that).rotation_q()))
			{
				translation() += rotation_q().rotate(std::forward<OtherSE3_elmt>(that).translation());
				rotation_q() *= std::forward<OtherSE3_elmt>(that).rotation_q();
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator*=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(inplace_mul_impl(that)))
			{				
				return inplace_mul_impl(that);
			}
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator*=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(inplace_mul_impl(std::move(that))))
			{
				return inplace_mul_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator/=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(*this *= that.inv()))
			{
				return *this *= that.inv();
			}
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR SE3_elmt& operator/=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(*this *= std::move(that).inv()))
			{
				return *this *= std::move(that).inv();
			}

		private:
			template <class ReturnType, class ThisType>
			JKL_GPU_EXECUTABLE static ReturnType log_impl(ThisType&& q)
			{
				auto rotation_part = std::forward<ThisType>(q).rotation_q().log();
				auto const angle = rotation_part.norm();
				ComponentType beta;
				if( almost_smaller(angle, jkj::math::zero<ComponentType>()) )
					beta = ComponentType(1) / 12 + angle / 360;
				else
					beta = (ComponentType(1) - (angle / 2) / tan(angle / 2)) / (angle * angle);

				auto cross_prod = cross(rotation_part, q.translation());
				auto translation_part = std::forward<ThisType>(q).translation() - cross_prod/2 +
					beta * cross(rotation_part, cross_prod);
				return{ std::move(rotation_part), std::move(translation_part) };
			}

		public:
			template <class OtherComponentType = ComponentType,
				class Otherso3Storage = OtherComponentType[3],
				class Otherso3StorageTraits = default_storage_traits,
				class OtherR3Storage = OtherComponentType[3],
				class OtherR3StorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> log() const&
			{
				return log_impl<se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
					OtherR3Storage, OtherR3StorageTraits>, SE3_elmt const&>(*this);
			}
			template <class OtherComponentType = ComponentType,
				class Otherso3Storage = OtherComponentType[3],
				class Otherso3StorageTraits = default_storage_traits,
				class OtherR3Storage = OtherComponentType[3],
				class OtherR3StorageTraits = default_storage_traits>
			JKL_GPU_EXECUTABLE se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> log() &&
			{
				return log_impl<se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
					OtherR3Storage, OtherR3StorageTraits>, SE3_elmt>(std::move(*this));
			}

			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) const
				noexcept(noexcept(translation() == that.translation() &&
				(rotation_q() == that.rotation_q() || rotation_q() == -that.rotation_q())))
			{
				return translation() == that.translation() &&
					(rotation_q() == that.rotation_q() || rotation_q() == -that.rotation_q());
			}
			template <class OtherComponentType, class OtherSU2Storage, class OtherSU2StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				SE3_elmt<OtherComponentType, OtherSU2Storage, OtherSU2StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) const
				noexcept(noexcept(!(*this == that)))
			{
				return !(*this == that);
			}

			JKL_GPU_EXECUTABLE static constexpr SE3_elmt unity()
				noexcept(noexcept(SE3_elmt{ rotation_type::unity(), translation_type::zero() }))
			{
				return{ rotation_type::unity(), translation_type::zero() };
			}

			// Action on R3
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) transform(
				R3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& p) const
				noexcept(noexcept(rotation_q().rotate(p) + translation()))
			{
				return rotation_q().rotate(p) + translation();
			}
		};


		namespace detail {
			template <class ComponentType, class so3Storage, class so3StorageTraits,
				class R3Storage, class R3StorageTraits>
			struct se3_elmt_base
			{
				using rotation_part_type = so3_elmt<ComponentType, so3Storage, so3StorageTraits>;
				using translation_part_type = R3_elmt<ComponentType, R3Storage, R3StorageTraits>;

				se3_elmt_base() = default;

			protected:
				rotation_part_type rot_;
				translation_part_type trans_;

				template <class RotArg, class TransArg>
				struct is_nothrow_constructible {
					static constexpr bool value =
						std::is_nothrow_constructible<rotation_part_type, RotArg>::value &&
						std::is_nothrow_constructible<translation_part_type, TransArg>::value;
				};

				template <class RotArg, class TransArg>
				JKL_GPU_EXECUTABLE constexpr se3_elmt_base(forward_to_storage_tag, RotArg&& r, TransArg&& t)
					noexcept(is_nothrow_constructible<RotArg, TransArg>::value) :
					rot_(std::forward<RotArg>(r)), trans_(std::forward<TransArg>(t)) {}
			};
		}

		template <class ComponentType, class so3Storage, class so3StorageTraits,
			class R3Storage, class R3StorageTraits>
		class se3_elmt : public tmp::generate_constructors<
			detail::se3_elmt_base<ComponentType, so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>,
			detail::forward_to_storage_tag,
			tmp::copy_or_move<so3_elmt<ComponentType, so3Storage, so3StorageTraits>,
			R3_elmt<ComponentType, R3Storage, R3StorageTraits>>>
		{
			using base_type = tmp::generate_constructors<
				detail::se3_elmt_base<ComponentType, so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>,
				detail::forward_to_storage_tag,
				tmp::copy_or_move<so3_elmt<ComponentType, so3Storage, so3StorageTraits>,
				R3_elmt<ComponentType, R3Storage, R3StorageTraits>>>;

		public:
			using component_type = ComponentType;
			static constexpr std::size_t components = 6;

			using rotation_part_type = typename base_type::rotation_part_type;
			using translation_part_type = typename base_type::translation_part_type;

			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR rotation_part_type& rotation_part() & noexcept {
				return base_type::rot_;
			}
			JKL_GPU_EXECUTABLE constexpr rotation_part_type const& rotation_part() const& noexcept {
				return base_type::rot_;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR rotation_part_type&& rotation_part() && noexcept {
				return std::move(base_type::rot_);
			}
			JKL_GPU_EXECUTABLE constexpr rotation_part_type const&& rotation_part() const&& noexcept {
				return std::move(base_type::rot_);
			}

			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR translation_part_type& translation_part() & noexcept {
				return base_type::trans_;
			}
			JKL_GPU_EXECUTABLE constexpr translation_part_type const& translation_part() const& noexcept {
				return base_type::trans_;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR translation_part_type&& translation_part() && noexcept {
				return std::move(base_type::trans_);
			}
			JKL_GPU_EXECUTABLE constexpr translation_part_type const&& translation_part() const&& noexcept {
				return std::move(base_type::trans_);
			}

			using base_type::base_type;

			// Default constructor; components might be filled with garbages
			se3_elmt() = default;

			// Convert from se3_elmt of other component type
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<se3_elmt,
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr se3_elmt(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(base_type{ detail::forward_to_storage_tag{},
					that.rotation_part(), that.translation_part() })) :
				base_type{ detail::forward_to_storage_tag{},
				that.rotation_part(), that.translation_part() } {}

			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<se3_elmt,
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr se3_elmt(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(base_type{ detail::forward_to_storage_tag{},
					std::move(that).rotation_part(), std::move(that).translation_part() })) :
				base_type{ detail::forward_to_storage_tag{},
				std::move(that).rotation_part(), std::move(that).translation_part() } {}

			// Copy and move
			se3_elmt(se3_elmt const&) = default;
			se3_elmt(se3_elmt&&) = default;
			se3_elmt& operator=(se3_elmt const&) & = default;
			se3_elmt& operator=(se3_elmt&&) & = default;

			// Assignment from se3_elmt of other component type
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<se3_elmt,
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) &
				noexcept(noexcept(rotation_part() = that.rotation_part()) &&
					noexcept(translation_part() = that.translation_part()))
			{
				rotation_part() = that.rotation_part();
				translation_part() = that.translation_part();
				return *this;
			}
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits,
				class = std::enable_if_t<!std::is_same<se3_elmt,
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that) &
				noexcept(noexcept(rotation_part() = std::move(that).rotation_part()) &&
					noexcept(translation_part() = std::move(that).translation_part()))
			{
				rotation_part() = std::move(that).rotation_part();
				translation_part() = std::move(that).translation_part();
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr se3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<se3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR se3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<se3_elmt>::value)
			{
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr se3_elmt operator-() const&
				noexcept(noexcept(se3_elmt{ -rotation_part(), -translation_part() }))
			{
				return{ -rotation_part(), -translation_part() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR se3_elmt operator-() &&
				noexcept(noexcept(se3_elmt{ -std::move(*this).rotation_part(),
					-std::move(*this).translation_part() }))
			{
				return{ -std::move(*this).rotation_part(), -std::move(*this).translation_part() };
			}

			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator+=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(rotation_part() += that.rotation_part()) &&
					noexcept(translation_part() += that.translation_part()))
			{
				rotation_part() += that.rotation_part();
				translation_part() += that.translation_part();
				return *this;
			}
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator+=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(rotation_part() += std::move(that).rotation_part()) &&
					noexcept(translation_part() += std::move(that).translation_part()))
			{
				rotation_part() += std::move(that).rotation_part();
				translation_part() += std::move(that).translation_part();
				return *this;
			}

			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator-=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that)
				noexcept(noexcept(rotation_part() -= that.rotation_part()) &&
					noexcept(translation_part() -= that.translation_part()))
			{
				rotation_part() -= that.rotation_part();
				translation_part() -= that.translation_part();
				return *this;
			}
			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator-=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits>&& that)
				noexcept(noexcept(rotation_part() -= std::move(that).rotation_part()) &&
					noexcept(translation_part() -= std::move(that).translation_part()))
			{
				rotation_part() -= std::move(that).rotation_part();
				translation_part() -= std::move(that).translation_part();
				return *this;
			}

			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(rotation_part() *= k) && noexcept(translation_part() *= k))
			{
				rotation_part() *= k;
				translation_part() *= k;
				return *this;
			}
			template <class OtherComponentType>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR se3_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(rotation_part() /= k) && noexcept(translation_part() /= k))
			{
				rotation_part() /= k;
				translation_part() /= k;
				return *this;
			}

			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) const
				noexcept(noexcept(rotation_part() == that.rotation_part() &&
					translation_part() == that.translation_part()))
			{
				return rotation_part() == that.rotation_part() &&
					translation_part() == that.translation_part();
			}

			template <class OtherComponentType, class Otherso3Storage, class Otherso3StorageTraits,
				class OtherR3Storage, class OtherR3StorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				se3_elmt<OtherComponentType, Otherso3Storage, Otherso3StorageTraits,
				OtherR3Storage, OtherR3StorageTraits> const& that) const
				noexcept(noexcept(!(*this == that)))
			{
				return !(*this == that);
			}

			JKL_GPU_EXECUTABLE static constexpr se3_elmt zero() noexcept {
				return{ rotation_part_type::zero(), translation_part_type::zero() };
			}
		};


		//// Binary operations between SE3_elmt's & se3_elmt's

		namespace detail {
			template <class LeftOperand, class RightOperand>
			struct get_SE3_elmt_binary_result_impl {};

			template <class LeftComponentType, class LeftSU2Storage, class LeftSU2StorageTraits,
				class LeftR3Storage, class LeftR3StorageTraits,
				class RightComponentType, class RightSU2Storage, class RightSU2StorageTraits,
				class RightR3Storage, class RightR3StorageTraits>
			struct get_SE3_elmt_binary_result_impl<
				SE3_elmt<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
				LeftR3Storage, LeftR3StorageTraits>,
				SE3_elmt<RightComponentType, RightSU2Storage, RightSU2StorageTraits,
				RightR3Storage, RightR3StorageTraits>>
			{
				using type = SE3_elmt_binary_result<
					LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
					LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, RightSU2Storage, RightSU2StorageTraits,
					RightR3Storage, RightR3StorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_SE3_elmt_binary_result = typename get_SE3_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class LeftOperand, class RightOperand>
			struct get_se3_elmt_binary_result_impl {};

			template <class LeftComponentType, class Leftso3Storage, class Leftso3StorageTraits,
				class LeftR3Storage, class LeftR3StorageTraits,
				class RightComponentType, class Rightso3Storage, class Rightso3StorageTraits,
				class RightR3Storage, class RightR3StorageTraits>
			struct get_se3_elmt_binary_result_impl<
				se3_elmt<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
				LeftR3Storage, LeftR3StorageTraits>,
				se3_elmt<RightComponentType, Rightso3Storage, Rightso3StorageTraits,
				RightR3Storage, RightR3StorageTraits>>
			{
				using type = se3_elmt_binary_result<
					LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
					LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, Rightso3Storage, Rightso3StorageTraits,
					RightR3Storage, RightR3StorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_se3_elmt_binary_result = typename get_se3_elmt_binary_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class Scalar, class Vector, bool from_left>
			struct get_se3_elmt_scalar_mult_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class Scalar, bool from_left,
				class ComponentType, class so3Storage, class so3StorageTraits,
				class R3Storage, class R3StorageTraits>
			struct get_se3_elmt_scalar_mult_result_impl_impl<Scalar,
				se3_elmt<ComponentType, so3Storage, so3StorageTraits,
				R3Storage, R3StorageTraits>, from_left>
			{
				using type = se3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, so3Storage, so3StorageTraits,
					R3Storage, R3StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, class Vector, bool from_left>
			struct get_se3_elmt_scalar_mult_result_impl : std::conditional_t<
				get_se3_elmt_scalar_mult_result_impl_impl<Scalar, Vector, from_left>::value,
				get_se3_elmt_scalar_mult_result_impl_impl<Scalar, Vector, from_left>,
				get_se3_elmt_scalar_mult_result_impl_impl<void, void, false>> {};

			template <class Scalar, class Vector, bool from_left>
			using get_se3_elmt_scalar_mult_result = typename get_se3_elmt_scalar_mult_result_impl<
				tmp::remove_cvref_t<Scalar>,
				tmp::remove_cvref_t<Vector>, from_left>::type;
		}

		// Binary multiplication of SE3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator*(LeftOperand&& s, RightOperand&& t)
			noexcept(noexcept(detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>{
			s.rotation_q() * std::forward<RightOperand>(t).rotation_q(),
				s.rotation_q().rotate(std::forward<RightOperand>(t).translation()) +
				std::forward<LeftOperand>(s).translation() }))
			-> detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot multiply two SE3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply two SE3_elmt's; failed to deduce the resulting storage type");

			return{
				s.rotation_q() * std::forward<RightOperand>(t).rotation_q(),
				s.rotation_q().rotate(std::forward<RightOperand>(t).translation()) +
				std::forward<LeftOperand>(s).translation()
			};
		}

		// Binary division of SE3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator/(LeftOperand&& p, RightOperand&& q)
			noexcept(noexcept(std::forward<LeftOperand>(p) * std::forward<RightOperand>(q).inv()))
			-> detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot divide two SE3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot divide two SE3_elmt's; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(p) * std::forward<RightOperand>(q).inv();
		}

		// Interpolation
		template <class LeftOperand, class RightOperand, class Parameter>
		JKL_GPU_EXECUTABLE auto SU2_interpolation(LeftOperand&& p, RightOperand&& q, Parameter&& t)
			noexcept(noexcept(std::forward<LeftOperand>(p) *
				detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>::exp(
				(p.inv() * std::forward<RightOperand>(q)).log() * std::forward<Parameter>(t))))
			-> detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_SE3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot compute SE3 interpolation; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot compute SE3 interpolation; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(p) * result_type::exp(
				(p.inv() * std::forward<RightOperand>(q)).log() * std::forward<Parameter>(t));
		}


		// Binary addition of se3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator+(LeftOperand&& s, RightOperand&& t)
			noexcept(noexcept(detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(s).rotation_part() +
				std::forward<RightOperand>(t).rotation_part(),
				std::forward<LeftOperand>(s).translation_part() +
				std::forward<RightOperand>(t).translation_part() }))
			-> detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot add two se3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot add two se3_elmt's; failed to deduce the resulting storage type");

			return{
				std::forward<LeftOperand>(s).rotation_part() +
				std::forward<RightOperand>(t).rotation_part(),
				std::forward<LeftOperand>(s).translation_part() +
				std::forward<RightOperand>(t).translation_part()
			};
		}

		// Binary subtraction of se3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator-(LeftOperand&& s, RightOperand&& t)
			noexcept(noexcept(detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>{
			std::forward<LeftOperand>(s).rotation_part() -
				std::forward<RightOperand>(t).rotation_part(),
				std::forward<LeftOperand>(s).translation_part() -
				std::forward<RightOperand>(t).translation_part() }))
			-> detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot subtract two se3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot subtract two se3_elmt's; failed to deduce the resulting storage type");

			return{
				std::forward<LeftOperand>(s).rotation_part() -
				std::forward<RightOperand>(t).rotation_part(),
				std::forward<LeftOperand>(s).translation_part() -
				std::forward<RightOperand>(t).translation_part()
			};
		}

		// Commutator of se3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto commutator(LeftOperand&& s, RightOperand&& t)
			noexcept(noexcept(detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>{
			cross(s.rotation_part(), t.rotation_part()),
				cross(s.rotation_part(), std::forward<RightOperand>(t).translation_part()) -
				cross(t.rotation_part(), std::forward<LeftOperand>(s).translation_part()) }))
			-> detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_se3_elmt_binary_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkj::math: cannot compute commutator of se3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot compute commutator of se3_elmt's; failed to deduce the resulting storage type");

			return{
				cross(s.rotation_part(), t.rotation_part()),
				cross(s.rotation_part(), std::forward<RightOperand>(t).translation_part()) -
				cross(t.rotation_part(), std::forward<LeftOperand>(s).translation_part())
			};
		
		}

		// Scalar multiplication of se3_elmt's from right
		template <class Vector, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Vector&& v, Scalar const& k)
			noexcept(noexcept(detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>{
			std::forward<Vector>(v).rotation_part() * k,
				std::forward<Vector>(v).translation_part() * k }))
			-> detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>
		{
			using result_type = detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply se3_elmt with a scalar; failed to deduce the resulting storage type");

			return{
				std::forward<Vector>(v).rotation_part() * k,
				std::forward<Vector>(v).translation_part() * k
			};
		}

		// Scalar multiplication of se3_elmt's from left
		template <class Scalar, class Vector>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Scalar const& k, Vector&& v)
			noexcept(noexcept(detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>{
			k * std::forward<Vector>(v).rotation_part(),
				k * std::forward<Vector>(v).translation_part()}))
			-> detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, true>
		{
			using result_type = detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot multiply se3_elmt with a scalar; failed to deduce the resulting storage type");

			return{
				k * std::forward<Vector>(v).rotation_part(),
				k * std::forward<Vector>(v).translation_part()
			};
		}

		// Scalar division of se3_elmt's from right
		template <class Vector, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Vector&& v, Scalar const& k)
			noexcept(noexcept(detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>{
			std::forward<Vector>(v).rotation_part() / k,
				std::forward<Vector>(v).translation_part() / k }))
			-> detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>
		{
			using result_type = detail::get_se3_elmt_scalar_mult_result<Scalar, Vector, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkj::math: cannot divide se3_elmt by a scalar; failed to deduce the resulting storage type");

			return{
				std::forward<Vector>(v).rotation_part() / k,
				std::forward<Vector>(v).translation_part() / k
			};
		}
	}
}