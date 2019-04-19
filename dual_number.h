#pragma once
#include <type_traits>
#include <utility>
#include "tmp/forward.h"
#include "tmp/generate_constructors.h"
#include "numerical_lie_group.h"

namespace jkj {
	namespace math {
		namespace detail {
			// Due to lack of generalized constexpr support in MSVC 2015, we need a workaround to
			// fully utilize move semantics

			template <class DualNumber, class Prim, class Dual, class PrimInv>
			JKL_GPU_EXECUTABLE constexpr DualNumber inv_calc(Prim&& p, Dual&& d, PrimInv&& i)
				noexcept(noexcept(DualNumber{ std::forward<PrimInv>(i),
					i * (std::forward<Dual>(d) / std::forward<Prim>(p)) }))
			{
				return{ std::forward<PrimInv>(i), i * (std::forward<Dual>(d) / std::forward<Prim>(p)) };
			}

			template <class DualNumber, class PrimResult, class Prim1, class Dual1, class Prim2, class Dual2>
			JKL_GPU_EXECUTABLE constexpr DualNumber multi_calc(PrimResult&& r,
				Prim1&& p1, Dual1&& d1, Prim2&& p2, Dual2&& d2)
				noexcept(noexcept(DualNumber{ std::forward<PrimResult>(r),
					std::forward<Prim1>(p1) * std::forward<Dual2>(d2)
					+ std::forward<Dual1>(d1) * std::forward<Prim2>(p2) }))
			{
				return{ std::forward<PrimResult>(r), std::forward<Prim1>(p1) * std::forward<Dual2>(d2)
					+ std::forward<Dual1>(d1) * std::forward<Prim2>(p2) };
			}

			template <class DualNumber, class PrimResult, class First, class Second>
			JKL_GPU_EXECUTABLE constexpr DualNumber multi_calc(PrimResult&& r, First&& f, Second&& s)
				noexcept(noexcept(DualNumber{ std::forward<PrimResult>(r),
					std::forward<First>(f) * std::forward<Second>(s) }))
			{
				return{ std::forward<PrimResult>(r), std::forward<First>(f) * std::forward<Second>(s) };
			}

			template <class DualNumber, class PrimResult, class Dual1, class Dual2, class PrimDivisor>
			JKL_GPU_EXECUTABLE constexpr DualNumber div_calc(PrimResult&& aci,
				Dual1&& b, Dual2&& d, PrimDivisor&& c)
				noexcept(noexcept(DualNumber{ std::forward<PrimResult>(aci),
				(std::forward<Dual1>(b) - aci * std::forward<Dual2>(d)) / std::forward<PrimDivisor>(c) }))
			{
				return{ std::forward<PrimResult>(aci),
					(std::forward<Dual1>(b) - aci * std::forward<Dual2>(d)) / std::forward<PrimDivisor>(c) };
			}

			template <class DualNumber, class PrimResult, class Dividend, class Divisor>
			JKL_GPU_EXECUTABLE constexpr DualNumber div_calc(PrimResult&& r, Dividend&& x, Divisor&& y)
				noexcept(noexcept(DualNumber{ std::forward<PrimResult>(r),
					std::forward<Dividend>(x) / std::forward<Divisor>(y) }))
			{
				return{ std::forward<PrimResult>(r), std::forward<Dividend>(x) / std::forward<Divisor>(y) };
			}

			struct dual_number_forward_tag {};

			template <class RingElmt, class DualType>
			struct dual_number_base {
				dual_number_base() = default;

				RingElmt		prim;
				DualType		dual;

			protected:
				template <class...>
				struct is_nothrow_constructible : std::false_type {};

				template <class Prim, class Dual>
				struct is_nothrow_constructible<Prim, Dual> {
					static constexpr bool value =
						std::is_nothrow_constructible<RingElmt, Prim>::value &&
						std::is_nothrow_constructible<DualType, Dual>::value;
				};

				template <class Prim, class Dual>
				JKL_GPU_EXECUTABLE constexpr dual_number_base(dual_number_forward_tag,
					Prim&& prim, Dual&& dual) noexcept(is_nothrow_constructible<Prim, Dual>::value) :
					prim(std::forward<Prim>(prim)), dual(std::forward<Dual>(dual)) {}
			};
		}

		template <class RingElmt, class DualType = RingElmt>
		class dual_number : public tmp::generate_constructors<
			detail::dual_number_base<RingElmt, DualType>,
			detail::dual_number_forward_tag,
			tmp::copy_or_move<RingElmt, DualType>,
			tmp::copy_or_move<RingElmt>>
		{
			using base_type = detail::dual_number_base<RingElmt, DualType>;
			using constructor_provider = tmp::generate_constructors<
				detail::dual_number_base<RingElmt, DualType>,
				detail::dual_number_forward_tag,
				tmp::copy_or_move<RingElmt, DualType>,
				tmp::copy_or_move<RingElmt>>;

		public:
			using constructor_provider::prim;
			using constructor_provider::dual;

			using prim_type = RingElmt;
			using dual_type = DualType;

			using constructor_provider::constructor_provider;

			// Default constructor
			dual_number() = default;

			// Standard constructor taking two elements (implicit)
			template <class Prim, class Dual, class = std::enable_if_t<
				std::is_constructible<RingElmt, Prim>::value &&
				std::is_constructible<DualType, Dual>::value &&
				std::is_convertible<Prim, RingElmt>::value &&
				std::is_convertible<Dual, DualType>::value>>
			JKL_GPU_EXECUTABLE constexpr dual_number(Prim&& prim, Dual&& dual)
				noexcept(base_type::template is_nothrow_constructible<Prim, Dual>::value) :
				constructor_provider(detail::dual_number_forward_tag{},
					std::forward<Prim>(prim), std::forward<Dual>(dual)) {}

			// Standard constructor taking two elements (explicit)
			template <class Prim, class Dual, class = void, class = std::enable_if_t<
				std::is_constructible<RingElmt, Prim>::value &&
				std::is_constructible<DualType, Dual>::value &&
				!(std::is_convertible<Prim, RingElmt>::value &&
					std::is_convertible<Dual, DualType>::value)>>
			JKL_GPU_EXECUTABLE explicit constexpr dual_number(Prim&& prim, Dual&& dual)
				noexcept(base_type::template is_nothrow_constructible<Prim, Dual>::value) :
				constructor_provider(detail::dual_number_forward_tag{},
					std::forward<Prim>(prim), std::forward<Dual>(dual)) {}

			// Leave the dual part zero (implicit)
			template <class Prim,
				class = std::enable_if_t<
				std::is_constructible<RingElmt, Prim>::value &&
				std::is_convertible<Prim, RingElmt>::value>,
				class = jkj::tmp::prevent_too_perfect_fwd<dual_number, Prim>>
			JKL_GPU_EXECUTABLE constexpr dual_number(Prim&& prim)
				noexcept(base_type::template is_nothrow_constructible<Prim,
					decltype(jkj::math::zero<DualType>())>::value) :
				constructor_provider(detail::dual_number_forward_tag{},
					std::forward<Prim>(prim), jkj::math::zero<DualType>()) {}

			// Leave the dual part zero (explicit)
			template <class Prim, class = void,
				class = std::enable_if_t<
				std::is_constructible<RingElmt, Prim>::value &&
				!std::is_convertible<Prim, RingElmt>::value>,
				class = jkj::tmp::prevent_too_perfect_fwd<dual_number, Prim>>
			JKL_GPU_EXECUTABLE explicit constexpr dual_number(Prim&& prim)
				noexcept(base_type::template is_nothrow_constructible<Prim,
					decltype(jkj::math::zero<DualType>())>::value) :
				constructor_provider(detail::dual_number_forward_tag{},
					std::forward<Prim>(prim), jkj::math::zero<DualType>()) {}


			/// Nullary operations

			// Zero
			JKL_GPU_EXECUTABLE static constexpr dual_number zero()
			{
				return{ jkj::math::zero<RingElmt>(), jkj::math::zero<DualType>() };
			}

			// Unity
			JKL_GPU_EXECUTABLE static constexpr dual_number unity()
			{
				return{ jkj::math::unity<RingElmt>(), jkj::math::zero<DualType>() };
			}


			/// Unary operations

			// Unary plus
			JKL_GPU_EXECUTABLE constexpr dual_number operator+() const&
				noexcept(std::is_nothrow_copy_constructible<dual_number>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator+() &&
				noexcept(std::is_nothrow_move_constructible<dual_number>::value)
			{
				return std::move(*this);
			}

			// Unary negation
			JKL_GPU_EXECUTABLE constexpr dual_number operator-() const&
				noexcept(noexcept(dual_number{ -prim, -dual }))
			{
				return{ -prim, -dual };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator-() &&
				noexcept(noexcept(dual_number{ -std::move(prim), -std::move(dual) }))
			{
				return{ -std::move(prim), -std::move(dual) };
			}

			// Reciprocal
			JKL_GPU_EXECUTABLE constexpr dual_number inv() const&
				noexcept(noexcept(detail::inv_calc<dual_number>(prim, dual, jkj::math::general_inverse(prim))))
			{
				return detail::inv_calc<dual_number>(prim, dual, jkj::math::general_inverse(prim));
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number inv() &&
				noexcept(noexcept(detail::inv_calc<dual_number>(std::move(prim), std::move(dual), jkj::math::general_inverse(prim))))
			{
				return detail::inv_calc<dual_number>(std::move(prim), std::move(dual), jkj::math::general_inverse(prim));
			}


			/// Addition

			// Inplace addition with dual_number
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator+=(dual_number const& that) &
				noexcept(noexcept(prim += that.prim))
			{
				return prim += that.prim, dual += that.dual, *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator+=(dual_number&& that) &
				noexcept(noexcept(prim += std::move(that.prim)))
			{
				return prim += std::move(that.prim), dual += std::move(that.dual), *this;
			}

			// Inplace addition with RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator+=(Prim&& that) &
				noexcept(noexcept(prim += std::forward<Prim>(that)))
			{
				return prim += std::forward<Prim>(that), *this;
			}

			// Addition with dual_number
			JKL_GPU_EXECUTABLE constexpr dual_number operator+(dual_number const& that) const&
				noexcept(noexcept(dual_number{ prim + that.prim, dual + that.dual }))
			{
				return{ prim + that.prim, dual + that.dual };
			}
			JKL_GPU_EXECUTABLE constexpr dual_number operator+(dual_number&& that) const&
				noexcept(noexcept(dual_number{ prim + std::move(that.prim), dual + std::move(that.dual) }))
			{
				return{ prim + std::move(that.prim), dual + std::move(that.dual) };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator+(dual_number const& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) + that.prim, std::move(dual) + that.dual }))
			{
				return{ std::move(prim) + that.prim, std::move(dual) + that.dual };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator+(dual_number&& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) + std::move(that.prim), 
					std::move(dual) + std::move(that.dual) }))
			{
				return{ std::move(prim) + std::move(that.prim), std::move(dual) + std::move(that.dual) };
			}

			// Addition with RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE constexpr dual_number operator+(Prim&& that) const&
				noexcept(noexcept(dual_number{ prim + std::forward<Prim>(that), dual }))
			{
				return{ prim + std::forward<Prim>(that), dual };
			}
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator+(Prim&& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) + std::forward<Prim>(that), std::move(dual) }))
			{
				return{ std::move(prim) + std::forward<Prim>(that), std::move(dual) };
			}


			/// Subtraction

			// Inplace subtraction by dual_number
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator-=(dual_number const& that) &
				noexcept(noexcept(prim -= that.prim))
			{
				return prim -= that.prim, dual -= that.dual, *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator-=(dual_number&& that) &
				noexcept(noexcept(prim -= std::move(that.prim)))
			{
				return prim -= std::move(that.prim), dual -= std::move(that.dual), *this;
			}

			// Inplace subtraction by RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator-=(Prim&& that) &
				noexcept(noexcept(prim -= std::forward<Prim>(that)))
			{
				return prim -= std::forward<Prim>(that), *this;
			}

			// Subtraction by dual_number
			JKL_GPU_EXECUTABLE constexpr dual_number operator-(dual_number const& that) const&
				noexcept(noexcept(dual_number{ prim - that.prim, dual - that.dual }))
			{
				return{ prim - that.prim, dual - that.dual };
			}
			JKL_GPU_EXECUTABLE constexpr dual_number operator-(dual_number&& that) const&
				noexcept(noexcept(dual_number{ prim - std::move(that.prim), dual - std::move(that.dual) }))
			{
				return{ prim - std::move(that.prim), dual - std::move(that.dual) };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator-(dual_number const& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) - that.prim, std::move(dual) - that.dual }))
			{
				return{ std::move(prim) - that.prim, std::move(dual) - that.dual };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator-(dual_number&& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) - std::move(that.prim),
					std::move(dual) - std::move(that.dual) }))
			{
				return{ std::move(prim) - std::move(that.prim), std::move(dual) - std::move(that.dual) };
			}

			// Subtraction by RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE constexpr dual_number operator-(Prim&& that) const&
				noexcept(noexcept(dual_number{ prim - std::forward<Prim>(that), dual }))
			{
				return{ prim - std::forward<Prim>(that), dual };
			}
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator-(Prim&& that) &&
				noexcept(noexcept(dual_number{ std::move(prim) - std::forward<Prim>(that), std::move(dual) }))
			{
				return{ std::move(prim) - std::forward<Prim>(that), std::move(dual) };
			}


			/// Multipliaction

			// Inplace multiplication with dual_number
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator*=(dual_number const& that) &
				noexcept(noexcept(dual *= that.prim) && noexcept(dual += prim * that.dual))
			{
				return dual *= that.prim, 
					dual += prim * that.dual, 
					prim *= that.prim, 
					*this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator*=(dual_number&& that) &
				noexcept(noexcept(dual *= that.prim) && noexcept(dual += prim * std::move(that.dual))
					&& noexcept(prim *= std::move(that.prim)))
			{
				return dual *= that.prim, 
					dual += prim * std::move(that.dual), 
					prim *= std::move(that.prim), 
					*this;
			}

			// Inplace multiplication with RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator*=(Prim&& that) &
			{
				return prim *= that,
					dual *= std::forward<Prim>(that),
					*this;
			}

			// Multiplication with dual_number
			JKL_GPU_EXECUTABLE constexpr dual_number operator*(dual_number const& that) const&
				noexcept(noexcept(detail::multi_calc<dual_number>(prim * that.prim, prim, dual, that.prim, that.dual)))
			{
				return detail::multi_calc<dual_number>(prim * that.prim, prim, dual, that.prim, that.dual);
			}
			JKL_GPU_EXECUTABLE constexpr dual_number operator*(dual_number&& that) const&
				noexcept(noexcept(detail::multi_calc<dual_number>(prim * that.prim, prim, dual, std::move(that.prim), std::move(that.dual))))
			{
				return detail::multi_calc<dual_number>(prim * that.prim, prim, dual, std::move(that.prim), std::move(that.dual));
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator*(dual_number const& that) &&
				noexcept(noexcept(detail::multi_calc<dual_number>(prim * that.prim, std::move(prim), std::move(dual), that.prim, that.dual)))
			{
				return detail::multi_calc<dual_number>(prim * that.prim, std::move(prim), std::move(dual), that.prim, that.dual);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator*(dual_number&& that) &&
				noexcept(noexcept(detail::multi_calc<dual_number>(prim * that.prim, std::move(prim), std::move(dual),
					std::move(that.prim), std::move(that.dual))))
			{
				return detail::multi_calc<dual_number>(prim * that.prim, std::move(prim), std::move(dual), 
					std::move(that.prim), std::move(that.dual));
			}

			// Multiplication with RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE constexpr dual_number operator*(Prim&& that) const&
				noexcept(noexcept(detail::multi_calc<dual_number>(prim * that, dual, std::forward<Prim>(that))))
			{
				return detail::multi_calc<dual_number>(prim * that, dual, std::forward<Prim>(that));
			}
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator*(Prim&& that) &&
				noexcept(noexcept(detail::multi_calc<dual_number>(std::move(prim) * that, std::move(dual), std::forward<Prim>(that))))
			{
				return detail::multi_calc<dual_number>(std::move(prim) * that, std::move(dual), std::forward<Prim>(that));
			}


			/// Division
			/// Error handling for the case of non-invertible elements is a responibility of RingElmt

			// Inplace division by dual_number			
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator/=(dual_number const& that) &
				noexcept(noexcept(prim /= that.prim) && noexcept(dual -= prim * that.dual))
			{
				return prim /= that.prim,
					dual -= prim * that.dual,
					dual /= that.prim,
					*this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator/=(dual_number&& that) &
			{
				return prim /= that.prim,
					dual -= prim * std::move(that.dual),
					dual /= std::move(that.prim),
					*this;
			}

			// Inplace division by RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number& operator/=(Prim&& that) &
			{
				return prim /= that,
					dual /= std::forward<Prim>(that),
					*this;
			}

			// Division by dual_number
			JKL_GPU_EXECUTABLE constexpr dual_number operator/(dual_number const& that) const&
				noexcept(noexcept(detail::div_calc<dual_number>(prim / that.prim, dual, that.dual, that.prim)))
			{
				return detail::div_calc<dual_number>(prim / that.prim, dual, that.dual, that.prim);
			}
			JKL_GPU_EXECUTABLE constexpr dual_number operator/(dual_number&& that) const&
				noexcept(noexcept(detail::div_calc<dual_number>(prim / that.prim, dual, std::move(that.dual), std::move(that.prim))))
			{
				return detail::div_calc<dual_number>(prim / that.prim, dual, std::move(that.dual), std::move(that.prim));
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator/(dual_number const& that) &&
				noexcept(noexcept(detail::div_calc<dual_number>(std::move(prim) / that.prim, std::move(dual), that.dual, that.prim)))
			{
				return detail::div_calc<dual_number>(std::move(prim) / that.prim, std::move(dual), that.dual, that.prim);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator/(dual_number&& that) &&
				noexcept(noexcept(detail::div_calc<dual_number>(std::move(prim) / that.prim,
					std::move(dual), std::move(that.dual), std::move(that.prim))))
			{
				return detail::div_calc<dual_number>(std::move(prim) / that.prim, 
					std::move(dual), std::move(that.dual), std::move(that.prim));
			}

			// Division by RingElmt
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE constexpr dual_number operator/(Prim&& that) const&
				noexcept(noexcept(detail::div_calc<dual_number>(prim / that, dual, std::forward<Prim>(that))))
			{
				return detail::div_calc<dual_number>(prim / that, dual, std::forward<Prim>(that));
			}
			template <class Prim,
				class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR dual_number operator/(Prim&& that) &&
				noexcept(noexcept(detail::div_calc<dual_number>(std::move(prim) / that, std::move(dual), std::forward<Prim>(that))))
			{
				return detail::div_calc<dual_number>(std::move(prim) / that, std::move(dual), std::forward<Prim>(that));
			}


			/// Relations

			JKL_GPU_EXECUTABLE constexpr bool operator==(dual_number const& that) const& {
				return prim == that.prim && dual == that.dual;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator==(dual_number&& that) const& {
				return prim == std::move(that.prim) && dual == std::move(that.dual);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator==(dual_number const& that) && {
				return std::move(prim) == that.prim && std::move(dual) == that.dual;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator==(dual_number&& that) && {
				return std::move(prim) == std::move(that.prim) && std::move(dual) == std::move(that.dual);
			}

			JKL_GPU_EXECUTABLE constexpr bool operator!=(dual_number const& that) const& {
				return prim != that.prim || dual != that.dual;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator!=(dual_number&& that) const& {
				return prim != std::move(that.prim) || dual != std::move(that.dual);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator!=(dual_number const& that) && {
				return std::move(prim) != that.prim || std::move(dual) != that.dual;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator!=(dual_number&& that) && {
				return std::move(prim) != std::move(that.prim) || std::move(dual) != std::move(that.dual);
			}

			JKL_GPU_EXECUTABLE constexpr bool operator<(dual_number const& that) const& {
				return prim < that.prim;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator<(dual_number&& that) const& {
				return prim < std::move(that.prim);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator<(dual_number const& that) && {
				return std::move(prim) < that.prim;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator<(dual_number&& that) && {
				return std::move(prim) < std::move(that.prim);
			}

			JKL_GPU_EXECUTABLE constexpr bool operator>(dual_number const& that) const& {
				return prim > that.prim;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator>(dual_number&& that) const& {
				return prim > std::move(that.prim);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator>(dual_number const& that) && {
				return std::move(prim) > that.prim;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator>(dual_number&& that) && {
				return std::move(prim) > std::move(that.prim);
			}

			JKL_GPU_EXECUTABLE constexpr bool operator<=(dual_number const& that) const& {
				return prim <= that.prim;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator<=(dual_number&& that) const& {
				return prim <= std::move(that.prim);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator<=(dual_number const& that) && {
				return std::move(prim) <= that.prim;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator<=(dual_number&& that) && {
				return std::move(prim) <= std::move(that.prim);
			}

			JKL_GPU_EXECUTABLE constexpr bool operator>=(dual_number const& that) const& {
				return prim >= that.prim;
			}
			JKL_GPU_EXECUTABLE constexpr bool operator>=(dual_number&& that) const& {
				return prim >= std::move(that.prim);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator>=(dual_number const& that) && {
				return std::move(prim) >= that.prim;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool operator>=(dual_number&& that) && {
				return std::move(prim) >= std::move(that.prim);
			}


			/// Selected functions in <cmath>
			/// Here we suppose the ring is commutative.

			/// Trigonometric functions

			JKL_GPU_EXECUTABLE friend dual_number cos(dual_number const& x) {
				using std::cos;
				using std::sin;
				return{ cos(x.prim), -x.dual * sin(x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number sin(dual_number const& x) {
				using std::cos;
				using std::sin;
				return{ sin(x.prim), x.dual * cos(x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number tan(dual_number const& x) {
				using std::tan;
				auto t = tan(x.prim);
				return{ t, x.dual * (1 + t * t) };
			}
			JKL_GPU_EXECUTABLE friend dual_number acos(dual_number const& x) {
				using std::acos;
				using std::sqrt;
				return{ acos(x.prim), -x.dual / sqrt(1 - x.prim * x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number asin(dual_number const& x) {
				using std::asin;
				using std::sqrt;
				return{ asin(x.prim), x.dual / sqrt(1 - x.prim * x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number atan(dual_number const& x) {
				using std::atan;
				return{ atan(x.prim), x.dual / (1 + x.prim * x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number atan2(dual_number const& y, dual_number const& x) {
				using std::atan2;
				auto sq_rad = x.prim*x.prim + y.prim*y.prim;
				return{ atan2(y.prim, x.prim), (y.dual * x.prim - x.dual * y.prim) / sq_rad };
			}


			/// Hyperbolic functions
			
			JKL_GPU_EXECUTABLE friend dual_number cosh(dual_number const& x) {
				using std::cosh;
				using std::sinh;
				return{ cosh(x.prim), x.dual * sinh(x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number sinh(dual_number const& x) {
				using std::cosh;
				using std::sinh;
				return{ sinh(x.prim), x.dual * cosh(x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number tanh(dual_number const& x) {
				using std::tanh;
				auto t = tanh(x.prim);
				return{ t, x.dual * (1 - t * t) };
			}
			JKL_GPU_EXECUTABLE friend dual_number acosh(dual_number const& x) {
				using std::acosh;
				using std::sqrt;
				return{ acosh(x.prim), x.dual / sqrt(x.prim * x.prim - 1) };
			}
			JKL_GPU_EXECUTABLE friend dual_number asinh(dual_number const& x) {
				using std::asinh;
				using std::sqrt;
				return{ asinh(x.prim), x.dual / sqrt(x.prim * x.prim + 1) };
			}
			JKL_GPU_EXECUTABLE friend dual_number atanh(dual_number const& x) {
				using std::atanh;
				return{ atanh(x.prim), x.dual / (1 - x.prim * x.prim) };
			}


			/// Exponential & logarithmic functions

			JKL_GPU_EXECUTABLE friend dual_number exp(dual_number const& x) {
				using std::exp;
				auto e = exp(x.prim);
				return{ e, x.dual * e };
			}
			JKL_GPU_EXECUTABLE friend dual_number log(dual_number const& x) {
				using std::log;
				return{ log(x.prim), x.dual / x.prim };
			}
			JKL_GPU_EXECUTABLE friend dual_number log10(dual_number const& x) {
				using std::log10;
				using std::log;
				return{ log10(x.prim), x.dual / (x.prim * log(10)) };
			}
			JKL_GPU_EXECUTABLE friend dual_number exp2(dual_number const& x) {
				using std::exp2;
				using std::log;
				auto e = exp2(x.prim);
				return{ e, x.dual * e * log(2) };
			}
			JKL_GPU_EXECUTABLE friend dual_number expm1(dual_number const& x) {
				using std::expm1;
				auto e = expm1(x.prim);
				return{ e, x.dual * (e + 1) };
			}
			JKL_GPU_EXECUTABLE friend dual_number log1p(dual_number const& x) {
				using std::log1p;
				return{ log1p(x.prim), x.dual / (1 + x.prim) };
			}
			JKL_GPU_EXECUTABLE friend dual_number log2(dual_number const& x) {
				using std::log2;
				using std::log;
				return{ log2(x.prim), x.dual / (x.prim * log(2)) };
			}


			/// Power functions

			JKL_GPU_EXECUTABLE friend dual_number pow(dual_number const& base, dual_number const& exponent) {
				using std::pow;
				using std::log;
				auto p = pow(base.prim, exponent.prim);
				return{ p, base.dual * p * exponent.prim / base.prim + exponent.dual * log(base.prim) };
			}

			JKL_GPU_EXECUTABLE friend dual_number sqrt(dual_number const& x) {
				using std::sqrt;
				auto r = sqrt(x.prim);
				return{ r, x.dual / (2 * r) };
			}

			JKL_GPU_EXECUTABLE friend dual_number cbrt(dual_number const& x) {
				using std::cbrt;
				auto r = cbrt(x.prim);
				return{ r, x.dual / (3 * r * r) };
			}

			JKL_GPU_EXECUTABLE friend dual_number hypot(dual_number const& x, dual_number const& y) {
				using std::hypot;
				auto h = hypot(x.prim, y.prim);
				return{ h, (x.dual * x.prim + y.dual * y.prim) / h };
			}


			/// Error and gamma functions

			JKL_GPU_EXECUTABLE friend dual_number erf(dual_number const& x) {
				using std::erf;
				using std::exp;
				return{ erf(x.prim), x.dual * 2 / constants<RingElmt>::sqrt_pi * exp(-x.prim * x.prim) };
			}

			JKL_GPU_EXECUTABLE friend dual_number erfc(dual_number const& x) {
				using std::erfc;
				using std::exp;
				return{ erfc(x.prim), -x.dual * 2 / constants<RingElmt>::sqrt_pi * exp(-x.prim * x.prim) };
			}

			
			/// Other functions

			JKL_GPU_EXECUTABLE friend dual_number abs(dual_number const& x) {
				using std::abs;
				return{ abs(x.prim), x.dual * (x.prim > 0 ? 1 : x.prim < 0 ? -1 : 0) };
			}
		};


		// RingElmt + dual_number
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator+(Prim&& x, dual_number<RingElmt, DualType> const& y)
			noexcept(noexcept(dual_number<RingElmt, DualType>{ std::forward<Prim>(x) + y.prim, y.dual }))
		{
			return{ std::forward<Prim>(x) + y.prim, y.dual };
		}
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator+(Prim&& x, dual_number<RingElmt, DualType>&& y)
			noexcept(noexcept(dual_number<RingElmt, DualType>{ std::forward<Prim>(x) + std::move(y.prim), std::move(y.dual) }))
		{
			return{ std::forward<Prim>(x) + std::move(y.prim), std::move(y.dual) };
		}

		// RingElmt - dual_number
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator-(Prim&& x, dual_number<RingElmt, DualType> const& y)
			noexcept(noexcept(dual_number<RingElmt, DualType>{ std::forward<Prim>(x) - y.prim, -y.dual }))
		{
			return{ std::forward<Prim>(x) - y.prim, -y.dual };
		}
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator-(Prim&& x, dual_number<RingElmt, DualType>&& y)
			noexcept(noexcept(dual_number<RingElmt, DualType>{ std::forward<Prim>(x) - std::move(y.prim), -std::move(y.dual) }))
		{
			return{ std::forward<Prim>(x) - std::move(y.prim), -std::move(y.dual) };
		}

		// RingElmt * dual_number
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator*(Prim&& x, dual_number<RingElmt, DualType> const& y)
			noexcept(noexcept(detail::multi_calc<dual_number<RingElmt, DualType>>(
				x * y.prim, std::forward<Prim>(x), y.dual)))
		{
			return detail::multi_calc<dual_number<RingElmt, DualType>>(x * y.prim, 
				std::forward<Prim>(x), y.dual);
		}
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator*(Prim&& x, dual_number<RingElmt, DualType>&& y)
			noexcept(noexcept(detail::multi_calc<dual_number<RingElmt, DualType>>(
				x * std::move(y.prim), std::forward<Prim>(x), std::move(y.dual))))
		{
			return detail::multi_calc<dual_number<RingElmt, DualType>>(x * std::move(y.prim), 
				std::forward<Prim>(x), std::move(y.dual));
		}

		// RingElmt / dual_number
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator/(Prim&& x, dual_number<RingElmt, DualType> const& y)
			noexcept(noexcept(detail::div_calc<dual_number<RingElmt, DualType>>(
				x / std::forward<Prim>(y.prim), x, std::forward<Prim>(y.dual))))
		{
			return detail::div_calc<dual_number<RingElmt, DualType>>(x / y.prim,
				std::forward<Prim>(x), y.dual);
		}
		template <class Prim, class RingElmt, class DualType,
			class = std::enable_if_t<std::is_convertible<Prim, RingElmt>::value>>
		JKL_GPU_EXECUTABLE constexpr dual_number<RingElmt, DualType>
			operator/(Prim&& x, dual_number<RingElmt, DualType>&& y)
			noexcept(noexcept(detail::div_calc<dual_number<RingElmt, DualType>>(
				x / std::move(y.prim), std::forward<Prim>(x), std::move(y.dual))))
		{
			return detail::div_calc<dual_number<RingElmt, DualType>>(x / std::move(y.prim),
				std::forward<Prim>(x), std::move(y.dual));
		}

		// Get primary / dual part of a dual number
		namespace detail {
			template <class DualNumber>
			struct primary_dual_part_impl {
				template <class RingElmt, class = std::enable_if_t<std::is_lvalue_reference<RingElmt>::value>>
				JKL_GPU_EXECUTABLE static constexpr auto&& prim(RingElmt&& x) noexcept {
					return x;
				}
				template <class RingElmt, class = std::enable_if_t<!std::is_lvalue_reference<RingElmt>::value>, class = void>
				JKL_GPU_EXECUTABLE static constexpr auto prim(RingElmt&& x) noexcept {
					return std::move(x);
				}

				template <class RingElmt>
				JKL_GPU_EXECUTABLE static constexpr auto dual(RingElmt&& x) noexcept {
					return RingElmt(0);
				}
			};

			template <class RingElmt, class DualType>
			struct primary_dual_part_impl<dual_number<RingElmt, DualType>> {
				template <class DualNumber, class = std::enable_if_t<std::is_lvalue_reference<DualNumber>::value>>
				JKL_GPU_EXECUTABLE static constexpr auto&& prim(DualNumber&& x) noexcept {
					return x.prim;
				}
				template <class DualNumber, class = std::enable_if_t<!std::is_lvalue_reference<DualNumber>::value>, class = void>
				JKL_GPU_EXECUTABLE static constexpr auto prim(DualNumber&& x) noexcept {
					return std::move(x.prim);
				}

				template <class DualNumber, class = std::enable_if_t<std::is_lvalue_reference<DualNumber>::value>>
				JKL_GPU_EXECUTABLE static constexpr auto&& dual(DualNumber&& x) noexcept {
					return x.dual;
				}
				template <class DualNumber, class = std::enable_if_t<!std::is_lvalue_reference<DualNumber>::value>, class = void>
				JKL_GPU_EXECUTABLE static constexpr auto dual(DualNumber&& x) noexcept {
					return std::move(x.dual);
				}
			};
		}
		template <class DualNumber>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) primary_part(DualNumber&& x) noexcept {
			return detail::primary_dual_part_impl<std::decay_t<DualNumber>>::prim(std::forward<DualNumber>(x));
		}
		template <class DualNumber>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) dual_part(DualNumber&& x) noexcept {
			return detail::primary_dual_part_impl<std::decay_t<DualNumber>>::dual(std::forward<DualNumber>(x));
		}
	}
}
