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
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "../portability.h"
#include "../tmp/generate_constructors.h"
#include "../tmp/remove_cvref.h"
#include "../tmp/forward.h"
#include "../tmp/is_convertible.h"
#include "storage_traits.h"

namespace jkl {
	namespace math {
		//// Get the additive identity of the given type
		namespace detail {
			template <class T, class = void>
			struct has_zero : std::false_type {
				using type = void;
			};

			template <class T>
			struct has_zero<T, VOID_T<decltype(T::zero())>> : std::true_type {
				using type = decltype(T::zero());
			};
		}
		// If there is a static member function zero(), call it
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto zero() noexcept(noexcept(T::zero()))
			-> std::enable_if_t<detail::has_zero<tmp::remove_cvref_t<T>>::value,
			typename detail::has_zero<tmp::remove_cvref_t<T>>::type>
		{
			return T::zero();
		}
		// Otherwise, construct from the literal constant 0
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto zero() noexcept(noexcept(T(0)))
			->std::enable_if_t<!detail::has_zero<tmp::remove_cvref_t<T>>::value &&
			std::is_constructible<T, int>::value, tmp::remove_cvref_t<T>>
		{
			return T(0);
		}

		//// Get the multiplicative identity of the given type
		namespace detail {
			template <class T, class = void>
			struct has_unity : std::false_type {
				using type = void;
			};

			template <class T>
			struct has_unity<T, VOID_T<decltype(T::unity())>> : std::true_type {
				using type = decltype(T::unity());
			};
		}
		// If there is a static member function unity(), call it
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto unity() noexcept(noexcept(T::unity()))
			-> std::enable_if_t<detail::has_unity<tmp::remove_cvref_t<T>>::value,
			typename detail::has_unity<tmp::remove_cvref_t<T>>::type>
		{
			return T::unity();
		}
		// Otherwise, construct from the literal constant 1
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto unity() noexcept(noexcept(T(1)))
			->std::enable_if_t<!detail::has_unity<tmp::remove_cvref_t<T>>::value &&
			std::is_constructible<T, int>::value, tmp::remove_cvref_t<T>>
		{
			return T(1);
		}

		//// Get the multiplicative inverse
		// The name "general_inverse" is somewhat ugly, but I think the name "inv" can be potentially dangerous,
		// since there can be other functions with the same name in jkl::math.
		// Universal reference is too greedy to define the default fallback behavior.
		namespace detail {
			template <class T, class = void>
			struct has_mem_inv : std::false_type {
				using type = void;
			};

			template <class T>
			struct has_mem_inv<T, VOID_T<decltype(std::declval<T>().inv())>> : std::true_type {
				using type = decltype(std::declval<T>().inv());
			};
		}
		// If there is a member funtion inv(), call it
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto general_inverse(T&& x)
			noexcept(noexcept(std::declval<T>().inv()))
			-> std::enable_if_t<detail::has_mem_inv<T>::value, typename detail::has_mem_inv<T>::type>
		{
			return std::forward<T>(x).inv();
		}
		// Otherwise, calculate 1/x
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto general_inverse(T&& x)
			noexcept(noexcept(unity<tmp::remove_cvref_t<T>>() / std::forward<T>(x)))
			-> std::enable_if_t<!detail::has_mem_inv<T>::value,
			decltype(unity<tmp::remove_cvref_t<T>>() / std::forward<T>(x))>
		{
			return unity<tmp::remove_cvref_t<T>>() / std::forward<T>(x);
		}

		//// Check if an element is invertible
		namespace detail {
			template <class T, class = void>
			struct has_is_invertible : std::false_type {};

#if defined(_MSC_VER) && _MSC_VER <= 1900
			// Workaround for void_t bug; seems to be MSVC2015-only
			struct void_t_workaround_has_is_invertible {};

			template <class T>
			struct has_is_invertible<T, VOID_T<decltype(std::declval<T>().is_invertible()),
				void_t_workaround_has_is_invertible>> : std::true_type {};
#else
			template <class T>
			struct has_is_invertible<T, VOID_T<decltype(std::declval<T>().is_invertible())>> :
				std::true_type {};
#endif
		}
		// When there is member function is_invertible(), call it
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto is_invertible(T&& x)
			noexcept(noexcept(std::forward<T>(x).is_invertible()))
			-> std::enable_if_t<detail::has_is_invertible<T>::value, bool>
		{
			return std::forward<T>(x).is_invertible();
		}
		// For signed integral types, only 1 and -1 are invertible
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto is_invertible(T&& x) noexcept
			-> std::enable_if_t<std::is_integral<std::remove_reference_t<T>>::value &&
			std::is_signed<std::remove_reference_t<T>>::value, bool>
		{
			return x == T(1) || x == T(-1);
		}
		// For unsigned integral types, only 1 is invertible
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto is_invertible(T&& x) noexcept
			-> std::enable_if_t<std::is_integral<std::remove_reference_t<T>>::value &&
			std::is_unsigned<std::remove_reference_t<T>>::value, bool>
		{
			return x == T(1);
		}
		// Assuming IEEE-754/IEC-559,
		// std::numeric_limits<T>::epsilon() +
		//    (T(1) / std::numeric_limits<T>::max()) * std::numeric_limits<T>::max() is precisely T(1).
		// Hence, we may regard std::numeric_limits<T>::max() an invertible number.
		// On the other hand, T(1) / (T(1) / std::numeric_limits<T>::max()) is infinity,
		// so T(1) / std::numeric_limits<T>::max() is not an invertible number.
		// (T(1) / std::numeric_limits<T>::max()) + std::numeric_limits<T>::denorm_min() on the other hand
		// satisfies the identity 1 = (1/x)*x = x*(1/x), so any finite number larger than
		// (T(1) / std::numeric_limits<T>::max()) can be regarded as invertible.
		// The same is true for negative numbers.
		namespace detail {
			// I believe this is in general faster than the fallback implementation
			template <class T, bool iec_559 = std::numeric_limits<T>::is_iec559>
			struct is_invertible_floating_point_impl {
				static constexpr T eps = T(1) / std::numeric_limits<T>::max();
				static constexpr T m = std::numeric_limits<T>::max();
				JKL_GPU_EXECUTABLE static constexpr bool impl(T x) noexcept {
					return (x > eps && x <= m) || (x < -eps && x >= -m);
				}
			};
			template <class T>
			struct is_invertible_floating_point_impl<T, false> {
				JKL_GPU_EXECUTABLE static constexpr bool impl(T x) noexcept {
					return T(1) == (T(1) / x) * x;
				}
			};
		}
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto is_invertible(T&& x) noexcept
			-> std::enable_if_t<std::is_floating_point<std::remove_reference_t<T>>::value, bool>
		{
			return detail::is_invertible_floating_point_impl<tmp::remove_cvref_t<T>>::impl(x);
		}
		// When there is a function is_invertible() that can be found by the ADL, call it
		namespace detail {
			template <class T>
			struct has_adl_is_invertible {
				struct has_no_adl {};
				template <class U, bool is_noexcept_>
				struct wrapper {
					using type = U;
					static constexpr bool is_noexcept = is_noexcept_;
				};
				JKL_GPU_EXECUTABLE static constexpr has_no_adl is_invertible(T&&) noexcept { return{}; }
				template <class U, class = std::enable_if_t<
					!has_is_invertible<U>::value &&
					!std::is_integral<U>::value &&
					!std::is_floating_point<U>::value>>
				JKL_GPU_EXECUTABLE static constexpr auto check(U&&) {
					using ::jkl::math::is_invertible;
					return wrapper<decltype(is_invertible(std::declval<U>())),
						noexcept(is_invertible(std::declval<U>()))>{};
				}
				JKL_GPU_EXECUTABLE static constexpr wrapper<has_no_adl, false> check(...) { return {}; }

				using ret_type = decltype(check(std::declval<T>()));
				using type = typename ret_type::type;
				static constexpr bool is_noexcept = ret_type::is_noexcept;
				static constexpr bool value = !std::is_same<type, has_no_adl>::value;
			};
		}
		template <class T>
		static constexpr auto is_invertible(T&& x)
			noexcept(detail::has_adl_is_invertible<T>::is_noexcept)
			-> std::enable_if_t<!detail::has_is_invertible<T>::value &&
			!std::is_integral<std::remove_reference_t<T>>::value &&
			!std::is_floating_point<std::remove_reference_t<T>>::value &&
			detail::has_adl_is_invertible<T>::value,
			typename detail::has_adl_is_invertible<T>::type>
		{
			return is_invertible(std::forward<T>(x));
		}

		//// Compute square
		namespace detail {
			template <class T, class = void>
			struct has_mem_square : std::false_type {
				using type = void;
			};

			template <class T>
			struct has_mem_square<T, VOID_T<decltype(std::declval<T>().square())>> : std::true_type {
				using type = decltype(std::declval<T>().square());
			};
		}
		// If there is a member funtion square(), call it
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto square(T&& x)
			noexcept(noexcept(std::declval<T>().inv()))
			-> std::enable_if_t<detail::has_mem_square<T>::value, typename detail::has_mem_square<T>::type>
		{
			return std::forward<T>(x).sqaure();
		}
		// Otherwise, calculate x*x
		template <class T>
		JKL_GPU_EXECUTABLE static constexpr auto square(T&& x)
			noexcept(noexcept(x * x))
			-> std::enable_if_t<!detail::has_mem_square<T>::value,
			decltype(x * std::forward<T>(x))>
		{
			return x * x;
		}

		template <class Float = float>
		struct constants {
			using number_type = Float;
			static constexpr Float pi = Float(3.141592653589793238462643383279502884197169399375105820974944592307816406286);
			static constexpr Float sqrt_pi = Float(1.77245385090551602729816748334114518279754945612238712821380778985291128459103);
			static constexpr Float exp1 = Float(2.71828182845904523536028747135266249775724709369995957496696762772407663035354);
			static constexpr Float eps = std::numeric_limits<Float>().epsilon() * 16;
		};

		// These functions are useful when fallback routines are needed for too small values.
		// The tolerance bound (the eps above) is just chosen arbitrarily.
		// Usage of these function should be replaced later by more precise bounds.
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE static constexpr bool close_to(Float1 const& x, Float2 const& y) noexcept {
			return x - y <= constants<tmp::remove_cvref_t<decltype(x - y)>>::eps &&
				y - x <= constants<tmp::remove_cvref_t<decltype(x - y)>>::eps;
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool close_to_zero(Float const& x) noexcept {
			return close_to(x, jkl::math::zero<Float>());
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool close_to_one(Float const& x) noexcept {
			return close_to(x, jkl::math::unity<Float>());
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool slightly_larger(Float1 const& x, Float2 const& y) noexcept {
			return x > y + constants<tmp::remove_cvref_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool slightly_smaller(Float1 const& x, Float2 const& y) noexcept {
			return x < y - constants<tmp::remove_cvref_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool almost_larger(Float1 const& x, Float2 const& y) noexcept {
			return x >= y - constants<tmp::remove_cvref_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool almost_smaller(Float1 const& x, Float2 const& y) noexcept {
			return x <= y + constants<tmp::remove_cvref_t<decltype(x - y)>>::eps;
		}

		// A tag type indicating that the arguments will be passed directly to the internal storage
		struct direct_construction {};

		// An exception class indicating that the arguments into a constructor
		// have failed to satisfy the preconditions
		class input_validity_error_general : public std::invalid_argument {
		public:
			// Indicates that the constructor didn't try to reform the arguments
			// to satisfy the preconditions; if it is true, then it means
			// the constructor had tried, but failed.
			bool const only_check;

			input_validity_error_general(char const* msg, bool only_check = false) :
				std::invalid_argument{ msg }, only_check{ only_check } {}
		};

		// An exception class deriving from the above, with the type information
		// (encoded in the exception type itself) indicating to which request was made
		template <class WhichType>
		class input_validity_error : public input_validity_error_general {
		public:
			using which_type = WhichType;
			using input_validity_error_general::input_validity_error_general;
		};

		// A tag type to request to constructors not to check preconditions
		// A constructor may throw input_validity_error if this flag is not set and
		// the input violates the preconditions
		struct no_validity_check {};

		// A tag type to request to constructors to try to reform the input data
		// to satisfy the preconditions; a constructor with this flag might still
		// input_validity_error if the reform process fails
		// [NOTE]
		// Although using auto_reform can be very convenient in some cases,
		// it is generally not recommended to use it, because just blindly using it
		// potentially hide some serious flaws in the user code's logic.
		// Do not use auto_reform, and check the precondition and try to
		// reform the input data yourself, if you are not sure about what's happening
		// inside the constructor when auto_reform tag is given.
		struct auto_reform {};

		// Internally used empty type
		namespace detail {
			struct empty_type {};
		}


		// To reduce some code duplication
		// However, use of this class would make debugging harder, because of possibly excessive depth of
		// inheritance chain. It would be harder to identify actual data members stored!
		// Please use the provided .natvis file to ease your debugging.
		namespace detail {
			// This tag type will be passed to the storage base
			struct forward_to_storage_tag {};

			template <std::size_t components_, class ComponentType, class Terminal>
			class constructor_provider : public tmp::generate_constructors<Terminal,
				forward_to_storage_tag, tmp::copy_or_move_n<ComponentType, components_>>
			{
			public:
				static constexpr std::size_t components = components_;
				using component_type = ComponentType;
				using storage_type = typename Terminal::storage_type;
				using storage_traits = typename Terminal::storage_traits;

			private:
				using base_type = tmp::generate_constructors<Terminal, forward_to_storage_tag,
					tmp::copy_or_move_n<ComponentType, components_>>;

			public:
				// Default constructor
				constructor_provider() = default;

				// Generated component-wise constructors + constructor defined in Terminal
				using base_type::base_type;

				// Direct access to the internal storage

#if defined(_MSC_VER) && _MSC_VER <= 1900
				// MSVC2015 has a bug that it is not possible to move a built-in array
				// So for MSVC2015, we don't define rvalue ref-qualified storage()'s...
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type& storage() noexcept {
					return storage_traits::get_storage(base_type::r_);
				}
				JKL_GPU_EXECUTABLE constexpr storage_type const& storage() const noexcept {
					return storage_traits::get_storage(base_type::r_);
				}
#else
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type& storage() & noexcept {
					return storage_traits::get_storage(base_type::r_);
				}
				JKL_GPU_EXECUTABLE constexpr storage_type const& storage() const& noexcept {
					return storage_traits::get_storage(base_type::r_);
				}
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR storage_type&& storage() && noexcept {
					return storage_traits::get_storage(std::move(*this).base_type::r_);
				}
				JKL_GPU_EXECUTABLE constexpr storage_type const&& storage() const&& noexcept {
					return storage_traits::get_storage(std::move(*this).base_type::r_);
				}
#endif
			};
		}


		//// Invoke unchecking constructor
		namespace detail {
			template <class ResultType>
			struct call_unchecking {};
		}


		//// Forward declarations

		// N-dimensional vector
		template <std::size_t N,
			class ComponentType,
			class Storage = ComponentType[N],
			class StorageTraits = default_storage_traits>
		class Rn_elmt;	// partial support

		template <std::size_t N>
		using Rn_elmtf = Rn_elmt<N, float>;
		template <std::size_t N>
		using Rn_elmtd = Rn_elmt<N, double>;
		template <std::size_t N>
		using Rn_elmtld = Rn_elmt<N, long double>;

		// 2-dimensional vector
		template <class ComponentType,
			class Storage = ComponentType[2],
			class StorageTraits = default_storage_traits>
		using R2_elmt = Rn_elmt<2, ComponentType, Storage, StorageTraits>;

		using R2_elmtf = R2_elmt<float>;
		using R2_elmtd = R2_elmt<double>;
		using R2_elmtld = R2_elmt<long double>;

		// 3-dimensional vector
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits>
		using R3_elmt = Rn_elmt<3, ComponentType, Storage, StorageTraits>;

		using R3_elmtf = R3_elmt<float>;
		using R3_elmtd = R3_elmt<double>;
		using R3_elmtld = R3_elmt<long double>;

		// 4-dimensional vector
		template <class ComponentType,
			class Storage = ComponentType[4],
			class StorageTraits = default_storage_traits>
		using R4_elmt = Rn_elmt<4, ComponentType, Storage, StorageTraits>;

		using R4_elmtf = R4_elmt<float>;
		using R4_elmtd = R4_elmt<double>;
		using R4_elmtld = R4_elmt<long double>;

		// 2x2 matrix
		template <class ComponentType,
			class Storage = R2_elmt<ComponentType>[2],
			class StorageTraits = default_storage_traits>
		class gl2_elmt;

		using gl2_elmtf = gl2_elmt<float>;
		using gl2_elmtd = gl2_elmt<double>;
		using gl2_elmtld = gl2_elmt<long double>;

		// 2x2 invertible matrix
		template <class ComponentType,
			class Storage = R2_elmt<ComponentType>[2],
			class StorageTraits = default_storage_traits>
		class GL2_elmt;

		using GL2_elmtf = GL2_elmt<float>;
		using GL2_elmtd = GL2_elmt<double>;
		using GL2_elmtld = GL2_elmt<long double>;

		// 2x2 symmetric matrix
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits>
		class sym2_elmt;

		using sym2_elmtf = sym2_elmt<float>;
		using sym2_elmtd = sym2_elmt<double>;
		using sym2_elmtld = sym2_elmt<long double>;

		// 2x2 positive-definite symmetric matrix
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits>
		class posdef2_elmt;

		using posdef2_elmtf = posdef2_elmt<float>;
		using posdef2_elmtd = posdef2_elmt<double>;
		using posdef2_elmtld = posdef2_elmt<long double>;

		// 3x3 matrix
		template <class ComponentType,
			class Storage = R3_elmt<ComponentType>[3],
			class StorageTraits = default_storage_traits>
		class gl3_elmt;

		using gl3_elmtf = gl3_elmt<float>;
		using gl3_elmtd = gl3_elmt<double>;
		using gl3_elmtld = gl3_elmt<long double>;

		// 3x3 invertible matrix
		template <class ComponentType,
			class Storage = R3_elmt<ComponentType>[3],
			class StorageTraits = default_storage_traits>
		class GL3_elmt;

		using GL3_elmtf = GL3_elmt<float>;
		using GL3_elmtd = GL3_elmt<double>;
		using GL3_elmtld = GL3_elmt<long double>;

		// 3x3 symmetric matrix
		template <class ComponentType,
			class Storage = ComponentType[6],
			class StorageTraits = default_storage_traits>
		class sym3_elmt;

		using sym3_elmtf = sym3_elmt<float>;
		using sym3_elmtd = sym3_elmt<double>;
		using sym3_elmtld = sym3_elmt<long double>;

		// 3x3 positive-definite symmetric matrix
		template <class ComponentType,
			class Storage = ComponentType[6],
			class StorageTraits = default_storage_traits>
		class posdef3_elmt;

		using posdef3_elmtf = posdef3_elmt<float>;
		using posdef3_elmtd = posdef3_elmt<double>;
		using posdef3_elmtld = posdef3_elmt<long double>;

		// An element in the unit circle, represented by a unit complex number
		template <class ComponentType,
			class Storage = std::complex<ComponentType>,
			class StorageTraits = default_storage_traits>
		class U1_elmt;	// stub

		using U1_elmtf = U1_elmt<float>;
		using U1_elmtd = U1_elmt<double>;
		using U1_elmtld = U1_elmt<long double>;

		// Lie algebra for the above
		template <class ComponentType>
		using u1_elmt = ComponentType;	// stub

		using u1_elmtf = u1_elmt<float>;
		using u1_elmtd = u1_elmt<double>;
		using u1_elmtld = u1_elmt<long double>;

		// 2-dimensional rigid transform
		template <class ComponentType,
			class SO2_Part = u1_elmt<ComponentType>,
			class R2_Part = R2_elmt<ComponentType>>
		class SE2_elmt;	// stub

		using SE2_elmtf = SE2_elmt<float>;
		using SE2_elmtd = SE2_elmt<double>;
		using SE2_elmtld = SE2_elmt<long double>;

		// Lie algebra for the above
		template <class ComponentType,
			class so2_Part = u1_elmt<ComponentType>,
			class R2_Part = R2_elmt<ComponentType>>
		class se2_elmt;	// stub

		using se2_elmtf = se2_elmt<float>;
		using se2_elmtd = se2_elmt<double>;
		using se2_elmtld = se2_elmt<long double>;

		// Unit quaternion
		template <class ComponentType,
			class Storage = ComponentType[4],
			class StorageTraits = default_storage_traits>
		class SU2_elmt;

		using SU2_elmtf = SU2_elmt<float>;
		using SU2_elmtd = SU2_elmt<double>;
		using SU2_elmtld = SU2_elmt<long double>;

		// Lie algebra for the above
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits>
		using su2_elmt = R3_elmt<ComponentType, Storage, StorageTraits>;

		using su2_elmtf = su2_elmt<float>;
		using su2_elmtd = su2_elmt<double>;
		using su2_elmtld = su2_elmt<long double>;

		// 3x3 rotation matrix
		template <class ComponentType,
			class Storage = R3_elmt<ComponentType>[3],
			class StorageTraits = default_storage_traits>
		class SO3_elmt;

		using SO3_elmtf = SO3_elmt<float>;
		using SO3_elmtd = SO3_elmt<double>;
		using SO3_elmtld = SO3_elmt<long double>;

		// Lie algebra for the above
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits>
		using so3_elmt = su2_elmt<ComponentType, Storage, StorageTraits>;

		using so3_elmtf = so3_elmt<float>;
		using so3_elmtd = so3_elmt<double>;
		using so3_elmtld = so3_elmt<long double>;

		// 3-dimensional rigid transform
		template <class ComponentType,
			class SU2Storage = ComponentType[4],
			class SU2StorageTraits = default_storage_traits,
			class R3Storage = ComponentType[3],
			class R3StorageTraits = default_storage_traits>
		class SE3_elmt;

		using SE3_elmtf = SE3_elmt<float>;
		using SE3_elmtd = SE3_elmt<double>;
		using SE3_elmtld = SE3_elmt<long double>;

		// Lie algebra for the above
		template <class ComponentType,
			class so3Storage = ComponentType[3],
			class so3StorageTraits = default_storage_traits,
			class R3Storage = ComponentType[3],
			class R3StorageTraits = default_storage_traits>
		class se3_elmt;

		using se3_elmtf = se3_elmt<float>;
		using se3_elmtd = se3_elmt<double>;
		using se3_elmtld = se3_elmt<long double>;


		//// Customization points for determining storage types of the
		//// results of binary operations

		// Describe why operation might not succeed
		enum class no_operation_reason {
			succeeded,
			component_type_not_compatible,
			storage_not_compatible
		};

		// A tag type indicating an operation between two types cannot succeed
		template <no_operation_reason reason>
		struct no_operation_tag {};

		// Some utilities used internally
		namespace detail {
			// Due to a bug in std::common_type of MSVC2015, we should check if
			// (some condition) ? T1 : T2 is ill-formed or not before std::common_type<T1, T2>
			// is instantiated... std::common_type of MSVC2015 is totally broken!
			// Note that this will not in general result in the right common type,
			// because it always returns false when the ternary operator cannot be
			// evaluated for T1 and T2!
#if defined(_MSC_VER) && _MSC_VER <= 1900
			template <class dummy>
			struct dummy_value_for_common_type_or_int : std::false_type {};

			template <class T1, class T2, class = void>
			struct common_type_or_int : std::false_type {
				using type = int;
			};

			template <class T1, class T2>
			struct common_type_or_int<T1, T2, VOID_T<decltype(
				dummy_value_for_common_type_or_int<T1>::value
				? std::declval<T1>() : std::declval<T2>())>>
			{
				template <class, class, class = void>
				struct impl : std::false_type {
					using type = int;
				};

				template <class T1_, class T2_>
				struct impl<T1_, T2_,
					VOID_T<typename std::common_type<T1_, T2_>::type>> : std::true_type
				{
					using type = typename std::common_type<T1_, T2_>::type;
				};

				static constexpr bool value = impl<T1, T2>::value;
				using type = typename impl<T1, T2>::type;
			};
#else
			template <class T1, class T2, class = void>
			struct common_type_or_int : std::false_type {
				using type = int;
			};
			template <class T1, class T2>
			struct common_type_or_int<T1, T2,
				VOID_T<typename std::common_type<T1, T2>::type>> : std::true_type {
				using type = typename std::common_type<T1, T2>::type;
			};
#endif

			template <class Scalar, class ComponentType, bool from_left, class = void>
			struct check_if_component_is_multipliable : std::false_type {
				using type = int;
			};

			template <class Scalar, class ComponentType>
			struct check_if_component_is_multipliable<Scalar, ComponentType, true,
				VOID_T<decltype(std::declval<Scalar>() * std::declval<ComponentType>())>> :
				detail::common_type_or_int<
				ComponentType, decltype(std::declval<Scalar>() * std::declval<ComponentType>())> {};

			template <class Scalar, class ComponentType>
			struct check_if_component_is_multipliable<Scalar, ComponentType, false,
				VOID_T<decltype(std::declval<ComponentType>() * std::declval<Scalar>())>> :
				detail::common_type_or_int<
				ComponentType, decltype(std::declval<ComponentType>() * std::declval<Scalar>())> {};
		}

		// Apply to addition, subtraction, and 3D cross-product of Rn_elmt's
		template <std::size_t N,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_Rn_elmt_binary_result {
				// The default algorithm is the following:
				//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
				//     exist, then generate an error.
				//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
				//     are defaults (that is, storages are built-in arrays of
				//     their component types and storage traits classes are default_storage_traits),
				//     then use default for the result also.
				//  3. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
				//     LeftComponentType, then use LeftStorage and LeftStorageTraits.
				//  4. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
				//     RightComponentType, then use RightStorage and RightStorageTraits.
				//  5. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			private:
				static constexpr bool all_defaults =
					std::is_same<LeftStorage, LeftComponentType[N]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, RightComponentType[N]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value;

				static constexpr bool is_left_broader =
					std::is_same<component_type, LeftComponentType>::value;

				static constexpr bool is_right_broader =
					std::is_same<component_type, RightComponentType>::value;

				using storage_pair = std::conditional_t<all_defaults,
					std::pair<component_type[N], default_storage_traits>,
					std::conditional_t<is_left_broader,
					std::pair<LeftStorage, LeftStorageTraits>,
					std::pair<RightStorage, RightStorageTraits>>>;

			public:
				static constexpr no_operation_reason value =
					!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_left_broader && !is_right_broader && !all_defaults) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		template <std::size_t N, class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using Rn_elmt_binary_result = std::conditional_t<
			find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
			Rn_elmt<N,
			typename find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::component_type,
			typename find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_type,
			typename find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
			no_operation_tag<
			find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value>>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using R2_elmt_binary_result = Rn_elmt_binary_result<2,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using R3_elmt_binary_result = Rn_elmt_binary_result<3,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to scalar multiplication and division of Rn_elmt's
		template <std::size_t N, class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		struct find_Rn_elmt_scalar_mult_result {
				// The default algorithm is the following:
				//  1. First, inspect the result type of Scalar * ComponentType or ComponentType * Scalar
				//     depending on from_left. If std::common_type_t of ComponentType with that type
				//     does not exist, generate an error. That common type will be the component type.
				//  2. Otherwise, if Storage and StorageTraits are the defaults
				//     (that is, storages are built-in arrays of
				//     their component types and storage traits classes are default_storage_traits),
				//     then use the default also for the result type.
				//  3. Otherwise, if the deduced component type is exactly ComponentType, then use the passed Rn_elmt.
				//  4. Otherwise, generate an error.

			using component_type = typename detail::check_if_component_is_multipliable<
				Scalar, ComponentType, from_left>::type;

			private:
				static constexpr bool all_defaults =
					std::is_same<Storage, ComponentType[N]>::value &&
					std::is_same<StorageTraits, default_storage_traits>::value;

				static constexpr bool is_component_type_broader =
					std::is_same<component_type, ComponentType>::value;

				using storage_pair = std::conditional_t<all_defaults,
					std::pair<component_type[N], default_storage_traits>,
					std::pair<Storage, StorageTraits>>;

			public:
				static constexpr no_operation_reason value =
					!detail::check_if_component_is_multipliable<Scalar, ComponentType, from_left>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_component_type_broader && !all_defaults) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		template <std::size_t N, class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		using Rn_elmt_scalar_mult_result = std::conditional_t<
			find_Rn_elmt_scalar_mult_result<N, Scalar, from_left,
			ComponentType, Storage, StorageTraits>::value == no_operation_reason::succeeded,
			Rn_elmt<N,
			typename find_Rn_elmt_scalar_mult_result<N, Scalar, from_left,
			ComponentType, Storage, StorageTraits>::component_type,
			typename find_Rn_elmt_scalar_mult_result<N, Scalar, from_left,
			ComponentType, Storage, StorageTraits>::storage_type,
			typename find_Rn_elmt_scalar_mult_result<N, Scalar, from_left,
			ComponentType, Storage, StorageTraits>::storage_traits>,
			no_operation_tag<
			find_Rn_elmt_scalar_mult_result<N, Scalar, from_left,
			ComponentType, Storage, StorageTraits>::value>>;

		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using R2_elmt_scalar_mult_result = Rn_elmt_scalar_mult_result<2, Scalar, from_left,
			ComponentType, Storage, StorageTraits>;

		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using R3_elmt_scalar_mult_result = Rn_elmt_scalar_mult_result<3, Scalar, from_left,
			ComponentType, Storage, StorageTraits>;


		// Apply to 2D & 3D outer product of Rn_elmt's
#if defined(_MSC_VER) && _MSC_VER <= 1900
		namespace detail {
			struct void_t_workaround_find_Rn_elmt_outer_result {};
		}
#endif
		template <std::size_t N,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_Rn_elmt_outer_result {
			// The default algorithm is the following:
			//  1. The result type of LeftComponentType * RightComponentType will be the new component type;
			//     if that is ill-formed, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults (that is, storages are built-in arrays of
			//     their component types and storage traits classes are default_storage_traits),
			//     then use default for the result also.
			//  3. Otherwise, if the component type is RightComponentType then
			//     use Rn_elmt<component_type, RightStorage, RightStorageTraits>[N]
			//     as the storage and default_storage_traits as the storage traits.
			//  4. Otherwise, generate an error.

		private:
			template <class dummy = void, class = void>
			struct find_component_type : std::false_type {
				using type = int;	// dummy
			};

			template <class dummy>
			struct find_component_type<dummy,
				VOID_T<decltype(std::declval<LeftComponentType>() * std::declval<RightComponentType>())
#if defined(_MSC_VER) && _MSC_VER <= 1900
				, detail::void_t_workaround_find_Rn_elmt_outer_result
#endif
				>>
				: std::true_type
			{
				using type = tmp::remove_cvref_t<
					decltype(std::declval<LeftComponentType>() * std::declval<RightComponentType>())>;
			};

		public:
			using component_type = typename find_component_type<>::type;

		private:
			static constexpr bool all_defaults =
				std::is_same<LeftStorage, LeftComponentType[N]>::value &&
				std::is_same<LeftStorageTraits, default_storage_traits>::value &&
				std::is_same<RightStorage, RightComponentType[N]>::value &&
				std::is_same<RightStorageTraits, default_storage_traits>::value;

			static constexpr bool is_right_broader =
				std::is_same<component_type, RightComponentType>::value;

			using storage_pair = std::conditional_t<all_defaults,
				std::pair<Rn_elmt<N, component_type>[N], default_storage_traits>,
				std::pair<Rn_elmt<N, component_type, RightStorage, RightStorageTraits>[N], default_storage_traits>>;

		public:
			static constexpr no_operation_reason value =
				!find_component_type<>::value ?
				no_operation_reason::component_type_not_compatible :
				(!is_right_broader && !all_defaults) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;

			using storage_type = typename storage_pair::first_type;
			using storage_traits = typename storage_pair::second_type;
		};

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using R2_elmt_outer_result = std::conditional_t<
			find_Rn_elmt_outer_result<2, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
			gl2_elmt<
			typename find_Rn_elmt_outer_result<2, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::component_type,
			typename find_Rn_elmt_outer_result<2, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_type,
			typename find_Rn_elmt_outer_result<2, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
			no_operation_tag<
			find_Rn_elmt_outer_result<2, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value>>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using R3_elmt_outer_result = std::conditional_t<
			find_Rn_elmt_outer_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
			gl3_elmt<
			typename find_Rn_elmt_outer_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::component_type,
			typename find_Rn_elmt_outer_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_type,
			typename find_Rn_elmt_outer_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
			no_operation_tag<
			find_Rn_elmt_outer_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value>>;


		// Apply to addition, subtraction, multiplication, and division of gl2_elmt's
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_gl2_elmt_binary_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults (that is, storages are built-in arrays of
			//     their component types and storage traits classes are default_storage_traits),
			//     then use default for the result also.
			//  3. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     LeftComponentType, then use LeftStorage and LeftStorageTraits.
			//  4. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     RightComponentType, then use RightStorage and RightStorageTraits.
			//  5. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			private:
				static constexpr bool all_defaults =
					std::is_same<LeftStorage, R2_elmt<LeftComponentType>[2]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, R2_elmt<RightComponentType>[2]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value;

				static constexpr bool is_left_broader =
					std::is_same<component_type, LeftComponentType>::value;

				static constexpr bool is_right_broader =
					std::is_same<component_type, RightComponentType>::value;

				using storage_pair = std::conditional_t<all_defaults,
					std::pair<R2_elmt<component_type>[2], default_storage_traits>,
					std::conditional_t<is_left_broader,
					std::pair<LeftStorage, LeftStorageTraits>,
					std::pair<RightStorage, RightStorageTraits>>>;

			public:
				static constexpr no_operation_reason value =
					!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_left_broader && !is_right_broader && !all_defaults) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using gl2_elmt_binary_result_impl = std::conditional_t<
				find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using gl2_elmt_binary_result = detail::gl2_elmt_binary_result_impl<gl2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using GL2_elmt_binary_result = detail::gl2_elmt_binary_result_impl<GL2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to scalar multiplication and division of gl2_elmt's
		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		struct find_gl2_elmt_scalar_mult_result {
			// The default algorithm is the following:
			//  1. Deduce the component type just like the Rn_elmt case; if that fails, generate an error.
			//  2. Otherwise, check if Storage is an R2_elmt of ComponentType's and
			//     StorageTraits is default_storage_traits:
			//     - if that is the case, check if that R2_elmt can be multiplied with Scalar;
			//       - if that is the case, use built-in array of that resulting R2_elmt's as storage
			//         and default_storage_traits as the storage traits;
			//       - otherwise, go to the case 3;
			//     - otherwise, go to the case 3.
			//  3. Otherwise, if std::common_type_t<Scalar, ComponentType> is
			//     ComponentType, then use the passed gl2_elmt.
			//  4. Otherwise, generate an error.

			using component_type = typename detail::check_if_component_is_multipliable<
				Scalar, ComponentType, from_left>::type;

			private:
				template <class T>
				struct check_if_row_is_multipliable_impl : std::false_type {
					using type = std::pair<int, int>;	// dummy
				};

				template <class RowStorage, class RowStorageTraits>
				struct check_if_row_is_multipliable_impl<
					R2_elmt<ComponentType, RowStorage, RowStorageTraits>[2]>
				{
					static constexpr bool value =
						find_Rn_elmt_scalar_mult_result<2, Scalar, from_left,
						ComponentType, RowStorage, RowStorageTraits>::value == no_operation_reason::succeeded;

					using type = std::pair<R2_elmt_scalar_mult_result<Scalar, from_left,
						ComponentType, RowStorage, RowStorageTraits>[2], default_storage_traits>;
				};

				static constexpr bool is_row_multiplicable =
					std::is_same<StorageTraits, default_storage_traits>::value &&
					check_if_row_is_multipliable_impl<Storage>::value;

				static constexpr bool is_component_type_broader =
					std::is_same<component_type, ComponentType>::value;

				using storage_pair = std::conditional_t<is_row_multiplicable,
					typename check_if_row_is_multipliable_impl<Storage>::type,
					std::pair<Storage, StorageTraits>>;

			public:
				static constexpr no_operation_reason value =
					!detail::check_if_component_is_multipliable<Scalar, ComponentType, from_left>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_row_multiplicable && !is_component_type_broader) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			using gl2_elmt_scalar_mult_result_impl = std::conditional_t<
				find_gl2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_gl2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::component_type,
				typename find_gl2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_type,
				typename find_gl2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_traits>,
				no_operation_tag<
				find_gl2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value>>;
		}
		
		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using gl2_elmt_scalar_mult_result = detail::gl2_elmt_scalar_mult_result_impl<gl2_elmt,
			Scalar, from_left, ComponentType, Storage, StorageTraits>;

		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using GL2_elmt_scalar_mult_result = detail::gl2_elmt_scalar_mult_result_impl<GL2_elmt,
			Scalar, from_left, ComponentType, Storage, StorageTraits>;

		
		// Apply to addition and subtraction of sym2_elmt's
		// Default algorithm is the same as Rn_elmt
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym2_elmt_binary_result :
			find_Rn_elmt_binary_result<3, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits> {};
		
		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
				using sym2_elmt_binary_result_impl = std::conditional_t<
				find_sym2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym2_elmt_binary_result =
			detail::sym2_elmt_binary_result_impl<sym2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using posdef2_elmt_binary_result =
			detail::sym2_elmt_binary_result_impl<posdef2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;
		

		// Apply to scalar multiplication and division of sym2_elmt's
		// Default algorithm is same as Rn_elmt
		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		struct find_sym2_elmt_scalar_mult_result : find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
			ComponentType, Storage, StorageTraits> {};

		namespace detail {
			template <template <class, class, class> class Template,
				class Scalar, bool from_left,
				class ComponentType, class Storage, class StorageTraits>
			using sym2_elmt_scalar_mult_result_impl = std::conditional_t<
				find_sym2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::component_type,
				typename find_sym2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_type,
				typename find_sym2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym2_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value>>;
		}

		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		using sym2_elmt_scalar_mult_result = detail::sym2_elmt_scalar_mult_result_impl<
			sym2_elmt, Scalar, from_left, ComponentType, Storage, StorageTraits>;

		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		using posdef2_elmt_scalar_mult_result = detail::sym2_elmt_scalar_mult_result_impl<
			posdef2_elmt, Scalar, from_left, ComponentType, Storage, StorageTraits>;

		
		// Apply to addition, subtraction, multiplication, and division of
		// a sym2_elmt and a gl2_elmt
		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym2_elmt_gl2_elmt_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults, then use default for the result also.
			//  3. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     LeftComponentType and left_is_sym == false, then use LeftStorage and LeftStorageTraits.
			//  4. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     RightComponentTyp and left_is_sym == true, then use RightStorage and RightStorageTraits.
			//  5. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

		private:
			static constexpr bool all_defaults = left_is_sym
				? (std::is_same<LeftStorage, LeftComponentType[3]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, R2_elmt<RightComponentType>[2]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value)
				: (std::is_same<LeftStorage, R2_elmt<LeftComponentType>[2]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, RightComponentType[3]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value);

			static constexpr bool is_left_broader =
				std::is_same<component_type, LeftComponentType>::value && !left_is_sym;

			static constexpr bool is_right_broader =
				std::is_same<component_type, RightComponentType>::value && left_is_sym;

			// Suppress generation of gl2_elmt with sym2_elmt-style storage
			template <bool, class>
			struct select_left_right;
			template <class dummy>
			struct select_left_right<false, dummy> {
				using type = std::pair<LeftStorage, LeftStorageTraits>;
			};
			template <class dummy>
			struct select_left_right<true, dummy> {
				using type = std::pair<RightStorage, RightStorageTraits>;
			};

			using storage_pair = std::conditional_t<all_defaults,
				std::pair<R2_elmt<component_type>[2], default_storage_traits>,
				typename select_left_right<left_is_sym, void>::type>;

		public:
			static constexpr no_operation_reason value =
				!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
				no_operation_reason::component_type_not_compatible :
				(!is_left_broader && !is_right_broader && !all_defaults) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;

			using storage_type = typename storage_pair::first_type;
			using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template, bool left_is_sym,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym2_elmt_gl2_elmt_result_impl = std::conditional_t<
				find_sym2_elmt_gl2_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym2_elmt_gl2_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym2_elmt_gl2_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym2_elmt_gl2_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym2_elmt_gl2_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym2_elmt_gl2_elmt_result =
			detail::sym2_elmt_gl2_elmt_result_impl<gl2_elmt, left_is_sym,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using posdef2_elmt_GL2_elmt_result =
			detail::sym2_elmt_gl2_elmt_result_impl<GL2_elmt, left_is_sym,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to multiplication and division of sym2_elmt's
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym2_mult_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults, then use default for the result also.
			//  3. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

		private:
			static constexpr bool all_defaults =
				std::is_same<LeftStorage, LeftComponentType[3]>::value &&
				std::is_same<LeftStorageTraits, default_storage_traits>::value &&
				std::is_same<RightStorage, RightComponentType[3]>::value &&
				std::is_same<RightStorageTraits, default_storage_traits>::value;

			using storage_pair = std::pair<R2_elmt<component_type>[2], default_storage_traits>;

		public:
			static constexpr no_operation_reason value =
				!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
				no_operation_reason::component_type_not_compatible :
				!all_defaults ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;

			using storage_type = typename storage_pair::first_type;
			using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
				using sym2_elmt_mult_result_impl = std::conditional_t<
				find_sym2_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym2_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym2_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym2_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym2_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym2_elmt_mult_result =
			detail::sym2_elmt_mult_result_impl<gl2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
			using posdef2_elmt_mult_result =
			detail::sym2_elmt_mult_result_impl<GL2_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to addition, subtraction, multiplication, and division of gl3_elmt's
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_gl3_elmt_binary_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults (that is, storages are built-in arrays of
			//     their component types and storage traits classes are default_storage_traits),
			//     then use default for the result also.
			//  3. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     LeftComponentType, then use LeftStorage and LeftStorageTraits.
			//  4. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     RightComponentType, then use RightStorage and RightStorageTraits.
			//  5. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			private:
				static constexpr bool all_defaults =
					std::is_same<LeftStorage, R3_elmt<LeftComponentType>[3]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, R3_elmt<RightComponentType>[3]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value;

				static constexpr bool is_left_broader =
					std::is_same<component_type, LeftComponentType>::value;

				static constexpr bool is_right_broader =
					std::is_same<component_type, RightComponentType>::value;

				using storage_pair = std::conditional_t<all_defaults,
					std::pair<R3_elmt<component_type>[3], default_storage_traits>,
					std::conditional_t<is_left_broader,
					std::pair<LeftStorage, LeftStorageTraits>,
					std::pair<RightStorage, RightStorageTraits>>>;

			public:
				static constexpr no_operation_reason value =
					!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_left_broader && !is_right_broader && !all_defaults) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using gl3_elmt_binary_result_impl = std::conditional_t<
				find_gl3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_gl3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_gl3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_gl3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_gl3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using gl3_elmt_binary_result = detail::gl3_elmt_binary_result_impl<gl3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using GL3_elmt_binary_result = detail::gl3_elmt_binary_result_impl<GL3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using SO3_elmt_binary_result = detail::gl3_elmt_binary_result_impl<SO3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to scalar multiplication and division of gl3_elmt's
		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		struct find_gl3_elmt_scalar_mult_result {
			// The default algorithm is the following:
			//  1. Deduce the component type just like the Rn_elmt case; if that fails, generate an error.
			//  2. Otherwise, check if Storage is an R2_elmt of ComponentType's and
			//     StorageTraits is default_storage_traits:
			//     - if that is the case, check if that R2_elmt can be multiplied with Scalar;
			//       - if that is the case, use built-in array of that resulting R2_elmt's as storage
			//         and default_storage_traits as the storage traits;
			//       - otherwise, go to the case 3;
			//     - otherwise, go to the case 3.
			//  3. Otherwise, if std::common_type_t<Scalar, ComponentType> is
			//     ComponentType, then use the passed gl2_elmt.
			//  4. Otherwise, generate an error.

			using component_type = typename detail::check_if_component_is_multipliable<
				Scalar, ComponentType, from_left>::type;

		private:
			template <class T>
			struct check_if_row_is_multipliable_impl : std::false_type {
				using type = std::pair<int, int>;	// dummy
			};

			template <class RowStorage, class RowStorageTraits>
			struct check_if_row_is_multipliable_impl<
				R3_elmt<ComponentType, RowStorage, RowStorageTraits>[3]>
			{
				static constexpr bool value =
					find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
					ComponentType, RowStorage, RowStorageTraits>::value == no_operation_reason::succeeded;

				using type = std::pair<R3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, RowStorage, RowStorageTraits>[3], default_storage_traits>;
			};

			static constexpr bool is_row_multiplicable =
				std::is_same<StorageTraits, default_storage_traits>::value &&
				check_if_row_is_multipliable_impl<Storage>::value;

			static constexpr bool is_component_type_broader =
				std::is_same<component_type, ComponentType>::value;

			using storage_pair = std::conditional_t<is_row_multiplicable,
				typename check_if_row_is_multipliable_impl<Storage>::type,
				std::pair<Storage, StorageTraits>>;

		public:
			static constexpr no_operation_reason value =
				!detail::check_if_component_is_multipliable<Scalar, ComponentType, from_left>::value ?
				no_operation_reason::component_type_not_compatible :
				(!is_row_multiplicable && !is_component_type_broader) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;

			using storage_type = typename storage_pair::first_type;
			using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			using gl3_elmt_scalar_mult_result_impl = std::conditional_t<
				find_gl3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_gl3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::component_type,
				typename find_gl3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_type,
				typename find_gl3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_traits>,
				no_operation_tag<
				find_gl3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value>>;
		}

		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using gl3_elmt_scalar_mult_result = detail::gl3_elmt_scalar_mult_result_impl<gl3_elmt,
			Scalar, from_left, ComponentType, Storage, StorageTraits>;

		template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
		using GL3_elmt_scalar_mult_result = detail::gl3_elmt_scalar_mult_result_impl<GL3_elmt,
			Scalar, from_left, ComponentType, Storage, StorageTraits>;


		// Apply to addition and subtraction of sym3_elmt's
		// Default algorithm is the same as Rn_elmt
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym3_elmt_binary_result :
			find_Rn_elmt_binary_result<6, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits> {};

		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym3_elmt_binary_result_impl = std::conditional_t<
				find_sym3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym3_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using sym3_elmt_binary_result =
			detail::sym3_elmt_binary_result_impl<sym3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using posdef3_elmt_binary_result =
			detail::sym3_elmt_binary_result_impl<posdef3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to scalar multiplication and division of sym3_elmt's
		// Default algorithm is same as Rn_elmt
		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		struct find_sym3_elmt_scalar_mult_result : find_Rn_elmt_scalar_mult_result<6, Scalar, from_left,
			ComponentType, Storage, StorageTraits> {};

		namespace detail {
			template <template <class, class, class> class Template,
				class Scalar, bool from_left,
				class ComponentType, class Storage, class StorageTraits>
			using sym3_elmt_scalar_mult_result_impl = std::conditional_t<
				find_sym3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::component_type,
				typename find_sym3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_type,
				typename find_sym3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym3_elmt_scalar_mult_result<Scalar, from_left,
				ComponentType, Storage, StorageTraits>::value>>;
		}

		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		using sym3_elmt_scalar_mult_result = detail::sym3_elmt_scalar_mult_result_impl<
			sym3_elmt, Scalar, from_left, ComponentType, Storage, StorageTraits>;

		template <class Scalar, bool from_left,
			class ComponentType, class Storage, class StorageTraits>
		using posdef3_elmt_scalar_mult_result = detail::sym3_elmt_scalar_mult_result_impl<
			posdef3_elmt, Scalar, from_left, ComponentType, Storage, StorageTraits>;


		// Apply to addition, subtraction, multiplication, and division of
		// a sym3_elmt and a gl3_elmt
		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym3_elmt_gl3_elmt_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults, then use default for the result also.
			//  3. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     LeftComponentType and left_is_sym == false, then use LeftStorage and LeftStorageTraits.
			//  4. Otherwise, if std::common_type_t<LeftComponentType, RightComponentType> is
			//     RightComponentTyp and left_is_sym == true, then use RightStorage and RightStorageTraits.
			//  5. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			private:
				static constexpr bool all_defaults = left_is_sym
					? (std::is_same<LeftStorage, LeftComponentType[6]>::value &&
						std::is_same<LeftStorageTraits, default_storage_traits>::value &&
						std::is_same<RightStorage, R3_elmt<RightComponentType>[3]>::value &&
						std::is_same<RightStorageTraits, default_storage_traits>::value)
					: (std::is_same<LeftStorage, R3_elmt<LeftComponentType>[3]>::value &&
						std::is_same<LeftStorageTraits, default_storage_traits>::value &&
						std::is_same<RightStorage, RightComponentType[6]>::value &&
						std::is_same<RightStorageTraits, default_storage_traits>::value);

				static constexpr bool is_left_broader =
					std::is_same<component_type, LeftComponentType>::value && !left_is_sym;

				static constexpr bool is_right_broader =
					std::is_same<component_type, RightComponentType>::value && left_is_sym;

				// Suppress generation of gl3_elmt with sym3_elmt-style storage
				template <bool, class>
				struct select_left_right;
				template <class dummy>
				struct select_left_right<false, dummy> {
					using type = std::pair<LeftStorage, LeftStorageTraits>;
				};
				template <class dummy>
				struct select_left_right<true, dummy> {
					using type = std::pair<RightStorage, RightStorageTraits>;
				};

				using storage_pair = std::conditional_t<all_defaults,
					std::pair<R3_elmt<component_type>[3], default_storage_traits>,
					typename select_left_right<left_is_sym, void>::type>;

			public:
				static constexpr no_operation_reason value =
					!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
					no_operation_reason::component_type_not_compatible :
					(!is_left_broader && !is_right_broader && !all_defaults) ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template, bool left_is_sym,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym3_elmt_gl3_elmt_result_impl = std::conditional_t<
				find_sym3_elmt_gl3_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym3_elmt_gl3_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym3_elmt_gl3_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym3_elmt_gl3_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym3_elmt_gl3_elmt_result<left_is_sym,
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using sym3_elmt_gl3_elmt_result =
			detail::sym3_elmt_gl3_elmt_result_impl<gl3_elmt, left_is_sym,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <bool left_is_sym,
			class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using posdef3_elmt_GL3_elmt_result =
			detail::sym3_elmt_gl3_elmt_result_impl<GL3_elmt, left_is_sym,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to multiplication and division of sym3_elmt's
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_sym3_mult_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, if all LeftStorage, LeftStorageTraits, RightStorage, RightStorageTraits
			//     are defaults, then use default for the result also.
			//  3. Otherwise, generate an error.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			private:
				static constexpr bool all_defaults =
					std::is_same<LeftStorage, LeftComponentType[6]>::value &&
					std::is_same<LeftStorageTraits, default_storage_traits>::value &&
					std::is_same<RightStorage, RightComponentType[6]>::value &&
					std::is_same<RightStorageTraits, default_storage_traits>::value;

				using storage_pair = std::pair<R3_elmt<component_type>[3], default_storage_traits>;

			public:
				static constexpr no_operation_reason value =
					!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
					no_operation_reason::component_type_not_compatible :
					!all_defaults ?
					no_operation_reason::storage_not_compatible :
					no_operation_reason::succeeded;

				using storage_type = typename storage_pair::first_type;
				using storage_traits = typename storage_pair::second_type;
		};

		namespace detail {
			template <template <class, class, class> class Template,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			using sym3_elmt_mult_result_impl = std::conditional_t<
				find_sym3_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
				Template<
				typename find_sym3_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::component_type,
				typename find_sym3_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_type,
				typename find_sym3_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
				no_operation_tag<
				find_sym3_mult_result<
				LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value>>;
		}

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using sym3_elmt_mult_result =
			detail::sym3_elmt_mult_result_impl<gl3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using posdef3_elmt_mult_result =
			detail::sym3_elmt_mult_result_impl<GL3_elmt,
			LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>;


		// Apply to multiplication, division, and interpolation of SU2_elmt's
		// Default algorithm is the same as Rn_elmt
		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		struct find_SU2_elmt_binary_result :
			find_Rn_elmt_binary_result<4, LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits> {};

		template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
			class RightComponentType, class RightStorage, class RightStorageTraits>
		using SU2_elmt_binary_result = std::conditional_t<
			find_SU2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded,
			SU2_elmt<
			typename find_SU2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::component_type,
			typename find_SU2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_type,
			typename find_SU2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::storage_traits>,
			no_operation_tag<
			find_SU2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
			RightComponentType, RightStorage, RightStorageTraits>::value>>;


		// Apply to multiplication, division, and interpolation of SE3_elmt's
		template <class LeftComponentType, class LeftSU2Storage, class LeftSU2StorageTraits,
			class LeftR3Storage, class LeftR3StorageTraits,
			class RightComponentType, class RightSU2Storage, class RightSU2StorageTraits,
			class RightR3Storage, class RightR3StorageTraits>
		struct find_SE3_elmt_binary_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, rely on find_SU2_elmt_binary_result and
			//     find_Rn_elmt_binary_result.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;
			
			using SU2_storage_type = typename find_SU2_elmt_binary_result<
				LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
				RightComponentType, RightSU2Storage, RightSU2StorageTraits>::storage_type;
			using SU2_storage_traits = typename find_SU2_elmt_binary_result<
				LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
				RightComponentType, RightSU2Storage, RightSU2StorageTraits>::storage_traits;
			using R3_storage_type = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
				RightComponentType, RightR3Storage, RightR3StorageTraits>::storage_type;
			using R3_storage_traits = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
				RightComponentType, RightR3Storage, RightR3StorageTraits>::storage_traits;

			static constexpr no_operation_reason value =
				!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
				no_operation_reason::component_type_not_compatible :
				(find_SU2_elmt_binary_result<
					LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
					RightComponentType, RightSU2Storage, RightSU2StorageTraits>::value
					== no_operation_reason::component_type_not_compatible ||
					find_Rn_elmt_binary_result<3,
					LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, RightR3Storage, RightR3StorageTraits>::value
					== no_operation_reason::component_type_not_compatible) ?
				no_operation_reason::component_type_not_compatible :
				(find_SU2_elmt_binary_result<
					LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
					RightComponentType, RightSU2Storage, RightSU2StorageTraits>::value
					== no_operation_reason::storage_not_compatible ||
					find_Rn_elmt_binary_result<3,
					LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, RightR3Storage, RightR3StorageTraits>::value
					== no_operation_reason::storage_not_compatible) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;
		};

		template <class LeftComponentType, class LeftSU2Storage, class LeftSU2StorageTraits,
			class LeftR3Storage, class LeftR3StorageTraits,
			class RightComponentType, class RightSU2Storage, class RightSU2StorageTraits,
			class RightR3Storage, class RightR3StorageTraits>
		using SE3_elmt_binary_result = std::conditional_t<
			find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::value == no_operation_reason::succeeded,
			SE3_elmt<
			typename find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::component_type,
			typename find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::SU2_storage_type,
			typename find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::SU2_storage_traits,
			typename find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::R3_storage_type,
			typename find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::R3_storage_traits>,
			no_operation_tag<
			find_SE3_elmt_binary_result<LeftComponentType, LeftSU2Storage, LeftSU2StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, RightSU2Storage, RightSU2StorageTraits,
			RightR3Storage, RightR3StorageTraits>::value>>;


		// Apply to addition, subtraction, and commutator of se3_elmt's
		template <class LeftComponentType, class Leftso3Storage, class Leftso3StorageTraits,
			class LeftR3Storage, class LeftR3StorageTraits,
			class RightComponentType, class Rightso3Storage, class Rightso3StorageTraits,
			class RightR3Storage, class RightR3StorageTraits>
		struct find_se3_elmt_binary_result {
			// The default algorithm is the following:
			//  1. If std::common_type_t<LeftComponentType, RightComponentType> does not
			//     exist, then generate an error.
			//  2. Otherwise, rely on find_Rn_elmt_binary_result.

			using component_type = typename detail::common_type_or_int<
				LeftComponentType, RightComponentType>::type;

			using so3_storage_type = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
				RightComponentType, Rightso3Storage, Rightso3StorageTraits>::storage_type;
			using so3_storage_traits = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
				RightComponentType, Rightso3Storage, Rightso3StorageTraits>::storage_traits;
			using R3_storage_type = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
				RightComponentType, RightR3Storage, RightR3StorageTraits>::storage_type;
			using R3_storage_traits = typename find_Rn_elmt_binary_result<3,
				LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
				RightComponentType, RightR3Storage, RightR3StorageTraits>::storage_traits;

			static constexpr no_operation_reason value =
				!detail::common_type_or_int<LeftComponentType, RightComponentType>::value ?
				no_operation_reason::component_type_not_compatible :
				(find_Rn_elmt_binary_result<3,
					LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
					RightComponentType, Rightso3Storage, Rightso3StorageTraits>::value
					== no_operation_reason::component_type_not_compatible ||
					find_Rn_elmt_binary_result<3,
					LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, RightR3Storage, RightR3StorageTraits>::value
					== no_operation_reason::component_type_not_compatible) ?
				no_operation_reason::component_type_not_compatible :
				(find_Rn_elmt_binary_result<3,
					LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
					RightComponentType, Rightso3Storage, Rightso3StorageTraits>::value
					== no_operation_reason::storage_not_compatible ||
					find_Rn_elmt_binary_result<3,
					LeftComponentType, LeftR3Storage, LeftR3StorageTraits,
					RightComponentType, RightR3Storage, RightR3StorageTraits>::value
					== no_operation_reason::storage_not_compatible) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;
		};

		template <class LeftComponentType, class Leftso3Storage, class Leftso3StorageTraits,
			class LeftR3Storage, class LeftR3StorageTraits,
			class RightComponentType, class Rightso3Storage, class Rightso3StorageTraits,
			class RightR3Storage, class RightR3StorageTraits>
		using se3_elmt_binary_result = std::conditional_t<
			find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::value == no_operation_reason::succeeded,
			SE3_elmt<
			typename find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::component_type,
			typename find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::so3_storage_type,
			typename find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::so3_storage_traits,
			typename find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::R3_storage_type,
			typename find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::R3_storage_traits>,
			no_operation_tag<
			find_se3_elmt_binary_result<LeftComponentType, Leftso3Storage, Leftso3StorageTraits,
			LeftR3Storage, LeftR3StorageTraits,
			RightComponentType, Rightso3Storage, Rightso3StorageTraits,
			RightR3Storage, RightR3StorageTraits>::value>>;


		// Apply to scalar multiplication and division of se3_elmt's
		template <class Scalar, bool from_left,
			class ComponentType, class so3Storage, class so3StorageTraits,
			class R3Storage, class R3StorageTraits>
		struct find_se3_elmt_scalar_mult_result {
			// The default algorithm is the following:
			//  1. First, inspect the result type of Scalar * ComponentType or ComponentType * Scalar
			//     depending on from_left. If std::common_type_t of ComponentType with that type
			//     does not exist, generate an error. That common type will be the component type.
			//  2. Otherwise, rely on find_Rn_elmt_scalar_mult_result.

			using component_type = typename detail::check_if_component_is_multipliable<
				Scalar, ComponentType, from_left>::type;
			
			using so3_storage_type = typename find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
				ComponentType, so3Storage, so3StorageTraits>::storage_type;
			using so3_storage_traits = typename find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
				ComponentType, so3Storage, so3StorageTraits>::storage_traits;
			using R3_storage_type = typename find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
				ComponentType, R3Storage, R3StorageTraits>::storage_type;
			using R3_storage_traits = typename find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
				ComponentType, R3Storage, R3StorageTraits>::storage_traits;

			static constexpr no_operation_reason value =
				!detail::check_if_component_is_multipliable<Scalar, ComponentType, from_left>::value ?
				no_operation_reason::component_type_not_compatible :
				(find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
					ComponentType, so3Storage, so3StorageTraits>::value
					== no_operation_reason::component_type_not_compatible ||
					find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
					ComponentType, R3Storage, R3StorageTraits>::value
					== no_operation_reason::component_type_not_compatible) ?
				no_operation_reason::component_type_not_compatible :
				(find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
					ComponentType, so3Storage, so3StorageTraits>::value
					== no_operation_reason::storage_not_compatible ||
					find_Rn_elmt_scalar_mult_result<3, Scalar, from_left,
					ComponentType, R3Storage, R3StorageTraits>::value
					== no_operation_reason::storage_not_compatible) ?
				no_operation_reason::storage_not_compatible :
				no_operation_reason::succeeded;
		};

		template <class Scalar, bool from_left,
			class ComponentType, class so3Storage, class so3StorageTraits,
			class R3Storage, class R3StorageTraits>
		using se3_elmt_scalar_mult_result = std::conditional_t<
			find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::value == no_operation_reason::succeeded,
			SE3_elmt<
			typename find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::component_type,
			typename find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::so3_storage_type,
			typename find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::so3_storage_traits,
			typename find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::R3_storage_type,
			typename find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::R3_storage_traits>,
			no_operation_tag<
			find_se3_elmt_scalar_mult_result<Scalar, from_left, ComponentType,
			so3Storage, so3StorageTraits, R3Storage, R3StorageTraits>::value>>;
		

		//// Specialization of std::common_type
		//// If a binary operation between two instances of a template type can be successful,
		//// and the result type is the same regardless of the order of operation, then
		//// that result type is the common type between two.
		//// !!!!!!!!!!!!!!!!!!!!!!(This part is far from complete)!!!!!!!!!!!!!!!!!!!!!!
	
		namespace detail {
			template <std::size_t N,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits,
				bool has_result =
				find_Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded
				&& std::is_same<
				Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>,
				Rn_elmt_binary_result<N, RightComponentType, RightStorage, RightStorageTraits,
				LeftComponentType, LeftStorage, LeftStorageTraits>>::value>
			struct Rn_elmt_common_type_base {};

			template <std::size_t N,
				class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct Rn_elmt_common_type_base<N, LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits, true>
			{
				using type = Rn_elmt_binary_result<N, LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits,
				bool has_result =
				find_gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>::value == no_operation_reason::succeeded
				&& std::is_same<
				gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits>,
				gl2_elmt_binary_result<RightComponentType, RightStorage, RightStorageTraits,
				LeftComponentType, LeftStorage, LeftStorageTraits>>::value>
			struct gl2_elmt_common_type_base {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct gl2_elmt_common_type_base<LeftComponentType, LeftStorage, LeftStorageTraits,
				RightComponentType, RightStorage, RightStorageTraits, true>
			{
				using type = gl2_elmt_binary_result<LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};
		}
	}
}

namespace std {
	template <std::size_t N,
		class LeftComponentType, class LeftStorage, class LeftStorageTraits,
		class RightComponentType, class RightStorage, class RightStorageTraits>
	struct common_type<jkl::math::Rn_elmt<N, LeftComponentType, LeftStorage, LeftStorageTraits>,
		jkl::math::Rn_elmt<N, RightComponentType, RightStorage, RightStorageTraits>> :
		jkl::math::detail::Rn_elmt_common_type_base<N,
		LeftComponentType, LeftStorage, LeftStorageTraits,
		RightComponentType, RightStorage, RightStorageTraits> {};

	template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
		class RightComponentType, class RightStorage, class RightStorageTraits>
	struct common_type<jkl::math::gl2_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
		jkl::math::gl2_elmt<RightComponentType, RightStorage, RightStorageTraits>> :
		jkl::math::detail::gl2_elmt_common_type_base<
		LeftComponentType, LeftStorage, LeftStorageTraits,
		RightComponentType, RightStorage, RightStorageTraits> {};
}

namespace jkl {
	namespace math {
		//// Since CUDA doesn't support std::tuple, we need an alternative for std::tuple.
		//// This will be used only internally, and is not meant to be exposed to users.
		//// Thus, the implementation is very much minimal.
		namespace detail {
			template <class... T>
			class tuple;

			template <>
			class tuple<> {};

			template <std::size_t>
			struct get_impl;

			template <std::size_t I>
			struct get_impl {
				template <class Tuple>
				JKL_GPU_EXECUTABLE static constexpr decltype(auto) get(Tuple&& t) noexcept {
					return std::forward<Tuple>(t).template get<I>();
				}
			};

			template <std::size_t I, class... T>
			JKL_GPU_EXECUTABLE constexpr std::tuple_element_t<I, tuple<T...>>&
				get(tuple<T...>& t) noexcept
			{
				return get_impl<I>::get(t);
			}
			template <std::size_t I, class... T>
			JKL_GPU_EXECUTABLE constexpr std::tuple_element_t<I, tuple<T...>> const&
				get(tuple<T...> const& t) noexcept
			{
				return get_impl<I>::get(t);
			}
			template <std::size_t I, class... T>
			JKL_GPU_EXECUTABLE constexpr std::tuple_element_t<I, tuple<T...>>&&
				get(tuple<T...>&& t) noexcept
			{
				return get_impl<I>::get(std::move(t));
			}
			template <std::size_t I, class... T>
			JKL_GPU_EXECUTABLE constexpr std::tuple_element_t<I, tuple<T...>> const&&
				get(tuple<T...> const&& t) noexcept
			{
				return get_impl<I>::get(std::move(t));
			}

			template <class First, class... Remainings>
			class tuple<First, Remainings...> : private tuple<Remainings...>
			{
				using base_type = tuple<Remainings...>;
				First first;

				template <std::size_t I, class = std::enable_if_t<I == 0>>
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR First& get() & noexcept {
					return first;
				}
				template <std::size_t I, class = std::enable_if_t<I == 0>>
				JKL_GPU_EXECUTABLE constexpr First const& get() const& noexcept {
					return first;
				}
				template <std::size_t I, class = std::enable_if_t<I == 0>>
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR First&& get() && noexcept {
					return static_cast<First&&>(first);
				}
				template <std::size_t I, class = std::enable_if_t<I == 0>>
				JKL_GPU_EXECUTABLE constexpr First const&& get() const&& noexcept {
					return static_cast<First const&&>(first);
				}

				template <std::size_t I, class = std::enable_if_t<I != 0>, class = void>
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() & noexcept {
					return static_cast<base_type&>(*this).template get<I - 1>();
				}
				template <std::size_t I, class = std::enable_if_t<I != 0>, class = void>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const& noexcept {
					return static_cast<base_type const&>(*this).template get<I - 1>();
				}
				template <std::size_t I, class = std::enable_if_t<I != 0>, class = void>
				JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() && noexcept {
					return static_cast<base_type&&>(*this).template get<I - 1>();
				}
				template <std::size_t I, class = std::enable_if_t<I != 0>, class = void>
				JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&& noexcept {
					return static_cast<base_type const&&>(*this).template get<I - 1>();
				}

			public:
				tuple() = default;

				template <class FirstArg,
					class = std::enable_if_t<std::is_convertible<FirstArg, First>::value &&
					sizeof...(Remainings) == 0>,
					class = tmp::prevent_too_perfect_fwd<tuple, FirstArg>>
				JKL_GPU_EXECUTABLE constexpr tuple(FirstArg&& first_arg)
					noexcept(std::is_nothrow_constructible<First, FirstArg>::value) :
					first(std::forward<FirstArg>(first_arg)) {}

				template <class FirstArg,
					class = std::enable_if_t<!std::is_convertible<FirstArg, First>::value &&
					sizeof...(Remainings) == 0 &&
					tmp::is_explicitly_convertible<FirstArg, First>::value>,
					class = tmp::prevent_too_perfect_fwd<tuple, FirstArg>, class = void>
				JKL_GPU_EXECUTABLE constexpr explicit tuple(FirstArg&& first_arg)
					noexcept(std::is_nothrow_constructible<First, FirstArg>::value) :
					first(std::forward<FirstArg>(first_arg)) {}

				template <class FirstArg, class... RemainingArgs,
					class = std::enable_if_t<std::is_convertible<FirstArg, First>::value &&
					tmp::is_convertible<base_type, RemainingArgs...>::value>>
				JKL_GPU_EXECUTABLE constexpr tuple(FirstArg&& first_arg, RemainingArgs&&... remaining_args)
					noexcept(std::is_nothrow_constructible<First, FirstArg>::value &&
						std::is_nothrow_constructible<base_type, RemainingArgs...>::value) :
					base_type(std::forward<RemainingArgs>(remaining_args)...),
					first(std::forward<FirstArg>(first_arg)) {}

				template <class FirstArg, class... RemainingArgs,
					class = std::enable_if_t<!(std::is_convertible<FirstArg, First>::value &&
						tmp::is_convertible<base_type, RemainingArgs...>::value) &&
					tmp::is_explicitly_convertible<FirstArg, First>::value &&
					std::is_constructible<base_type, RemainingArgs...>::value>, class = void>
				JKL_GPU_EXECUTABLE constexpr explicit tuple(FirstArg&& first_arg, RemainingArgs&&... remaining_args)
					noexcept(std::is_nothrow_constructible<First, FirstArg>::value &&
						std::is_nothrow_constructible<base_type, RemainingArgs...>::value) :
					base_type(std::forward<RemainingArgs>(remaining_args)...),
					first(std::forward<FirstArg>(first_arg)) {}

				template <std::size_t>
				friend struct get_impl;

				template <class...>
				friend class tuple;
			};

			template <std::size_t I, class...>
			struct tuple_element_impl;

			template <class First, class... Remainings>
			struct tuple_element_impl<0, First, Remainings...> {
				using type = First;
			};

			template <std::size_t I, class First, class... Remainings>
			struct tuple_element_impl<I, First, Remainings...> {
				using type = typename tuple_element_impl<I - 1, Remainings...>::type;
			};

			// Used for several row-wise constructors of matrix-like classes
			template <class StorageTraits, class... Rows>
			struct row_ref_tuple_traits {
				// Check if a type can be considered as "row-like"
				template <class Row, std::size_t I = sizeof...(Rows)>
				struct is_row_like {
					static constexpr bool value =
						(storage_traits_inspector<StorageTraits>::
							template choose_appropriate<I - 1, Row&&>() !=
							storage_access_method::no_way) &&
						is_row_like<Row, I - 1>::value;
				};
				template <class Row>
				struct is_row_like<Row, 0> {
					static constexpr bool value = true;
				};

				// Check if all types in Rows can be considered as "row-like"
				template <std::size_t I = sizeof...(Rows), class = void>
				struct are_row_like {
					using row_type = std::tuple_element_t<I - 1, tuple<Rows...>>;
					static constexpr bool value =
						is_row_like<row_type>::value &&
						are_row_like<I - 1>::value;
				};
				template <class dummy>
				struct are_row_like<0, dummy> {
					static constexpr bool value = true;
				};

				static constexpr bool value = are_row_like<>::value;

				
				// For detail::tuple<Rows&&...>, use tuple-like access
				template <std::size_t I>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr
					std::tuple_element_t<I, tuple<Rows&&...>>& get(tuple<Rows&&...>& s) noexcept {
					return ::jkl::math::detail::get<I>(s);
				}
				template <std::size_t I>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr
					std::tuple_element_t<I, tuple<Rows&&...>>& get(tuple<Rows&&...> const& s) noexcept {
					return ::jkl::math::detail::get<I>(s);
				}
				template <std::size_t I>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr
					std::tuple_element_t<I, tuple<Rows&&...>> get(tuple<Rows&&...>&& s) noexcept {
					return ::jkl::math::detail::get<I>(std::move(s));
				}
				template <std::size_t I>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr
					std::tuple_element_t<I, tuple<Rows&&...>> get(tuple<Rows&&...> const&& s) noexcept {
					return ::jkl::math::detail::get<I>(std::move(s));
				}

				// For other types, use StorageTraits
				template <std::size_t I, class Storage>
				FORCEINLINE JKL_GPU_EXECUTABLE static constexpr auto get(Storage&& s)
					noexcept(noexcept(StorageTraits::template get<I>(std::forward<Storage>(s))))
					-> decltype(StorageTraits::template get<I>(std::forward<Storage>(s)))
				{
					return StorageTraits::template get<I>(std::forward<Storage>(s));
				}

				template <std::size_t I, class Storage>
				struct tuple_element : StorageTraits::template tuple_element<I, Storage> {};
				
				template <std::size_t I>
				struct tuple_element<I, tuple<Rows&&...>> {
					using type = std::tuple_element_t<I, tuple<Rows&&...>>;
				};

				template <std::size_t I>
				struct tuple_element<I, tuple<Rows&&...>&> {
					using type = std::tuple_element_t<I, tuple<Rows&&...>>&;
				};
				template <std::size_t I>
				struct tuple_element<I, tuple<Rows&&...>&&> {
					using type = std::tuple_element_t<I, tuple<Rows&&...>>;
				};


				template <class Storage, class TargetType>
				using storage_wrapper = Storage;

				template <class StorageWrapper>
				JKL_GPU_EXECUTABLE static constexpr auto&& get_storage(StorageWrapper&& s) noexcept {
					return std::forward<StorageWrapper>(s);
				}
			};
		}
	}
}

namespace std {
	template <class... T>
	struct tuple_size<jkl::math::detail::tuple<T...>> {
		static constexpr std::size_t value = sizeof...(T);
	};

	template <std::size_t I, class... T>
	struct tuple_element<I, jkl::math::detail::tuple<T...>> {
		static_assert(I < tuple_size<jkl::math::detail::tuple<T...>>::value,
			"jkl::math: index out of range");
		using type = typename jkl::math::detail::tuple_element_impl<I, T...>::type;
	};
}