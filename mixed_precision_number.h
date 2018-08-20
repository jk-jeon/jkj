/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2017 Junekey Jeon                                                                   ///
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
#include <type_traits>

namespace jkl {
	namespace math {
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Here is a union-like class template providing a unified way to treat integral type and floating-point type data.
		/// Division (or taking reciprocal) may change the internal operating mode of the union from integer to floating-point, 
		/// while other operations (addition, subtraction, multiplication, etc.) do not.
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			// Classes having trivial destructors may have better opportunites of being optimized.
			// Even empty destructor may give some penalty, especially when compiled by Visual C++ with exception-handling enabled.
			// The purpose of this class template is to generate the trivial destructor for mixed_precision_number whenever possible.
			template<
				typename IntType, typename FloatType,
				template<class> class IntClassifier, template<class> class FloatClassifier,
				bool is_trivially_destructible = std::is_trivially_destructible<IntType>::value &&
				std::is_trivially_destructible<FloatType>::value
			>
			class mixed_precision_number_base {
			protected:
				union {
					IntType		int_value;
					FloatType	float_value;
				};
				static constexpr bool is_nothrow_destructible = noexcept(std::declval<IntType>().~IntType()) &&
					noexcept(std::declval<FloatType>().~FloatType());

				struct valueless_mode_construction {};
				mixed_precision_number_base(valueless_mode_construction) noexcept : mode(valueless_mode) {}

			public:
				// valueless_mode is activated only under some exceptional circumstances; e.g., exception is thrown during operation
				enum operating_mode : unsigned char {
					int_mode, float_mode, valueless_mode
				} mode;

				// Default constructor
				// If IntType is default-constructible, then default-construct it
				template <class = std::enable_if_t<std::is_default_constructible<IntType>::value>>
				constexpr mixed_precision_number_base() noexcept : int_value{}, mode(int_mode) {}
				// If IntType is not default-constructible but FloatType is default-constructible, default-construct it
				template <class = std::enable_if_t<
					!std::is_default_constructible<IntType>::value && std::is_default_constructible<FloatType>::value>, 
					class = void>
				constexpr mixed_precision_number_base() noexcept : float_value{}, mode(float_mode) {}
				// If both IntType and FloatType are not default-constructible, go to valueless modes
				template <class = std::enable_if_t<
					!std::is_default_constructible<IntType>::value && !std::is_default_constructible<FloatType>::value>,
					class = void, class = void>
				constexpr mixed_precision_number_base() noexcept : mode(valueless_mode) {}
				// SFINAE constructors
				template <class T, class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value>>
				constexpr mixed_precision_number_base(T&& value) noexcept(noexcept(IntType(std::forward<T>(value))))
					: int_value(std::forward<T>(value)), mode(int_mode) {}
				template <class T, class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
				constexpr mixed_precision_number_base(T&& value) noexcept(noexcept(FloatType(std::forward<T>(value))))
					: float_value(std::forward<T>(value)), mode(float_mode) {}
			};

			template<typename IntType, typename FloatType,
				template<class> class IntClassifier, template<class> class FloatClassifier>
			class mixed_precision_number_base<IntType, FloatType, IntClassifier, FloatClassifier, false> {
			protected:
				union {
					IntType		int_value;
					FloatType	float_value;
				};
				static constexpr bool is_nothrow_destructible = noexcept(std::declval<IntType>().~IntType()) &&
					noexcept(std::declval<FloatType>().~FloatType());

				struct valueless_mode_construction {};
				mixed_precision_number_base(valueless_mode_construction) noexcept : mode(valueless_mode) {}

			public:
				// valueless_mode is activated only under some exceptional circumstances; e.g., exception is thrown during operation
				enum operating_mode : unsigned char {
					int_mode, float_mode, valueless_mode
				} mode;

				// Default constructor
				// If IntType is default-constructible, then default-construct it
				template <class = std::enable_if_t<std::is_default_constructible<IntType>::value>>
				constexpr mixed_precision_number_base() noexcept : int_value{}, mode(int_mode) {}
				// If IntType is not default-constructible but FloatType is default-constructible, default-construct it
				template <class = std::enable_if_t<
					!std::is_default_constructible<IntType>::value && std::is_default_constructible<FloatType>::value>,
					class = void>
					constexpr mixed_precision_number_base() noexcept : float_value{}, mode(float_mode) {}
				// If both IntType and FloatType are not default-constructible, go to valueless modes
				template <class = std::enable_if_t<
					!std::is_default_constructible<IntType>::value && !std::is_default_constructible<FloatType>::value>,
					class = void, class = void>
					constexpr mixed_precision_number_base() noexcept : mode(valueless_mode) {}
				// SFINAE constructors
				template <class T, class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value>>
				constexpr mixed_precision_number_base(T&& value) noexcept(noexcept(IntType(std::forward<T>(value))))
					: int_value(std::forward<T>(value)), mode(int_mode) {}
				template <class T, class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
				constexpr mixed_precision_number_base(T&& value) noexcept(noexcept(FloatType(std::forward<T>(value))))
					: float_value(std::forward<T>(value)), mode(float_mode) {}

				// Destructor
				~mixed_precision_number_base() noexcept(is_nothrow_destructible) {
					switch( mode ) {
					case int_mode:
						int_value.~IntType();
						break;
					case float_mode:
						float_value.~FloatType();
						break;
					}
				}
			};
		}

		// The main class
		template<typename IntType = long long, typename FloatType = long double,
			template<class> class IntClassifier = std::is_integral,
			template<class> class FloatClassifier = std::is_floating_point>
		class mixed_precision_number : public detail::mixed_precision_number_base<IntType, FloatType, IntClassifier, FloatClassifier> {
		protected:
			using base_type = detail::mixed_precision_number_base<IntType, FloatType, IntClassifier, FloatClassifier>;
			using base_type::int_value;
			using base_type::float_value;
			using base_type::mode;
			using base_type::is_nothrow_destructible;
			using typename base_type::valueless_mode_construction;

			mixed_precision_number(valueless_mode_construction x) noexcept : base_type(x) {}

		public:
			using operating_mode = base_type::operating_mode;
			using base_type::int_mode;
			using base_type::float_mode;
			using base_type::valueless_mode;

			using integral_type = IntType;
			using floating_point_type = FloatType;
			template <class T>
			using is_integral = IntClassifier<T>;
			template <class T>
			using is_floating_point = FloatClassifier<T>;

			// Default constructor
			constexpr mixed_precision_number() noexcept {}
			// Inherited constructors
			using base_type::base_type;
			// Copy constructor
			mixed_precision_number(const mixed_precision_number& that) noexcept(std::is_nothrow_copy_constructible<IntType>::value &&
				std::is_nothrow_copy_constructible<FloatType>::value) {
				switch( that.mode ) {
				case int_mode:
					new(&int_value) IntType(that.int_value);
					mode = int_mode;
					break;
				case float_mode:
					new(&float_value) FloatType(that.float_value);
					mode = float_mode;
					break;
				}
			}
			// Move constructor
			mixed_precision_number(mixed_precision_number&& that) noexcept(std::is_nothrow_copy_constructible<IntType>::value &&
				std::is_nothrow_copy_constructible<FloatType>::value) {
				switch( that.mode ) {
				case int_mode:
					new(&int_value) IntType(std::move(that.int_value));
					mode = int_mode;
					break;
				case float_mode:
					new(&float_value) FloatType(std::move(that.float_value));
					mode = float_mode;
					break;
				}
			}

			// Copy assignment operator
			mixed_precision_number& operator=(const mixed_precision_number& that) & noexcept(is_nothrow_destructible &&
				std::is_nothrow_copy_constructible<IntType>::value &&
				std::is_nothrow_copy_constructible<FloatType>::value &&
				std::is_nothrow_copy_assignable<IntType>::value &&
				std::is_nothrow_copy_assignable<FloatType>::value) {
				switch( that.mode ) {
				case int_mode:
					switch( mode ) {
					case int_mode:
						mode = valueless_mode;
						int_value = that.int_value;
						break;

					case float_mode:
						float_value.~FloatType();
						mode = valueless_mode;

					default:
						new(&int_value) IntType(that.int_value);
					}
					mode = int_mode;
					break;

				case float_mode:
					switch( mode ) {
					case float_mode:
						mode = valueless_mode;
						float_value = that.float_value;
						break;

					case int_mode:
						int_value.~IntType();
						mode = valueless_mode;

					default:
						new(&float_value) FloatType(that.float_value);
					}
					mode = float_mode;
					break;

				default:
					this->~mixed_precision_number();
				}
				return *this;
			}
			// Move assignment operator
			mixed_precision_number& operator=(mixed_precision_number&& that) & noexcept(is_nothrow_destructible &&
				std::is_nothrow_move_constructible<IntType>::value &&
				std::is_nothrow_move_constructible<FloatType>::value &&
				std::is_nothrow_move_assignable<IntType>::value &&
				std::is_nothrow_move_assignable<FloatType>::value) {
				switch( that.mode ) {
				case int_mode:
					switch( mode ) {
					case int_mode:
						mode = valueless_mode;
						int_value = std::move(that.int_value);
						break;

					case float_mode:
						float_value.~FloatType();
						mode = valueless_mode;

					default:
						new(&int_value) IntType(std::move(that.int_value));
					}
					mode = int_mode;
					break;

				case float_mode:
					switch( mode ) {
					case float_mode:
						mode = valueless_mode;
						float_value = std::move(that.float_value);
						break;

					case int_mode:
						int_value.~IntType();
						mode = valueless_mode;

					default:
						new(&float_value) FloatType(std::move(that.float_value));
					}
					mode = float_mode;
					break;

				default:
					this->~mixed_precision_number();
				}
				return *this;
			}
			// SFINAE assignment operators
			template <class T, class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value>>
			mixed_precision_number& operator=(T&& value) & noexcept(is_nothrow_destructible &&
				std::is_nothrow_assignable<IntType, std::add_rvalue_reference_t<T>>::value &&
				std::is_nothrow_constructible<IntType, std::add_rvalue_reference_t<T>>::value) {
				switch( mode ) {
				case int_mode:
					mode = valueless_mode;
					int_value = std::forward<T>(value);
					break;

				case float_mode:
					float_value.~FloatType();
					mode = valueless_mode;

				default:
					new(&int_value) IntType(std::forward<T>(value));
				}
				mode = int_mode;
				return *this;
			}
			template <class T, class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			mixed_precision_number& operator=(T&& value) & noexcept(is_nothrow_destructible &&
				std::is_nothrow_assignable<FloatType, std::add_rvalue_reference_t<T>>::value &&
				std::is_nothrow_constructible<FloatType, std::add_rvalue_reference_t<T>>::value) {
				switch( mode ) {
				case float_mode:
					mode = valueless_mode;
					float_value = std::forward<T>(value);
					break;

				case int_mode:
					int_value.~IntType();
					mode = valueless_mode;

				default:
					new(&float_value) FloatType(std::forward<T>(value));
				}
				mode = float_mode;
				return *this;
			}

			// Utililies
			bool is_int() const noexcept { return mode == int_mode; }
			bool is_float() const noexcept { return mode == float_mode; }
			bool is_undefined() const noexcept { return mode == valueless_mode; }
			operating_mode is_on() const noexcept { return mode; }
			// Get raw values (unsafe)
			const IntType& as_int() const& noexcept { return int_value; }
			const FloatType& as_float() const& noexcept { return float_value; }
			IntType&& as_int() && noexcept { return std::move(int_value); }
			FloatType&& as_float() && noexcept { return std::move(float_value); }

			// SFINAE casting operators
			template <class T, class = std::enable_if_t<
				IntClassifier<std::remove_reference_t<T>>::value || FloatClassifier<std::remove_reference_t<T>>::value> >
			operator T() const& noexcept(std::is_nothrow_constructible<T, const IntType&>::value &&
				std::is_nothrow_constructible<T, const FloatType&>::value && std::is_nothrow_default_constructible<T>::value) {
				switch( mode ) {
				case int_mode:
					return (T)int_value;
				case float_mode:
					return (T)float_value;
				default:
					return T();
				}
			}
			template <class T, class = std::enable_if_t<
				IntClassifier<std::remove_reference_t<T>>::value || FloatClassifier<std::remove_reference_t<T>>::value> >
			operator T() && noexcept(std::is_nothrow_constructible<T, IntType&&>::value &&
				std::is_nothrow_constructible<T, FloatType&&>::value && std::is_nothrow_default_constructible<T>::value) {
				switch( mode ) {
				case int_mode:
					return (T)std::move(int_value);
				case float_mode:
					return (T)std::move(float_value);
				default:
					return T();
				}
			}

			// Unary arithmetic operators
			mixed_precision_number operator-() const& noexcept(noexcept(mixed_precision_number(-int_value)) && noexcept(mixed_precision_number(-float_value))) {
				switch( mode ) {
				case int_mode:
					return mixed_precision_number(-int_value);
				case float_mode:
					return mixed_precision_number(-float_value);
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}
			mixed_precision_number operator-() && noexcept(noexcept(mixed_precision_number(-std::move(int_value))) && noexcept(mixed_precision_number(-std::move(float_value)))) {
				switch( mode ) {
				case int_mode:
					return mixed_precision_number(-std::move(int_value));
				case float_mode:
					return mixed_precision_number(-std::move(float_value));
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}
			mixed_precision_number reciprocal() const& noexcept(is_nothrow_convertible && noexcept(mixed_precision_number(std::declval<FloatType>()))) {
				switch( mode ) {
				case int_mode:
					if( int_value == 1 || int_value == -1 )
						return *this;
					else
						return mixed_precision_number((FloatType)1 / (FloatType)int_value);
				case float_mode:
					return mixed_precision_number((FloatType)1 / float_value);
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}
			mixed_precision_number reciprocal() && noexcept(is_nothrow_convertible && noexcept(mixed_precision_number(std::declval<FloatType>()))) {
				switch( mode ) {
				case int_mode:
					if( int_value == 1 || int_value == -1 )
						return std::move(*this);
					else
						return mixed_precision_number((FloatType)1 / (FloatType)std::move(int_value));
				case float_mode:
					return mixed_precision_number((FloatType)1 / std::move(float_value));
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}

			// Binary arithmetic operators (const lvalue & non-const rvalue for +,-,*,/)
			template <class T>
			mixed_precision_number operator+(T&& that) const& noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<const IntType&>() + std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() + std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() + std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() + std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() + std::declval<FloatType>()))) {
				return binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) + std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator+(T&& that) && noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<IntType>() + std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() + std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() + std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() + std::declval<FloatType>()))) {
				return binary_op_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) + std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator-(T&& that) const& noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<const IntType&>() - std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() - std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() - std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() - std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() - std::declval<FloatType>()))) {
				return binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) - std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator-(T&& that) && noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<IntType>() - std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() - std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() - std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() - std::declval<FloatType>()))) {
				return binary_op_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) - std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator*(T&& that) const& noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<const IntType&>() * std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() * std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() * std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() * std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() * std::declval<FloatType>()))) {
				return binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator*(T&& that) && noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<IntType>() * std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() * std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() * std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() * std::declval<FloatType>()))) {
				return binary_op_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return mixed_precision_number(std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b)); });
			}
			template <class T>
			mixed_precision_number operator/(T&& that) const& noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<const IntType&>() / std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() / std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() / std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() / std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() / std::declval<FloatType>()))) {
				return binary_op(std::forward<T>(that), DivisionFunctor());
			}
			template <class T>
			mixed_precision_number operator/(T&& that) && noexcept(is_nothrow_convertible && is_nothrow_move_constructible && (
				noexcept(std::declval<IntType>() / std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() / std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() / std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() / std::declval<FloatType>()))) {
				return binary_op_rvalue(std::forward<T>(that), DivisionFunctor());
			}

			// Inplace binary arithmetic operators (non-const lvalue for +=, -=, *=, /=)
			template <class T>
			mixed_precision_number& operator+=(T&& that) & noexcept(is_nothrow_convertible && is_nothrow_destructible && (
				noexcept(std::declval<IntType&>() += std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType&>() += std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType&>() += std::declval<IntType>()) &&
				noexcept(std::declval<FloatType&>() += std::declval<FloatType>()))) {
				inplace_binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { std::forward<decltype(a)>(a) += std::forward<decltype(b)>(b); });
				return *this;
			}
			template <class T>
			mixed_precision_number& operator-=(T&& that) & noexcept(is_nothrow_convertible && is_nothrow_destructible && (
				noexcept(std::declval<IntType&>() -= std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType&>() -= std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType&>() -= std::declval<IntType>()) &&
				noexcept(std::declval<FloatType&>() -= std::declval<FloatType>()))) {
				inplace_binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { std::forward<decltype(a)>(a) -= std::forward<decltype(b)>(b); });
				return *this;
			}
			template <class T>
			mixed_precision_number& operator*=(T&& that) & noexcept(is_nothrow_convertible && is_nothrow_destructible && (
				noexcept(std::declval<IntType&>() *= std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType&>() *= std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType&>() *= std::declval<IntType>()) &&
				noexcept(std::declval<FloatType&>() *= std::declval<FloatType>()))) {
				inplace_binary_op(std::forward<T>(that),
					[](auto&& a, auto&& b) { std::forward<decltype(a)>(a) *= std::forward<decltype(b)>(b); });
				return *this;
			}
			template <class T>
			mixed_precision_number& operator/=(T&& that) & noexcept(is_nothrow_convertible && is_nothrow_move_constructible && is_nothrow_destructible && (
				noexcept(std::declval<IntType&>() /= std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType&>() /= std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType&>() /= std::declval<IntType>()) &&
				noexcept(std::declval<FloatType&>() /= std::declval<FloatType>()))) {
				inplace_binary_op(std::forward<T>(that), InplaceDivisionFunctor(this));
				return *this;
			}

			// Binary relations (const lvalue and non-const rvalue for ==, !=, <=, >=, <, >)
			template <class T>
			bool operator==(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() == std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() == std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() == std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() == std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() == std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) == std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator==(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() == std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() == std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() == std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() == std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) == std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator!=(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() != std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() != std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() != std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() != std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() != std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) != std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator!=(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() != std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() != std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() != std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() != std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) != std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator<=(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() <= std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() <= std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() <= std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() <= std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() <= std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) <= std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator<=(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() <= std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() <= std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() <= std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() <= std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) <= std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator>=(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() >= std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() >= std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() >= std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() >= std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() >= std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) >= std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator>=(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() >= std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() >= std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() >= std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() >= std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) >= std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator<(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() < std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() < std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() < std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() < std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() < std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) < std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator<(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() < std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() < std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() < std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() < std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) < std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator>(T&& that) const& noexcept(is_nothrow_convertible && (
				noexcept(std::declval<const IntType&>() > std::declval<const IntType&>()) &&
				noexcept(std::declval<const FloatType&>() > std::declval<const FloatType&>()) &&
				noexcept(std::declval<const IntType&>() > std::declval<IntType>()) &&
				noexcept(std::declval<const FloatType&>() > std::declval<FloatType>()) &&
				noexcept(std::declval<FloatType>() > std::declval<FloatType>()))) {
				return binary_rel(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) > std::forward<decltype(b)>(b); });
			}
			template <class T>
			bool operator>(T&& that) && noexcept(is_nothrow_convertible && (
				noexcept(std::declval<IntType>() > std::declval<const IntType&>()) &&
				noexcept(std::declval<FloatType>() > std::declval<const FloatType&>()) &&
				noexcept(std::declval<IntType>() > std::declval<IntType>()) &&
				noexcept(std::declval<FloatType>() > std::declval<FloatType>()))) {
				return binary_rel_rvalue(std::forward<T>(that),
					[](auto&& a, auto&& b) { return std::forward<decltype(a)>(a) > std::forward<decltype(b)>(b); });
			}

		private:
			static constexpr bool is_nothrow_move_constructible = std::is_nothrow_move_constructible<IntType>::value &&
				std::is_nothrow_move_constructible<FloatType>::value;
			static constexpr bool is_nothrow_convertible = noexcept((FloatType)std::declval<IntType>()) &&
				noexcept((FloatType)std::declval<const IntType&>());

			struct DivisionFunctor {
				// For integral operands
				template <typename T1, typename T2,
					class = std::enable_if_t< IntClassifier<std::remove_reference_t<T1>>::value &&
					IntClassifier<std::remove_reference_t<T2>>::value >>
				mixed_precision_number operator()(T1&& a, T2&& b) const {
					auto q = a / b;
					if( q * b != a )
						return mixed_precision_number((FloatType)std::forward<T1>(a) / (FloatType)std::forward<T2>(b));
					else
						return mixed_precision_number(q);
				}
				// For floating-point operands
				template <typename T1, typename T2,
					class = std::enable_if_t< FloatClassifier<std::remove_reference_t<T1>>::value ||
					FloatClassifier<std::remove_reference_t<T2>>::value >, class = void>
				mixed_precision_number operator()(T1&& a, T2&& b) const {
					return mixed_precision_number(a / b);
				}
			};

			struct InplaceDivisionFunctor {
				mixed_precision_number* n;
				InplaceDivisionFunctor(mixed_precision_number* n) : n(n) {}
				// For integral operands
				template <typename T1, typename T2,
					class = std::enable_if_t< IntClassifier<std::remove_reference_t<T1>>::value &&
					IntClassifier<std::remove_reference_t<T2>>::value >>
				void operator()(T1&& a, T2&& b) const {
					auto q = a / b;
					if( q * b != a ) {
						n->int_value.~IntType();
						n->mode = valueless_mode;
						new(&n->float_value) FloatType((FloatType)std::forward<T1>(a) / (FloatType)std::forward<T2>(b));
						n->mode = float_mode;
					} else
						a = q;
				}
				// For floating-point operands
				template <typename T1, typename T2,
					class = std::enable_if_t< FloatClassifier<std::remove_reference_t<T1>>::value ||
					FloatClassifier<std::remove_reference_t<T2>>::value >, class = void>
				void operator()(T1&& a, T2&& b) const {
					a /= b;
				}
			};

			// Binary operation for lvalue mixed_precision_number operand
			template <typename Functor>
			mixed_precision_number binary_op(const mixed_precision_number& that, Functor&& func) const {
				if( mode == valueless_mode )
					return mixed_precision_number(valueless_mode_construction{});
				else if( that.is_undefined() )
					return mixed_precision_number(valueless_mode_construction{});
				else if( mode == int_mode && that.is_int() )
					return func(int_value, that.int_value);
				else if( mode == int_mode && that.is_float() )
					return func(FloatType(int_value), that.float_value);
				else if( mode == float_mode && that.is_int() )
					return func(float_value, (FloatType)that.int_value);
				else
					return func(float_value, that.float_value);
			}
			// Binary operation for rvalue mixed_precision_number operand
			template <typename Functor>
			mixed_precision_number binary_op(mixed_precision_number&& that, Functor&& func) const {
				if( mode == valueless_mode )
					return mixed_precision_number(valueless_mode_construction{});
				else if( that.is_undefined() )
					return mixed_precision_number(valueless_mode_construction{});
				else if( mode == int_mode && that.is_int() )
					return func(int_value, std::move(that).int_value);
				else if( mode == int_mode && that.is_float() )
					return func(FloatType(int_value), std::move(that).float_value);
				else if( mode == float_mode && that.is_int() )
					return func(float_value, (FloatType)std::move(that).int_value);
				else
					return func(float_value, std::move(that).float_value);
			}
			// Binary operation for integral operand
			template <typename T, typename Functor,
				class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value >>
			mixed_precision_number binary_op(T&& that, Functor&& func) const {
				switch( mode ) {
				case int_mode:
					return func(int_value, std::forward<T>(that));
				case float_mode:
					return func(float_value, (FloatType)std::forward<T>(that));
				default:
					return mixed_precision_number(valueless_mode_construction{});;
				}
			}
			// Binary operation for floating-point operand
			template <typename T, typename Functor,
				class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			mixed_precision_number binary_op(T&& that, Functor&& func) const {
				switch( mode ) {
				case int_mode:
					return func((FloatType)int_value, std::forward<T>(that));
				case float_mode:
					return func(float_value, std::forward<T>(that));
				default:
					return mixed_precision_number(valueless_mode_construction{});;
				}
			}

			// Binary operation for lvalue mixed_precision_number operand
			template <typename Functor>
			mixed_precision_number binary_op_rvalue(const mixed_precision_number& that, Functor&& func) {
				if( mode == valueless_mode )
					return mixed_precision_number(valueless_mode_construction{});
				else if( that.is_undefined() )
					return mixed_precision_number(valueless_mode_construction{});
				else if( mode == int_mode && that.is_int() )
					return func(std::move(int_value), that.int_value);
				else if( mode == int_mode && that.is_float() )
					return func((FloatType)std::move(int_value), that.float_value);
				else if( mode == float_mode && that.is_int() )
					return func(std::move(float_value), (FloatType)that.int_value);
				else
					return func(std::move(float_value), that.float_value);
			}
			// Binary operation for rvalue mixed_precision_number operand
			template <typename Functor>
			mixed_precision_number binary_op_rvalue(mixed_precision_number&& that, Functor&& func) {
				if( mode == valueless_mode )
					return mixed_precision_number(valueless_mode_construction{});
				else if( that.is_undefined() )
					return mixed_precision_number(valueless_mode_construction{});
				else if( mode == int_mode && that.is_int() )
					return func(std::move(int_value), std::move(that).int_value);
				else if( mode == int_mode && that.is_float() )
					return func((FloatType)std::move(int_value), std::move(that).float_value);
				else if( mode == float_mode && that.is_int() )
					return func(std::move(float_value), (FloatType)std::move(that).int_value);
				else
					return func(std::move(float_value), std::move(that).float_value);
			}
			// Binary operation for integral operand
			template <typename T, typename Functor,
				class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value >>
			mixed_precision_number binary_op_rvalue(T&& that, Functor&& func) {
				switch( mode ) {
				case int_mode:
					return func(std::move(int_value), std::forward<T>(that));
				case float_mode:
					return func(std::move(float_value), (FloatType)std::forward<T>(that));
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}
			// Binary operation for floating-point operand
			template <typename T, typename Functor,
				class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			mixed_precision_number binary_op_rvalue(T&& that, Functor&& func) {
				switch( mode ) {
				case int_mode:
					return func((FloatType)std::move(int_value), std::forward<T>(that));
				case float_mode:
					return func(std::move(float_value), std::forward<T>(that));
				default:
					return mixed_precision_number(valueless_mode_construction{});
				}
			}

			// Inplace binary operation for lvalue mixed_precision_number operand
			template <typename Functor>
			void inplace_binary_op(const mixed_precision_number& that, Functor&& func) {
				if( mode == valueless_mode || that.is_undefined() )
					return;
				else if( mode == int_mode && that.is_int() )
					func(int_value, that.int_value);
				else if( mode == int_mode && that.is_float() ) {
					int_value.~IntType();
					mode = valueless_mode;
					new(&float_value) FloatType((FloatType)int_value);
					mode = float_mode;
					func(float_value, that.float_value);
				} else if( mode == float_mode && that.is_int() )
					func(float_value, (FloatType)that.int_value);
				else
					func(float_value, that.float_value);
			}
			// Inplace binary operation for rvalue mixed_precision_number operand
			template <typename Functor>
			void inplace_binary_op(mixed_precision_number&& that, Functor&& func) {
				if( mode == valueless_mode || that.is_undefined() )
					return;
				else if( mode == int_mode && that.is_int() )
					func(int_value, std::move(that).int_value);
				else if( mode == int_mode && that.is_float() ) {
					int_value.~IntType();
					mode = valueless_mode;
					new(&float_value) FloatType((FloatType)int_value);
					mode = float_mode;
					func(float_value, std::move(that).float_value);
				} else if( mode == float_mode && that.is_int() )
					func(float_value, (FloatType)std::move(that).int_value);
				else
					func(float_value, std::move(that).float_value);
			}
			// Inplace binary operation for integral operand
			template <typename T, typename Functor,
				class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value >>
			void inplace_binary_op(T&& that, Functor&& func) {
				switch( mode ) {
				case int_mode:
					func(int_value, std::forward<T>(that));
					break;
				case float_mode:
					func(float_value, (FloatType)std::forward<T>(that));
					break;
				default:
					break;
				}
			}
			// Inplace binary operation for floating-point operand
			template <typename T, typename Functor,
				class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			void inplace_binary_op(T&& that, Functor&& func) {
				switch( mode ) {
				case int_mode:
					int_value.~IntType();
					mode = valueless_mode;
					new(&float_value) FloatType((FloatType)int_value);
					mode = float_mode;
					func(float_value, std::forward<T>(that));
					break;
				case float_mode:
					func(float_value, std::forward<T>(that));
					break;
				default:
					break;
				}
			}

			// Binary relation for lvalue mixed_precision_number operand
			template <typename Functor>
			bool binary_rel(const mixed_precision_number& that, Functor&& rel) const {
				if( mode == valueless_mode || that.is_undefined() )
					return false;
				else if( mode == int_mode && that.is_int() )
					return rel(int_value, that.int_value);
				else if( mode == int_mode && that.is_float() )
					return rel((FloatType)int_value, that.float_value);
				else if( mode == float_mode && that.is_int() )
					return rel(float_value, (FloatType)that.int_value);
				else
					return rel(float_value, that.float_value);
			}
			// Binary relation for rvalue mixed_precision_number operand
			template <typename Functor>
			bool binary_rel(mixed_precision_number&& that, Functor&& rel) const {
				if( mode == valueless_mode || that.is_undefined() )
					return false;
				else if( mode == int_mode && that.is_int() )
					return rel(int_value, std::move(that).int_value);
				else if( mode == int_mode && that.is_float() )
					return rel((FloatType)int_value, std::move(that).float_value);
				else if( mode == float_mode && that.is_int() )
					return rel(float_value, (FloatType)std::move(that).int_value);
				else
					return rel(float_value, std::move(that).float_value);
			}
			// Binary relation for rvalue integral operand
			template <typename T, typename Functor,
				class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value >>
			bool binary_rel(T&& that, Functor&& rel) const {
				switch( mode ) {
				case int_mode:
					return rel(int_value, std::forward<T>(that));
				case float_mode:
					return rel(float_value, (FloatType)std::forward<T>(that));
				default:
					return false;
				}
			}
			// Binary relation for rvalue floating-point operand
			template <typename T, typename Functor,
				class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			bool binary_rel(T&& that, Functor&& rel) const {
				switch( mode ) {
				case int_mode:
					return rel((FloatType)int_value, std::forward<T>(that));
				case float_mode:
					return rel(float_value, std::forward<T>(that));
				default:
					return false;
				}
			}

			// Binary relation for lvalue mixed_precision_number operand
			template <typename Functor>
			bool binary_rel_rvalue(const mixed_precision_number& that, Functor&& rel) {
				if( mode == valueless_mode || that.is_undefined() )
					return false;
				else if( mode == int_mode && that.is_int() )
					return rel(std::move(int_value), that.int_value);
				else if( mode == int_mode && that.is_float() )
					return rel((FloatType)std::move(int_value), that.float_value);
				else if( mode == float_mode && that.is_int() )
					return rel(std::move(float_value), (FloatType)that.int_value);
				else
					return rel(std::move(float_value), that.float_value);
			}
			// Binary relation for rvalue mixed_precision_number operand
			template <typename Functor>
			bool binary_rel_rvalue(mixed_precision_number&& that, Functor&& rel) {
				if( mode == valueless_mode || that.is_undefined() )
					return false;
				else if( mode == int_mode && that.is_int() )
					return rel(std::move(int_value), std::move(that).int_value);
				else if( mode == int_mode && that.is_float() )
					return rel((FloatType)std::move(int_value), std::move(that).float_value);
				else if( mode == float_mode && that.is_int() )
					return rel(std::move(float_value), (FloatType)std::move(that).int_value);
				else
					return rel(std::move(float_value), std::move(that).float_value);
			}
			// Binary relation for rvalue integral operand
			template <typename T, typename Functor,
				class = std::enable_if_t<IntClassifier<std::remove_reference_t<T>>::value >>
			bool binary_rel_rvalue(T&& that, Functor&& rel) {
				switch( mode ) {
				case int_mode:
					return rel(std::move(int_value), std::forward<T>(that));
				case float_mode:
					return rel(std::move(float_value), (FloatType)std::forward<T>(that));
				default:
					return false;
				}
			}
			// Binary relation for rvalue floating-point operand
			template <typename T, typename Functor,
				class = std::enable_if_t<FloatClassifier<std::remove_reference_t<T>>::value>, class = void>
			bool binary_rel_rvalue(T&& that, Functor&& rel) {
				switch( mode ) {
				case int_mode:
					return rel((FloatType)std::move(int_value), std::forward<T>(that));
				case float_mode:
					return rel(std::move(float_value), std::forward<T>(that));
				default:
					return false;
				}
			}
		};
		// Addition
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator+(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() + std::declval<decltype(a)>())) {
			return b + std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator+(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() + std::declval<decltype(a)>())) {
			return std::move(b) + std::forward<T>(a);
		}
		// Subtraction
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator-(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(-std::declval<decltype(b)>() + std::declval<decltype(a)>())) {
			return -b + std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator-(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(-std::declval<decltype(b)>() + std::declval<decltype(a)>())) {
			return -std::move(b) + std::forward<T>(a);
		}
		// Multiplication
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator*(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() * std::declval<decltype(a)>())) {
			return b * std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator*(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() * std::declval<decltype(a)>())) {
			return std::move(b) * std::forward<T>(a);
		}
		// Division
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator/(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>().reciprocal() * std::declval<decltype(a)>())) {
			return b.reciprocal() * std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>
			operator/(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>().reciprocal() * std::declval<decltype(a)>())) {
			return std::move(b).reciprocal() * std::forward<T>(a);
		}
		// Equal
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator==(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() == std::declval<decltype(a)>())) {
			return b == std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator==(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() == std::declval<decltype(a)>())) {
			return std::move(b) == std::forward<T>(a);
		}
		// Not equal
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator!=(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() != std::declval<decltype(a)>())) {
			return b != std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator!=(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() != std::declval<decltype(a)>())) {
			return std::move(b) != std::forward<T>(a);
		}
		// Less-than or equal
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator<=(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() >= std::declval<decltype(a)>())) {
			return b >= std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator<=(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() >= std::declval<decltype(a)>())) {
			return std::move(b) >= std::forward<T>(a);
		}
		// Greater-than or equal
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator>=(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() <= std::declval<decltype(a)>())) {
			return b <= std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator>=(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() <= std::declval<decltype(a)>())) {
			return std::move(b) <= std::forward<T>(a);
		}
		// Less-than
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator<(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() > std::declval<decltype(a)>())) {
			return b > std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator<(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() > std::declval<decltype(a)>())) {
			return std::move(b) > std::forward<T>(a);
		}
		// Greater-than
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator>(T&& a, const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& b)
			noexcept(noexcept(std::declval<decltype(b)>() < std::declval<decltype(a)>())) {
			return b < std::forward<T>(a);
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class T,
			class = std::enable_if_t< !std::is_same<mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>,
			std::remove_reference_t<T>>::value >>
		inline bool operator>(T&& a, mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& b)
			noexcept(noexcept(std::declval<decltype(b)>() < std::declval<decltype(a)>())) {
			return std::move(b) < std::forward<T>(a);
		}
		// Stream output
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class TChar, class Char_Policy>
		inline std::basic_ostream<TChar, Char_Policy>& operator<<(std::basic_ostream<TChar, Char_Policy>& stream,
			const mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>& number) noexcept(
			noexcept(std::declval<decltype(stream)>() << std::declval<IntType>()) &&
			noexcept(std::declval<decltype(stream)>() << std::declval<FloatType>())) {
			switch( number.is_on() ) {
			case mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>::int_mode:
				return stream << number.as_int();
			case mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>::float_mode:
				return stream << number.as_float();
			default:
				return stream;
			}
		}
		template <class IntType, class FloatType,
			template<class> class IntClassifier, template<class> class FloatClassifier, class TChar, class Char_Policy>
		inline std::basic_ostream<TChar, Char_Policy>& operator<<(std::basic_ostream<TChar, Char_Policy>& stream,
			mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>&& number) noexcept(
			noexcept(std::declval<decltype(stream)>() << std::declval<IntType>()) &&
			noexcept(std::declval<decltype(stream)>() << std::declval<FloatType>())) {
			switch( number.is_on() ) {
			case mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>::int_mode:
				return stream << std::move(number).as_int();
			case mixed_precision_number<IntType, FloatType, IntClassifier, FloatClassifier>::float_mode:
				return stream << std::move(number).as_float();
			default:
				return stream;
			}
		}
	}
}