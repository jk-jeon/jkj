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
#include <cstdint>
#include "portability.h"

namespace jkl {
	namespace math {
		/// Calculate the 2-based logarithm of an integer n
		/// The result is the largest nonnegative integer e such that 2^e <= n
		/// When n = 0, this function produces 0, though this is not the correct answer
		/// See http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
		template <class UIntType>
		constexpr UIntType ilog2(UIntType n) noexcept;

		namespace detail {
			template <typename UIntType, class = void>
			struct log2_table;

			template <class dummy>
			struct log2_table<std::uint8_t, dummy> {
				static constexpr std::uint8_t table[256] = {
					0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
					4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
					5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
				};
			};
			template <class dummy>
			constexpr std::uint8_t log2_table<std::uint8_t, dummy>::table[256];

			template <class dummy>
			struct log2_table<std::uint16_t, dummy> {
				static constexpr std::uint16_t table[16] = {
					0, 7, 1, 13, 8, 10, 2, 14, 6, 12, 9, 5, 11, 4, 3, 15
				};
			};
			template <class dummy>
			constexpr std::uint16_t log2_table<std::uint16_t, dummy>::table[16];

			template <class dummy>
			struct log2_table<std::uint32_t, dummy> {
				static constexpr std::uint32_t table[32] = {
					0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
					8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
				};
			};
			template <class dummy>
			constexpr std::uint32_t log2_table<std::uint32_t, dummy>::table[32];

			template <class dummy>
			struct log2_table<std::uint64_t, dummy> {
				static constexpr std::uint64_t table[64] = {
					0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54, 33, 42, 3, 61,
					51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4, 62,
					57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21, 56,
					45, 25, 31, 35, 16, 9, 12, 44, 24, 15, 8, 23, 7, 6, 5, 63
				};
			};
			template <class dummy>
			constexpr std::uint64_t log2_table<std::uint64_t, dummy>::table[64];

			template <typename UIntType>
			struct set_under_msb {
				template <class shift = std::integral_constant<UIntType, 1>, class = void>
				struct impl {
					static constexpr UIntType set(UIntType n) noexcept {
						return impl<std::integral_constant<UIntType, shift::value * 2>>::set(UIntType(n | (n >> shift::value)));
					}
				};

				template <class dummy>
				struct impl<std::integral_constant<UIntType, sizeof(UIntType) * 8>, dummy> {
					static constexpr UIntType set(UIntType n) noexcept {
						return n;
					}
				};
			};
		}

		/// 8-bit version
		template <>
		constexpr std::uint8_t ilog2<std::uint8_t>(std::uint8_t n) noexcept {
			return detail::log2_table<std::uint8_t>::table[n];
		}

		/// 16-bit version
		template <>
		constexpr std::uint16_t ilog2<std::uint16_t>(std::uint16_t n) noexcept {
			return detail::log2_table<std::uint16_t>::table[(std::uint16_t)(
				detail::set_under_msb<std::uint16_t>::template impl<>::set(n)
				* 0xf2d) >> 12];
		}

		/// 32-bit version
		template <>
		constexpr std::uint32_t ilog2<std::uint32_t>(std::uint32_t n) noexcept {
			return detail::log2_table<std::uint32_t>::table[(std::uint32_t)(
				detail::set_under_msb<std::uint32_t>::template impl<>::set(n)
				* 0x07c4acdd) >> 27];
		}

		/// 64-bit version
		template <>
		constexpr std::uint64_t ilog2<std::uint64_t>(std::uint64_t n) noexcept {
			return detail::log2_table<std::uint64_t>::table[(std::uint64_t)(
				detail::set_under_msb<std::uint64_t>::template impl<>::set(n)
				* 0x03f6eaf2cd271461) >> 58];
		}

		/// General version (wider than 64-bit)
		template <class UIntType, std::size_t bytes>
		constexpr UIntType ilog2(UIntType const& n) noexcept {
			static_assert(bytes > 0, "Byte numbers must be positive!");
			static_assert(!std::is_signed<UIntType>::value, "Signed integers cannot be used as the arguments!");
			// If bytes is less than or equal to 8, call the 8-bit, 16-bit, 32-bit, and 64-bit versions
			// This branching is eliminated at compile-time, since bytes is a compile-time constant
			return bytes == 1 ?
				UIntType{ ilog2<std::uint8_t>(*(std::uint8_t const*)&n) } : (
				bytes == 2 ?
				UIntType{ ilog2<std::uint16_t>(*(std::uint16_t const*)&n) } : (
				bytes <= 4 ?
				UIntType{ ilog2<std::uint32_t>((*(std::uint32_t const*)&n) >> (4 - bytes)) } : (
				bytes <= 8 ?
				UIntType{ ilog2<std::uint64_t>((*(std::uint64_t const*)&n) >> (8 - bytes)) } : (
				// Otherwise, find the first 64-bits that are nonzero, and then calculate the log2 of them
				*(std::uint64_t const*)&n == 0 ?
				ilog2<UIntType, bytes - 8>(*(UIntType *)((std::uint64_t const*)&n + 1)) :
				UIntType{ ilog2<std::uint64_t>(*(std::uint64_t const*)&n) + (bytes - 8) * 8 }))));
		}
	}
	
	namespace util {
		/// Useful bitwise mask operations
		template <typename IntegralType>
		JKL_GPU_EXECUTABLE constexpr IntegralType upper_bits(IntegralType x, std::size_t n) noexcept {
			return x - (x & (((IntegralType)-1) >> n));
		}
		template <typename IntegralType>
		JKL_GPU_EXECUTABLE constexpr IntegralType lower_bits(IntegralType x, std::size_t n) noexcept {
			return x & ((1 << n) - 1);
		}
		template <typename IntegralType>	// subtract upper_bits(x, upper_bound) and lower_bits(x, lower_bound)
		JKL_GPU_EXECUTABLE constexpr IntegralType middle_bits(IntegralType x, int upper_bound, int lower_bound) noexcept {
			return x - upper_bits(x, upper_bound) - lower_bits(x, lower_bound);
		}
		// n increases from LSB to MSB
		template <typename IntegralType>
		JKL_GPU_EXECUTABLE constexpr bool bit_at(IntegralType x, std::size_t n) noexcept {
			return ((x >> n) & 1) != 0;
		}
		template <typename IntegralType>
		JKL_GPU_EXECUTABLE constexpr std::uint8_t byte_at(IntegralType x, std::size_t n) noexcept {
			return (std::uint8_t)((x >> (n * 8)) & 0xFF);
		}

		// Binary composer; return type is the smallest possible integral type that can hold the result
		// The first argument is the LSB, while the last argument is the MSB
		namespace detail {
			template <std::size_t number_of_bits, bool exact = (number_of_bits % 8 == 0)>
			struct smallest_possible_uint_t {
				static_assert(number_of_bits <= 64, "Too many arguments!");
				using type = typename smallest_possible_uint_t<number_of_bits + 1>::type;
			};
			template <>
			struct smallest_possible_uint_t<0, true> {
				using type = std::uint8_t;
			};
			template <>
			struct smallest_possible_uint_t<8, true> {
				using type = std::uint8_t;
			};
			template <>
			struct smallest_possible_uint_t<16, true> {
				using type = std::uint16_t;
			};
			template <>
			struct smallest_possible_uint_t<32, true> {
				using type = std::uint32_t;
			};
			template <>
			struct smallest_possible_uint_t<64, true> {
				using type = std::uint64_t;
			};
		}
		JKL_GPU_EXECUTABLE constexpr std::uint8_t bits() noexcept {
			return std::uint8_t{ 0 };
		}
		template <typename... Bools>
		JKL_GPU_EXECUTABLE constexpr auto bits(bool first_bit, Bools... remaining_bits) noexcept {
			using return_type = typename detail::smallest_possible_uint_t<sizeof...(remaining_bits)+1>::type;
			return return_type(return_type(first_bit) | return_type(bits((remaining_bits != 0)...) << 1));
		}

		// Calculate the number of ones
		namespace detail {
			template <std::size_t size>
			struct half_size_uint_t;
			template <>
			struct half_size_uint_t<16> {
				using type = std::uint8_t;
			};
			template <>
			struct half_size_uint_t<32> {
				using type = std::uint16_t;
			};
			template <>
			struct half_size_uint_t<64> {
				using type = std::uint32_t;
			};

			template <typename IntegralType, bool is_8_bits = sizeof(IntegralType) == 1>
			struct number_of_ones_helper {
				JKL_GPU_EXECUTABLE static constexpr std::uint8_t calculate(IntegralType x) noexcept {
					using half_size = typename half_size_uint_t<sizeof(IntegralType) * 8>::type;
					return number_of_ones_helper<half_size>::calculate(half_size(x))
						+ number_of_ones_helper<half_size>::calculate(half_size(x >> (sizeof(IntegralType) * 4)));
				}
			};
			template <typename IntegralType>
			struct number_of_ones_helper<IntegralType, true> {
				JKL_GPU_EXECUTABLE static constexpr std::uint8_t calculate(IntegralType x) noexcept {
					return std::uint8_t(bit_at(x, 0) + bit_at(x, 1) + bit_at(x, 2) + bit_at(x, 3) +
						bit_at(x, 4) + bit_at(x, 5) + bit_at(x, 6) + bit_at(x, 7));
				}
			};
		}
		template <typename IntegralType>
		JKL_GPU_EXECUTABLE constexpr std::uint8_t number_of_ones(IntegralType x) noexcept {
			return detail::number_of_ones_helper<IntegralType>::calculate(x);
		}
	}
}