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
#include <cstddef>

namespace jkj {
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Compile-time string
	/// This class became obsolete as std::string_view is now a standard.
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename CharType>
	constexpr std::size_t constexpr_strlen(CharType const* str) noexcept {
		return str[0] == CharType('\0') ? 0 : constexpr_strlen(str + 1);
	}

	template <typename CharType>
	class basic_constexpr_str {
		CharType const* const	str_ = nullptr;
		std::size_t				size_ = 0;

	public:
		// The trailing '\0' will not be treated as a part of the string
		template<std::size_t N>
		constexpr basic_constexpr_str(CharType const(&str)[N]) noexcept
			: str_(str), size_(std[N - 1] == CharType('\0') ? N - 1 : N)
		{
			static_assert(N >= 1, "not a string literal");
		}

		constexpr basic_constexpr_str(CharType const* str, std::size_t size) noexcept
			: str_(str), size_(size) {}

		/// Utilities

		constexpr CharType operator[](std::size_t n) const noexcept {
			return str_[n];
		}
		constexpr operator CharType const*() const noexcept {
			return str_;
		}
		constexpr std::size_t size() const noexcept {
			return size_;
		}

		/// Relational operators

		constexpr bool operator==(basic_constexpr_str const& that) const noexcept {
			return size() == 0 ? (that.size() == 0 ? true : false) :
				(that.size() == 0 ? false :
					basic_constexpr_str(str_ + 1, size() - 1) == basic_constexpr_str(that.str_ + 1, that.size() - 1));
		}
		constexpr bool operator!=(basic_constexpr_str const& that) const noexcept {
			return !(*this == that);
		}
		constexpr bool operator<=(basic_constexpr_str const& that) const noexcept {
			return size() == 0 ? true :
				(that.size() == 0 ? false :
				(str_[0] == that[0] ?
					basic_constexpr_str(str_ + 1, size() - 1) <= basic_constexpr_str(that.str_ + 1, that.size() - 1) :
					(str_[0] < that[0] ? true : false)));
		}
		constexpr bool operator<(basic_constexpr_str const& that) const noexcept {
			return size() == 0 ? (that.size() == 0 ? false : true) :
				(that.size() == 0 ? false :
				(str_[0] == that[0] ?
					basic_constexpr_str(str_ + 1, size() - 1) < basic_constexpr_str(that.str_ + 1, that.size() - 1) :
					(str_[0] < that[0] ? true : false)));
		}
		constexpr bool operator>=(basic_constexpr_str const& that) const noexcept {
			return that <= *this;
		}
		constexpr bool operator>(basic_constexpr_str const& that) const noexcept {
			return that < *this;
		}
	};

	template <typename CharType, size_t N>
	constexpr auto make_constexpr_str(CharType const(&str)[N]) noexcept {
		return basic_constexpr_str<CharType>{ str };
	}

	using constexpr_str = basic_constexpr_str<char>;
	using constexpr_wstr = basic_constexpr_str<wchar_t>;
	using constexpr_u16str = basic_constexpr_str<char16_t>;
	using constexpr_u32str = basic_constexpr_str<char32_t>;
}