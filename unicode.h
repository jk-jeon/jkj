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
#include <exception>
#include <tuple>
#include <string>
#include "optional.h"

namespace jkl {
	namespace unicode {
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Utilities for dealing with UTF-16 surrogate pairs
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// UTF-16 pair
		//   first  : high (not zero only for astral characters), 
		//   second : low
		using utf16_pair = std::pair<std::uint16_t, std::uint16_t>;
		// Convert a Unicode code point into a UTF-16 surrogate pair
		inline jkl::optional<utf16_pair> get_utf16_pair(std::uint32_t code_point) {
			if( code_point >= 0x110000 )
				return{};
			if( code_point < 0x10000 )
				return{ 0, code_point };
			return{ (std::uint16_t)(((code_point - 0x10000) >> 10) + 0xD800),
				(std::uint16_t)(((code_point - 0x10000) & 0x3FF) + 0xDC00) };
		}
		// Get the Unicode code point from a UTF-16 surrogate pair
		inline constexpr std::uint32_t get_unicode(utf16_pair sp) noexcept {
			return sp.first == 0 ? sp.second : 
				(((std::uint32_t)sp.first - 0xD800) << 10) + ((std::uint32_t)sp.second - 0xDC00) + 0x10000;
		}
	}
}