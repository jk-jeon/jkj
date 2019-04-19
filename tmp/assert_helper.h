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
#include <type_traits>

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Assert helper
		// There may be a template types such that any attempt to instantiate those types are 
		// certainly an error. In those cases, one may want to use static_assert to report an error message, 
		// but static_assert(false, ...) always produces an error even if the type is never instantiated.
		// In that case, use assert_helper.
		//
		// [Usage example]
		// template <int N, class dummy = void>
		// struct a_type {};
		// template <class dummy>
		// struct a_type<-1, dummy> { // a_type must not be instantitated with N = -1
		//   static_assert(assert_helper<a_type<-1, dummy>>::value, "N should not be negative!");
		// };
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename... T>
		struct assert_helper : std::false_type {};
	}
}