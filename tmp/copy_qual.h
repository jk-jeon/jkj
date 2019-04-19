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
#include <functional>
#include <type_traits>
#include <utility>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy-and-paste the ref-qualifiers and the top level cv qualifiers from one type to another
// Ex) copy_qual_t<int const&, float> = float const&
// Ex) copy_qual_t<int const* volatile, float> = float volatile
/////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace jkj {
	namespace tmp {
		template <class From, class To>
		struct copy_const {
			using type = std::conditional_t<
				std::is_const<From>::value,
				std::add_const_t<To>, To>;
		};
		template <class From, class To>
		using copy_const_t = typename copy_const<From, To>::type;

		template <class From, class To>
		struct copy_volatile {
			using type = std::conditional_t<
				std::is_volatile<From>::value,
				std::add_volatile_t<To>, To>;
		};
		template <class From, class To>
		using copy_volatile_t = typename copy_volatile<From, To>::type;

		template <class From, class To>
		struct copy_cv {
			using type = copy_volatile_t<From, copy_const_t<From, To>>;
		};
		template <class From, class To>
		using copy_cv_t = typename copy_cv<From, To>::type;

		template <class From, class To>
		struct copy_ref {
			using type = std::conditional_t<
				std::is_lvalue_reference<From>::value,
				std::add_lvalue_reference_t<To>,
				std::conditional_t<
				std::is_rvalue_reference<From>::value,
				std::add_rvalue_reference_t<To>,
				To>>;
		};
		template <class From, class To>
		using copy_ref_t = typename copy_ref<From, To>::type;

		template <class From, class To>
		struct copy_qual {
			using type = copy_ref_t<From,
				copy_cv_t<std::remove_reference_t<From>, To>>;
		};
		template <class From, class To>
		using copy_qual_t = typename copy_qual<From, To>::type;
	}
}