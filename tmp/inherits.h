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
#include "typelist.h"

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// A proxy class inheriting from a list of base classes;
		// pre-eliminate multiple instances of the same class in the list.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <class Typelist>
			struct inherits_without_multiplicity;

			template <class... Bases>
			struct inherits_without_multiplicity<typelist<Bases...>>
				: Bases...
			{
				template <std::size_t I>
				using base_type = typename typelist<Bases...>::template type<I>;
			};
		}

		template <class... Bases>
		using inherits_without_multiplicity =
			detail::inherits_without_multiplicity<remove_multiplicity_t<typelist<Bases...>>>;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// A proxy class inheriting from a list of base classes;
		// if there are multiple instances of a same class in the list,
		// inherit multiple time from that class.
		// TBD later...
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <class... Bases>
		struct inherits_with_multiplicity;
	}
}