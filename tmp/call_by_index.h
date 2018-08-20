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
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include "../portability.h"

namespace jkl {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Runtime selection of functions with integer template parameter (from 0 to N)
		/// Build a constexpr table of function pointers, and choose one of them according to a 
		/// index variable given in runtime.
		/// For example, what we want to achieve is the following: we have a functor
		///
		/// struct functor {
		///   template <std::size_t index>
		///   void operator()() { // do something with index }
		/// };
		/// then the code 
		///
		/// call_by_index<10>(functor{}, i);
		///
		/// will effectively invoke the entry table[i] in the array
		///
		/// table[10] = { &functor::operator()<0>, ... , &functor::operator()<9> };
		///
		/// of member function pointers that is statically built.
		/// Note that all template specializations of operator() must have the same signature.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail {
			template <class Functor, class... Args, std::size_t... indices>
			auto call_by_index(Functor&& functor, std::size_t index, std::index_sequence<indices...>, Args&&... args) {
				static constexpr auto temp = &std::decay_t<Functor>::TEMPLATE_FUNCTION_CALL_OPERATOR<0>;
				static constexpr decltype(temp) table[] = { &std::decay_t<Functor>::TEMPLATE_FUNCTION_CALL_OPERATOR<indices>... };
				return (functor.*table[index])(std::forward<Args>(args)...);
			}

			template <std::size_t N, class Functor, class... Args>
			struct call_by_index_noexcept {
				static constexpr bool value = call_by_index_noexcept<N - 1, Functor, Args...>::value
					&& noexcept(std::declval<Functor>().TEMPLATE_FUNCTION_CALL_OPERATOR<N - 1>(std::declval<Args>()...));
			};

			template <class Functor, class... Args>
			struct call_by_index_noexcept<0, Functor, Args...> {
				static constexpr bool value = true;
			};
		}
		template <std::size_t N, class Functor, class... Args>
		FORCEINLINE auto call_by_index(Functor&& functor, std::size_t index, Args&&... args)
			noexcept(detail::call_by_index_noexcept<N, Functor, Args...>::value)
		{
			assert(index < N);
			return detail::call_by_index<Functor, Args...>(
				std::forward<Functor>(functor), index, std::make_index_sequence<N>{}, std::forward<Args>(args)...);
		}
	}
}