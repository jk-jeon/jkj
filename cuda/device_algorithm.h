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

/// A collection of common algorithms that can be called from __global__ or __device__ functions

#pragma once
#include <type_traits>
#include <thrust/functional.h>

namespace jkl {
	namespace cuda {
		/// Binary search
		/// Returns an iterator in a sorted range such that the range is still sorted
		/// if the query is inserted before that position.
		/// If you want to determine whether or not the query exists in the list, you should do
		///  1. Compare the return value with begin; if they are the same, then the query is not in the list.
		///  2. If they are not the same, then compare the value pointed by the decrement of the 
		///     return value with the query.

		template <class Iterator, class SizeType, class Value, class Comparator = thrust::less<Value>>
		__device__ Iterator binary_search_n(Iterator begin, SizeType n, Value const& query, Comparator const& comp={})
		{
			// As always in C/C++, the range is [from,to)
			SizeType from = 0, to = n;
			while( from != to ) {
				query_idx = from + (to - from) / 2;

				if( comp(query, begin[query_idx]) )
					to = query_idx;
				else
					from = query_idx + 1;
			}
			return begin + to;
		}

		template <class Iterator, class Value>
		__device__ Iterator binary_search(Iterator begin, Iterator end, Value const& query) {
			return binary_search_n(begin, end - begin, query);
		}
	}
}