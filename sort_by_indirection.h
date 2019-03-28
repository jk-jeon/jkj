/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2019 Junekey Jeon                                                                   ///
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
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <functional>
#include <vector>

namespace jkl {
	// When sorting a vector of items, std::sort somewhat performs very poorly
	// if move or swap is slow. In this case, it is often much faster to perform
	// sorting on a vector of pointers, and then reflect the result back to the original vector.
	// For array of typical struct with about thirty std::string members and more,
	// sort_by_indirection_t's performance is about 10x faster than that of std::sort,
	// when the length is about 1300.
	// This feature was suggested by Jae-Hun Kim.
	// [CAUTION] There is no exception guarantee, if the comparator and/or
	// move constructor/assignment might throw.
	template <class RandomAccessIterator, class Allocator = std::allocator<RandomAccessIterator>>
	class sort_by_indirection_t {
		std::vector<RandomAccessIterator, Allocator>	buffer;

		// For generation of range of RandomAccessIterator's
		struct iterator_returning_iterator {
			RandomAccessIterator	current_itr;

			// Strictly speaking, this iterator is not a RandomAccessIterator, because
			// the multipass guarantee does not hold (thus is not a ForwardIterator).
			// In fact, any kind of "generating" iterator can never satisfy the multipass guarantee.
			// This restriction really sucks.
			using iterator_category = std::random_access_iterator_tag;
			using value_type = RandomAccessIterator;
			using reference = RandomAccessIterator;
			using pointer = RandomAccessIterator const*;
			using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;

			/* Dereference */

			reference operator*() const {
				return current_itr;
			}
			pointer operator->() const {
				return &current_itr;
			}
			reference operator[](difference_type n) const {
				return current_itr + n;
			}

			/* Forward movement */

			iterator_returning_iterator& operator++() {
				++current_itr;
				return *this;
			}
			iterator_returning_iterator operator++(int) {
				return{ current_itr++ };
			}
			iterator_returning_iterator& operator+=(difference_type n) {
				current_itr += n;
				return *this;
			}
			iterator_returning_iterator operator+(difference_type n) const {
				return{ current_itr + n };
			}
			friend iterator_returning_iterator operator+(difference_type n,
				iterator_returning_iterator const& itr)
			{
				return itr + n;
			}

			/* Backward movement */

			iterator_returning_iterator& operator--() {
				--current_itr;
				return *this;
			}
			iterator_returning_iterator operator--(int) {
				return{ current_itr-- };
			}
			iterator_returning_iterator& operator-=(difference_type n) {
				current_itr -= n;
				return *this;
			}
			iterator_returning_iterator operator-(difference_type n) const {
				return{ current_itr - n };
			}

			/* Distance */

			difference_type operator-(iterator_returning_iterator const& itr) const {
				return current_itr - itr.current_itr;
			}

			/* Relation */

			bool operator==(iterator_returning_iterator const& itr) const {
				return current_itr == itr.current_itr;
			}
			bool operator!=(iterator_returning_iterator const& itr) const {
				return current_itr != itr.current_itr;
			}
			bool operator<(iterator_returning_iterator const& itr) const {
				return current_itr < itr.current_itr;
			}
			bool operator<=(iterator_returning_iterator const& itr) const {
				return current_itr <= itr.current_itr;
			}
			bool operator>(iterator_returning_iterator const& itr) const {
				return current_itr > itr.current_itr;
			}
			bool operator>=(iterator_returning_iterator const& itr) const {
				return current_itr >= itr.current_itr;
			}
		};

		using iterator_value_type = typename
			std::iterator_traits<RandomAccessIterator>::value_type;

	public:
		// This function is not const (thus is not thread-safe),
		// because it modifies the internal buffer.
		// To achieve thread-safe behavior without locking mechanisms,
		// each thread might hold an independent instance of sort_by_indirection_t.
		// A quick (and dirty) solution to achieve that is to declare an instance
		// with a thread_local storage duration.
		template <class Comparator = std::less<iterator_value_type>>
		void operator()(RandomAccessIterator first, RandomAccessIterator last,
			Comparator&& c = {})
		{
			// Prepare indirection buffer
			buffer.assign(iterator_returning_iterator{ first },
				iterator_returning_iterator{ last });

			// Sort indirection buffer
			std::sort(buffer.begin(), buffer.end(),
				[&c](auto const& itr1, auto const& itr2) {
				return c(*itr1, *itr2);
			});

			// Rearrange the original range
			for( std::size_t cycle_start_pos = 0; cycle_start_pos < buffer.size(); ++cycle_start_pos ) {
				// Perform cyclic swap
				// The value `last` indicates that the position has already been visited.
				if( buffer[cycle_start_pos] != last ) {
					auto dst_itr = buffer[cycle_start_pos];
					auto src_idx = std::size_t(dst_itr - first);

					// Avoid useless self-swap for length 1-cycle
					if( src_idx != cycle_start_pos ) {
						auto tmp = std::move(*dst_itr);

						do {
							// Do move and update destination
							*dst_itr = std::move(*buffer[src_idx]);
							dst_itr = buffer[src_idx];

							// Indicate move and update source
							buffer[src_idx] = last;
							src_idx = std::size_t(dst_itr - first);
						} while( src_idx != cycle_start_pos );

						*dst_itr = std::move(tmp);
					}
				}
			}
		}

		void release_buffer() noexcept {
			buffer.clear();
			buffer.shrink_to_fit();
		}
	};
}
