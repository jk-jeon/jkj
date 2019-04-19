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
#include <iterator>
#include "../../numerical_lie_group/Rn_elmt.h"

namespace jkj {
	namespace gui {
		// Iterate over a 2D region in row-major order.
		// For given y, it is assumed that the corresponding row
		// can be written as an interval of the form [x_min, x_max),
		// thus giving the name row_connected_region_iterator.
		// Hence, for example, a region that is not simply-connected cannot be
		// iterated using this iterator alone.
		template <class XRangeCalculator>
		class row_connected_region_iterator : private XRangeCalculator {
			math::R2_elmt<int> current_pos;
			int current_x_max;
			int y_max;

			decltype(auto) x_range(int y) {
				return static_cast<XRangeCalculator&>(*this)(y);
			}
			decltype(auto) x_range(int y) const {
				return static_cast<XRangeCalculator const&>(*this)(y);
			}

		public:
			using iterator_category = std::input_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = math::R2_elmt<int>;
			using pointer = math::R2_elmt<int> const*;
			using reference = math::R2_elmt<int> const&;

			row_connected_region_iterator() = default;

			template <class... Args>
			row_connected_region_iterator(int y, int y_max, Args&&... args) :
				XRangeCalculator(std::forward<Args>(args)...), y_max{ y_max }
			{
				assert(y <= y_max);
				current_pos.y() = y;
				while( current_pos.y() < y_max ) {
					std::tie(current_pos.x(), current_x_max) = x_range(current_pos.y());
					if( current_pos.x() == current_x_max )
						++current_pos.y();
					else
						break;
				}
			}

			// Dereference
			reference operator*() const noexcept {
				return current_pos;
			}

			// Member access
			pointer operator->() const noexcept {
				return &current_pos;
			}

			// Increment
			row_connected_region_iterator& operator++() {
				if( ++current_pos.x() == current_x_max ) {
					while( current_pos.y() < y_max ) {
						std::tie(current_pos.x(), current_x_max) = x_range(++current_pos.y());
						if( current_pos.x() < current_x_max )
							break;
					}
				}
				return *this;
			}
			row_connected_region_iterator operator++(int) {
				auto prev = *this;
				++*this;
				return prev;
			}

			// Relations
			bool operator==(row_connected_region_iterator const& itr) const noexcept {
				assert(y_max == itr.y_max);
				if( current_pos.y() == y_max )
					return itr.current_pos.y() == y_max;
				return current_pos == itr.current_pos;
			}
			bool operator!=(row_connected_region_iterator const& itr) const noexcept {
				return !(*this == itr);
			}
		};

		template <class XRangeCalculator>
		std::pair<row_connected_region_iterator<XRangeCalculator>, row_connected_region_iterator<XRangeCalculator>>
			make_row_connected_region_range(int y_min, int y_max, XRangeCalculator&& c)
		{
			return{ { y_min, y_max, c }, { y_max, y_max, c } };
		}

		// Iterate over a rectangular region
		class rectangular_region_iterator {
			math::R2_elmt<int>		current_pos;
			math::R2_elmt<int>		top_left;
			math::R2_elmt<int>		bottom_right;

		public:
			using iterator_category = std::input_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = math::R2_elmt<int>;
			using pointer = math::R2_elmt<int> const*;
			using reference = math::R2_elmt<int> const&;

			rectangular_region_iterator() = default;
			rectangular_region_iterator(int y,
				math::R2_elmt<int> const& top_left,
				math::R2_elmt<unsigned int> const& size) :
				top_left{ top_left }, bottom_right{ decltype(bottom_right)(top_left + size) }
			{
				assert(y >= top_left.y() && y <= bottom_right.y());
				current_pos = { top_left.x(), y };
				if( top_left.x() == bottom_right.x() ) {
					current_pos.y() = bottom_right.y();
				}
			}

			// Dereference
			reference operator*() const noexcept {
				return current_pos;
			}

			// Member access
			pointer operator->() const noexcept {
				return &current_pos;
			}

			// Increment
			rectangular_region_iterator& operator++() {
				if( ++current_pos.x() == bottom_right.x() ) {
					++current_pos.y();
					current_pos.x() = top_left.x();
				}
				return *this;
			}
			rectangular_region_iterator operator++(int) {
				auto prev = *this;
				++*this;
				return prev;
			}

			// Relations
			bool operator==(rectangular_region_iterator const& itr) const noexcept {
				assert(top_left == itr.top_left);
				assert(bottom_right == itr.bottom_right);
				return current_pos == itr.current_pos;
			}
			bool operator!=(rectangular_region_iterator const& itr) const noexcept {
				return !(*this == itr);
			}
		};

		std::pair<rectangular_region_iterator, rectangular_region_iterator>
			make_rectangular_region_range(
				math::R2_elmt<int> const& top_left,
				math::R2_elmt<unsigned int> const& size)
		{
			return{
				{ top_left.y(), top_left, size },
			{ int(top_left.y() + size.y()), top_left, size }
			};
		}
	}
}