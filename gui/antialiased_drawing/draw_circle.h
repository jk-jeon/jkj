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
#include <algorithm>
#include <bitset>
#include <cmath>
#include "draw.h"
#include "region_iterator.h"

namespace jkl {
	namespace gui {
		namespace draw_detail {
			// For radius <= 1
			template <class ValueType, class PixelBuffer>
			void draw_small_circle_impl(PixelBuffer& g,
				math::R2_elmt<ValueType> const& center, ValueType radius,
				nana::color const& color)
			{
				using std::floor;
				using std::sqrt;
				using std::acos;

				// First, find the pixel the center belongs to
				auto center_pixel_x = int(floor(center.x()));
				auto center_pixel_y = int(floor(center.y()));

				// Next, determine whether or not each four endpoints of that pixel belongs to the circle
				auto is_in_circle = [&center, radius_sq = radius * radius]
				(math::R2_elmt<ValueType> const& p)
				{
					return normsq(p - center) < radius_sq;
				};

				std::bitset<4> sign_flags;
				sign_flags[0] = is_in_circle({ ValueType(center_pixel_x + 1), ValueType(center_pixel_y + 1) });
				sign_flags[1] = is_in_circle({ ValueType(center_pixel_x), ValueType(center_pixel_y + 1) });
				sign_flags[2] = is_in_circle({ ValueType(center_pixel_x), ValueType(center_pixel_y) });
				sign_flags[3] = is_in_circle({ ValueType(center_pixel_x + 1), ValueType(center_pixel_y) });

				// Not like general drawing using SDF, distance to the corners are not so useful.
				// Instead, the following values are used for calculating intersection ratios.
				ValueType edge_margins[] = {
					ValueType(center.x()) + radius - ValueType(center_pixel_x + 1),
					ValueType(center.y()) + radius - ValueType(center_pixel_y + 1),
					ValueType(center_pixel_x) + radius - ValueType(center.x()),
					ValueType(center_pixel_y) + radius - ValueType(center.y())
				};

				auto calculate_edge_segment = [radius](ValueType edge_margin)
				{
					return acos((radius - edge_margin) / radius) * radius * radius -
						(radius - edge_margin) * sqrt(2 * radius * edge_margin - edge_margin * edge_margin);
				};
				auto calculate_corner_segment = [&calculate_edge_segment, radius]
				(ValueType edge_margin1, ValueType edge_margin2, ValueType edge_segment1, ValueType edge_segment2)
				{
					return (edge_segment1 + edge_segment2) / 2 +
						(radius - edge_margin1) * (radius - edge_margin2) -
						math::constants<ValueType>::pi / 4 * (radius * radius);
				};

				ValueType ir_center = math::constants<ValueType>::pi * radius * radius;
				// xp-yp-xm-ym
				ValueType ir_edges[] = { 0, 0, 0, 0 };
				// xpyp-xmyp-xmym-xpym
				ValueType ir_corners[] = { 0, 0, 0, 0 };

				// When only one corner is in the circle
				auto deal_with_one_corner_case = [&](auto c) {
					auto e1 = c;
					auto e2 = (c + 1) % 4;

					auto s1 = calculate_edge_segment(edge_margins[e1]);
					auto s2 = calculate_edge_segment(edge_margins[e2]);
					ir_corners[c] = calculate_corner_segment(edge_margins[e1], edge_margins[e2], s1, s2);
					ir_edges[e1] = s1 - ir_corners[c];
					ir_edges[e2] = s2 - ir_corners[c];

					auto e3 = (c + 2) % 4;
					auto e4 = (c + 3) % 4;
					if( edge_margins[e3] > 0 )
						ir_edges[e3] = calculate_edge_segment(edge_margins[e3]);
					if( edge_margins[e4] > 0 )
						ir_edges[e4] = calculate_edge_segment(edge_margins[e4]);

					ir_center -= (s1 + s2 - ir_corners[c] + ir_edges[e3] + ir_edges[e4]);
				};
				// When two consecutive corners are in the circle
				auto deal_with_two_corner_case = [&](auto e) {
					auto c1 = (e + 3) % 4;
					auto c2 = e;
					auto e1 = (e + 3) % 4;
					auto e2 = (e + 1) % 4;

					auto s = calculate_edge_segment(edge_margins[e]);
					auto s1 = calculate_edge_segment(edge_margins[e1]);
					auto s2 = calculate_edge_segment(edge_margins[e2]);

					ir_corners[c1] = calculate_corner_segment(edge_margins[e1], edge_margins[e], s1, s);
					ir_corners[c2] = calculate_corner_segment(edge_margins[e], edge_margins[e2], s, s2);
					ir_edges[e] = s - ir_corners[c1] - ir_corners[c2];
					ir_edges[e1] = s1 - ir_corners[c1];
					ir_edges[e2] = s2 - ir_corners[c2];

					auto e_opposite = (e + 2) % 4;
					if( edge_margins[e_opposite] > 0 )
						ir_edges[e_opposite] = calculate_edge_segment(edge_margins[e_opposite]);

					ir_center -= (s1 + s2 + ir_edges[e] + ir_edges[e_opposite]);
				};
				// When three corners are in the circle
				auto deal_with_three_corner_case = [&](auto c_inside) {
					auto c1 = (c_inside + 3) % 4;
					auto c2 = (c_inside + 1) % 4;
					auto e1 = (c_inside + 3) % 4;
					auto e2 = (c_inside + 2) % 4;
					auto e01 = c_inside;
					auto e02 = (c_inside + 1) % 4;

					auto s1 = calculate_edge_segment(edge_margins[e1]);
					auto s2 = calculate_edge_segment(edge_margins[e2]);
					auto s01 = calculate_edge_segment(edge_margins[e01]);
					auto s02 = calculate_edge_segment(edge_margins[e02]);

					ir_corners[c_inside] = calculate_corner_segment(edge_margins[e01], edge_margins[e02], s01, s02);
					ir_corners[c1] = calculate_corner_segment(edge_margins[e1], edge_margins[e01], s1, s01);
					ir_corners[c2] = calculate_corner_segment(edge_margins[e2], edge_margins[e02], s2, s02);
					ir_edges[e1] = s1 - ir_corners[c1];
					ir_edges[e2] = s2 - ir_corners[c2];
					ir_edges[e01] = s01 - ir_corners[c1] - ir_corners[c_inside];
					ir_edges[e02] = s02 - ir_corners[c2] - ir_corners[c_inside];
					ir_center -= (s1 + s2 + ir_edges[e01] + ir_edges[e02] + ir_corners[c_inside]);
				};

				switch( sign_flags.to_ulong() ) {
				case 0:
					for( auto q = 0; q < 4; ++q ) {
						if( edge_margins[q] > 0 ) {
							ir_edges[q] = calculate_edge_segment(edge_margins[q]);
							ir_center -= ir_edges[q];
						}
					}
					break;


				case 1:
					deal_with_one_corner_case(0);
					break;
				case 2:
					deal_with_one_corner_case(1);
					break;
				case 4:
					deal_with_one_corner_case(2);
					break;
				case 8:
					deal_with_one_corner_case(3);
					break;


				case 9:
					deal_with_two_corner_case(0);
					break;
				case 3:
					deal_with_two_corner_case(1);
					break;
				case 6:
					deal_with_two_corner_case(2);
					break;
				case 12:
					deal_with_two_corner_case(3);
					break;


				case 11:
					deal_with_three_corner_case(0);
					break;
				case 7:
					deal_with_three_corner_case(1);
					break;
				case 14:
					deal_with_three_corner_case(2);
					break;
				case 13:
					deal_with_three_corner_case(3);
					break;


				case 15:
					ir_center = 1;
					for( auto e = 0; e < 4; ++e )
						ir_edges[e] = calculate_edge_segment(edge_margins[e]);
					for( auto c = 0; c < 4; ++c ) {
						auto e1 = c;
						auto e2 = (c + 1) % 4;
						ir_corners[c] = calculate_corner_segment(edge_margins[e1], edge_margins[e2],
							ir_edges[e1], ir_edges[e2]);
					}
					for( auto e = 0; e < 4; ++e ) {
						auto c1 = (e + 3) % 4;
						auto c2 = e;
						ir_edges[e] -= (ir_corners[c1] + ir_corners[c2]);
					}
					break;


				default:
					// Cannot reach here
					assert(false);
				}

				assert(ir_center >= 0 && ir_center <= 1);
				for( auto q = 0; q < 4; ++q ) {
					assert(ir_edges[q] >= 0 && ir_edges[q] <= 1);
					assert(ir_corners[q] >= 0 && ir_corners[q] <= 1);
				}

				// Write
				auto set_pixel = [&g, &color](math::R2_elmt<int> const& p, ValueType intersection_ratio) {
					if( !(p.x() >= 0 && p.x() < int(g.size().width) &&
						p.y() >= 0 && p.y() < int(g.size().height)) )
						return;

					g[p.y()][p.x()] = color_blender{ color }(g, p, intersection_ratio, ValueType(0));
				};

				set_pixel({ center_pixel_x, center_pixel_y }, ir_center);
				set_pixel({ center_pixel_x + 1, center_pixel_y }, ir_edges[0]);
				set_pixel({ center_pixel_x, center_pixel_y + 1 }, ir_edges[1]);
				set_pixel({ center_pixel_x - 1, center_pixel_y }, ir_edges[2]);
				set_pixel({ center_pixel_x, center_pixel_y - 1 }, ir_edges[3]);
				set_pixel({ center_pixel_x + 1, center_pixel_y + 1 }, ir_corners[0]);
				set_pixel({ center_pixel_x - 1, center_pixel_y + 1 }, ir_corners[1]);
				set_pixel({ center_pixel_x - 1, center_pixel_y - 1 }, ir_corners[2]);
				set_pixel({ center_pixel_x + 1, center_pixel_y - 1 }, ir_corners[3]);
			}

			// For radius > 1
			template <class ValueType, class PixelBuffer>
			void draw_large_circle_impl(PixelBuffer& g,
				math::R2_elmt<ValueType> const& center, ValueType radius, ValueType border_width,
				nana::color const& color, nana::color const& border_color)
			{
				using std::floor;
				using std::ceil;
				using std::max;
				using std::min;
				using std::sqrt;

				auto augmented_radius = radius + border_width / 2;

				auto y_min = max(int(floor(center.y() - augmented_radius)), 0);
				auto y_max = min(int(ceil(center.y() + augmented_radius)), int(g.size().height));

				if( y_min >= y_max )
					return;

				auto range = make_row_connected_region_range(y_min, y_max,
					[&center, r_sq = augmented_radius * augmented_radius, w = int(g.size().width)](int y)
				{
					auto dy = ValueType(y) - ValueType(center.y());
					if( y < center.y() )
						dy += ValueType(1);
					auto dx = sqrt(r_sq - dy * dy);

					return std::make_pair(
						min(max(int(floor(ValueType(center.x()) - dx)), 0), w),
						min(int(ceil(ValueType(center.x()) + dx)), w));
				});

				auto sdf = [&center, radius](math::R2_elmt<ValueType> const& p) {
					return ValueType(norm(p - center) - radius);
				};

				draw<ValueType>(g, range.first, range.second, sdf, border_width, color, border_color);
			}

			template <class ValueType, class PixelBuffer>
			void draw_circle_impl(PixelBuffer& g,
				math::R2_elmt<ValueType> const& center, ValueType radius, nana::color const& color)
			{
				if( radius <= 1 ) {
					draw_small_circle_impl<ValueType>(g, center, radius, color);
				}
				else {
					draw_large_circle_impl<ValueType>(g, center, radius, ValueType(0), color, color);
				}
			}

			template <class ValueType, class PixelBuffer>
			void draw_circle_impl(PixelBuffer& g,
				math::R2_elmt<ValueType> const& center, ValueType radius, ValueType border_width,
				nana::color const& color, nana::color const& border_color)
			{
				auto augmented_radius = radius + border_width / 2;
				auto diminished_radius = radius - border_width / 2;

				if( augmented_radius <= 1 ) {
					draw_small_circle_impl<ValueType>(g, center, augmented_radius, border_color);
					if( diminished_radius >= 0 )
						draw_small_circle_impl<ValueType>(g, center, diminished_radius, color);
				}
				else if( radius <= 1 ) {
					draw_large_circle_impl<ValueType>(g, center, augmented_radius, 0, border_color, border_color);
					if( diminished_radius >= 0 )
						draw_small_circle_impl<ValueType>(g, center, diminished_radius, color);
				}
				else {
					draw_large_circle_impl<ValueType>(g, center, radius, border_width, color, border_color);
				}
			}
		}

		template <class ValueType = double, class PixelBuffer, class Point, class Radius>
		void draw_circle(PixelBuffer& g, Point const& center, Radius radius, nana::color const& color)
		{
			assert(radius > 0);
			draw_detail::draw_circle_impl<ValueType>(g, center, ValueType(radius), color);
		}

		template <class ValueType = double, class PixelBuffer,
			class Integer, class Radius, class = std::enable_if_t<std::is_integral_v<Integer>>>
		void draw_circle(PixelBuffer& g, math::R2_elmt<Integer> const& center,
			Radius radius, nana::color const& color)
		{
			draw_circle<ValueType>(g,
				math::R2_elmt<ValueType>{ ValueType(center.x()) + ValueType(0.5),
				ValueType(center.y()) + ValueType(0.5) },
				radius, color);
		}

		template <class ValueType = double, class PixelBuffer, class Point, class Radius, class BorderWidth>
		void draw_circle(PixelBuffer& g, Point const& center,
			Radius radius, BorderWidth border_width, nana::color const& color, nana::color const& border_color)
		{
			assert(radius >= 0);
			assert(border_width >= 0);
			draw_detail::draw_circle_impl<ValueType>(g, center,
				ValueType(radius), ValueType(border_width), color, border_color);
		}

		template <class ValueType = double, class PixelBuffer,
			class Integer, class Radius, class BorderWidth, class = std::enable_if_t<std::is_integral_v<Integer>>>
		void draw_circle(PixelBuffer& g, math::R2_elmt<Integer> const& center,
			Radius radius, BorderWidth border_width, nana::color const& color, nana::color const& border_color)
		{
			draw_circle<ValueType>(g,
				math::R2_elmt<ValueType>{ ValueType(center.x()) + ValueType(0.5), ValueType(center.y()) + ValueType(0.5) },
				radius, border_width, color, border_color);
		}
	}
}