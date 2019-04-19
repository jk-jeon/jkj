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
#include "draw_circle.h"

namespace jkj {
	namespace gui {
		namespace draw_detail {
			// For thin rectangle drawing
			template <class ValueType>
			struct strip_ray {
				math::R2_elmt<ValueType>	center;
				math::R2_elmt<ValueType>	normalized_direction_vector;
				ValueType						half_width;

				bool contains(math::R2_elmt<ValueType> const& p) const {
					auto v = p - center;
					if( dot(v, normalized_direction_vector) < 0 )
						return false;
					else
						return abs(signed_area(v, normalized_direction_vector)) <= half_width;
				}

				ValueType calculate_intersection_ratio(math::R2_elmt<int> const& pixel_coord) const {
					// Since the intersection is convex, we only need to know its corner positions
					math::R2_elmt<ValueType> intersection_corners[7];
					unsigned int corner_counter = 0;

					// Square corners
					math::R2_elmt<ValueType> square_corners[] = {
						{ ValueType(pixel_coord.x() + 1), ValueType(pixel_coord.y() + 1) },
					{ ValueType(pixel_coord.x()), ValueType(pixel_coord.y() + 1) },
					{ ValueType(pixel_coord.x()), ValueType(pixel_coord.y()) },
					{ ValueType(pixel_coord.x() + 1), ValueType(pixel_coord.y()) }
					};
					for( auto q = 0; q < 4; ++q ) {
						if( contains(square_corners[q]) )
							intersection_corners[corner_counter++] = square_corners[q];
					}

					// Strip corners
					math::R2_elmt<ValueType> strip_corners[2] = {
						{ center.x() - half_width * normalized_direction_vector.y(),
						center.y() + half_width * normalized_direction_vector.x() },
					{ center.x() + half_width * normalized_direction_vector.y(),
					center.y() - half_width * normalized_direction_vector.x() }
					};
					bool strip_corner_inside_flags[2] = { false, false };
					for( auto q = 0; q < 2; ++q ) {
						if( is_in_pixel(strip_corners[q], pixel_coord) ) {
							intersection_corners[corner_counter++] = strip_corners[q];
							strip_corner_inside_flags[q] = true;
						}
					}

					// Calculate crossing points between a ray and four edges when the ray center is outside the square
					auto calculate_crossing_pt_outside = [&](auto const& center, auto const& ray) {
						ValueType sdf_values[4];
						std::bitset<4> sign_flags;
						for( auto q = 0; q < 4; ++q ) {
							sdf_values[q] = signed_area(square_corners[q] - center, ray);
							sign_flags[q] = sdf_values[q] > ValueType(0);
						}

						// If signs differ at consecutive corners, use linear interpolation to find the position of zero
						for( auto q = 0; q < 4; ++q ) {
							auto next_q = (q + 1) % 4;
							if( sign_flags[q] != sign_flags[next_q] ) {
								auto value_diff = sdf_values[q] - sdf_values[next_q];

								auto zero = (sdf_values[q] * square_corners[next_q] - sdf_values[next_q] * square_corners[q])
									/ value_diff;

								// Is this zero not in the backside?
								if( dot(ray, zero - center) >= 0 )
									intersection_corners[corner_counter++] = zero;
							}
						}
					};

					// When both corners are inside the square
					if( strip_corner_inside_flags[0] && strip_corner_inside_flags[1] ) {
						for( auto q = 0; q < 2; ++q ) {
							intersection_corners[corner_counter++] =
								calculate_crossing_pt_inside(strip_corners[q], normalized_direction_vector, pixel_coord);
						}
					}
					else {
						// When only one corner is inside the square
						auto only_one_strip_corner = [&](auto q) {
							intersection_corners[corner_counter++] =
								calculate_crossing_pt_inside(strip_corners[q], normalized_direction_vector, pixel_coord);

							// The line joining two corners should cross a square edge
							auto const& the_other_corner = strip_corners[(q + 1) % 2];
							intersection_corners[corner_counter++] =
								calculate_crossing_pt_inside(strip_corners[q], the_other_corner - strip_corners[q], pixel_coord);

							// For the corner outside the square
							calculate_crossing_pt_outside(the_other_corner, normalized_direction_vector);
						};

						if( strip_corner_inside_flags[0] )
							only_one_strip_corner(0);
						else if( strip_corner_inside_flags[1] )
							only_one_strip_corner(1);

						// When no corner is inside the square
						else {
							calculate_crossing_pt_outside(strip_corners[0], normalized_direction_vector);
							calculate_crossing_pt_outside(strip_corners[1], normalized_direction_vector);

							// Only the short side of the strip is left
							// We can almost use the same trick with the ray case
							auto prev_counter = corner_counter;
							calculate_crossing_pt_outside(strip_corners[0], strip_corners[1] - strip_corners[0]);

							// There is a simple trick that can filter false zeros found above
							if( prev_counter < corner_counter ) {
								if( dot(intersection_corners[corner_counter - 1] - strip_corners[1],
									strip_corners[0] - strip_corners[1]) < 0 )
								{
									corner_counter = prev_counter;
								}
							}
						}
					}

					// Now we have calculated all the corners
					if( corner_counter < 3 ) {
						// It seems that corner_counter == 1 or 2 sometimes occurs for edge cases.
						// Perhaps due to some floating-point operation pitfalls? I don't know.
						return ValueType(0);
					}

					assert(corner_counter <= 7);

					// Pick one corner as the pivot and sort other corners in a counterclockwise order using bubble sort
					auto const& pivot = intersection_corners[corner_counter - 1];
					for( unsigned int i = 0; i < corner_counter - 2; ++i ) {
						for( unsigned int j = 0; j < corner_counter - i - 2; ++j ) {
							if( signed_area(intersection_corners[j] - pivot,
								intersection_corners[j + 1] - pivot) < 0 )
							{
								using std::swap;
								swap(intersection_corners[j], intersection_corners[j + 1]);
							}
						}
					}

					// Calculate the area
					auto area = ValueType(0);
					auto prev_segment = intersection_corners[0] - pivot;
					for( unsigned int c = 1; c < corner_counter - 1; ++c ) {
						auto segment = intersection_corners[c] - pivot;
						area += signed_area(prev_segment, segment) / ValueType(2);

						assert(signed_area(prev_segment, segment) >= 0);
						prev_segment = segment;
					}

					return area;
				}

			private:
				static bool is_in_pixel(math::R2_elmt<ValueType> const& p, math::R2_elmt<int> const& pixel_coord) {
					return p.x() >= ValueType(pixel_coord.x()) && p.x() <= ValueType(pixel_coord.x() + 1) &&
						p.y() >= ValueType(pixel_coord.y()) && p.y() <= ValueType(pixel_coord.y() + 1);
				}

				// Robust crossing point calculator
				enum class ray_direction { right, top, left, bottom };
				static ray_direction classify_ray_direction(math::R2_elmt<ValueType> const& center,
					math::R2_elmt<ValueType> const& ray, math::R2_elmt<int> const& pixel_coord)
				{
					// 1st quadrant
					if( ray.x() > 0 && ray.y() >= 0 ) {
						math::R2_elmt<ValueType> v{ ValueType(pixel_coord.x() + 1), ValueType(pixel_coord.y() + 1) };
						if( signed_area(ray, v - center) >= 0 )
							return ray_direction::right;
						else
							return ray_direction::top;
					}
					// 2nd quadrant
					else if( ray.x() <= 0 && ray.y() > 0 ) {
						math::R2_elmt<ValueType> v{ ValueType(pixel_coord.x()), ValueType(pixel_coord.y() + 1) };
						if( signed_area(ray, v - center) >= 0 )
							return ray_direction::top;
						else
							return ray_direction::left;
					}
					// 3rd quadrant
					else if( ray.x() < 0 && ray.y() <= 0 ) {
						math::R2_elmt<ValueType> v{ ValueType(pixel_coord.x()), ValueType(pixel_coord.y()) };
						if( signed_area(ray, v - center) >= 0 )
							return ray_direction::left;
						else
							return ray_direction::bottom;
					}
					// 4th quadrant
					else {
						assert(ray.x() >= 0 && ray.y() < 0);
						math::R2_elmt<ValueType> v{ ValueType(pixel_coord.x() + 1), ValueType(pixel_coord.y()) };
						if( signed_area(ray, v - center) >= 0 )
							return ray_direction::bottom;
						else
							return ray_direction::right;
					}
				}

				static math::R2_elmt<ValueType> calculate_crossing_pt_inside(math::R2_elmt<ValueType> const& center,
					math::R2_elmt<ValueType> const& ray, math::R2_elmt<int> const& pixel_coord)
				{
					switch( classify_ray_direction(center, ray, pixel_coord) ) {
					case ray_direction::right:
						return{ ValueType(pixel_coord.x() + 1),
							(ValueType(pixel_coord.x() + 1) - center.x()) * ray.y() / ray.x() + center.y() };

					case ray_direction::top:
						return{ (ValueType(pixel_coord.y() + 1) - center.y()) * ray.x() / ray.y() + center.x(),
							ValueType(pixel_coord.y() + 1) };

					case ray_direction::left:
						return{ ValueType(pixel_coord.x()),
							(ValueType(pixel_coord.x()) - center.x()) * ray.y() / ray.x() + center.y() };

					case ray_direction::bottom:
						return{ (ValueType(pixel_coord.y()) - center.y()) * ray.x() / ray.y() + center.x(),
							ValueType(pixel_coord.y()) };

					default:
						// Cannot reach here
						assert(false);
						return{};
					}
				}
			};

			template <class ValueType, class PixelBuffer>
			void draw_line_impl(PixelBuffer& g,
				math::R2_elmt<ValueType> const& from, math::R2_elmt<ValueType> const& to,
				ValueType linewidth, nana::color const& line_color)
			{
				using std::floor;
				using std::ceil;
				using std::max;
				using std::min;
				using std::abs;

				auto half_width = ValueType(linewidth) / 2;
				auto length = norm(to - from);

				// For extremely short line, just draw two circles
				if( length < std::numeric_limits<ValueType>::epsilon() * 16 ) {
					draw_circle<ValueType>(g, from, half_width, line_color);
					draw_circle<ValueType>(g, to, half_width, line_color);
					return;
				}
				auto normalized_direction_vector = normalize(to - from);

				auto x_min = max(int(floor(ValueType(min(from.x(), to.x())) - half_width)), 0);
				auto x_max = min(int(ceil(ValueType(max(from.x(), to.x())) + half_width)), int(g.size().width));
				auto y_min = max(int(floor(ValueType(min(from.y(), to.y())) - half_width)), 0);
				auto y_max = min(int(ceil(ValueType(max(from.y(), to.y())) + half_width)), int(g.size().height));

				if( x_max <= x_min || y_max <= y_min )
					return;

				auto call_draw = [&](auto const& range) {
					// Draw thick line
					if( half_width > 1 ) {
						auto sdf = [&](math::R2_elmt<ValueType> const& p) {
							if( dot(p - from, normalized_direction_vector) < 0 )
								return norm(p - from);

							if( dot(p - to, normalized_direction_vector) > 0 )
								return norm(p - to);

							return abs(signed_area(p - from, normalized_direction_vector));
						};
						draw<ValueType>(g, range.first, range.second, sdf, linewidth, line_color, line_color);
					}
					// Draw thin line
					else {
						auto sdf = [&from, &normalized_direction_vector](math::R2_elmt<ValueType> const& p) {
							return signed_area(p - from, normalized_direction_vector);
						};
						auto blender = [&](PixelBuffer& g, math::R2_elmt<int> const& pos,
							ValueType intersection_ratio, ValueType border_intersection_ratio)
						{
							auto is_near_pixel = [&](auto const& pos, auto const& ref) {
								return pos.x() <= ref.x() + 2 && pos.x() >= ref.x() - 2 &&
									pos.y() <= ref.y() + 2 && pos.y() >= ref.y() - 2;
							};

							if( is_near_pixel(pos, from) ) {
								strip_ray<ValueType> sr = { from, -normalized_direction_vector, half_width };
								border_intersection_ratio -= sr.calculate_intersection_ratio(pos);
							}
							if( is_near_pixel(pos, to) ) {
								strip_ray<ValueType> sr = { to, normalized_direction_vector, half_width };
								border_intersection_ratio -= sr.calculate_intersection_ratio(pos);
							}
							if( border_intersection_ratio < ValueType(0) )
								border_intersection_ratio = ValueType(0);
							assert(border_intersection_ratio <= 1);

							return color_blender{ {}, line_color }(g, pos, intersection_ratio, border_intersection_ratio);
						};
						draw<ValueType>(g, range.first, range.second, sdf, linewidth, blender);

						draw_small_circle_impl<ValueType>(g, from, half_width, line_color);
						draw_small_circle_impl<ValueType>(g, to, half_width, line_color);
					}
				};

				if( abs(normalized_direction_vector.y()) <= std::numeric_limits<ValueType>::epsilon() * 16 ) {
					call_draw(make_rectangular_region_range({ x_min, y_min },
						{ unsigned(x_max - x_min), unsigned(y_max - y_min) }));
				}
				else {
					call_draw(make_row_connected_region_range(y_min, y_max,
						[cot_theta = normalized_direction_vector.x() / normalized_direction_vector.y(),
						dx = abs(half_width / normalized_direction_vector.y()),
						&from, x_min, x_max](int y)
					{
						auto dy = ValueType(y) - ValueType(from.y());
						auto center_min = from.x() + dy * cot_theta;
						auto center_max = center_min;

						if( cot_theta >= 0 )
							center_max += cot_theta;
						else
							center_min += cot_theta;

						return std::make_pair(
							min(max(int(floor(center_min - dx)), x_min), x_max),
							max(min(int(ceil(center_max + dx)), x_max), x_min));
					}));
				}
			}
		}

		template <class ValueType = double, class PixelBuffer, class Point, class BorderWidth>
		void draw_line(PixelBuffer& g, Point const& from, Point const& to,
			BorderWidth linewidth, nana::color const& line_color)
		{
			assert(linewidth > 0);
			draw_detail::draw_line_impl<ValueType>(g,
				math::R2_elmt<ValueType>{ from },
				math::R2_elmt<ValueType>{ to },
				ValueType(linewidth), line_color);
		}
	}
}