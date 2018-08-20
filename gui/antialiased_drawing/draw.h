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
#include <nana/basic_types.hpp>
#include "../../numerical_lie_group/Rn_elmt.h"

namespace jkl {
	namespace gui {
		// Blender is an object that calculates color to be set at a given position,
		// given background and the pre-calculated ratio of intersection of that pixel
		// with the figure currently drawing. This color_blender class is such a class
		// when the target figure is of plain single color.
		// Users may implement their own blending function if necessary.
		// For example, one might implement gradation blender.
		struct color_blender {
			nana::color color;
			nana::color border_color;

			template <class T>
			static auto clamp(T v) {
				if( v < 0 )
					return (unsigned char)(0);
				else if( v > 255 )
					return (unsigned char)(255);
				else
					return (unsigned char)(v);
			};

			template <class ValueType, class PixelBuffer>
			nana::pixel_color_t operator()(PixelBuffer& g,
				math::R2_elmt<int> const& pos,
				ValueType intersection_ratio, ValueType border_intersection_ratio) const
			{
				intersection_ratio *= ValueType(color.a());
				border_intersection_ratio *= ValueType(border_color.a());

				auto residual_ratio = 1 - intersection_ratio - border_intersection_ratio;
				if( residual_ratio < 0 )
					residual_ratio = 0;

				nana::pixel_color_t c;
				auto const& bg_color = g[pos.y()][pos.x()];
				c.element.alpha_channel = bg_color.element.alpha_channel;
				c.element.red = clamp(residual_ratio * bg_color.element.red +
					intersection_ratio * color.r() + border_intersection_ratio * border_color.r());
				c.element.green = clamp(residual_ratio * bg_color.element.green +
					intersection_ratio * color.g() + border_intersection_ratio * border_color.g());
				c.element.blue = clamp(residual_ratio * bg_color.element.blue +
					intersection_ratio * color.b() + border_intersection_ratio * border_color.b());

				return c;
			}
		};

		namespace draw_detail {
			// The main drawing implementation with anti-aliasing feature.
			// For each pixel in the region, classify each 4 corner of the pixel
			// as one of interior, boundary, and exterior.
			// Using that information and linearly interpolation, the inner region of the pixel
			// is divided into those three classes (interior, boundary, and exterior),rnl
			// and the portion of interior and boundary regions are calculated.
			// The portion of interior region is called the "intersection ratio," while
			// the portion of boundary region is called the "border intersection ratio."
			// These values are passed into the blender object, then the blender object
			// accordingly calculates the color for that pixel.
			//
			// This function is intended to be general enough so that
			// most of implementations of various drawings, besides some extreme cases,
			// can be in turn end up with implementing the range iterator and the SDF function,
			// and occasionally the blender in addition.
			template <class ValueType, class PixelBuffer, class RegionIterator, class SdfType, class BlenderType>
			void draw_impl(PixelBuffer& g, RegionIterator first, RegionIterator last, SdfType&& sdf,
				ValueType const& border_width, BlenderType&& blender, std::size_t subdivision)
			{
				assert(border_width >= 0);

				auto calculate_intersection_ratio = [&sdf, half_width = border_width / 2]
				(ValueType const(&sdf_values)[4], math::R2_elmt<ValueType> const& center)
				{
					// Interior: sdf < -w/2
					// Boundary: sdf in [-w/2, w/2]
					// Exterior: sdf > w/2
					enum class region : unsigned char { intr = 0, bdy = 1, extr = 2 };
					auto aggregate = [](region r0, region r1, region r2, region r3) {
						return (unsigned(r0)) | (unsigned(r1) << 2) | (unsigned(r2) << 4) | (unsigned(r3) << 6);
					};

					region classifications[4];
					for( auto q = 0; q < 4; ++q ) {
						if( sdf_values[q] < -half_width )
							classifications[q] = region::intr;
						else if( sdf_values[q] > half_width )
							classifications[q] = region::extr;
						else
							classifications[q] = region::bdy;
					}

					ValueType intersection_ratio = 0;
					ValueType border_intersection_ratio = 0;

					auto bdy_to_extr_solve = [half_width](ValueType bdy_sdf, ValueType extr_sdf) {
						return (bdy_sdf - half_width) / (bdy_sdf - extr_sdf);
					};
					auto intr_to_bdy_solve = [half_width](ValueType intr_sdf, ValueType bdy_sdf) {
						return (intr_sdf + half_width) / (intr_sdf - bdy_sdf);
					};
					auto intr_to_extr_solve = [half_width](ValueType intr_sdf, ValueType extr_sdf) {
						return std::make_pair(
							(intr_sdf + half_width) / (intr_sdf - extr_sdf),	// Where the bdy starts?
							(intr_sdf - half_width) / (intr_sdf - extr_sdf));	// Where the bdy ends?
					};

					///////////////////////////////////////////////////////////////////////////
					////////////////////////// 21 canonical patterns //////////////////////////
					///////////////////////////////////////////////////////////////////////////

					// #0
					////////////
					// e -- e //
					// |    | //
					// e -- e //
					////////////
					// do nothing

					// #1
					////////////
					// e -- b //
					// |    | //
					// e -- e //
					////////////
					auto bdy_extr_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						border_intersection_ratio =
							bdy_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]) *
							bdy_to_extr_solve(permuted_sdfs[0], permuted_sdfs[1]) / 2;
					};

					// #2
					////////////
					// e -- i //
					// |    | //
					// e -- e //
					////////////
					auto intr_extr_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pr = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto pt = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[1]);

						intersection_ratio = pr.first * pt.first / 2;
						border_intersection_ratio = pr.second * pt.second / 2 - intersection_ratio;
					};

					// #3
					////////////
					// b -- b //
					// |    | //
					// e -- e //
					////////////
					auto bdy_bdy_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						border_intersection_ratio =
							(bdy_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]) +
								bdy_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2])) / 2;
					};

					// #4
					////////////
					// b -- i //
					// |    | //
					// e -- e //
					////////////
					auto intr_bdy_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pr = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);

						intersection_ratio =
							intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[1]) * pr.first / 2;
						border_intersection_ratio =
							(bdy_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]) + pr.second) / 2
							- intersection_ratio;
					};

					// #4'
					////////////
					// i -- b //
					// |    | //
					// e -- e //
					////////////
					auto bdy_intr_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						intr_bdy_extr_extr({ permuted_sdfs[1], permuted_sdfs[0], permuted_sdfs[3], permuted_sdfs[2] });
					};

					// #5
					////////////
					// i -- i //
					// |    | //
					// e -- e //
					////////////
					auto intr_intr_extr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pr = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto pl = intr_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]);

						intersection_ratio = (pr.first + pl.first) / 2;
						border_intersection_ratio = (pr.second + pl.second) / 2 - intersection_ratio;
					};

					// #6
					////////////                 ////////////
					// e -- b //                 // e -- b //
					// |\  \| //                 // |/  /| //  This configuration is impossible
					// b -- e //                 // b -- e //
					////////////                 ////////////
					auto bdy_extr_bdy_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto r = bdy_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto t = bdy_to_extr_solve(permuted_sdfs[0], permuted_sdfs[1]);
						auto l = bdy_to_extr_solve(permuted_sdfs[2], permuted_sdfs[1]);
						auto b = bdy_to_extr_solve(permuted_sdfs[2], permuted_sdfs[3]);

						border_intersection_ratio = (r * t + l * b) / 2;
					};

					// #7
					////////////                 ////////////
					// e -- i //                 // e -- i //
					// |\  \| //                 // |/  /| //  This configuration is impossible
					// b -- e //                 // b -- e //
					////////////                 ////////////
					auto intr_extr_bdy_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pr = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto pt = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[1]);

						intersection_ratio = pr.first * pt.first / 2;
						border_intersection_ratio = (
							(bdy_to_extr_solve(permuted_sdfs[2], permuted_sdfs[3]) *
								bdy_to_extr_solve(permuted_sdfs[2], permuted_sdfs[1])) +
							pt.second * pr.second) / 2 - intersection_ratio;
					};

					// #8 (need ambiguity resolution)
					////////////
					// e -- i //
					// |    | //
					// i -- e //
					////////////
					auto intr_extr_intr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pr = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto pt = intr_to_extr_solve(permuted_sdfs[0], permuted_sdfs[1]);
						auto pl = intr_to_extr_solve(permuted_sdfs[2], permuted_sdfs[1]);
						auto pb = intr_to_extr_solve(permuted_sdfs[2], permuted_sdfs[3]);

						// Ambiguity resolution
						auto center_sdf = sdf(center);
						if( center_sdf > 0 ) {
							////////////
							// e -- i //
							// |\  \| //
							// i -- e //
							////////////

							intersection_ratio = (pr.first * pt.first + pl.first * pb.first) / 2;
							border_intersection_ratio = (pr.second * pt.second + pl.second * pb.second) / 2
								- intersection_ratio;
						}
						else {
							////////////
							// e -- i //
							// |/  /| //
							// i -- e //
							////////////

							intersection_ratio = 1 -
								((1 - pt.first) * (1 - pl.first) + (1 - pr.first) * (1 - pb.first)) / 2;
							border_intersection_ratio = 1 -
								((1 - pt.second) * (1 - pl.second) + (1 - pr.second) * (1 - pb.second)) / 2
								- intersection_ratio;
						}
					};

					// #9
					////////////
					// b -- b //
					// |    | //
					// e -- b //
					////////////
					auto bdy_bdy_extr_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						border_intersection_ratio = 1 -
							((1 - bdy_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2])) *
							(1 - bdy_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2]))) / 2;
					};

					// #10
					////////////
					// b -- i //
					// |    | //
					// e -- b //
					////////////
					auto intr_bdy_extr_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						intersection_ratio =
							intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[3]) *
							intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[1]) / 2;

						border_intersection_ratio = 1 -
							((1 - bdy_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2])) *
							(1 - bdy_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2]))) / 2
							- intersection_ratio;
					};

					// #11
					////////////
					// b -- b //
					// |    | //
					// e -- i //
					////////////
					auto bdy_bdy_extr_intr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pb = intr_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2]);
						intersection_ratio = pb.first *
							intr_to_bdy_solve(permuted_sdfs[3], permuted_sdfs[0]) / 2;

						border_intersection_ratio = 1 -
							((1 - pb.second) * (1 - bdy_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]))) / 2
							- intersection_ratio;
					};

					// #11'
					////////////
					// b -- b //
					// |    | //
					// i -- e //
					////////////
					auto bdy_bdy_intr_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						bdy_bdy_extr_intr({ permuted_sdfs[1], permuted_sdfs[0], permuted_sdfs[3], permuted_sdfs[2] });
					};

					// #12
					////////////
					// i -- b //
					// |    | //
					// e -- i //
					////////////
					auto bdy_intr_extr_intr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pl = intr_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]);
						auto pb = intr_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2]);

						auto rt_triangle_times_2 =
							intr_to_bdy_solve(permuted_sdfs[3], permuted_sdfs[0]) *
							intr_to_bdy_solve(permuted_sdfs[1], permuted_sdfs[0]);
						auto lb_triangle_times_2 = (1 - pl.first) * (1 - pb.first);

						intersection_ratio = 1 - (rt_triangle_times_2 + lb_triangle_times_2) / 2;

						border_intersection_ratio = (rt_triangle_times_2 +
							lb_triangle_times_2 - (1 - pl.second) * (1 - pb.second)) / 2;
					};

					// #13
					////////////
					// i -- i //
					// |    | //
					// e -- b //
					////////////
					auto intr_intr_extr_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pl = intr_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]);

						intersection_ratio =
							(intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[3]) + pl.first) / 2;

						border_intersection_ratio = 1 -
							((1 - pl.second) * (1 - bdy_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2])) / 2
							+ intersection_ratio);
					};

					// #13'
					////////////
					// i -- i //
					// |    | //
					// b -- e //
					////////////
					auto intr_intr_bdy_extr = [&](ValueType const(&permuted_sdfs)[4]) {
						intr_intr_extr_bdy({ permuted_sdfs[1], permuted_sdfs[0], permuted_sdfs[3], permuted_sdfs[2] });
					};

					// #14
					////////////
					// i -- i //
					// |    | //
					// e -- i //
					////////////
					auto intr_intr_extr_intr = [&](ValueType const(&permuted_sdfs)[4]) {
						auto pl = intr_to_extr_solve(permuted_sdfs[1], permuted_sdfs[2]);
						auto pb = intr_to_extr_solve(permuted_sdfs[3], permuted_sdfs[2]);

						intersection_ratio = 1 - (1 - pl.first) * (1 - pb.first) / 2;
						border_intersection_ratio = 1 - ((1 - pl.second) * (1 - pb.second) / 2 + intersection_ratio);
					};

					// #15
					////////////
					// b -- b //
					// |    | //
					// b -- b //
					////////////
					auto bdy_bdy_bdy_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						border_intersection_ratio = 1;
					};

					// #16
					////////////
					// b -- i //
					// |    | //
					// b -- b //
					////////////
					auto intr_bdy_bdy_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						intersection_ratio =
							intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[3]) *
							intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[1]) / 2;
						border_intersection_ratio = 1 - intersection_ratio;
					};

					// #17
					////////////
					// i -- i //
					// |    | //
					// b -- b //
					////////////
					auto intr_intr_bdy_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						intersection_ratio =
							(intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[3]) +
								intr_to_bdy_solve(permuted_sdfs[1], permuted_sdfs[2])) / 2;
						border_intersection_ratio = 1 - intersection_ratio;
					};

					// #18 (need ambiguity resolution)
					////////////
					// b -- i //
					// |    | //
					// i -- b //
					////////////
					auto intr_bdy_intr_bdy = [&](ValueType const(&permuted_sdfs)[4]) {
						auto r = intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[3]);
						auto t = intr_to_bdy_solve(permuted_sdfs[0], permuted_sdfs[1]);
						auto l = intr_to_bdy_solve(permuted_sdfs[2], permuted_sdfs[1]);
						auto b = intr_to_bdy_solve(permuted_sdfs[2], permuted_sdfs[3]);

						// Ambiguity resolution
						auto center_sdf = sdf(center);
						if( center_sdf <= half_width && center_sdf >= -half_width ) {
							////////////
							// b -- i //
							// |\  \| //
							// i -- b //
							////////////
							intersection_ratio = (r * t + l * b) / 2;
							border_intersection_ratio = 1 - intersection_ratio;
						}
						else {
							////////////
							// b -- i //
							// |/  /| //
							// i -- b //
							////////////
							border_intersection_ratio = ((1 - t) * (1 - l) + (1 - b) * (1 - r)) / 2;
							intersection_ratio = 1 - border_intersection_ratio;
						}
					};

					// #19
					////////////
					// i -- i //
					// |    | //
					// b -- i //
					////////////
					auto intr_intr_bdy_intr = [&](ValueType const(&permuted_sdfs)[4]) {
						border_intersection_ratio =
							(1 - intr_to_bdy_solve(permuted_sdfs[1], permuted_sdfs[2])) *
							(1 - intr_to_bdy_solve(permuted_sdfs[3], permuted_sdfs[2])) / 2;
						intersection_ratio = 1 - border_intersection_ratio;
					};

					// #20
					////////////
					// i -- i //
					// |    | //
					// i -- i //
					////////////
					auto intr_intr_intr_intr = [&](ValueType const(&permuted_sdfs)[4]) {
						intersection_ratio = 1;
					};


					// Map 81 cases into 21 canonical cases using dihedral symmetry
					switch( aggregate(classifications[0], classifications[1], classifications[2], classifications[3]) )
					{
						// #0
						////////////
						// e -- e //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::extr, region::extr, region::extr, region::extr):
						break;

						// #1
						////////////
						// e -- b //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::bdy, region::extr, region::extr, region::extr):
						bdy_extr_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::bdy, region::extr, region::extr):
						bdy_extr_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::bdy, region::extr):
						bdy_extr_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::extr, region::extr, region::extr, region::bdy):
						bdy_extr_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #2
						////////////
						// e -- i //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::intr, region::extr, region::extr, region::extr):
						intr_extr_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::extr, region::extr):
						intr_extr_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::intr, region::extr):
						intr_extr_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::extr, region::extr, region::extr, region::intr):
						intr_extr_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #3
						////////////
						// b -- b //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::bdy, region::bdy, region::extr, region::extr):
						bdy_bdy_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::bdy, region::bdy, region::extr):
						bdy_bdy_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::bdy, region::bdy):
						bdy_bdy_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::extr, region::extr, region::bdy):
						bdy_bdy_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #4
						////////////
						// b -- i //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::intr, region::bdy, region::extr, region::extr):
						intr_bdy_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::bdy, region::extr):
						intr_bdy_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::intr, region::bdy):
						intr_bdy_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::extr, region::extr, region::intr):
						intr_bdy_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #4'
						////////////
						// i -- b //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::bdy, region::intr, region::extr, region::extr):
						bdy_intr_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::bdy, region::intr, region::extr):
						bdy_intr_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::bdy, region::intr):
						bdy_intr_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::extr, region::extr, region::bdy):
						bdy_intr_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #5
						////////////
						// i -- i //
						// |    | //
						// e -- e //
						////////////
					case aggregate(region::intr, region::intr, region::extr, region::extr):
						intr_intr_extr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::intr, region::extr):
						intr_intr_extr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::extr, region::intr, region::intr):
						intr_intr_extr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::extr, region::extr, region::intr):
						intr_intr_extr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #6
						////////////                 ////////////
						// e -- b //                 // e -- b //
						// |\  \| //                 // |/  /| //  This configuration is impossible
						// b -- e //                 // b -- e //
						////////////                 ////////////
					case aggregate(region::bdy, region::extr, region::bdy, region::extr):
						bdy_extr_bdy_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::bdy, region::extr, region::bdy):
						bdy_extr_bdy_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;

						// #7
						////////////                 ////////////
						// e -- i //                 // e -- i //
						// |\  \| //                 // |/  /| //  This configuration is impossible
						// b -- e //                 // b -- e //
						////////////                 ////////////
					case aggregate(region::intr, region::extr, region::bdy, region::extr):
						intr_extr_bdy_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::extr, region::bdy):
						intr_extr_bdy_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::bdy, region::extr, region::intr, region::extr):
						intr_extr_bdy_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::extr, region::bdy, region::extr, region::intr):
						intr_extr_bdy_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #8
						////////////
						// e -- i //
						// |    | //
						// i -- e //
						////////////
					case aggregate(region::intr, region::extr, region::intr, region::extr):
						intr_extr_intr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::extr, region::intr):
						intr_extr_intr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;

						// #9
						////////////
						// b -- b //
						// |    | //
						// e -- b //
						////////////
					case aggregate(region::bdy, region::bdy, region::extr, region::bdy):
						bdy_bdy_extr_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::bdy, region::bdy, region::extr):
						bdy_bdy_extr_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::bdy, region::bdy, region::bdy):
						bdy_bdy_extr_bdy({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::extr, region::bdy, region::bdy):
						bdy_bdy_extr_bdy({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #10
						////////////
						// b -- i //
						// |    | //
						// e -- b //
						////////////
					case aggregate(region::intr, region::bdy, region::extr, region::bdy):
						intr_bdy_extr_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::intr, region::bdy, region::extr):
						intr_bdy_extr_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::bdy, region::intr, region::bdy):
						intr_bdy_extr_bdy({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::extr, region::bdy, region::intr):
						intr_bdy_extr_bdy({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #11
						////////////
						// b -- b //
						// |    | //
						// e -- i //
						////////////
					case aggregate(region::bdy, region::bdy, region::extr, region::intr):
						bdy_bdy_extr_intr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::intr, region::bdy, region::bdy, region::extr):
						bdy_bdy_extr_intr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::intr, region::bdy, region::bdy):
						bdy_bdy_extr_intr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::extr, region::intr, region::bdy):
						bdy_bdy_extr_intr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #11'
						////////////
						// b -- b //
						// |    | //
						// i -- e //
						////////////
					case aggregate(region::bdy, region::bdy, region::intr, region::extr):
						bdy_bdy_intr_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::bdy, region::bdy, region::intr):
						bdy_bdy_intr_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::intr, region::extr, region::bdy, region::bdy):
						bdy_bdy_intr_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::intr, region::extr, region::bdy):
						bdy_bdy_intr_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #12
						////////////
						// i -- b //
						// |    | //
						// e -- i //
						////////////
					case aggregate(region::bdy, region::intr, region::extr, region::intr):
						bdy_intr_extr_intr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::intr, region::bdy, region::intr, region::extr):
						bdy_intr_extr_intr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::intr, region::bdy, region::intr):
						bdy_intr_extr_intr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::extr, region::intr, region::bdy):
						bdy_intr_extr_intr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #13
						////////////
						// i -- i //
						// |    | //
						// e -- b //
						////////////
					case aggregate(region::intr, region::intr, region::extr, region::bdy):
						intr_intr_extr_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::intr, region::intr, region::extr):
						intr_intr_extr_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::bdy, region::intr, region::intr):
						intr_intr_extr_bdy({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::extr, region::bdy, region::intr):
						intr_intr_extr_bdy({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #13'
						////////////
						// i -- i //
						// |    | //
						// b -- e //
						////////////
					case aggregate(region::intr, region::intr, region::bdy, region::extr):
						intr_intr_bdy_extr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::extr, region::intr, region::intr, region::bdy):
						intr_intr_bdy_extr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::bdy, region::extr, region::intr, region::intr):
						intr_intr_bdy_extr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::bdy, region::extr, region::intr):
						intr_intr_bdy_extr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #14
						////////////
						// i -- i //
						// |    | //
						// e -- i //
						////////////
					case aggregate(region::intr, region::intr, region::extr, region::intr):
						intr_intr_extr_intr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::intr, region::intr, region::intr, region::extr):
						intr_intr_extr_intr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::extr, region::intr, region::intr, region::intr):
						intr_intr_extr_intr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::extr, region::intr, region::intr):
						intr_intr_extr_intr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #15
						////////////
						// b -- b //
						// |    | //
						// b -- b //
						////////////
					case aggregate(region::bdy, region::bdy, region::bdy, region::bdy):
						bdy_bdy_bdy_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;

						// #16
						////////////
						// b -- i //
						// |    | //
						// b -- b //
						////////////
					case aggregate(region::intr, region::bdy, region::bdy, region::bdy):
						intr_bdy_bdy_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::intr, region::bdy, region::bdy):
						intr_bdy_bdy_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::bdy, region::bdy, region::intr, region::bdy):
						intr_bdy_bdy_bdy({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::bdy, region::bdy, region::bdy, region::intr):
						intr_bdy_bdy_bdy({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #17
						////////////
						// i -- i //
						// |    | //
						// b -- b //
						////////////
					case aggregate(region::intr, region::intr, region::bdy, region::bdy):
						intr_intr_bdy_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::intr, region::intr, region::bdy):
						intr_intr_bdy_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::bdy, region::bdy, region::intr, region::intr):
						intr_intr_bdy_bdy({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::bdy, region::bdy, region::intr):
						intr_intr_bdy_bdy({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #18
						////////////
						// b -- i //
						// |    | //
						// i -- b //
						////////////
					case aggregate(region::intr, region::bdy, region::intr, region::bdy):
						intr_bdy_intr_bdy({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::bdy, region::intr, region::bdy, region::intr):
						intr_bdy_intr_bdy({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;

						// #19
						////////////
						// i -- i //
						// |    | //
						// b -- i //
						////////////
					case aggregate(region::intr, region::intr, region::bdy, region::intr):
						intr_intr_bdy_intr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;
					case aggregate(region::intr, region::intr, region::intr, region::bdy):
						intr_intr_bdy_intr({ sdf_values[1], sdf_values[2], sdf_values[3], sdf_values[0] });
						break;
					case aggregate(region::bdy, region::intr, region::intr, region::intr):
						intr_intr_bdy_intr({ sdf_values[2], sdf_values[3], sdf_values[0], sdf_values[1] });
						break;
					case aggregate(region::intr, region::bdy, region::intr, region::intr):
						intr_intr_bdy_intr({ sdf_values[3], sdf_values[0], sdf_values[1], sdf_values[2] });
						break;

						// #20
						////////////
						// i -- i //
						// |    | //
						// i -- i //
						////////////
					case aggregate(region::intr, region::intr, region::intr, region::intr):
						intr_intr_intr_intr({ sdf_values[0], sdf_values[1], sdf_values[2], sdf_values[3] });
						break;

					default:
						// Cannot reach here
						assert(false);
					}

					assert(intersection_ratio >= 0 && intersection_ratio <= 1);
					assert(border_intersection_ratio >= 0 && border_intersection_ratio <= 1);
					return std::make_pair(intersection_ratio, border_intersection_ratio);
				};

				auto const subpixel_length = ValueType(1) / subdivision;
				auto const subpixel_area = subpixel_length * subpixel_length;

				for( auto itr = first; itr != last; ++itr ) {
					math::R2_elmt<ValueType> ref_point{ ValueType(itr->x()), ValueType(itr->y()) };
					auto intersection_ratio = ValueType(0);
					auto border_intersection_ratio = ValueType(0);

					for( std::size_t subdiv_y = 0; subdiv_y < subdivision; ++subdiv_y, ref_point.y() += subpixel_length ) {
						for( std::size_t subdiv_x = 0; subdiv_x < subdivision; ++subdiv_x, ref_point.x() += subpixel_length ) {
							ValueType sdf_values[] = {
								sdf({ ref_point.x() + subpixel_length, ref_point.y() + subpixel_length }),
								sdf({ ref_point.x(), ref_point.y() + subpixel_length }),
								sdf(ref_point),
								sdf({ ref_point.x() + subpixel_length, ref_point.y() })
							};
							auto increments = calculate_intersection_ratio(sdf_values,
								{ ref_point.x() + subpixel_length / 2, ref_point.y() + subpixel_length / 2 });

							intersection_ratio += increments.first * subpixel_area;
							border_intersection_ratio += increments.second * subpixel_area;
						}
						ref_point.x() = ValueType(itr->x());
					}

					g[itr->y()][itr->x()] = blender(g, *itr, intersection_ratio, border_intersection_ratio);
				}
			}
		}

		template <class ValueType = double, class PixelBuffer,
			class RegionIterator, class SdfType, class BorderWidth, class BlenderType>
			void draw(PixelBuffer& g, RegionIterator first, RegionIterator last, SdfType&& sdf,
				BorderWidth border_width, BlenderType&& blender, std::size_t subdivision = 1)
		{
			draw_detail::draw_impl(g, std::move(first), std::move(last),
				std::forward<SdfType>(sdf), ValueType(border_width), std::forward<BlenderType>(blender),
				subdivision);
		}

		template <class ValueType = double, class PixelBuffer, class RegionIterator, class SdfType, class BorderWidth>
		void draw(PixelBuffer& g, RegionIterator first, RegionIterator last, SdfType const& sdf,
			BorderWidth border_width, nana::color const& color, nana::color const& border_color,
			std::size_t subdivision = 1)
		{
			draw_detail::draw_impl(g, std::move(first), std::move(last),
				sdf, ValueType(border_width), color_blender{ color, border_color }, subdivision);
		}
	}
}