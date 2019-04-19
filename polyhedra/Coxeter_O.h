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

#pragma once
#include <array>
#include <tuple>
#include "../bit_twiddling.h"
#include "../numerical_lie_group.h"

namespace jkj {
	namespace math {
		/// Rotational octahedral symmetry group
		/// This is the group of rotational symmetry of cube (or equivalently, octahedron).
		/// The group has 24 elements, and is isomorphic to the symmetry group S4 of 4 elements.
		/// This corresponds to permutation of four space diagonals (of length sqrt(3)).
		///
		///      v5----e6----v7
		///     /|          /|         |z
		///   e3 |        e4 |         |
		///   /  e9       / e11        /------ y
		///  v6----e8----v8  |        /
		///  |   |       |   |        x
		///  |   v1----e5|---v3
		/// e10 /       e12 /
		///  | e1        | e2
		///  |/          |/
		///  v2----e7----v4
		///

		namespace Coxeter_O_detail {
			struct vertex {
				// The first bit (LSB) is the sign along the x-axis. (0: minus, 1: plus)
				// The second bit is the sign along the y-axis. (0: minus, 1: plus)
				// The third bit is the sign along the z-axis. (0: minus, 1: plus)
				// All the other bits should be zero.
				using internal_type = std::uint8_t;
				JKL_GPU_EXECUTABLE constexpr vertex(internal_type internal_value) noexcept
					: internal_value{ internal_value } {}
				vertex() = default;

				JKL_GPU_EXECUTABLE constexpr operator internal_type() const noexcept {
					return internal_value;
				}
				
				JKL_GPU_EXECUTABLE constexpr bool operator==(vertex v) const noexcept {
					return internal_type(*this) == internal_type(v);
				}

			private:
				internal_type		internal_value;
			};

			struct edge_base {
				// The last two bits (next-to-MSB and MSB) represents the direction.
				// The first two bits (lsb and next-to-LSB) represents two coordinates; for example, 
				// For an edge along x-axis, 
				// - 00 means that the edge intersect with the yz-plane at (-1, -1), 
				// - 10 means that the edge intersect with the yz-plane at (+1, -1),
				// - 01 means that the edge intersect with the yz-plane at (-1, +1), and
				// - 11 means that the edge intersect with the yz-plane at (+1, +1).
				using internal_type = std::uint8_t;
				JKL_GPU_EXECUTABLE constexpr edge_base(internal_type internal_value) noexcept
					: internal_value{ internal_value } {}
				edge_base() = default;

				JKL_GPU_EXECUTABLE constexpr operator internal_type() const noexcept {
					return internal_value;
				}

				enum direction_type : std::uint8_t { x = 0, y = 1, z = 2 };

				// Get the last two bits
				JKL_GPU_EXECUTABLE constexpr direction_type direction() const noexcept {
					return direction_type(internal_value >> 2);
				}

				// Runtime calculation of endpoints
				JKL_GPU_EXECUTABLE constexpr vertex calculate_first_end() const noexcept {
					using jkj::util::bit_at;
					return jkj::util::bits(
						// first bit
						(bit_at(internal_value, 2) && bit_at(internal_value, 1)) ||
						(bit_at(internal_value, 3) && bit_at(internal_value, 0)),
						// second bit
						(bit_at(internal_value, 3) && bit_at(internal_value, 1)) ||
						(!bit_at(internal_value, 2) && !bit_at(internal_value, 3) && bit_at(internal_value, 0)),
						// third bit
						(!bit_at(internal_value, 2) && !bit_at(internal_value, 3) && bit_at(internal_value, 1)) ||
						(bit_at(internal_value, 2) && bit_at(internal_value, 0)));
				}
				JKL_GPU_EXECUTABLE constexpr vertex calculate_second_end() const noexcept {
					using jkj::util::bit_at;
					return jkj::util::bits(
						// First bit
						(bit_at(internal_value, 2) && bit_at(internal_value, 1)) ||
						(bit_at(internal_value, 3) && bit_at(internal_value, 0)) ||
						(!bit_at(internal_value, 2) && !bit_at(internal_value, 3)),
						// Second bit
						(bit_at(internal_value, 3) && bit_at(internal_value, 1)) ||
						(!bit_at(internal_value, 2) && !bit_at(internal_value, 3) && bit_at(internal_value, 0)) ||
						bit_at(internal_value, 2),
						// Third bit
						(!bit_at(internal_value, 2) && !bit_at(internal_value, 3) && bit_at(internal_value, 1)) ||
						(bit_at(internal_value, 2) && bit_at(internal_value, 0)) ||
						bit_at(internal_value, 3));
				}

			private:
				internal_type		internal_value;
			};

			template <std::uint8_t... i>
			static constexpr auto generate_endpoint_table(std::integer_sequence<std::uint8_t, i...>) noexcept
			{
				return std::make_pair(
					std::array<vertex, sizeof...(i)>{ { edge_base{ i }.calculate_first_end()... } },
					std::array<vertex, sizeof...(i)>{ { edge_base{ i }.calculate_second_end()... } });
			}

			struct edge : edge_base {
				using edge_base::edge_base;
				JKL_GPU_EXECUTABLE constexpr operator internal_type() const noexcept {
					return static_cast<edge_base const&>(*this);
				}

				// Calling these with internal_value >= 12 results in undefined-behavior
				constexpr vertex first_end() const noexcept {
					return endpoint_table.first[*this];
				}
				constexpr vertex second_end() const noexcept {
					return endpoint_table.second[*this];
				}
				
				JKL_GPU_EXECUTABLE constexpr bool operator==(edge e) const noexcept {
					return internal_type(*this) == internal_type(e);
				}

			protected:
				static constexpr auto endpoint_table =
					generate_endpoint_table(std::make_integer_sequence<std::uint8_t, 12>{});
			};

			JKL_GPU_EXECUTABLE constexpr bool along_x(vertex v, vertex w) noexcept {
				return bool(jkj::util::bit_at(v, 0) ^ jkj::util::bit_at(w, 0));
			}
			JKL_GPU_EXECUTABLE constexpr bool along_y(vertex v, vertex w) noexcept {
				return bool(jkj::util::bit_at(v, 1) ^ jkj::util::bit_at(w, 1));
			}
			JKL_GPU_EXECUTABLE constexpr bool along_z(vertex v, vertex w) noexcept {
				return bool(jkj::util::bit_at(v, 2) ^ jkj::util::bit_at(w, 2));
			}

			JKL_GPU_EXECUTABLE constexpr edge join(vertex v, vertex w) noexcept {
				using jkj::util::bit_at;
				return jkj::util::bits(
					// First bit
					(along_x(v, w) && bit_at(v, 1)) ||
					(along_y(v, w) && bit_at(v, 2)) ||
					(along_z(v, w) && bit_at(v, 0)),
					// Second bit
					(along_x(v, w) && bit_at(v, 2)) ||
					(along_y(v, w) && bit_at(v, 0)) ||
					(along_z(v, w) && bit_at(v, 1)),
					// Third bit
					!along_x(v, w) && (along_y(v, w) || !along_z(v, w)),
					// Fourth bit
					!along_x(v, w) && (!along_y(v, w) || along_z(v, w))
				);
			}

			struct element_base {
				// Elements in the Coxeter group O8 permute basis vectors of R^3, with or without sign inversion.
				// They can be represented as 3x3 rotation matrices whose entries are one of -1, 0, +1.
				// There are exactly 24 such matrices; this is nothing but the 3-dimensional faithful representation of S4.
				// The encoding of elements is done as follows:
				// The first two bits (lsb and the next) represent sign inversions.
				// - 00 means both x-axis and y-axis become some basis vectors.
				// - 10 means x-axis becomes the mirror image of a basis vector while y-axis doesn't.
				// - 01 means x-axis doesn't become the mirror image of a basis vector while y-axis does.
				// - 11 menas both x-axis and y-axis become the mirror images of some basis vectors.
				// The next three bits represent permutation of three basis vectors.
				// - 000 is the identity permutation
				// - 100 is the permutation (y z)
				// - 010 is the permutation (z x)
				// - 110 is the permutation (x y z)
				// - 001 is the permutation (x y)
				// - 101 is the permutation (x z x)
				// This encoding has been designed with the followings in mind:
				// - 100 means the x-axis is fixed
				// - 010 means the y-axis is fixed
				// - 001 means the z-axis is fixed
				// - 110 means the composition (100)(010); that is, (y z)(z x) = (x y z)
				// - 101 means the composition (100)(001); that is, (y z)(x y) = (x z y)
				// The order of composition was of course an arbitrary choice.
			private:
				using int_vector = jkj::math::R3_elmt<int>;
				using int_matrix = jkj::math::gl3_elmt<int>;
				
			public:
				using internal_type = std::uint8_t;
				JKL_GPU_EXECUTABLE constexpr element_base(internal_type internal_value) noexcept
					: internal_value{ internal_value } {}
				element_base() = default;

				JKL_GPU_EXECUTABLE constexpr operator internal_type() const noexcept {
					return internal_value;
				}

				using representation_type = jkj::math::SO3_elmt<int>;
				JKL_GPU_EXECUTABLE constexpr representation_type calculate_representation() const noexcept {
					return{ permute_coordinates(get_axis_permutation(),
						int_matrix{
						x_parity(), 0, 0,
						0, y_parity(), 0,
						0, 0, z_parity() }),
						jkj::math::no_validity_check{}
					};
				}

				JKL_GPU_EXECUTABLE constexpr vertex calculate_action(vertex v) const noexcept {
					return vector_to_vertex(calculate_representation() * vertex_to_vector(v));
				}

				JKL_GPU_EXECUTABLE constexpr edge calculate_action(edge e) const noexcept {
					return join(calculate_action(e.calculate_first_end()), calculate_action(e.calculate_second_end()));
				}

				JKL_GPU_EXECUTABLE static constexpr element_base calculate_from_representation(
					representation_type const& r) noexcept {
					// MSVC's built-in array is somewhat buggy.
					// I guess that's why the operator[] doesn't evaluate as constexpr here...
					using jkj::util::bits;
					/*return
						r[0][0] != 0 && r[1][1] != 0 ? bits(r[0][0] < 0, r[1][1] < 0, 0, 0, 0) :	// 0 ~ 3
						r[0][0] != 0 && r[2][1] != 0 ? bits(r[0][0] < 0, r[2][1] < 0, 1, 0, 0) :	// 4 ~ 7
						r[2][0] != 0 && r[1][1] != 0 ? bits(r[2][0] < 0, r[1][1] < 0, 0, 1, 0) :	// 8 ~ 11
						r[2][0] != 0 && r[0][1] != 0 ? bits(r[2][0] < 0, r[0][1] < 0, 1, 1, 0) :	// 12 ~ 15
						r[1][0] != 0 && r[0][1] != 0 ? bits(r[1][0] < 0, r[0][1] < 0, 0, 0, 1) :	// 16 ~ 19
						bits(r[1][0] < 0, r[2][1] < 0, 1, 0, 1);									// 20 ~ 23*/
					return
						r.get<0, 0>() != 0 && r.get<1, 1>() != 0 ?
						bits(r.get<0, 0>() < 0, r.get<1, 1>() < 0, 0, 0, 0) :	// 0 ~ 3
						r.get<0, 0>() != 0 && r.get<2, 1>() != 0 ?
						bits(r.get<0, 0>() < 0, r.get<2, 1>() < 0, 1, 0, 0) :	// 4 ~ 7
						r.get<2, 0>() != 0 && r.get<1, 1>() != 0 ?
						bits(r.get<2, 0>() < 0, r.get<1, 1>() < 0, 0, 1, 0) :	// 8 ~ 11
						r.get<2, 0>() != 0 && r.get<0, 1>() != 0 ?
						bits(r.get<2, 0>() < 0, r.get<0, 1>() < 0, 1, 1, 0) :	// 12 ~ 15
						r.get<1, 0>() != 0 && r.get<0, 1>() != 0 ?
						bits(r.get<1, 0>() < 0, r.get<0, 1>() < 0, 0, 0, 1) :	// 16 ~ 19
						bits(r.get<1, 0>() < 0, r.get<2, 1>() < 0, 1, 0, 1);	// 20 ~ 23
				}

				JKL_GPU_EXECUTABLE constexpr element_base calculate_multiplication(element_base e) const noexcept {
					return calculate_from_representation(calculate_representation() * e.calculate_representation());
				}

			private:
				internal_type		internal_value;

				JKL_GPU_EXECUTABLE constexpr std::uint8_t get_axis_permutation() const noexcept {
					return std::uint8_t(internal_value >> 2);
				}

				JKL_GPU_EXECUTABLE static constexpr int_matrix permute_coordinates(std::uint8_t axis_permutation, 
					int_matrix const& m) noexcept
				{
					using jkj::util::bits;
				#if !defined(HAS_GENERALIZED_CONSTEXPR)
					return axis_permutation == bits(0, 0, 0) ?
						m :
						axis_permutation == bits(1, 0, 0) ?
						int_matrix{ m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[1][0], m[1][1], m[1][2] } :
						axis_permutation == bits(0, 1, 0) ?
						int_matrix{ m[2][0], m[2][1], m[2][2], m[1][0], m[1][1], m[1][2], m[0][0], m[0][1], m[0][2] } :
						axis_permutation == bits(1, 1, 0) ?
						int_matrix{ m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[0][0], m[0][1], m[0][2] } :
						axis_permutation == bits(0, 0, 1) ?
						int_matrix{ m[1][0], m[1][1], m[1][2], m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2] } :
						// axis_permutation == bits(1, 0, 1) ?
						int_matrix{ m[2][0], m[2][1], m[2][2], m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2] };
				#else
					// MSVC's built-in array is somewhat buggy.
					// I guess that's why the operator[] doesn't evaluate as constexpr here...
					switch( axis_permutation ) {
					case bits(0, 0, 0):			// Identity
						return m;
					case bits(1, 0, 0):			// (y z)
						//return{ m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[1][0], m[1][1], m[1][2] };
						return{
							m.get<0, 0>(), m.get<0, 1>(), m.get<0, 2>(),
							m.get<2, 0>(), m.get<2, 1>(), m.get<2, 2>(),
							m.get<1, 0>(), m.get<1, 1>(), m.get<1, 2>() };
					case bits(0, 1, 0):			// (z x)
						//return{ m[2][0], m[2][1], m[2][2], m[1][0], m[1][1], m[1][2], m[0][0], m[0][1], m[0][2] };
						return{
							m.get<2, 0>(), m.get<2, 1>(), m.get<2, 2>(),
							m.get<1, 0>(), m.get<1, 1>(), m.get<1, 2>(),
							m.get<0, 0>(), m.get<0, 1>(), m.get<0, 2>() };
					case bits(1, 1, 0):			// (x y z)
						//return{ m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[0][0], m[0][1], m[0][2] };
						return{
							m.get<1, 0>(), m.get<1, 1>(), m.get<1, 2>(),
							m.get<2, 0>(), m.get<2, 1>(), m.get<2, 2>(),
							m.get<0, 0>(), m.get<0, 1>(), m.get<0, 2>() };
					case bits(0, 0, 1):			// (x y)
						//return{ m[1][0], m[1][1], m[1][2], m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2] };
						return{
							m.get<1, 0>(), m.get<1, 1>(), m.get<1, 2>(),
							m.get<0, 0>(), m.get<0, 1>(), m.get<0, 2>(),
							m.get<2, 0>(), m.get<2, 1>(), m.get<2, 2>() };
					default: // bits(1, 0, 1)	// (x z y)
						//return{ m[2][0], m[2][1], m[2][2], m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2] };
						return{
							m.get<2, 0>(), m.get<2, 1>(), m.get<2, 2>(),
							m.get<0, 0>(), m.get<0, 1>(), m.get<0, 2>(),
							m.get<1, 0>(), m.get<1, 1>(), m.get<1, 2>() };
					}
				#endif
				}

				JKL_GPU_EXECUTABLE constexpr int x_parity() const noexcept {
					return jkj::util::bit_at(internal_value, 0) ? -1 : 1;
				}
				JKL_GPU_EXECUTABLE constexpr int y_parity() const noexcept {
					return jkj::util::bit_at(internal_value, 1) ? -1 : 1;
				}
				JKL_GPU_EXECUTABLE constexpr int z_parity() const noexcept {
					return jkj::util::number_of_ones(internal_value) % 2 ? -1 : 1;
				}

				JKL_GPU_EXECUTABLE static constexpr int_vector vertex_to_vector(vertex v) noexcept {
					return{
						jkj::util::bit_at(v, 0) ? 1 : -1,
						jkj::util::bit_at(v, 1) ? 1 : -1,
						jkj::util::bit_at(v, 2) ? 1 : -1
					};
				}

				JKL_GPU_EXECUTABLE static constexpr vertex vector_to_vertex(int_vector const& v) noexcept {
					return jkj::util::bits(v.x() > 0, v.y() > 0, v.z() > 0);
				}
			};

			/* A workaround for NVCC constexpr bug */
			template <class To, class From, std::size_t N, std::size_t... I>
			constexpr std::array<To, N> convert_array_detail(std::array<From, N> const& arr, std::index_sequence<I...>) noexcept {
				return{ { arr[I]... } };
			}
			template <class To, class From, std::size_t N>
			constexpr std::array<To, N> convert_array(std::array<From, N> const& arr) noexcept {
				return convert_array_detail<To>(arr, std::make_index_sequence<N>{});
			}

			template <class object, std::uint8_t... i>
			constexpr auto generate_transform_row(element_base g, std::integer_sequence<std::uint8_t, i...>) noexcept {
				// I don't know why, but the following code is rejected by NVCC (not evaluated as a constant)
				//return std::array<object, sizeof...(i)>{ { g.calculate_action(object{ i })... } };
				return convert_array<object>(
					std::array<typename object::internal_type, sizeof...(i)>{ { g.calculate_action(object{ i })... } });
			}

			template <class object, std::size_t num_objects, std::uint8_t... g>
			constexpr std::array<std::array<object, num_objects>, sizeof...(g)>
				generate_transform_table(std::integer_sequence<std::uint8_t, g...>) noexcept
			{
				return{ {
						generate_transform_row<object>(g, std::make_integer_sequence<std::uint8_t, num_objects>{})...
					} };
			}

			template <std::uint8_t... g>
			constexpr std::array<element_base, sizeof...(g)>
				generate_inverse_table(std::integer_sequence<std::uint8_t, g...>) noexcept
			{
				return{ {
						element_base::calculate_from_representation(element_base{ g }.calculate_representation().inv())...
					} };
			}

			template <std::uint8_t... h>
			constexpr auto generate_multiplication_row(element_base g, std::integer_sequence<std::uint8_t, h...>) noexcept {
				// I don't know why, but the following code is rejected by NVCC (not evaluated as a constant)
				//return std::array<element_base, sizeof...(h)>{ { g.calculate_multiplication(element_base{ h })... } };
				return convert_array<element_base>(
					std::array<element_base::internal_type, sizeof...(h)>{ { g.calculate_multiplication(element_base{ h })... } });
			}

			template <std::uint8_t... g>
			constexpr std::array<std::array<element_base, 24>, sizeof...(g)>
				generate_multiplication_table(std::integer_sequence<std::uint8_t, g...>) noexcept
			{
				return{ {
						generate_multiplication_row(g, std::make_integer_sequence<std::uint8_t, 24>{})...
					} };
			}

			struct element : element_base {
				using element_base::element_base;
				JKL_GPU_EXECUTABLE constexpr operator internal_type() const noexcept {
					return static_cast<element_base const&>(*this);
				}

				JKL_GPU_EXECUTABLE static constexpr element calculate_from_representation(
					representation_type const& r) noexcept
				{
					return internal_type{ element_base::calculate_from_representation(r) };
				}

				JKL_GPU_EXECUTABLE constexpr element calculate_multiplication(element g) const noexcept {
					return internal_type{ element_base::calculate_multiplication(g) };
				}

				constexpr vertex operator*(vertex v) const noexcept {
					return vertex_transform_table[*this][v];
				}

				constexpr edge operator*(edge e) const noexcept {
					return edge_transform_table[*this][e];
				}

				constexpr element operator*(element g) const noexcept {
					return internal_type{ multiplication_table[*this][g] };
				}

				element& operator*=(element g) noexcept {
					return *this = *this * g;
				}

				constexpr element inv() const noexcept {
					return internal_type{ inverse_table[*this] };
				}

				friend constexpr element inv(element g) noexcept {
					return g.inv();
				}
				
				JKL_GPU_EXECUTABLE constexpr bool operator==(element g) const noexcept {
					return internal_type(*this) == internal_type(g);
				}

			protected:
				static constexpr auto vertex_transform_table =
					generate_transform_table<vertex, 8>(std::make_integer_sequence<std::uint8_t, 24>{});

				static constexpr auto edge_transform_table =
					generate_transform_table<edge, 12>(std::make_integer_sequence<std::uint8_t, 24>{});

				static constexpr auto inverse_table =
					generate_inverse_table(std::make_integer_sequence<std::uint8_t, 24>{});

				static constexpr auto multiplication_table =
					generate_multiplication_table(std::make_integer_sequence<std::uint8_t, 24>{});
			};
		}

		class Coxeter_O
		{
		public:
			using vertex = Coxeter_O_detail::vertex;
			using edge = Coxeter_O_detail::edge;
			using element = Coxeter_O_detail::element;
			using internal_type = element::internal_type;

			///      v5----e6----v7
			///     /|          /|         |z
			///   e3 |        e4 |         |
			///   /  e9       / e11        /------ y
			///  v6----e8----v8  |        /
			///  |   |       |   |        x
			///  |   v1----e5|---v3
			/// e10 /       e12 /
			///  | e1        | e2
			///  |/          |/
			///  v2----e7----v4
			///

			static constexpr vertex v1{ 0 };
			static constexpr vertex v2{ 1 };
			static constexpr vertex v3{ 2 };
			static constexpr vertex v4{ 3 };
			static constexpr vertex v5{ 4 };
			static constexpr vertex v6{ 5 };
			static constexpr vertex v7{ 6 };
			static constexpr vertex v8{ 7 };
			static constexpr vertex no_vertex{ 8 };

			static constexpr edge e1{ 0 };
			static constexpr edge e2{ 1 };
			static constexpr edge e3{ 2 };
			static constexpr edge e4{ 3 };
			static constexpr edge e5{ 4 };
			static constexpr edge e6{ 5 };
			static constexpr edge e7{ 6 };
			static constexpr edge e8{ 7 };
			static constexpr edge e9{ 8 };
			static constexpr edge e10{ 9 };
			static constexpr edge e11{ 10 };
			static constexpr edge e12{ 11 };
			static constexpr edge no_edge{ 12 };

			JKL_GPU_EXECUTABLE static constexpr edge join(vertex v, vertex w) noexcept {
				return Coxeter_O_detail::join(v, w);
			}
			JKL_GPU_EXECUTABLE static constexpr std::size_t order() noexcept {
				return 24;
			}

			static constexpr element Id{ 0 };				// identity
			static constexpr element _16_25_38_47{ 1 };		// 180 deg rotation along the y-axis
			static constexpr element _17_28_35_46{ 2 };		// 180 deg rotation along the x-axis
			static constexpr element _14_23_58_67{ 3 };		// 180 deg rotation along the z-axis
			static constexpr element _1375_2486{ 4 };		// 90 deg rotation along the x-axis
			static constexpr element _12_36_45_78{ 5 };		// 180 deg rotation along the axis e1-e4
			static constexpr element _1573_2684{ 6 };		// 270 deg rotation along the x-axis
			static constexpr element _18_27_34_56{ 7 };		// 180 deg rotation along the axis e2-e3
			static constexpr element _1265_3487{ 8 };		// 270 deg rotation along the y-axis
			static constexpr element _1562_3784{ 9 };		// 90 deg rotation along the y-axis
			static constexpr element _13_27_45_68{ 10 };	// 180 deg rotation along the axis e5-e8
			static constexpr element _18_24_36_57{ 11 };	// 180 deg rotation along the axis e6-e7
			static constexpr element _253_467{ 12 };		// 120 deg rotation along the axis v8->v1
			static constexpr element _176_238{ 13 };		// 120 deg rotation along the axis v5->v4
			static constexpr element _147_285{ 14 };		// 120 deg rotation along the axis v3->v6
			static constexpr element _164_358{ 15 };		// 120 deg rotation along the axis v2->v7
			static constexpr element _15_27_36_48{ 16 };	// 180 deg rotation along the axis e9-e12
			static constexpr element _1342_5786{ 17 };		// 270 deg rotation along the z-axis
			static constexpr element _1243_5687{ 18 };		// 90 deg rotation along the z-axis
			static constexpr element _18_26_37_45{ 19 };	// 180 deg rotation along the axis e10-e11
			static constexpr element _235_476{ 20 };		// 120 deg rotation along the axis v1->v8
			static constexpr element _146_385{ 21 };		// 120 deg rotation along the axis v7->v2
			static constexpr element _167_283{ 22 };		// 120 deg rotation along the axis v4->v5
			static constexpr element _174_258{ 23 };		// 120 deg rotation along the axis v6->v3
			static constexpr element no_element{ 24 };

			class iterator : public std::iterator<std::input_iterator_tag, 
				element, std::make_signed_t<internal_type>, internal_type, element> {
				internal_type index;
			public:
				constexpr iterator(internal_type index) noexcept : index{ index } {}

				constexpr bool operator==(iterator that) const noexcept { return that.index == index; }
				constexpr bool operator!=(iterator that) const noexcept { return that.index != index; }
				constexpr element operator*() const noexcept { return element(index); }
				iterator& operator++() noexcept { ++index; return *this; }
				iterator operator++(int) noexcept { return iterator{ index++ }; }
				iterator& operator--() noexcept { --index; return *this; }
				iterator operator--(int) noexcept { return iterator{ index-- }; }
			};

			static constexpr iterator begin() noexcept { return iterator(0); }
			static constexpr iterator end() noexcept { return iterator{ internal_type(order()) }; }
			static constexpr iterator rbegin() noexcept { return iterator{ internal_type(order() - 1) }; }
			static constexpr iterator rend() noexcept { return iterator{ internal_type(-1) }; }
		};
	}
}
