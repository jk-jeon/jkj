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
#include "draw_line.h"
#include "../../tmp/remove_cvref.h"
#include "../../dual_number.h"
#include "../../pseudo_ptr.h"

namespace jkl {
	namespace gui {
		// NOTE: here the range is inclusive; last is NOT excluded!
		template <class ValueType = double, class PixelBuffer, class ControlPointIterator, class BorderWidth>
		void draw_curve(PixelBuffer& g, ControlPointIterator first, ControlPointIterator last,
			BorderWidth linewidth, nana::color const& line_color)
		{
			if( first == last ) {
				draw_line(g, *first, *first, linewidth, line_color);
				return;
			}

			auto to = first;
			auto from = to++;
			for( ; from != last; from = to++ ) {
				draw_line(g, *from, *to, linewidth, line_color);
			}
		}

		template <class ValueType, class ParameterizedCurve>
		class uniform_interval_curve_iterator {
			std::size_t counter;
			ValueType t;
			ValueType dt;
			ParameterizedCurve curve;

		public:
			using iterator_category = std::random_access_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using reference = decltype(curve(std::declval<ValueType>()));
			using value_type = tmp::remove_cvref_t<reference>;
			using pointer = pseudo_ptr<reference>;

			uniform_interval_curve_iterator() = default;

			uniform_interval_curve_iterator(std::size_t counter, ValueType t,
				ValueType dt, ParameterizedCurve const& curve) :
				counter{ counter }, t{ t }, dt{ dt }, curve{ curve } {}

			reference operator*() const {
				return curve(t);
			}

			reference operator[](difference_type n) const {
				return curve(t + n * dt);
			}

			pointer operator->() const {
				return pointer{ **this };
			}

			uniform_interval_curve_iterator& operator++() {
				++counter;
				t += dt;
				return *this;
			}
			uniform_interval_curve_iterator operator++(int) {
				auto itr = *this;
				++*this;
				return itr;
			}

			uniform_interval_curve_iterator& operator--() {
				--counter;
				t -= dt;
				return *this;
			}
			uniform_interval_curve_iterator operator--(int) {
				auto itr = *this;
				--*this;
				return itr;
			}
			
			uniform_interval_curve_iterator& operator+=(difference_type n) {
				counter += n;
				t += n * dt;
				return *this;
			}
			uniform_interval_curve_iterator operator+(difference_type n) const {
				auto itr = *this;
				itr += n;
				return itr;
			}

			uniform_interval_curve_iterator& operator-=(difference_type n) {
				counter -= n;
				t -= n * dt;
				return *this;
			}
			uniform_interval_curve_iterator operator-(difference_type n) const {
				auto itr = *this;
				itr -= n;
				return itr;
			}

			friend uniform_interval_curve_iterator operator+(difference_type n,
				uniform_interval_curve_iterator const& itr) {
				return itr + n;
			}

			friend difference_type operator-(
				uniform_interval_curve_iterator const& itr1,
				uniform_interval_curve_iterator const& itr2) {
				return itr1.counter - itr2.counter;
			}

			bool operator==(uniform_interval_curve_iterator const& itr) const {
				return counter == itr.counter;
			}
			bool operator!=(uniform_interval_curve_iterator const& itr) const {
				return counter != itr.counter;
			}
			bool operator<(uniform_interval_curve_iterator const& itr) const {
				return counter < itr.counter;
			}
			bool operator>(uniform_interval_curve_iterator const& itr) const {
				return counter > itr.counter;
			}
			bool operator<=(uniform_interval_curve_iterator const& itr) const {
				return counter <= itr.counter;
			}
			bool operator>=(uniform_interval_curve_iterator const& itr) const {
				return counter >= itr.counter;
			}
		};

		template <class ValueType = double, class ParameterizedCurve>
		std::pair<uniform_interval_curve_iterator<ValueType, ParameterizedCurve>,
			uniform_interval_curve_iterator<ValueType, ParameterizedCurve>>
			make_uniform_interval_curve_range(ValueType from, ValueType to, std::size_t divisions,
				ParameterizedCurve&& curve)
		{
			auto dt = (to - from) / divisions;
			return{
				{ 0, from, dt, curve },
				{ divisions, to, dt, curve }
			};
		}


		template <class ValueType, class ParameterizedCurve>
		class uniform_length_curve_iterator {
			std::size_t counter;
			ValueType t;
			ValueType ds;
			jkl::math::R2_elmt<ValueType> point;
			jkl::math::R2_elmt<ValueType> tangent_vector;
			ParameterizedCurve curve;
			ValueType t_final;

			void evaluate() {
				auto combined_calc = curve(jkl::math::dual_number<ValueType>{ t, ValueType(1) });
				point = { combined_calc.x().prim, combined_calc.y().prim };
				tangent_vector = { combined_calc.x().dual, combined_calc.y().dual };
			}

		public:
			using iterator_category = std::input_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = jkl::math::R2_elmt<ValueType>;
			using reference = value_type const&;
			using pointer = value_type const*;

			uniform_length_curve_iterator() = default;

			uniform_length_curve_iterator(std::size_t counter, ValueType t,
				ValueType ds, ParameterizedCurve const& curve, ValueType t_final) :
				counter{ counter }, t{ t }, ds{ ds }, curve{ curve }, t_final{ t_final }
			{
				evaluate();
			}

			reference operator*() const {
				return point;
			}

			pointer operator->() const {
				return &point;
			}

			uniform_length_curve_iterator& operator++() {
				++counter;
				// 0.01 is a regularizer
				t += ds / (tangent_vector.norm() + ValueType(0.01));

				if( t > t_final )
					t = t_final;
				
				evaluate();

				return *this;
			}
			uniform_length_curve_iterator operator++(int) {
				auto itr = *this;
				++*this;
				return itr;
			}			

			bool operator==(uniform_length_curve_iterator const& itr) const {
				if( t >= t_final && itr.t >= itr.t_final )
					return true;
				else if( t >= t_final )
					return false;
				else if( itr.t >= itr.t_final )
					return false;
				else
					return counter == itr.counter;
			}

			bool operator!=(uniform_length_curve_iterator const& itr) const {
				return !(*this == itr);
			}
		};

		template <class ValueType = double, class ParameterizedCurve>
		std::pair<uniform_length_curve_iterator<ValueType, ParameterizedCurve>,
			uniform_length_curve_iterator<ValueType, ParameterizedCurve>>
			make_uniform_length_curve_range(ValueType from, ValueType to, ValueType ds,
				ParameterizedCurve&& curve)
		{
			return{
				{ 0, from, ds, curve, to },
				{ 0, to, ds, curve, to }
			};
		}


		template <class ValueType, class ParameterizedCurve>
		class uniform_angle_curve_iterator {
			std::size_t counter;
			ValueType t;
			ValueType dtheta;
			jkl::math::R2_elmt<ValueType> point;
			jkl::math::R2_elmt<ValueType> tangent_vector;
			jkl::math::R2_elmt<ValueType> second_derivative;
			ParameterizedCurve curve;
			ValueType t_final;

			void evaluate() {
				using dual_number_type = jkl::math::dual_number<ValueType>;
				using double_dual_number_type = jkl::math::dual_number<dual_number_type>;

				auto combined_calc = curve(
					double_dual_number_type{ { t, ValueType(1) }, { ValueType(1), ValueType(0) } });

				point = { combined_calc.x().prim.prim, combined_calc.y().prim.prim };
				tangent_vector = { combined_calc.x().prim.dual, combined_calc.y().prim.dual };
				second_derivative = { combined_calc.x().dual.dual, combined_calc.y().dual.dual };
			}

		public:
			using iterator_category = std::input_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = jkl::math::R2_elmt<ValueType>;
			using reference = value_type const&;
			using pointer = value_type const*;

			uniform_angle_curve_iterator() = default;

			uniform_angle_curve_iterator(std::size_t counter, ValueType t,
				ValueType dtheta, ParameterizedCurve const& curve, ValueType t_final) :
				counter{ counter }, t{ t }, dtheta{ dtheta }, curve{ curve }, t_final{ t_final }
			{
				evaluate();
			}

			reference operator*() const {
				return point;
			}

			pointer operator->() const {
				return &point;
			}

			uniform_angle_curve_iterator& operator++() {
				++counter;
				// 0.01 is a regularizer
				using std::abs;
				t += dtheta * (tangent_vector.normsq() + ValueType(0.01)) /
					(abs(signed_area(tangent_vector, second_derivative)) + ValueType(0.01));

				if( t > t_final )
					t = t_final;

				evaluate();

				return *this;
			}
			uniform_angle_curve_iterator operator++(int) {
				auto itr = *this;
				++*this;
				return itr;
			}

			bool operator==(uniform_angle_curve_iterator const& itr) const {
				if( t >= t_final && itr.t >= itr.t_final )
					return true;
				else if( t >= t_final )
					return false;
				else if( itr.t >= itr.t_final )
					return false;
				else
					return counter == itr.counter;
			}

			bool operator!=(uniform_angle_curve_iterator const& itr) const {
				return !(*this == itr);
			}
		};

		template <class ValueType = double, class ParameterizedCurve>
		std::pair<uniform_angle_curve_iterator<ValueType, ParameterizedCurve>,
			uniform_angle_curve_iterator<ValueType, ParameterizedCurve>>
			make_uniform_angle_curve_range(ValueType from, ValueType to, ValueType dtheta,
				ParameterizedCurve&& curve)
		{
			return{
				{ 0, from, dtheta, curve, to },
				{ 0, to, dtheta, curve, to }
			};
		}
	}
}