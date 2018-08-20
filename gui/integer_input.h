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
#include <nana/gui/widgets/panel.hpp>
#include <nana/gui/place.hpp>
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/spinbox.hpp>
#include "../cache.h"

namespace jkl {
	namespace gui {
		// Input panel for integers
		// [label]: [spinbox]
		// The spinbox is assumed to take integers as input
		// The range of the spinbox is specified as constructor arguments
		class integer_input : public nana::panel<true>
		{
		public:
			integer_input(nana::window const& parent, std::string const& label,
				int default_value, int from, int to);

			decltype(auto) events() const {
				return m_text.events();
			}

			void enabled(bool e) {
				m_text.enabled(e);
			}
			bool enabled() const {
				return m_text.enabled();
			}

			decltype(auto) caption(std::string& c) {
				return m_text.caption(c);
			}
			decltype(auto) caption() const {
				return m_text.caption();
			}

			decltype(auto) bgcolor(nana::color const& c) {
				return m_text.bgcolor(c);
			}
			decltype(auto) bgcolor() const {
				return m_text.bgcolor();
			}

			// get() is not const, as it may change the content of the input
			int get();
			operator int() {
				return get();
			}

			void set(int new_value);
			auto& operator=(int new_value) {
				set(new_value);
				return *this;
			}

		private:
			nana::place					m_place;
			nana::label					m_label;
			nana::spinbox				m_text;
			cache<int>					m_cache;
		};
	}
}