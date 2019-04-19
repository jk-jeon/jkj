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
#include <nana/gui/widgets/textbox.hpp>
#include "../numerical_lie_group.h"
#include "../cache.h"

namespace jkj {
	namespace gui {
		// Input panel for 2-dimensional coordinates
		// [label]: ([textbox],[textbox])
		// The textboxes are assumed to take real numbers as inputs
		// The condition for accepting the input is specified as a constructor argument
		class coordinate_input : public nana::panel<true>
		{
		public:
			coordinate_input(nana::window const& parent, std::string const& label,
				math::R2_elmtf default_value, unsigned int paran_comma_label_width = 10,
				std::function<bool(math::R2_elmtf const&)> predicate = [](auto&&) { return true; });

			decltype(auto) events_x() const {
				return m_text_x.events();
			}
			decltype(auto) events_y() const {
				return m_text_y.events();
			}

			void enabled(bool e) {
				m_text_x.enabled(e);
				m_text_y.enabled(e);
			}
			bool enabled() const {
				return m_text_x.enabled() && m_text_y.enabled();
			}

			decltype(auto) caption_x(std::string& c) {
				return m_text_x.caption(c);
			}
			decltype(auto) caption_x() const {
				return m_text_x.caption();
			}
			decltype(auto) caption_y(std::string& c) {
				return m_text_y.caption(c);
			}
			decltype(auto) caption_y() const {
				return m_text_y.caption();
			}
			
			decltype(auto) bgcolor_x(nana::color const& c) {
				return m_text_x.bgcolor(c);
			}
			decltype(auto) bgcolor_x() const {
				return m_text_x.bgcolor();
			}
			decltype(auto) bgcolor_y(nana::color const& c) {
				return m_text_y.bgcolor(c);
			}
			decltype(auto) bgcolor_y() const {
				return m_text_y.bgcolor();
			}
			void bgcolor(nana::color const& c) {
				m_text_x.bgcolor(c);
				m_text_y.bgcolor(c);
			}

			// get() is not const, as it may change the content of the input
			math::R2_elmtf get();
			operator math::R2_elmtf() {
				return get();
			}

			void set(math::R2_elmtf const& new_value);
			auto& operator=(math::R2_elmtf const& new_value) {
				set(new_value);
				return *this;
			}

		private:
			nana::place												m_place;
			nana::label												m_label;
			nana::label												m_openning_paran;
			nana::label												m_closing_paran;
			nana::label												m_comma;
			nana::textbox											m_text_x;
			nana::textbox											m_text_y;
			cache<math::R2_elmtf>									m_cache;
			std::function<bool(math::R2_elmtf const&)>				m_predicate;
			bool													m_invalidate_signal = true;
		};
	}
}