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
#include "../cache.h"

namespace jkl {
	namespace gui {
		// Input panel for integers
		// [label]: [textbox]
		// The textbox is assumed to take real numbers as input
		// The condition for accepting the input is specified as a constructor argument
		class number_input : public nana::panel<true>
		{
		public:
			number_input(nana::window const& parent, std::string const& label, float default_value,
				std::function<bool(float)> predicate = [](auto&&) { return true; });

			// Two widely used predicates
			struct nonnegative {
				bool operator()(float x) const noexcept {
					return x >= 0;
				}
			};
			struct positive {
				bool operator()(float x) const noexcept {
					return x > 0;
				}
			};

			decltype(auto) events() const {
				return m_text.events();
			}

			void enabled(bool e) {
				m_text.enabled(e);
			}
			bool enabled() const {
				return m_text.enabled();
			}

			decltype(auto) caption(std::string const& c) {
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
			float get();
			operator float() {
				return get();
			}

			void set(float new_value);
			auto& operator=(float new_value) {
				set(new_value);
				return *this;
			}

		private:
			nana::place					m_place;
			nana::label					m_label;
			nana::textbox				m_text;
			cache<float>				m_cache;
			std::function<bool(float)>	m_predicate;
			bool						m_invalidate_signal = true;
		};
	}
}