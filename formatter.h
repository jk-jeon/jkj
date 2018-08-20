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
#include <iostream>
#include <iterator>
#include "tmp/is_convertible.h"
#include "tmp/identity.h"

namespace jkl {
	// Object reference together with a formatting information
	// The stream output operator (operator<<) is overloaded for this template.
	// If the ADL chooses that overload, the print() function of the formatter is called.
	// A Formatter is a class satisfying the following conditions:
	//   1. Given an instance fmt of type Formatter, fmt(x) will produce
	//      formatted_object<Object, Formatter>,
	//      whose obj member refers to x and formatter member refers to fmt,
	//      whenever Object is a "printable type" defined by Formmater.
	//      Checking if a type is printable is not required.
	//   2. Whenever x is of a "printable type" defined by Formatter,
	//      fmt.print(out, x) will print the object x to the ostream out.
	template <class Object, class Formatter>
	struct formatted_object {
		Object const&		obj;
		Formatter const&	formatter;

		template <class CharType, class CharTraits>
		friend auto& operator<<(std::basic_ostream<CharType, CharTraits>& out,
			formatted_object const& fmt)
		{
			return fmt.formatter.print(out, fmt.obj);
		}
	};

	// [Example usage]
	// std::vector<int> v{ 1, 2, 3, 4, 5 };
	// std::cout << jkl::make_range_formatter()(v);
	//
	// [Output]
	// [1, 2, 3, 4, 5]

	// A formatter type that can print ranges
	template <class String, class ElementFormatter = tmp::identity_functor>
	struct range_formatter : private ElementFormatter {
		String		opening;
		String		closing;
		String		separator;
		
	private:
		// Some helpers to reduce duplication
		template <class OpeningArg, class ClosingArg, class SeparatorArg, class... ElementFormatterArgs>
		static constexpr bool is_convertible_v =
			tmp::is_convertible_v<String, OpeningArg> &&
			tmp::is_convertible_v<String, ClosingArg> &&
			tmp::is_convertible_v<String, SeparatorArg> &&
			tmp::is_convertible_v<ElementFormatter, ElementFormatterArgs...>;

		template <class OpeningArg, class ClosingArg, class SeparatorArg, class... ElementFormatterArgs>
		static constexpr bool is_nothrow_constructible_v =
			std::is_nothrow_constructible_v<String, OpeningArg> &&
			std::is_nothrow_constructible_v<String, ClosingArg> &&
			std::is_nothrow_constructible_v<String, SeparatorArg> &&
			std::is_nothrow_constructible_v<ElementFormatter, ElementFormatterArgs...>;

	public:
		// Construct ElementFormatter with additional arguments
		template <
			class OpeningArg, class ClosingArg, class SeparatorArg, class... ElementFormatterArgs,
			class = std::enable_if_t<is_convertible_v<OpeningArg, ClosingArg, SeparatorArg, ElementFormatterArgs...>>
		>
		constexpr range_formatter(
			OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator,
			ElementFormatterArgs&&... element_formatter_args)
			noexcept(is_nothrow_constructible_v<OpeningArg, ClosingArg, SeparatorArg, ElementFormatterArgs...>) :
			ElementFormatter(std::forward<ElementFormatterArgs>(element_formatter_args)...),
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		template <
			class OpeningArg, class ClosingArg, class SeparatorArg, class... ElementFormatterArgs,
			class = std::enable_if_t<!is_convertible_v<OpeningArg, ClosingArg, SeparatorArg, ElementFormatterArgs...>>,
			class = void
		>
		explicit constexpr range_formatter(
				OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator,
				ElementFormatterArgs&&... element_formatter_args)
			noexcept(is_nothrow_constructible_v<OpeningArg, ClosingArg, SeparatorArg, ElementFormatterArgs...>) :
			ElementFormatter(std::forward<ElementFormatterArgs>(element_formatter_args)...),
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		template <class CharType, class CharTraits, class Range>
		auto& print(std::basic_ostream<CharType, CharTraits>& out, Range const& range) const
		{
			auto next_itr = std::cbegin(range);
			auto const sentinel = std::cend(range);

			out << opening;
			if( next_itr != sentinel ) {
				auto itr = next_itr++;
				for( ; next_itr != sentinel; itr = next_itr++ )
					out << static_cast<ElementFormatter const&>(*this)(*itr) << separator;
				out << static_cast<ElementFormatter const&>(*this)(*itr) << closing;
			} else
				out << closing;

			return out;
		}

		template <class Object>
		FORCEINLINE formatted_object<Object, range_formatter> operator()(Object const& obj) const {
			return{ obj, *this };
		}
	};

	template <class String = char const*,
		class ElementFormatter = tmp::identity_functor,
		class OpeningString = char const* const&,
		class ClosingString = char const* const&,
		class SeparatorString = char const* const&
	>
	FORCEINLINE auto make_range_formatter(
			OpeningString&& opening = "[",
			ClosingString&& closing = "]",
			SeparatorString&& separator = ", ",
			ElementFormatter&& element_formatter = {})
	{
		return range_formatter<String, ElementFormatter>{
			std::forward<OpeningString>(opening),
			std::forward<ClosingString>(closing),
			std::forward<SeparatorString>(separator),
			std::forward<ElementFormatter>(element_formatter)
		};
	}

	// A formatter type that can print std::pair
	template <class String,
		class FirstFormatter = tmp::identity_functor,
		class SecondFormatter = tmp::identity_functor
	>
	struct pair_formatter : private std::pair<FirstFormatter, SecondFormatter> {
		String		opening;
		String		closing;
		String		separator;

	private:
		// Some helpers to improve readability
		template <class OpeningArg, class ClosingArg, class SeparatorArg>
		static constexpr bool is_convertible_default_v =
			tmp::is_convertible_v<String, OpeningArg> &&
			tmp::is_convertible_v<String, ClosingArg> &&
			tmp::is_convertible_v<String, SeparatorArg> &&
			tmp::is_convertible_v<FirstFormatter> &&
			tmp::is_convertible_v<SecondFormatter>;

		template <class OpeningArg, class ClosingArg, class SeparatorArg,
			class FirstFormatterArg, class SecondFormatterArg>
		static constexpr bool is_convertible_v =
			tmp::is_convertible_v<String, OpeningArg> &&
			tmp::is_convertible_v<String, ClosingArg> &&
			tmp::is_convertible_v<String, SeparatorArg> &&
			tmp::is_convertible_v<FirstFormatter, FirstFormatterArg> &&
			tmp::is_convertible_v<SecondFormatter, SecondFormatterArg>;

		template <class OpeningArg, class ClosingArg, class SeparatorArg>
		static constexpr bool is_nothrow_constructible_default_v =
			std::is_nothrow_constructible_v<String, OpeningArg> &&
			std::is_nothrow_constructible_v<String, ClosingArg> &&
			std::is_nothrow_constructible_v<String, SeparatorArg> &&
			std::is_nothrow_constructible_v<FirstFormatter> &&
			std::is_nothrow_constructible_v<SecondFormatter>;

		template <class OpeningArg, class ClosingArg, class SeparatorArg,
			class FirstFormatterArg, class SecondFormatterArg>
		static constexpr bool is_nothrow_constructible_v =
			std::is_nothrow_constructible_v<String, OpeningArg> &&
			std::is_nothrow_constructible_v<String, ClosingArg> &&
			std::is_nothrow_constructible_v<String, SeparatorArg> &&
			std::is_nothrow_constructible_v<FirstFormatter, FirstFormatterArg> &&
			std::is_nothrow_constructible_v<SecondFormatter, SecondFormatterArg>;

	public:
		// Default construct FirstFormatter & SecondFormatter
		template <class OpeningArg, class ClosingArg, class SeparatorArg,
			class = std::enable_if_t<is_convertible_default_v<OpeningArg, ClosingArg, SeparatorArg>>
		>
		constexpr pair_formatter(
			OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator)
			noexcept(is_nothrow_constructible_default_v<OpeningArg, ClosingArg, SeparatorArg>) :
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		template <class OpeningArg, class ClosingArg, class SeparatorArg,
			class = std::enable_if_t<!is_convertible_default_v<OpeningArg, ClosingArg, SeparatorArg>>,
			class = void
		>
		explicit constexpr pair_formatter(
			OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator)
			noexcept(is_nothrow_constructible_default_v<OpeningArg, ClosingArg, SeparatorArg>) :
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		// Copy/move construct FirstFormatter & SecondFormatter
		template <class OpeningArg, class ClosingArg, class SeparatorArg, 
			class FirstFormatterArg, class SecondFormatterArg,
			class = std::enable_if_t<is_convertible_v<OpeningArg, ClosingArg, SeparatorArg,
			FirstFormatterArg, SecondFormatterArg>>
		>
		constexpr pair_formatter(
			OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator,
			FirstFormatterArg&& first_formmater_arg, SecondFormatterArg&& second_formmater_arg)
			noexcept(is_nothrow_constructible_v<OpeningArg, ClosingArg, SeparatorArg,
				FirstFormatterArg, SecondFormatterArg>) :
			std::pair<FirstFormatter, SecondFormatter>(
				std::forward<FirstFormatterArg>(first_formmater_arg),
				std::forward<SecondFormatterArg>(second_formmater_arg)),
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		template <class OpeningArg, class ClosingArg, class SeparatorArg,
			class FirstFormatterArg, class SecondFormatterArg,
			class = std::enable_if_t<!is_convertible_v<OpeningArg, ClosingArg, SeparatorArg,
			FirstFormatterArg, SecondFormatterArg>>, class = void
		>
		explicit constexpr pair_formatter(
			OpeningArg&& opening, ClosingArg&& closing, SeparatorArg&& separator,
			FirstFormatterArg&& first_formmater_arg, SecondFormatterArg&& second_formmater_arg)
			noexcept(is_nothrow_constructible_v<OpeningArg, ClosingArg, SeparatorArg,
			FirstFormatterArg, SecondFormatterArg>) :
			std::pair<FirstFormatter, SecondFormatter>(
				std::forward<FirstFormatterArg>(first_formmater_arg),
				std::forward<SecondFormatterArg>(second_formmater_arg)),
			opening(std::forward<OpeningArg>(opening)),
			closing(std::forward<ClosingArg>(closing)),
			separator(std::forward<SeparatorArg>(separator)) {}

		template <class CharType, class CharTraits, class Pair>
		auto& print(std::basic_ostream<CharType, CharTraits>& out, Pair const& p) const
		{
			auto const& formatter_pair = static_cast<std::pair<FirstFormatter, SecondFormatter> const&>(*this);

			out << opening
				<< formatter_pair.first(p.first)
				<< separator
				<< formatter_pair.second(p.second)
				<< closing;

			return out;
		}

		template <class Object>
		FORCEINLINE formatted_object<Object, pair_formatter> operator()(Object const& obj) const {
			return{ obj, *this };
		}
	};

	template <class String = char const*,
		class FirstFormatter = tmp::identity_functor,
		class SecondFormatter = tmp::identity_functor,
		class OpeningString = char const* const&,
		class ClosingString = char const* const&,
		class SeparatorString = char const* const&
	>
	FORCEINLINE auto make_pair_formatter(
		OpeningString&& opening = "[",
		ClosingString&& closing = "]",
		SeparatorString&& separator = ", ",
		FirstFormatter&& first_formatter = {},
		SecondFormatter&& second_formatter = {})
	{
		return pair_formatter<String, FirstFormatter, SecondFormatter>{
			std::forward<OpeningString>(opening),
			std::forward<ClosingString>(closing),
			std::forward<SeparatorString>(separator),
			std::forward<FirstFormatter>(first_formatter),
			std::forward<SecondFormatter>(second_formatter)
		};
	}
}