/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2017 Junekey Jeon                                                                   ///
/// Permission is hereby granted, free of chargse, to any person obtaining a copy of this software     ///
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

// To be improved:
// - Reporting of error location is not very correct right now
// - Parsing requires a complete string; it may be better to support istream as well as ostream

#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include "tmp.h"
#include "bit_twiddling.h"
#include "mixed_precision_number.h"
#include "unicode.h"

namespace jkl {
	namespace json {
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// To specify rules for generating string from JSON entries, following types are defined
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Indentation policy for output
		enum class indent_policy : std::uint8_t {
			no_indent = 0,
			_1 = 1,
			space = _1,
			single_space = _1,
			_2 = 2, 
			double_space = _2,
			_3 = 3,
			triple_space = _3,
			_4 = 4,
			four_spaces = _4,
			_5 = 5,
			five_spaces = _5,
			_6 = 6,
			six_spaces = _6,
			_7 = 7,
			seven_spaces = _7,
			_8 = 8,
			eight_spaces = _8,
			tab, 
			double_tab
		};

		// End-Of-Line policy for output
		enum class eol_policy : std::uint8_t {
			no_eol,
			LF,
			CR,
			CRLF,
			LFCR,
			UNIX = LF,
			ClassicMac = CR,
			Windows = CRLF,
		};

		// Upper case or lower case when showing hexadecimal number
		enum class hex_policy : int {
			lower_case,
			upper_case
		};

		// Aggregation of the aboves
		struct text_format {
			indent_policy	indentation;
			eol_policy		eol;
			hex_policy		hex;
			constexpr text_format(indent_policy indentation = indent_policy::tab,
				eol_policy eol = eol_policy::LF,
				hex_policy hex = hex_policy::lower_case) noexcept
				: indentation(indentation), eol(eol), hex(hex) {}
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Type id of JSON entry
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		enum class type : std::uint8_t {
			null,
			boolean,
			number,
			string,
			array,
			object,
			general
		};
		static constexpr char const* json_type_name[] = {
			"null", "boolean", "number", "string", "array", "object"
		};
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Exception classes
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		class json_exception : virtual public std::runtime_error
		{
		public:
			template <class MessageType>
			json_exception(MessageType const& message) : runtime_error(message) {}
		};

		class text_parsing_error : virtual public json_exception {
		public:
			enum error_case {
				empty_string,
				invalid_char,
				incomplete_word,
				ill_formed_number,
				quote_not_closed,
				invalid_escape_syntax,
				invalid_utf16_codepoint,
				broken_utf8_encoding,
				ill_formed_array,
				ill_foremd_object
			} reason;
			char const* text_begin_parsed;
			char const* error_location;

			template <class MessageType>
			text_parsing_error(MessageType const& message, error_case reason,
				char const* text_begin_parsed, char const* error_location)
				: runtime_error(message), json_exception(message), reason(reason), 
				text_begin_parsed(text_begin_parsed), error_location(error_location) {}
		};

		class type_mismatch : virtual public json_exception {
		public:
			type entry_type;
			type queried_type;

			template <class MessageType>
			type_mismatch(MessageType const& message, type entry_type, type queried_type)
				: runtime_error(message), json_exception(message), entry_type(entry_type), queried_type(queried_type) {}
		};

		template <class JsonEntry>
		class entry_not_found : virtual public json_exception {
		public:
			JsonEntry& entry;
			typename std::decay_t<JsonEntry>::string_type queried_key;

			template <class MessageType, class KeyType>
			entry_not_found(MessageType const& message, JsonEntry& entry, KeyType&& queried_key)
				: runtime_error(message), json_exception(message),
				entry{ entry }, queried_key{ std::forward<KeyType>(queried_key) } {}
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Abstract polymorphic base class for general JSON entry
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		template <type tid_, class Policy>
		class json_entry;

		namespace detail {
			// Check if the Policy support stream input/output
			template <class Policy>
			struct support_stream {
			private:
				template <class = typename Policy::istream_type>
				static constexpr bool check_istream(int) { return true; }
				static constexpr bool check_istream(float) { return false; }
				template <class = typename Policy::ostream_type>
				static constexpr bool check_ostream(int) { return true; }
				static constexpr bool check_ostream(float) { return false; }
			public:
				static constexpr bool istream = check_istream(0);
				static constexpr bool ostream = check_ostream(0);
			};

			// If supported, json_entry must provide the virtual function to_stream
			template <class Policy, bool support = support_stream<Policy>::ostream>
			struct to_stream_interface {
				using ostream_type = typename Policy::ostream_type;
				virtual ostream_type& to_stream(ostream_type& out, std::size_t indentation_level, text_format format) const = 0;
			};

			template <class Policy>
			struct to_stream_interface<Policy, false> {
				struct ostream_type {
					template <class T>
					ostream_type& operator<<(T&&) { return *this; }
				};

				template <class OstreamType>
				OstreamType& to_stream(OstreamType& out, std::size_t, text_format) const {
					static_assert(jkl::tmp::assert_helper<Policy>::value, 
						"The specified JSON policy class does not support stream output.");
					return out;
				}
			};
		}

		template <class Policy>
		class json_entry<type::general, Policy> : public detail::to_stream_interface<Policy>
		{
		protected:
			template <type requested>
			static void bad_json_get_request(type real) {
				throw type_mismatch(std::string("Attempt to get a JSON entry of type ") +
					json_type_name[(int)requested] + " from a JSON entry of type " +
					json_type_name[(int)real] + '!', real, requested);
			}

		public:
			using policy_type = Policy;
			using ostream_type = typename detail::to_stream_interface<Policy>::ostream_type;

			virtual ~json_entry() = default;
			virtual type tid() const noexcept = 0;
			virtual typename Policy::string_type to_string(std::size_t indentation_level = 0,
				text_format format = text_format()) const = 0;
			typename Policy::string_type to_string(text_format format) const {
				return to_string(0, format);
			}
			using detail::to_stream_interface<Policy>::to_stream;
			ostream_type& to_stream(ostream_type& out, text_format format) const {
				return to_stream(out, 0, format);
			}

			// Convert the entry to a specific JSON entry type
			template <type tid_>
			auto& as() {
				if( tid() != tid_ && tid_ != type::general )
					bad_json_get_request<tid_>(tid());
				return static_cast<json_entry<tid_, Policy>&>(*this);
			}

			template <type tid_>
			auto& as() const {
				if( tid() != tid_ && tid_ != type::general )
					bad_json_get_request<tid_>(tid());
				return static_cast<json_entry<tid_, Policy> const&>(*this);
			}

			template <class TargetEntry>
			auto& as() {
				if( tid() != TargetEntry::entry_type_id )
					bad_json_get_request<TargetEntry::entry_type_id>(tid());
				return static_cast<TargetEntry&>(*this);
			}

			template <class TargetEntry>
			auto& as() const {
				if( tid() != TargetEntry::entry_type_id )
					bad_json_get_request<TargetEntry::entry_type_id>(tid());
				return static_cast<TargetEntry const&>(*this);
			}

			// Unsafe (no-checking) versions of above
			template <type tid_>
			auto& unsafe_as() {
				return static_cast<json_entry<tid_, Policy>&>(*this);
			}
			template <type tid_>
			auto& unsafe_as() const {
				return static_cast<json_entry<tid_, Policy> const&>(*this);
			}
			template <class TargetEntry>
			auto& unsafe_as() {
				return static_cast<TargetEntry&>(*this);
			}
			template <class TargetEntry>
			auto& unsafe_as() const {
				return static_cast<TargetEntry const&>(*this);
			}
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// JSON Policy classes
		/// These classes define how contents in JSON entries are treated
		///
		/// Following types must be defined:
		///   -- entry_type					: the type of abstract polymorphic base class; usually set to json_entry<type::general, policy_class_name>
		///                                   this class must have methods tid() and to_string()
		///   -- entry_pointer				: the type actually being stored; the library performs no direct pointer management, 
		///									  so the user must provide the way of doing it; usually std::shared_ptr suffices
		///   -- entry_const_pointer		: const version of the above
		///
		/// To specify the actual type to hold a given JSON entry type, the user must write 
		/// "ASSOCIATE_JSON_TYPE(JSON_types, actual_types);"
		/// for all JSON_types (null, boolean, number, string, array, and object) between 
		/// "BEGIN_TYPE_ASSOCIATION();" and "END_TYPE_ASSOCIATION();"
		/// Once this is done, a template type alias entry_data_type<type tid> is automatically defined
		/// Template specializations of this alias for each tid are also automatically aliased as 
		/// null_type, boolean_type, number_type, string_type, array_type, and object_type.
		/// Type members value_type of string_type is also aliased as char_type.
		/// Stream output can be opt-in by specifying additional type alias ostream_type; in this case, the expression 
		/// (variable of type ostream_type) << (variable of type string_type const&) must make a valid output.
		/// These macros are doing template specialization internally.
		///
		/// The type associated to string should support followings:
		///   -- typename value_type, which is the type of characters; it should be able to construct/assign value_type from char
		///   -- operator+= with another string
		///   -- operator+= with char & unsigned char
		///   -- iterator (iterator types, begin(), cbegin(), etc.)
		///   -- default constructor, and the result of default construction should be the empty string
		///   -- Optional support of stream output into ostream_type
		/// The type associated to array should support followings:
		///   -- typename value_type, which should be compatible with entry_pointer
		///   -- operator[] with integer index (both const & non-const versions)
		///   -- size() const method
		///   -- emplace_back(...) method
		///   -- iterator (iterator types, begin(), cbegin(), etc.)
		/// The type associated to object should support followings:
		///   -- typename value_type, which should be able to be constructible from { string_type, entry_pointer }
		///   -- operator[] with string_type index (both const & non-const versions)
		///   -- size() const method
		///   -- emplace() method taking a single std::pair<string_type, entry_pointer> argument
		///   -- iterator (iterator types, begin(), cbegin(), etc.)
		/// The type associated to number may deal both with integer & floating point data, but this is not mandatory
		///
		/// To perform string conversion operations, following static methods should be provided: ("..." just for abbreviation)
		///   -- make_entry(...)				: for creating an object of specified JSON type, which can be converted to entry_pointer
		///   -- to_string()					: for converting a null-type entry to text (text is represented by string_type)
		///   -- to_string(boolean_type const&)	: for converting a boolean-type entry to text
		///   -- to_string(number_type const&)	: for converting a number-type entry to text
		///   -- get_code_point(...)			: to take care of Unicode escape, way of getting Unicode code point from a string should be known
		///   -- code_point_to_string(...)		: similarly, the reverse operation also should be known
		///   -- eol_string(eol_policy)			: get End-Of-Line string according to the specified policy
		///   -- indent_string(indent_policy)	: get indentation string according to the specified policy
		///
		/// The following is an example of JSONPolicy classes
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BEGIN_TYPE_ASSOCIATION()			template<type tid_, class = void> struct entry_data_type_
	#define ASSOCIATE_JSON_TYPE(_enum, ...)		template<class dummy> struct entry_data_type_<type::_enum, dummy>{ using type = __VA_ARGS__; }
	#define END_TYPE_ASSOCIATION()				template<type tid_> using entry_data_type = typename entry_data_type_<tid_>::type; \
												using null_type = entry_data_type<type::null>; \
												using boolean_type = entry_data_type<type::boolean>; \
												using number_type = entry_data_type<type::number>; \
												using string_type = entry_data_type<type::string>; \
												using array_type = entry_data_type<type::array>; \
												using object_type = entry_data_type<type::object>; \
												using char_type = typename string_type::value_type
		struct default_policy {
			using entry_pointer = std::shared_ptr<json_entry<type::general, default_policy>>;
			using entry_const_pointer = std::shared_ptr<json_entry<type::general, default_policy> const>;
			using ostream_type = std::basic_ostream<char>;

			BEGIN_TYPE_ASSOCIATION();
			ASSOCIATE_JSON_TYPE(null, void);
			ASSOCIATE_JSON_TYPE(boolean, bool);
			ASSOCIATE_JSON_TYPE(number, jkl::math::mixed_precision_number<long long, long double>);
			ASSOCIATE_JSON_TYPE(string, std::string);
			ASSOCIATE_JSON_TYPE(array, std::vector<entry_pointer>);
			ASSOCIATE_JSON_TYPE(object, std::map<std::string, entry_pointer>);
			END_TYPE_ASSOCIATION();

			// The basic factory function
			template <type tid, typename... Args>
			static auto	make_entry(Args&&... args) {
				return std::make_shared<json_entry<tid, default_policy>>(std::forward<Args>(args)...);
			}

			// Convert Null-type to text
			static string_type			to_string() {
				return std::string("null");
			}

			// Convert boolean-type to text
			static string_type			to_string(boolean_type const& data) {
				if( data )
					return std::string("true");
				else
					return std::string("false");
			}

			// Convert number-type to text
			static string_type			to_string(number_type const& data) {
				if( data.is_int() )
					return std::to_string((number_type::integral_type)data);
				if( data.is_float() )
					return std::to_string((number_type::floating_point_type)data);
				else
					return std::string();
			}

			// Get Unicode code point from a character inside a string
			static std::uint32_t	to_code_point(string_type::const_iterator& itr, string_type::const_iterator const& end) {
				auto const original_itr = itr;

				auto unexpected_termination = [&itr, &original_itr] {
					throw text_parsing_error("String unexpectedly ends!", 
						text_parsing_error::broken_utf8_encoding, &*original_itr, &*itr);
				};
				auto invalid_second_byte = [&itr, &original_itr] {
					throw text_parsing_error("The second byte is not of the form 10xxxxxx!", 
						text_parsing_error::broken_utf8_encoding, &*original_itr, &*itr);
				};
				auto invalid_third_byte = [&itr, &original_itr] {
					throw text_parsing_error("The third byte is not of the form 10xxxxxx!", 
						text_parsing_error::broken_utf8_encoding, &*original_itr, &*itr);
				};
				auto invalid_fourth_byte = [&itr, &original_itr] {
					throw text_parsing_error("The fourth byte is not of the form 10xxxxxx!", 
						text_parsing_error::broken_utf8_encoding, &*original_itr, &*itr);
				};

				unsigned char value1, value2, value3, value4;
				value1 = (unsigned char)(*itr);
				if( jkl::util::upper_bits(value1, 1) == 0 )			// U+0000 - U+007F
					return value1;
				if( jkl::util::upper_bits(value1, 3) == 0xC0 ) {	// U+0080 - U+07FF
					if( ++itr == end )
						unexpected_termination();
					value2 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value2, 2) != 0x80 )
						invalid_second_byte();
					return ((std::uint32_t)jkl::util::lower_bits(value1, 5) << 6) + jkl::util::lower_bits(value2, 6);
				}
				if( jkl::util::upper_bits(value1, 4) == 0xE0 ) {	// U+0800 - U+FFFF
					if( ++itr == end )
						unexpected_termination();
					value2 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value2, 2) != 0x80 )
						invalid_second_byte();
					if( ++itr == end )
						unexpected_termination();
					value3 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value3, 2) != 0x80 )
						invalid_third_byte();
					return ((std::uint32_t)jkl::util::lower_bits(value1, 4) << 12) +
						((std::uint32_t)jkl::util::lower_bits(value2, 6) << 6) + jkl::util::lower_bits(value3, 6);
				}
				if( jkl::util::upper_bits(value1, 5) == 0xF0 ) {	// U+10000 - U+1FFFFF
					if( ++itr == end )
						unexpected_termination();
					value2 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value2, 2) != 0x80 )
						invalid_second_byte();
					if( ++itr == end )
						unexpected_termination();
					value3 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value3, 2) != 0x80 )
						invalid_third_byte();
					if( ++itr == end )
						unexpected_termination();
					value4 = (unsigned char)(*itr);
					if( jkl::util::upper_bits(value4, 2) != 0x80 )
						invalid_fourth_byte();
					return ((std::uint32_t)jkl::util::lower_bits(value1, 3) << 18) + ((std::uint32_t)jkl::util::lower_bits(value2, 6) << 12) +
						((std::uint32_t)jkl::util::lower_bits(value3, 6) << 6) + jkl::util::lower_bits(value4, 6);
				}

				throw text_parsing_error("Out of the range specified by RFC 3629!",
					text_parsing_error::broken_utf8_encoding, &*original_itr, &*itr);
				return 0;
			}

			// Get string from a single Unicode code point
			static string_type			code_point_to_string(std::uint32_t code_point) {
				if( code_point < 0x0080 )
					return{ (char)code_point };
				else if( code_point < 0x800 )
					return{ (char)(unsigned char)(0xC0 + (jkl::util::upper_bits(code_point, 26) >> 6)),
					(char)(unsigned char)(0x80 + jkl::util::lower_bits(code_point, 6)) };
				else if( code_point < 0x10000 )
					return{ (char)(unsigned char)(0xE0 + (jkl::util::upper_bits(code_point, 20) >> 12)),
					(char)(unsigned char)(0x80 + (jkl::util::middle_bits(code_point, 20, 6) >> 6)),
					(char)(unsigned char)(0x80 + jkl::util::lower_bits(code_point, 6)) };
				else
					return{ (char)(unsigned char)(0xF0 + (jkl::util::upper_bits(code_point, 14) >> 18)),
					(char)(unsigned char)(0x80 + (jkl::util::middle_bits(code_point, 14, 12) >> 12)),
					(char)(unsigned char)(0x80 + (jkl::util::middle_bits(code_point, 20, 6) >> 6)),
					(char)(unsigned char)(0x80 + jkl::util::lower_bits(code_point, 6)) };
			}

			// End-of-line string
			static string_type			eol_string(eol_policy policy) {
				switch( policy ) {
				case eol_policy::LF:
					return std::string("\n");
				case eol_policy::CR:
					return std::string("\r");
				case eol_policy::CRLF:
					return std::string("\r\n");
				case eol_policy::LFCR:
					return std::string("\n\r");
				default:
					return std::string();
				}
			}

			// Indentation string
			static string_type			indent_string(indent_policy policy) {
				switch( policy ) {
				case indent_policy::no_indent:
					return std::string();
				case indent_policy::tab:
					return std::string("\t");
				case indent_policy::double_tab:
					return std::string("\t\t");
				default:
					string_type str;
					for( auto i = (std::uint8_t)indent_policy::single_space; i <= (std::uint8_t)policy; i++ )
						str += ' ';
					return str;
				}
			}
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// json_entry class template
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		namespace detail
		{
			// For things common for all entry types
			template <type tid_, class Policy>
			class json_entry_base : public json_entry<type::general, Policy>
			{
			public:
				using ostream_type = typename json_entry<type::general, Policy>::ostream_type;
				static constexpr type entry_type_id = tid_;
				type tid() const noexcept { return tid_; }

				template <type tid_requested, class = std::enable_if_t<tid_requested == tid_ || tid_requested == type::general>>
				auto& as() {
					return static_cast<json_entry<tid_requested, Policy>&>(*this);
				}
				template <type tid_requested, class = std::enable_if_t<tid_requested == tid_ || tid_requested == type::general>>
				auto& as() const {
					return static_cast<json_entry<tid_requested, Policy> const&>(*this);
				}
				template <class EntryPtr, class = std::enable_if_t<std::is_convertible<json_entry<tid_, Policy>&, EntryPtr&>::value>>
				auto& as() {
					return static_cast<EntryPtr&>(*this);
				}
				template <class EntryPtr, class = std::enable_if_t<std::is_convertible<json_entry<tid_, Policy> const&, EntryPtr const&>::value>>
				auto& as() const {
					return static_cast<EntryPtr const&>(*this);
				}

				template <type tid_requested>
				auto& unsafe_as() {
					return as<tid_requested>();
				}
				template <type tid_requested>
				auto& unsafe_as() const {
					return as<tid_requested>();
				}
				template <class EntryPtr>
				auto& unsafe_as() {
					return as<EntryPtr>();
				}
				template <class EntryPtr>
				auto& unsafe_as() const {
					return as<EntryPtr>();
				}
			};

			// to_string() implementaion for boolean and numerical data; calls to_string() of the specified policy
			template <type tid_, class Policy>
			struct to_string_impl
			{
				using data_type_ = typename Policy::template entry_data_type<tid_>;
				using string_type_ = typename Policy::string_type;
				static string_type_ to_string(data_type_ const& data, text_format) {
					return Policy::to_string(data);
				}
			};

			// to_string() implementation for string data; take care of escape syntax
			template <class Policy>
			struct to_string_impl <type::string, Policy>
			{
				using data_type_ = typename Policy::string_type;
				using string_type_ = typename Policy::string_type;

				// Convert std::uint16_t to hexadecimal digits
				static string_type_ to_hex_string(std::uint16_t hex, hex_policy hex_policy) {
					string_type_ str;
					auto q0 = (unsigned char)(hex % 16);
					auto q1 = (unsigned char)((hex / 16) % 16);
					auto q2 = (unsigned char)((hex / 256) % 16);
					auto q3 = (unsigned char)(hex / 4096);

					switch( hex_policy ) {
					case hex_policy::lower_case:
						str += (q3<10) ? (q3 + '0') : (q3 - 10 + 'a');
						str += (q2<10) ? (q2 + '0') : (q2 - 10 + 'a');
						str += (q1<10) ? (q1 + '0') : (q1 - 10 + 'a');
						str += (q0<10) ? (q0 + '0') : (q0 - 10 + 'a');
						break;
					case hex_policy::upper_case:
						str += (q3<10) ? (q3 + '0') : (q3 - 10 + 'A');
						str += (q2<10) ? (q2 + '0') : (q2 - 10 + 'A');
						str += (q1<10) ? (q1 + '0') : (q1 - 10 + 'A');
						str += (q0<10) ? (q0 + '0') : (q0 - 10 + 'A');
						break;
					}

					return str;
				}

				static string_type_ to_string(const data_type_& data, text_format format) {
					string_type_ str;
					str += '\"';
					for( auto itr = data.cbegin(); itr < data.cend(); ++itr ) {
						// For normal character
						if( *itr >= 32 && *itr < 128 && *itr != '\\' && *itr != '\"' )
							str += *itr;
						// For escape syntax
						else {
							str += '\\';
							switch( *itr ) {
							case '\\': str += '\\'; break;
							case '\"': str += '\"'; break;
							case '\b': str += 'b'; break;
							case '\f': str += 'f'; break;
							case '\n': str += 'n'; break;
							case '\r': str += 'r'; break;
							case '\t': str += 't'; break;
							default:
								str += 'u';
								auto code_point = Policy::to_code_point(itr, data.cend());
								// If to_code_point do not return a valid code point, behaviour is undefined
								auto sp = *jkl::unicode::get_utf16_pair(code_point);
								if( sp.first != 0 ) {
									str += to_hex_string(sp.first, format.hex);
									str += '\\';
									str += 'u';
								}
								str += to_hex_string(sp.second, format.hex);
							}
						}
					}
					str += '\"';
					return str;
				}
			};
		}

		// JSON entry class for boolean, number, and string data
		template <type tid_, class Policy = default_policy>
		class json_entry : public detail::json_entry_base<tid_, Policy>, private detail::to_string_impl<tid_, Policy>
		{
		public:
			using data_type = typename Policy::template entry_data_type<tid_>;
			using string_type = typename Policy::string_type;

			// Default contructor (if any)
			json_entry() = default;
			// Perfect forwarding constructors
			template <typename Arg, class = jkl::tmp::prevent_too_perfect_fwd<json_entry, Arg>>
			explicit json_entry(Arg&& arg) : data_(std::forward<Arg>(arg)) {}
			template <typename FirstArg, typename SecondArg, typename... RemainingArgs>
			json_entry(FirstArg&& first_arg, SecondArg&& second_arg, RemainingArgs&&... remaining_args) 
				: data_(std::forward<FirstArg>(first_arg), std::forward<SecondArg>(second_arg), 
				std::forward<RemainingArgs>(remaining_args)...) {}

			// Casting operators for numerical data; return by value
			template <class T, class = std::enable_if_t<tid_ == type::number && std::is_convertible<data_type, T>::value>>
			operator T() const noexcept { return data_; }
			// lvalue casting operators; return by reference
			operator data_type&() & noexcept { return data_; }
			operator data_type const&() const& noexcept { return data_; }
			auto& data() & noexcept { return data_; }
			auto const& data() const& noexcept { return data_; }
			// rvalue casting operators; return by value
			operator data_type() && noexcept { return std::move(data_); }
			auto data() && noexcept { return std::move(data_); }

			// Perfect forwarding assignment
			template <typename T, class = std::enable_if_t<std::is_assignable<data_type, T>::value>>
			json_entry& operator=(T&& data) & { this->data_ = std::forward<T>(data); return *this; }

			// Factory function
			template <typename... Args>
			static auto create(Args&&... args) {
				return Policy::template make_entry<tid_>(std::forward<Args>(args)...);
			}

			// To string
			using json_entry<type::general, Policy>::to_string;
			string_type to_string(std::size_t = 0, text_format format = text_format()) const {
				return detail::to_string_impl<tid_, Policy>::to_string(data_, format);
			}

			// To stream
			using ostream_type = typename json_entry<type::general, Policy>::ostream_type;
			using json_entry<type::general, Policy>::to_stream;
			ostream_type& to_stream(ostream_type& out, std::size_t = 0, text_format format = text_format()) const {
				return out << to_string(0, format);
			}

		private:
			data_type data_;
		};

		// JSON entry for null data
		template <class Policy>
		class json_entry<type::null, Policy> : public detail::json_entry_base<type::null, Policy>
		{
		public:
			using data_type = typename Policy::null_type;
			using string_type = typename Policy::string_type;

			// Factory function
			static auto create() {
				return Policy::template make_entry<type::null>();
			}

			// To string
			using json_entry<type::general, Policy>::to_string;
			string_type to_string(std::size_t = 0, text_format = text_format()) const {
				return Policy::to_string();
			}

			// To stream
			using ostream_type = typename json_entry<type::general, Policy>::ostream_type;
			using json_entry<type::general, Policy>::to_stream;
			ostream_type& to_stream(ostream_type& out, std::size_t, text_format) const {
				return out << to_string();
			}
		};

		namespace detail {
			// For things common for array and object type
			template <class Policy, class Impl, type tid_>
			class array_object_crtp : public json_entry_base<tid_, Policy>
			{
			public:
				using data_type = typename Policy::template entry_data_type<tid_>;
				using string_type = typename Policy::string_type;

				// Default contructor (if any)
				array_object_crtp() = default;
				// Initializer list constructor
				array_object_crtp(std::initializer_list<typename data_type::value_type> init_list) : data_array(init_list) {}
				// Perfect forwarding constructors
				template <typename Arg, class = jkl::tmp::prevent_too_perfect_fwd<Impl, Arg>>
				explicit array_object_crtp(Arg&& arg) : data_array(std::forward<Arg>(arg)) {}
				template <typename FirstArg, typename SecondArg, typename... RemainingArgs>
				array_object_crtp(FirstArg&& first_arg, SecondArg&& second_arg, RemainingArgs&&... remaining_args)
					: data_array(std::forward<FirstArg>(first_arg), std::forward<SecondArg>(second_arg),
						std::forward<RemainingArgs>(remaining_args)...) {}
				// Size
				std::size_t size() const noexcept { return data_array.size(); }
				// Get the underlying array of pointer
				auto& data() noexcept { return data_array; }
				auto const& data() const noexcept { return data_array; }

				// Factory functions
				static auto create(std::initializer_list<typename data_type::value_type> init_list) {
					return Policy::template make_entry<tid_>(init_list);
				}
				template <typename... Args>
				static auto create(Args&&... args) {
					return Policy::template make_entry<tid_>(std::forward<Args>(args)...);
				}

				// To string
				using json_entry<type::general, Policy>::to_string;
				string_type to_string(std::size_t indentation_level = 0, text_format format = text_format()) const {
					string_type str;
					static_cast<Impl const*>(this)->to_string_impl([&str](auto&& s) {
						str += s;
					}, [&str, indentation_level, format](auto&& entry_ptr) {
						str += entry_ptr->to_string(indentation_level + 1, format);
					}, indentation_level, format);
					return str;
				}

				// To stream
				using ostream_type = typename json_entry<type::general, Policy>::ostream_type;
				using json_entry<type::general, Policy>::to_stream;
				ostream_type& to_stream(ostream_type& out, std::size_t indentation_level = 0, text_format format = text_format()) const {
					static_cast<Impl const*>(this)->to_string_impl([&out](auto&& s) {
						out << s;
					}, [&out, indentation_level, format](auto&& entry_ptr) {
						entry_ptr->to_stream(out, indentation_level + 1, format);
					}, indentation_level, format);
					return out;
				}

				// Iterators
				using iterator = typename data_type::iterator;
				using const_iterator = typename data_type::const_iterator;
				iterator begin() noexcept { return data_array.begin(); }
				const_iterator begin() const noexcept { return data_array.begin(); }
				const_iterator cbegin() const noexcept { return data_array.cbegin(); }
				iterator end() noexcept { return data_array.end(); }
				const_iterator end() const noexcept { return data_array.end(); }
				const_iterator cend() const noexcept { return data_array.cend(); }

			private:
				data_type					data_array;
			};
		}

		// JSON entry for array data
		template <class Policy>
		class json_entry<type::array, Policy> : public detail::array_object_crtp<Policy, json_entry<type::array, Policy>, type::array>
		{
			using crtp_base_type = detail::array_object_crtp<Policy, json_entry<type::array, Policy>, type::array>;
			friend crtp_base_type;
		public:
			using data_type = typename Policy::array_type;
			using string_type = typename Policy::string_type;

			using crtp_base_type::crtp_base_type;
			using crtp_base_type::data;
			using crtp_base_type::size;
			operator data_type&() noexcept { return data(); }
			operator data_type const&() const noexcept { return data(); }

			// Accessors; returns reference to entry_type, while the underlying array holds pointers
			auto& operator[](std::size_t idx) {
				return static_cast<json_entry<type::general, Policy>&>(*data_array[idx]);
			}
			auto& operator[](std::size_t idx) const {
				return static_cast<json_entry<type::general, Policy> const&>(*data_array[idx]);
			}

			// Create a new entry and append it to the array
			template <typename... Args>
			decltype(auto) emplace(Args&&... args) {
				return data().emplace_back(jkl::json::create<Policy>(std::forward<Args>(args)...));
			}
			// Append an existing JSON entry; entry must be convertible to entry_pointer
			template <class EntryPtr>
			decltype(auto) append(EntryPtr&& entry) {
				return data().emplace_back(std::forward<EntryPtr>(entry));
			}

			// Copy assignment from data_type
			json_entry& operator=(data_type const& data_array) & { data() = data_array; return *this; }
			// Move assignment from data_type
			json_entry& operator=(data_type&& data_array) & { data() = std::move(data_array); return *this; }

		private:
			template <class BasicBuilder, class RecursiveBuilder>
			void to_string_impl(BasicBuilder basic, RecursiveBuilder recursive, 
				std::size_t indentation_level, text_format format) const
			{
				if( size() == 0 ) {
					basic('[');
					basic(']');
					return;
				}
				string_type indent_str;
				for( std::size_t i = 0; i < indentation_level; i++ )
					indent_str += Policy::indent_string(format.indentation);
				basic('[');
				basic(Policy::eol_string(format.eol));
				auto last = --(data().cend());
				for( auto itr = data().cbegin(); itr != last; ++itr ) {
					basic(indent_str);
					basic(Policy::indent_string(format.indentation));
					recursive(*itr);
					basic(',');
					basic(Policy::eol_string(format.eol));
				}
				basic(indent_str);
				basic(Policy::indent_string(format.indentation));
				recursive(*last);
				basic(Policy::eol_string(format.eol));
				basic(indent_str);
				basic(']');
			}
		};

		// JSON entry for object data
		template <class Policy>
		class json_entry<type::object, Policy> : public detail::array_object_crtp<Policy, json_entry<type::object, Policy>, type::object>
		{
			using crtp_base_type = detail::array_object_crtp<Policy, json_entry<type::object, Policy>, type::object>;
			friend crtp_base_type;
		public:
			using data_type = typename Policy::object_type;
			using string_type = typename Policy::string_type;

			using crtp_base_type::crtp_base_type;
			using crtp_base_type::data;
			using crtp_base_type::size;
			operator data_type&() noexcept { return data(); }
			operator data_type const&() const noexcept { return data(); }

			// Accessors; returns reference to entry_type, while the underlying array holds pointers
			// If there is no entry found, throw
			auto& operator[](string_type const& key) {
				return static_cast<json_entry<type::general, Policy>&>(access_impl(*this, key));
			}
			auto& operator[](string_type const& key) const {
				return static_cast<json_entry<type::general, Policy> const&>(access_impl(*this, key));
			}

			// Create a new entry and append it to the array
			template <class String, typename... Args>
			decltype(auto) emplace(String&& name, Args&&... args) {
				return data().emplace(typename data_type::value_type(std::forward<String>(name), jkl::json::create<Policy>(std::forward<Args>(args)...)));
			}
			// Append an existing JSON entry; entry must be convertible to entry_pointer
			template <class String, class EntryPtr>
			decltype(auto) append(String&& name, EntryPtr&& entry) {
				return data().emplace(typename data_type::value_type(std::forward<String>(name), std::forward<EntryPtr>(entry)));
			}

			// Copy assignment from data_type
			json_entry& operator=(data_type const& data_array) { data() = data_array; return *this; }
			// Move assignment from data_type
			json_entry& operator=(data_type&& data_array) { data() = std::move(data_array); return *this; }

		private:
			template <class BasicBuilder, class RecursiveBuilder>
			void to_string_impl(BasicBuilder basic, RecursiveBuilder recursive,
				std::size_t indentation_level, text_format format) const
			{
				string_type indent_str;
				for( std::size_t i = 0; i < indentation_level; i++ )
					indent_str += Policy::indent_string(format.indentation);
				basic('{');
				basic(Policy::eol_string(format.eol));
				if( size() == 0 ) {
					basic('}');
					return;
				}
				auto last = --(data().cend());
				for( auto itr = data().cbegin(); itr != last; ++itr ) {
					basic(indent_str);
					basic(Policy::indent_string(format.indentation));
					basic('\"');
					basic((*itr).first);
					basic('\"');
					basic(' ');
					basic(':');
					basic(' ');
					recursive((*itr).second);
					basic(',');
					basic(Policy::eol_string(format.eol));
				}
				basic(indent_str);
				basic(Policy::indent_string(format.indentation));
				basic('\"');
				basic((*last).first);
				basic('\"');
				basic(' ');
				basic(':');
				basic(' ');
				recursive((*last).second);
				basic(Policy::eol_string(format.eol));
				basic(indent_str);
				basic('}');
			}

			template <class EntryType>
			static auto& access_impl(EntryType&& entry, string_type const& key) {
				auto itr = entry.data().find(key);
				if( itr == entry.data().end() ) {
					throw entry_not_found<EntryType>{ string_type{ "Can't find the entry " } +key,
						std::forward<EntryType>(entry), key };
				}
				return *std::get<1>(*itr);
			}
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Stream output ability (only supported when Policy enables it)
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// An aggregation of a JSON entry with output options
		template <class Policy>
		struct printable_entry_type {
			json_entry<type::general, Policy> const&	entry;
			std::size_t									indentation_level;
			text_format									format;

			printable_entry_type(json_entry<type::general, Policy> const& entry,
				std::size_t indentation_level = 0, text_format format = text_format())
				: entry(entry), indentation_level(indentation_level), format(format) {}
			printable_entry_type(json_entry<type::general, Policy> const& entry, text_format format)
				: printable_entry_type(entry, 0, format) {}
		};

		// Helper functions to make printable_entry_type
		// These functions do not actually print anything
		template <type tid_, class Policy>
		printable_entry_type<Policy> print(json_entry<tid_, Policy> const& entry,
			std::size_t indentation_level = 0, text_format format = text_format()) {
			return{ entry, indentation_level, format };
		}
		template <type tid_, class Policy>
		printable_entry_type<Policy> print(json_entry<tid_, Policy> const& entry, text_format format) {
			return{ entry, format };
		}

		// The actual output function
		template <class OstreamType, class Policy>
		auto& operator<<(OstreamType& out, printable_entry_type<Policy> const& printable_entry) {
			return printable_entry.entry.to_stream(out, printable_entry.indentation_level, printable_entry.format);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Factory functions
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Creation of Null-type entries
		template <class Policy = default_policy>
		inline auto create() {
			return json_entry<type::null, Policy>::create();
		}

		// Creation of boolean-type entries
		template <class Policy = default_policy, class BoolType = typename Policy::boolean_type,
			class = std::enable_if_t< std::is_same<std::decay_t<typename Policy::boolean_type>, std::decay_t<BoolType>>::value >>
		inline auto create(BoolType&& data) {
			return json_entry<type::boolean, Policy>::create(std::forward<BoolType>(data));
		}

		// Creation of number-type entries
		template <class Policy = default_policy>
		inline auto create(typename Policy::number_type const& data) {
			return json_entry<type::number, Policy>::create(data);
		}
		template <class Policy = default_policy>
		inline auto create(typename Policy::number_type&& data) {
			return json_entry<type::number, Policy>::create(std::move(data));
		}

		// Creation of string-type entries
		template <class Policy = default_policy>
		inline auto create(typename Policy::string_type const& data) {
			return json_entry<type::string, Policy>::create(data);
		}
		template <class Policy = default_policy>
		inline auto create(typename Policy::string_type&& data) {
			return json_entry<type::string, Policy>::create(std::move(data));
		}

		// Creation of Int-type entries (bound to number-type)
		template <class Policy = default_policy, typename IntType = typename Policy::integral_type,
			class = std::enable_if_t<Policy::number_type::template is_integral<std::remove_reference_t<IntType>>::value &&
			!std::is_same< std::decay_t<IntType>, std::decay_t<typename Policy::boolean_type> >::value &&
			!std::is_same< std::decay_t<IntType>, std::decay_t<typename Policy::string_type::value_type> >::value>, class = void>
		inline auto create(IntType&& data) {
			return json_entry<type::number, Policy>::create(std::forward<IntType>(data));
		}

		// Creation of Float-type entries (bound to number-type)
		template <class Policy = default_policy, typename FloatType = typename Policy::floating_point_type,
			class = std::enable_if_t<Policy::number_type::template is_floating_point<std::remove_reference_t<FloatType>>::value &&
			!std::is_same< std::decay_t<FloatType>, std::decay_t<typename Policy::boolean_type> >::value &&
			!std::is_same< std::decay_t<FloatType>, std::decay_t<typename Policy::string_type::value_type> >::value>, class = void, class = void>
		inline auto create(FloatType&& data) {
			return json_entry<type::number, Policy>::create(std::forward<FloatType>(data));
		}

		// Creation of Character-type entries (bound to string-type); be careful that "unsigned character" will not bound to this overload
		template <class Policy = default_policy>
		inline auto create(typename Policy::char_type const& data) {
			return json_entry<type::string, Policy>::create(std::initializer_list<typename Policy::char_type>(&data, &data + 1));
		}
		template <class Policy = default_policy>
		inline auto create(typename Policy::char_type&& data) {
			return json_entry<type::string, Policy>::create(std::initializer_list<typename Policy::char_type>(&data, &data + 1));
		}
		
		// Creation of array-type entries
		template <class Policy = default_policy>
		inline auto create(typename Policy::array_type const& data) {
			return json_entry<type::array, Policy>::create(data);
		}
		template <class Policy = default_policy>
		inline auto create(typename Policy::array_type&& data) {
			return json_entry<type::array, Policy>::create(std::move(data));
		}
		template <class Policy = default_policy>
		inline auto create(std::initializer_list<typename Policy::array_type::value_type> data) {
			return json_entry<type::array, Policy>::create(data);
		}

		// Creation of object-type entries
		template <class Policy = default_policy>
		inline auto create(typename Policy::object_type const& data) {
			return json_entry<type::object, Policy>::create(data);
		}
		template <class Policy = default_policy>
		inline auto create(typename Policy::object_type&& data) {
			return json_entry<type::object, Policy>::create(std::move(data));
		}
		template <class Policy = default_policy>
		inline auto create(std::initializer_list<typename Policy::object_type::value_type> data) {
			return json_entry<type::object, Policy>::create(data);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Parse a UTF-8 encoded string and create a JSON entry
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Parser class
		template <class Policy = default_policy>
		typename Policy::entry_pointer parse(typename Policy::string_type const& input);

		namespace detail {
			template <class Policy>
			class json_parser {
				using entry_pointer = typename Policy::entry_pointer;
				using boolean_type = typename Policy::boolean_type;
				using number_type = typename Policy::number_type;
				using string_type = typename Policy::string_type;
				using char_type = typename Policy::char_type;
				using const_iterator = typename string_type::const_iterator;

				friend entry_pointer jkl::json::parse<Policy>(string_type const& input);
				json_parser() = delete;	// This class cannot be instantiated

				static entry_pointer parse(const_iterator& itr, const_iterator const& end) {
					// Remove spaces
					skip_space(itr, end);
					auto prev_pos = itr;
					// If 'n' is found
					if( *itr == 'n' ) {
						// Looking for "ull"
						if( word_test(++itr, end, "ull") )
							return create();
						else {
							throw text_parsing_error("The character 'n' must be followed by \"ull\"!", 
								text_parsing_error::incomplete_word, (char const*)&*prev_pos, (char const*)&*itr);
						}
					}
					// If 'f' is found
					else if( *itr == 'f' ) {
						// Looking for "alse"
						if( word_test(++itr, end, "alse") )
							return create(static_cast<boolean_type>(false));
						else {
							throw text_parsing_error("The character 'f' must be followed by \"alse\"!",
								text_parsing_error::incomplete_word, (char const*)&*prev_pos, (char const*)&*itr);
						}
					}
					// If 't' is found
					else if( *itr == 't' ) {
						// Looking for "rue"
						if( word_test(++itr, end, "rue") )
							return create(static_cast<boolean_type>(true));
						else {
							throw text_parsing_error("The character 't' must be followed by \"rue\"!",
								text_parsing_error::incomplete_word, (char const*)&*prev_pos, (char const*)&*itr);
						}
					} else if( *itr == '+' )
						// Positive number
						return parse_number(++itr, end, 1, prev_pos);
					else if( *itr == '-' )
						// Negative number
						return parse_number(++itr, end, -1, prev_pos);
					else if( *itr == '.' || (*itr >= '0' && *itr <= '9') )
						// Positive number
						return parse_number(itr, end, 1, prev_pos);
					else if( *itr == '\"' )
						// string
						return create(parse_string(++itr, end, prev_pos));
					else if( *itr == '[' )
						// array
						return parse_array(++itr, end, prev_pos);
					else if( *itr == '{' )
						// object
						return parse_object(++itr, end, prev_pos);

					return nullptr;
				}

				static bool is_space(char_type const& character) noexcept {
					if( character == ' ' || character == '\t' || character == '\r' || character == '\n' )
						return true;
					else
						return false;
				}

				static void skip_space(const_iterator& itr, const_iterator const& end) noexcept {
					for( ; itr != end; ++itr )
						if( !is_space(*itr) )
							return;
				}

				// Returns true iff the string starts with the given word
				template<std::size_t length>
				static bool word_test(const_iterator& itr, const_iterator const& end, char_type const(&word)[length]) noexcept {
					if( itr != end && *itr++ == word[0] )
						return word_test(itr, end, reinterpret_cast<const char_type(&)[length - 1]>(word[1]));
					return false;
				}
				template <>
				static bool word_test<1>(const_iterator&, const_iterator const&, char_type const(&)[1]) noexcept {
					return true;	// Do not care about the trailing null character
				}

				static entry_pointer parse_number(const_iterator& itr, 
					const_iterator const& end, int sign, const_iterator const& prev_pos) {
					enum state {
						before_dot,
						after_dot,
						just_after_exp,
						after_exp
					} s = before_dot;
					number_type value = 0;
					number_type multiplier = 0.1;
					int exponent = 0;
					int exponent_sign = 1;
					bool is_integer = true;
					while( itr != end && !is_space(*itr) && *itr != ',' && *itr != ']' && *itr != '}' ) {
						switch( s ) {
						case before_dot:
							if( *itr >= '0' && *itr <= '9' ) {
								value *= 10;
								value += *itr - '0';
							} else if( *itr == '.' ) {
								s = after_dot;
								is_integer = false;
							} else if( *itr == 'e' || *itr == 'E' ) {
								s = just_after_exp;
								is_integer = false;
							} else {
								throw text_parsing_error(std::string("Expected '0'-'9', '.', 'e', or 'E', but got '") + (char)*itr + "'!", 
									text_parsing_error::ill_formed_number, (char const*)&*prev_pos, (char const*)&*itr);
							}
							break;

						case after_dot:
							if( *itr >= '0' && *itr <= '9' ) {
								value += multiplier * (*itr - '0');
								multiplier /= 10;
							} else if( *itr == 'e' || *itr == 'E' )
								s = just_after_exp;
							else {
								throw text_parsing_error(std::string("Expected '0'-'9', 'e', or 'E', but got '") + (char)*itr + "'!", 
									text_parsing_error::ill_formed_number, (char const*)&*prev_pos, (char const*)&*itr);
							}
							break;

						case just_after_exp:
							if( *itr == '+' ) {
								s = after_exp;
								break;
							} else if( *itr == '-' ) {
								exponent_sign = -1;
								s = after_exp;
								break;
							}
						case after_exp:
							if( *itr >= '0' && *itr <= '9' ) {
								s = after_exp;
								exponent *= 10;
								exponent += *itr - '0';
							} else {
								if( s == just_after_exp ) {
									throw text_parsing_error(std::string("Expected '0'-'9', '+', or '-', but got '") + (char)*itr + "'!", 
										text_parsing_error::ill_formed_number, (char const*)&*prev_pos, (char const*)&*itr);
								}
								else {
									throw text_parsing_error(std::string("Expected '0'-'9', but got '") + (char)*itr + "'!", 
										text_parsing_error::ill_formed_number, (char const*)&*prev_pos, (char const*)&*itr);
								}
							}
							break;
						}

						++itr;
					}
					if( is_integer )
						return create(sign * value);
					else
						return create(sign * value * pow(10, exponent_sign * exponent));
				}

				template <class MessageType>
				static void throw_invalid_escape_exception(MessageType const& message, 
					const_iterator& itr, const_iterator const& prev_pos) {
					throw text_parsing_error(message, 
						text_parsing_error::invalid_escape_syntax, (char const*)&*prev_pos, (char const*)&*itr);
				}
				static std::uint16_t hex_char_to_value(const_iterator& itr, 
					const_iterator const& end, const_iterator const& prev_pos) {
					if( itr == end )
						throw_invalid_escape_exception("Expected a hexadecimal number, but reached the end of the string!", itr, prev_pos);
					if( *itr >= '0' && *itr <= '9' )
						return *itr - '0';
					else if( *itr >= 'A' && *itr <= 'F' )
						return *itr - 'A' + 10;
					else if( *itr >= 'a' && *itr <= 'f' )
						return *itr - 'a' + 10;
					throw_invalid_escape_exception(std::string("Expected '0'-'9', 'A'-'F', or 'a'-'f', but got '") + (char)*itr + "'!", itr, prev_pos);
					return 0;
				}
				static char value_to_hex_char(std::uint8_t c) noexcept {
					if( c < 10 )
						return (char)c + '0';
					else
						return (char)c + 'A';
				}
				static std::uint16_t four_hex_to_value(const_iterator& itr,
					const_iterator const& end, const_iterator const& prev_pos) {
					return 4096 * hex_char_to_value(itr, end, prev_pos) + 256 * hex_char_to_value(++itr, end, prev_pos) +
						16 * hex_char_to_value(++itr, end, prev_pos) + hex_char_to_value(++itr, end, prev_pos);
				}

				static string_type parse_string(const_iterator& itr, 
					const_iterator const& end, const_iterator const& prev_pos) {
					string_type str;
					std::uint32_t code_point;
					while( itr != end && *itr != '\"' ) {
						// If non-ASCII character is found
						if( *itr < 0 || *itr >= 128 ) {
							// Treat all connected bytes as a whole
							auto from = itr;
							Policy::to_code_point(itr, end);
							str += string_type(from, ++itr);	// Take a substring
							continue;
						}
						// Escape syntax
						else if( *itr == '\\' ) {
							if( ++itr == end ) {
								throw_invalid_escape_exception("Expected one of '/', '\\', '\"', 'b', 'f', 'n', 'r', 't', or 'u', but "
									"reached the end of the string!", itr, prev_pos);
							}
							switch( *itr ) {
							case '/': str += '/'; break;
							case '\\': str += '\\'; break;
							case '\"': str += '\"'; break;
							case 'b': str += '\b'; break;
							case 'f': str += '\f'; break;
							case 'n': str += '\n'; break;
							case 'r': str += '\r'; break;
							case 't': str += '\t'; break;
							case 'u':
								code_point = four_hex_to_value(++itr, end, prev_pos);
								// For surrogate pair
								if( code_point >= 0xD800 && code_point < 0xE000 ) {
									if( ++itr == end || *itr != '\\' || ++itr == end || *itr != 'u' ) {
										throw text_parsing_error(std::string("The UTF-16 code point 0x") +
											value_to_hex_char(jkl::util::byte_at(code_point, 3)) +
											value_to_hex_char(jkl::util::byte_at(code_point, 2)) +
											value_to_hex_char(jkl::util::byte_at(code_point, 1)) +
											value_to_hex_char(jkl::util::byte_at(code_point, 0)) +
											" should come as a surrogate pair, but the second code point is not found!",
											text_parsing_error::invalid_utf16_codepoint, (char const*)&*prev_pos, (char const*)&*itr);
									}
									std::uint16_t low = four_hex_to_value(++itr, end, prev_pos);
									code_point = jkl::unicode::get_unicode({ (std::uint16_t)code_point, low });
								}
								str += Policy::code_point_to_string(code_point);
								break;
							default:
								throw_invalid_escape_exception(std::string("Unknown escape syntax! The letter after '\\' must be "
								"one of '/', '\\', '\"', 'b', 'f', 'n', 'r', 't', or 'u', but got '") + (char)*itr + "'!", itr, prev_pos);
							}
						}
						// ASCII characters
						else
							str += *itr;
						++itr;
					}
					if( itr == end ) {
						throw text_parsing_error("Expected '\"', but reached the end of the string!",
							text_parsing_error::quote_not_closed, (char const *)&*prev_pos, (char const*)&*itr);
					}
					++itr;
					return str;
				}

				static entry_pointer parse_array(const_iterator& itr, 
					const_iterator const& end, const_iterator const& prev_pos) {
					auto new_entry = json_entry<type::array, Policy>::create();

					// Before finding the first ','
					while( itr != end ) {
						auto new_item = parse(itr, end);
						if( new_item == nullptr ) {
							if( *itr == ']' )
								goto end_parsing;
							else {
								throw text_parsing_error(std::string("Expected ']', but got '") + (char)*itr + "'!", 
									text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
							}
						}
						new_entry->template unsafe_as<type::array>().append(new_item);

						// Find comma
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ']', but reached the end of the string!",
								text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr == ',' ) {
							++itr;
							break;
						}
						else if( *itr == ']' )
							goto end_parsing;
						else {
							throw text_parsing_error(std::string("Expected ',' or ']', but got '") + (char)*itr + "'!", 
								text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
						}
					}

					// After finding the first ','
					while( itr != end ) {
						auto new_item = parse(itr, end);
						if( new_item == nullptr ) {
							throw text_parsing_error("Cannot parse an entry!", text_parsing_error::invalid_char,
								(char const*)&*prev_pos, (char const*)&*itr);
						}
						new_entry->template unsafe_as<type::array>().append(new_item);

						// Find comma
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ']', but reached the end of the string!",
								text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr == ',' )
							++itr;
						else if( *itr == ']' )
							break;
						else {
							throw text_parsing_error(std::string("Expected ',' or ']', but got '") + (char)*itr + "'!",
								text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
						}
					}

					if( itr == end ) {
						throw text_parsing_error("Expected ']', but reached the end of the string!",
							text_parsing_error::ill_formed_array, (char const *)&*prev_pos, (char const*)&*itr);
					}

				end_parsing:					
					++itr;
					return std::move(new_entry);
				}

				static entry_pointer parse_object(const_iterator& itr, 
					const_iterator const& end, const_iterator const& prev_pos)
				{
					auto new_entry = json_entry<type::object, Policy>::create();

					// Before finding the first ','
					while( itr != end ) {
						// Find name
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected '\"' or '}', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr == '}' )
							goto end_parsing;

						else if( *itr != '\"' ) {
							throw text_parsing_error(std::string("Expected '\"' or '}', but got '") + (char)*itr + "'!", 
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						auto pos_before_name = itr;
						auto name = parse_string(++itr, end, pos_before_name);

						// Find colon
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ':', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr != ':' ) {
							throw text_parsing_error(std::string("Expected ':', but got '") + (char)*itr + "'!", 
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}

						// Find entry
						auto item = parse(++itr, end);
						if( item == nullptr ) {
							throw text_parsing_error("Cannot parse an entry!", text_parsing_error::invalid_char,
								(char const*)&*itr, (char const*)&*itr);
						}
						new_entry->template unsafe_as<type::object>().append(name, std::move(item));

						// Find comma
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ',' or '}', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr == ',' ) {
							++itr;
							break;
						}
						else if( *itr == '}' )
							goto end_parsing;
						else {
							throw text_parsing_error(std::string("Expected ',' or '}', but got '") + (char)*itr + "'!", 
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
					}

					while( itr != end ) {
						// Find name
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected '\"', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr != '\"' ) {
							throw text_parsing_error(std::string("Expected '\"', but got '") + (char)*itr + "'!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						auto pos_before_name = itr;
						auto name = parse_string(++itr, end, pos_before_name);

						// Find colon
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ':', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr != ':' ) {
							throw text_parsing_error(std::string("Expected ':', but got '") + (char)*itr + "'!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}

						// Find entry
						auto item = parse(++itr, end);
						if( item == nullptr ) {
							throw text_parsing_error("Cannot parse an entry!", text_parsing_error::invalid_char,
								(char const*)&*itr, (char const*)&*itr);
						}
						new_entry->template as<type::object>().append(name, std::move(item));

						// Find comma
						skip_space(itr, end);
						if( itr == end ) {
							throw text_parsing_error("Expected ',' or '}', but reached the end of the string!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
						else if( *itr == ',' )
							++itr;
						else if( *itr == '}' )
							break;
						else {
							throw text_parsing_error(std::string("Expected ',' or '}', but got '") + (char)*itr + "'!",
								text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
						}
					}

					if( itr == end ) {
						throw text_parsing_error("Expected '}', but reached the end of the string!",
							text_parsing_error::ill_foremd_object, (char const *)&*prev_pos, (char const*)&*itr);
					}

				end_parsing:
					++itr;
					return std::move(new_entry);
				}
			};
		}
		template <class Policy>
		typename Policy::entry_pointer parse(typename Policy::string_type const& input) {
			auto begin = input.begin();
			auto end = input.end();
			if( begin == end ) {
				throw text_parsing_error("Input string is empty!", text_parsing_error::empty_string, 
					(char const*)&*begin, (char const*)&*begin);
			}
			auto ret = detail::json_parser<Policy>::parse(begin, end);
			if( ret == nullptr ) {
				throw text_parsing_error("Cannot parse an entry!", text_parsing_error::invalid_char,
					(char const*)&*input.begin(), (char const*)&*begin);
			}
			detail::json_parser<Policy>::skip_space(begin, end);
			if( begin != end ) {
				throw text_parsing_error("Cannot parse an entry!", text_parsing_error::invalid_char,
					(char const*)&*begin, (char const*)&*begin);
			}
			return std::move(ret);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Convert the entry to a specific JSON entry type
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		template <class TargetEntry, class EntryPtr>
		auto& as(EntryPtr&& ptr) {
			return ptr->template as<TargetEntry>();
		}

		template <type tid_, class EntryPtr>
		auto& as(EntryPtr&& ptr) {
			return ptr->template as<tid_>();
		}

		template <class TargetEntry, class EntryPtr>
		auto& unsafe_as(EntryPtr&& ptr) {
			return ptr->template unsafe_as<TargetEntry>();
		}

		template <type tid_, class EntryPtr>
		auto& unsafe_as(EntryPtr&& ptr) {
			return ptr->template unsafe_as<tid_>();
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// Specializations for the case when JSONPolicy = default_policy
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <type tid_>
		using entry_type = json_entry<tid_, default_policy>;
		using entry_pointer = default_policy::entry_pointer;
		using null_entry = json_entry<type::null, default_policy>;
		using boolean_entry = json_entry<type::boolean, default_policy>;
		using number_entry = json_entry<type::number, default_policy>;
		using string_entry = json_entry<type::string, default_policy>;
		using array_entry = json_entry<type::array, default_policy>;
		using object_entry = json_entry<type::object, default_policy>;
	}
};