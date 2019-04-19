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

// A bitfield interface inspired by http://preshing.com/20150324/safe-bitfields-in-cpp/
// Elements are packed in lower bits -> upper bits order. Use this class like a tuple.
// Note that this class doesn't care overflow/underflow.

#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include "portability.h"
#include "tmp/forward.h"

namespace jkj {
	template <class UnderlyingType, std::size_t... field_lengths>
	class bitfield;

	template <class UnderlyingType, std::size_t... field_lengths>
	class bitfield_view;

	namespace detail {
		template <class UnderlyingType>
		class bitfield_base {
		public:
			static_assert(std::is_unsigned<UnderlyingType>::value,
				"jkj::bitfield can be only instantiated with unsigned integer types");

			using underlying_type = UnderlyingType;
			static constexpr std::size_t underlying_bits = sizeof(underlying_type) * 8;

			JKL_GPU_EXECUTABLE bitfield_base() = default;

			JKL_GPU_EXECUTABLE constexpr bitfield_base(UnderlyingType&, std::index_sequence<>) noexcept {}

			// For constexpr workaround of standard constructor
			template <std::size_t first_offset, std::size_t... remaining_offsets,
				class FirstElementType, class... RemainingElementTypes>
			JKL_GPU_EXECUTABLE constexpr bitfield_base(UnderlyingType& value,
				std::index_sequence<first_offset, remaining_offsets...>,
				FirstElementType&& first_element, RemainingElementTypes&&... remaining_elements)
				noexcept(noexcept(bitfield_base(
					value |= UnderlyingType(UnderlyingType(std::forward<FirstElementType>(first_element)) << first_offset),
					std::index_sequence<remaining_offsets...>{},
					std::forward<RemainingElementTypes>(remaining_elements)...))) :
				bitfield_base(
					value |= UnderlyingType(UnderlyingType(std::forward<FirstElementType>(first_element)) << first_offset),
					std::index_sequence<remaining_offsets...>{},
					std::forward<RemainingElementTypes>(remaining_elements)...) {}

			// Element type
			// Support conversion into the underlying type, basic arithmetic/bitwise operations
		private:
			template <std::size_t offset_, std::size_t field_length_, bool is_const>
			class element_reference_base;

		public:
			template <std::size_t offset, std::size_t field_length>
			class element_reference;

			template <std::size_t offset, std::size_t field_length>
			class element_const_reference;

			template <std::size_t offset_, std::size_t field_length_>
			class element_reference : public element_reference_base<offset_, field_length_, false> {
				using base_type = element_reference_base<offset_, field_length_, false>;
				using base_type::value;

				// Only bitfield and bitfield_view can construct this class
				template <class UnderlyingType, std::size_t... field_lengths>
				friend class bitfield;
				template <class UnderlyingType, std::size_t... field_lengths>
				friend class bitfield_view;

				JKL_GPU_EXECUTABLE constexpr element_reference(underlying_type& value) noexcept :
					base_type{ value } {}

			public:
				static constexpr std::size_t offset = base_type::offset;
				static constexpr std::size_t field_length = base_type::field_length;
				static constexpr underlying_type mask = base_type::mask;
				static constexpr underlying_type maximum = base_type::maximum;
				using reference = typename base_type::reference;
				using const_reference = typename base_type::const_reference;

				// Simple assignment
				JKL_GPU_EXECUTABLE element_reference& operator=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((that << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((that << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((std::move(that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((std::move(that) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// Addition assignment
				JKL_GPU_EXECUTABLE element_reference& operator+=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) + that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) + that) << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator+=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) + std::move(that)) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) + std::move(that)) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// Subtraction assignment
				JKL_GPU_EXECUTABLE element_reference& operator-=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) - that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) - that) << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator-=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) - std::move(that)) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) - std::move(that)) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// Multiplication assignment
				JKL_GPU_EXECUTABLE element_reference& operator*=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) * that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) * that) << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator*=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((((value >> offset) * std::move(that)) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type(((((value >> offset) * std::move(that)) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// Division assignment
				JKL_GPU_EXECUTABLE element_reference& operator/=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type((((((value & mask) >> offset) / that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((((((value & mask) >> offset) / that) << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator/=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type((((((value & mask) >> offset) / std::move(that)) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((((((value & mask) >> offset) / std::move(that)) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// Modulo assignment
				JKL_GPU_EXECUTABLE element_reference& operator%=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type((((((value & mask) >> offset) % that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((((((value & mask) >> offset) % that) << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator%=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type((((((value & mask) >> offset) % std::move(that)) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((((((value & mask) >> offset) % std::move(that)) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// AND assignment
				JKL_GPU_EXECUTABLE element_reference& operator&=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type((value & (that << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((value & (that << offset) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator&=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type((value & (std::move(that) << offset) & mask) | (value & ~mask))))
				{
					value = underlying_type((value & (std::move(that) << offset) & mask) | (value & ~mask));
					return *this;
				}

				// OR assignment
				JKL_GPU_EXECUTABLE element_reference& operator|=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((value | (that << offset)) & mask) | (value & ~mask))))
				{
					value = underlying_type(((value | (that << offset)) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator|=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((value | (std::move(that) << offset)) & mask) | (value & ~mask))))
				{
					value = underlying_type(((value | (std::move(that) << offset)) & mask) | (value & ~mask));
					return *this;
				}

				// XOR assignment
				JKL_GPU_EXECUTABLE element_reference& operator^=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type(((value ^ (that << offset)) & mask) | (value & ~mask))))
				{
					value = underlying_type(((value ^ (that << offset)) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator^=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type(((value ^ (std::move(that) << offset)) & mask) | (value & ~mask))))
				{
					value = underlying_type(((value ^ (std::move(that) << offset)) & mask) | (value & ~mask));
					return *this;
				}

				// Left-shift assignment
				JKL_GPU_EXECUTABLE element_reference& operator<<=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type((((value & mask) << that) & mask) | (value & ~mask))))
				{
					value = underlying_type((((value & mask) << that) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator<<=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type((((value & mask) << std::move(that)) & mask) | (value & ~mask))))
				{
					value = underlying_type((((value & mask) << std::move(that)) & mask) | (value & ~mask));
					return *this;
				}

				// Right-shift assignment
				JKL_GPU_EXECUTABLE element_reference& operator>>=(underlying_type const& that)
					noexcept(noexcept(value = underlying_type((((value & mask) >> that) & mask) | (value & ~mask))))
				{
					value = underlying_type((((value & mask) >> that) & mask) | (value & ~mask));
					return *this;
				}
				JKL_GPU_EXECUTABLE element_reference& operator>>=(underlying_type&& that)
					noexcept(noexcept(value = underlying_type((((value & mask) >> std::move(that)) & mask) | (value & ~mask))))
				{
					value = underlying_type((((value & mask) >> std::move(that)) & mask) | (value & ~mask));
					return *this;
				}

				// Increment
				JKL_GPU_EXECUTABLE element_reference& operator++()
					noexcept(noexcept(std::declval<element_reference>() += 1))
				{
					return *this += 1;
				}
				JKL_GPU_EXECUTABLE element_reference operator++(int)
					noexcept(noexcept(std::declval<element_reference>() += 1))
				{
					auto ret = *this;
					++*this;
					return ret;
				}

				// Decrement
				JKL_GPU_EXECUTABLE element_reference& operator--()
					noexcept(noexcept(std::declval<element_reference>() -= 1))
				{
					return *this -= 1;
				}
				JKL_GPU_EXECUTABLE element_reference operator--(int)
					noexcept(noexcept(std::declval<element_reference>() -= 1))
				{
					auto ret = *this;
					--*this;
					return ret;
				}
			};

			template <std::size_t offset_, std::size_t field_length_>
			class element_const_reference : public element_reference_base<offset_, field_length_, true> {
				using base_type = element_reference_base<offset_, field_length_, true>;

				// Only bitfield and bitfield_view can construct this class
				template <class UnderlyingType, std::size_t... field_lengths>
				friend class bitfield;
				template <class UnderlyingType, std::size_t... field_lengths>
				friend class bitfield_view;

				JKL_GPU_EXECUTABLE constexpr element_const_reference(underlying_type const& value) noexcept :
					base_type{ value } {}

			public:
				static constexpr std::size_t offset = base_type::offset;
				static constexpr std::size_t field_length = base_type::field_length;
				static constexpr underlying_type mask = base_type::mask;
				static constexpr underlying_type maximum = base_type::maximum;
				using reference = typename base_type::reference;
				using const_reference = typename base_type::const_reference;

				// Can be converted from non-const
				JKL_GPU_EXECUTABLE constexpr element_const_reference(reference that) noexcept
					: base_type{ that } {}
			};

		private:
			template <std::size_t offset_, std::size_t field_length_, bool is_const>
			class element_reference_base {
				// Hold the reference to the underlying type
				using underlying_reference = std::conditional_t<is_const, underlying_type const, underlying_type>&;
				underlying_reference value;

			public:
				static constexpr std::size_t offset = offset_;
				static constexpr std::size_t field_length = field_length_;
				static constexpr auto mask = underlying_type(
					underlying_type(underlying_type(-1) >> (underlying_bits - (offset + field_length))) &
					underlying_type(underlying_type(-1) << offset));
				static constexpr auto maximum = underlying_type((underlying_type(1) << field_length) - 1);

				using reference = element_reference<offset_, field_length_>;
				using const_reference = element_const_reference<offset_, field_length_>;

				friend class reference;

				// Constructors
				JKL_GPU_EXECUTABLE constexpr element_reference_base(underlying_reference value) noexcept : value{ value } {}
				JKL_GPU_EXECUTABLE constexpr element_reference_base(reference that) noexcept : value{ that.value } {}

				// Convert to underlying_type
				JKL_GPU_EXECUTABLE constexpr operator underlying_type() const
					noexcept(noexcept(underlying_type((value & mask) >> offset)))
				{
					return underlying_type((value & mask) >> offset);
				}
			};
		};

		template <class UnderlyingType>
		struct bitfield_value_container {
			UnderlyingType value;

			JKL_GPU_EXECUTABLE bitfield_value_container() = default;
			
			template <class T, class = jkj::tmp::prevent_too_perfect_fwd<bitfield_value_container, T>>
			JKL_GPU_EXECUTABLE bitfield_value_container(T&& v)
				noexcept(std::is_nothrow_constructible<UnderlyingType, T>::value) :
				value(std::forward<T>(v)) {}
		};
	}

	// Indicates that the constructor argument to bitfield will be given as a packed underlying_type
	struct bitpack_tag {};

	template <class UnderlyingType, std::size_t... field_lengths>
	class bitfield :
		private detail::bitfield_value_container<UnderlyingType>,
		private detail::bitfield_base<UnderlyingType>
	{
		using base_type = detail::bitfield_base<UnderlyingType>;
		using value_container = detail::bitfield_value_container<UnderlyingType>;

		using value_container::value;

	public:
		using underlying_type = typename base_type::underlying_type;
		static constexpr std::size_t underlying_bits = base_type::underlying_bits;
		static constexpr std::size_t number_of_fields = sizeof...(field_lengths);

	private:
		// Compute offset and field length
		template <class dummy, std::size_t idx, std::size_t... field_lengths>
		struct get_offset_and_length_impl;

		template <class dummy, std::size_t idx, std::size_t first_field_length, std::size_t... remaining_field_lengths>
		struct get_offset_and_length_impl<dummy, idx, first_field_length, remaining_field_lengths...> {
			using prev = get_offset_and_length_impl<dummy, idx - 1, remaining_field_lengths...>;

			static constexpr std::size_t offset = prev::offset + first_field_length;
			static constexpr std::size_t field_length = prev::field_length;
		};

		template <class dummy, std::size_t first_field_length, std::size_t... remaining_field_lengths>
		struct get_offset_and_length_impl<dummy, 0, first_field_length, remaining_field_lengths...> {
			static constexpr std::size_t offset = 0;
			static constexpr std::size_t field_length = first_field_length;
		};

		template <class dummy>
		struct get_offset_and_length_impl<dummy, 0> {
			static constexpr std::size_t offset = 0;
			static constexpr std::size_t field_length = 0;
		};

		template <std::size_t idx>
		using get_offset_and_length = get_offset_and_length_impl<void, idx, field_lengths...>;

	public:
		// Get offsets of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr std::size_t offset() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return get_offset_and_length<idx>::offset;
		}

		// Get field lengths of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr std::size_t field_length() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return get_offset_and_length<idx>::field_length;
		}

	private:
		struct helper_tag {};

		// Standard constructor helper
		template <class FirstElementType, class... RemainingElementTypes, std::size_t... I>
		JKL_GPU_EXECUTABLE constexpr bitfield(helper_tag, std::index_sequence<I...>,
			FirstElementType&& first_elmt, RemainingElementTypes&&... remaining_elmts)
			noexcept(std::is_nothrow_constructible<underlying_type, FirstElementType>::value &&
				std::is_nothrow_constructible<base_type, underlying_type&, std::index_sequence<offset<I>()...>,
				RemainingElementTypes...>::value) :
			value_container(std::forward<FirstElementType>(first_elmt)),
			base_type{ value, std::index_sequence<offset<I + 1>()...>{},
			std::forward<RemainingElementTypes>(remaining_elmts)... } {}
		
	public:
		static constexpr std::size_t total_bits = get_offset_and_length<number_of_fields>::offset;
		static_assert(total_bits <= underlying_bits,
			"jkj::bitfield: the specified type cannot hold the bitfield; too many bits are requested!");

		// Default constructor : random initial value
		JKL_GPU_EXECUTABLE bitfield() = default;

		// Pack constructor : initialize bits packed in an underlying_type
		// Remember that elements are packed in lower bits -> upper bits order.
		JKL_GPU_EXECUTABLE constexpr bitfield(bitpack_tag, underlying_type const& packed_bits)
			noexcept(std::is_nothrow_copy_constructible<underlying_type>::value) :
			value_container{ packed_bits } {}
		JKL_GPU_EXECUTABLE constexpr bitfield(bitpack_tag, underlying_type&& packed_bits)
			noexcept(std::is_nothrow_move_constructible<underlying_type>::value) :
			value_container{ std::move(packed_bits) } {}

		// Standard constructor: initialize the first element, when there is only one element
		template <class FirstElementType, class = std::enable_if_t<
			number_of_fields == 1 && !std::is_convertible<FirstElementType, bitpack_tag>::value>,
			class = jkj::tmp::prevent_too_perfect_fwd<bitfield, FirstElementType>>
		JKL_GPU_EXECUTABLE explicit constexpr bitfield(FirstElementType&& first_elmt)
			noexcept(std::is_nothrow_constructible<underlying_type, FirstElementType>::value) :
			value_container{ std::forward<FirstElementType>(first_elmt) } {}

		// Standard constructor: initialize elements, when there are more than one elements
		template <class FirstElementType, class SecondElementType, class... RemainingElementTypes,
			class = std::enable_if_t<number_of_fields == 2 + sizeof...(RemainingElementTypes)
			&& !std::is_convertible<FirstElementType, bitpack_tag>::value>>
		JKL_GPU_EXECUTABLE constexpr bitfield(FirstElementType&& first_elmt, SecondElementType&& second_elmt,
				RemainingElementTypes&&... remaining_elmts)
			noexcept(noexcept(bitfield{ helper_tag{}, std::make_index_sequence<sizeof...(RemainingElementTypes) + 1>{},
				std::forward<FirstElementType>(first_elmt), std::forward<SecondElementType>(second_elmt),
				std::forward<RemainingElementTypes>(remaining_elmts)... })) :
			bitfield{ helper_tag{}, std::make_index_sequence<sizeof...(RemainingElementTypes) + 1>{},
			std::forward<FirstElementType>(first_elmt), std::forward<SecondElementType>(second_elmt),
			std::forward<RemainingElementTypes>(remaining_elmts)... } {}


		// Element reference type
		template <std::size_t idx>
		using reference = typename base_type::template element_reference<offset<idx>(), field_length<idx>()>;
		template <std::size_t idx>
		using const_reference = typename base_type::template element_const_reference<offset<idx>(), field_length<idx>()>;

		// Get maximum possible values of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr underlying_type maximum() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return reference<idx>::maximum;
		}

		// Get element
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE constexpr reference<idx> get() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return{ value };
		}
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE constexpr const_reference<idx> get() const noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return{ value };
		}

		// Get packed value
		JKL_GPU_EXECUTABLE constexpr underlying_type const& get_packed() const noexcept {
			return value;
		}

	private:
		// These are for natvis

		template <class, std::size_t...>
		friend class bitfield;

		template <class dummy, bool length_zero = (number_of_fields == 0)>
		struct get_final_offset_and_mask {
			static constexpr std::size_t final_offset = offset<number_of_fields - 1>();
			static constexpr underlying_type final_mask = maximum<number_of_fields - 1>() << final_offset;
		};
		template <class dummy>
		struct get_final_offset_and_mask<dummy, true> {
			static constexpr std::size_t final_offset = 0;
			static constexpr underlying_type final_mask = 0;
		};

		static constexpr std::size_t final_offset = get_final_offset_and_mask<void>::final_offset;
		static constexpr underlying_type final_mask = get_final_offset_and_mask<void>::final_mask;

		template <class, std::size_t...>
		struct exclude_last_field;
		template <class dummy, std::size_t first, std::size_t... remainings>
		struct exclude_last_field<dummy, first, remainings...> {
			template <class T>
			struct add_to_first;

			template <std::size_t... I>
			struct add_to_first<bitfield<UnderlyingType, I...>> {
				using type = bitfield<UnderlyingType, first, I...>;
			};

			using type = typename
				add_to_first<typename bitfield<UnderlyingType, remainings...>::exclude_last_field_t>::type;
			using further = typename type::exclude_last_field_t;
		};
		template <class dummy, std::size_t first>
		struct exclude_last_field<dummy, first> {
			using type = bitfield<UnderlyingType>;
		};
		template <class dummy>
		struct exclude_last_field<dummy> {
			using type = bitfield<UnderlyingType>;
		};
		using exclude_last_field_t = typename exclude_last_field<void, field_lengths...>::type;
	};

	template <class UnderlyingType, std::size_t... field_lengths>
	class bitfield_view : private detail::bitfield_base<UnderlyingType> {
		using base_type = detail::bitfield_base<UnderlyingType>;

	public:
		using underlying_type = typename base_type::underlying_type;
		static constexpr std::size_t underlying_bits = base_type::underlying_bits;
		static constexpr std::size_t number_of_fields = sizeof...(field_lengths);

		using bitfield_type = bitfield<underlying_type, field_lengths...>;
		static constexpr std::size_t total_bits = bitfield_type::total_bits;

		// Constructor
		JKL_GPU_EXECUTABLE constexpr bitfield_view(underlying_type& value) noexcept
			: value{ value } {}

		// Get offsets of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr std::size_t offset() noexcept {
			return bitfield_type::template offset<idx>();
		}

		// Get field lengths of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr std::size_t field_length() noexcept {
			return bitfield_type::template field_length<idx>();
		}

		// Element reference type
		template <std::size_t idx>
		using reference = typename bitfield_type::template reference<idx>;
		template <std::size_t idx>
		using const_reference = typename bitfield_type::template const_reference<idx>;

		// Get maximum possible values of elements
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE static constexpr underlying_type maximum() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return reference<idx>::maximum;
		}

		// Get element
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE constexpr reference<idx> get() noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return{ value };
		}
		template <std::size_t idx>
		JKL_GPU_EXECUTABLE constexpr const_reference<idx> get() const noexcept {
			static_assert(idx < number_of_fields, "jkj::bitfield: index out of range!");
			return{ value };
		}

		// Get packed value
		JKL_GPU_EXECUTABLE constexpr underlying_type const& get_packed() const noexcept {
			return value;
		}

	private:
		underlying_type& value;

		// These are for natvis

		template <class, std::size_t...>
		friend class bitfield_view;

		template <class dummy, bool length_zero = (number_of_fields == 0)>
		struct get_final_offset_and_mask {
			static constexpr std::size_t final_offset = offset<number_of_fields - 1>();
			static constexpr underlying_type final_mask = maximum<number_of_fields - 1>() << final_offset;
		};
		template <class dummy>
		struct get_final_offset_and_mask<dummy, true> {
			static constexpr std::size_t final_offset = 0;
			static constexpr underlying_type final_mask = 0;
		};

		static constexpr std::size_t final_offset = get_final_offset_and_mask<void>::final_offset;
		static constexpr underlying_type final_mask = get_final_offset_and_mask<void>::final_mask;

		template <class, std::size_t...>
		struct exclude_last_field;
		template <class dummy, std::size_t first, std::size_t... remainings>
		struct exclude_last_field<dummy, first, remainings...> {
			template <class T>
			struct add_to_first;

			template <std::size_t... I>
			struct add_to_first<bitfield_view<UnderlyingType, I...>> {
				using type = bitfield_view<UnderlyingType, first, I...>;
			};

			using type = typename
				add_to_first<typename bitfield_view<UnderlyingType, remainings...>::exclude_last_field_t>::type;
			using further = typename type::exclude_last_field_t;
		};
		template <class dummy, std::size_t first>
		struct exclude_last_field<dummy, first> {
			using type = bitfield_view<UnderlyingType>;
		};
		template <class dummy>
		struct exclude_last_field<dummy> {
			using type = bitfield_view<UnderlyingType>;
		};
		using exclude_last_field_t = typename exclude_last_field<void, field_lengths...>::type;
	};

	template <class UnderlyingType, std::size_t... field_lengths>
	using const_bitfield_view = bitfield_view<std::add_const_t<UnderlyingType>, field_lengths...>;

	// std::get-like access
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield<UnderlyingType, field_lengths...>& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield<UnderlyingType, field_lengths...> const& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield<UnderlyingType, field_lengths...>&& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield<UnderlyingType, field_lengths...> const&& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield_view<UnderlyingType, field_lengths...>& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield_view<UnderlyingType, field_lengths...> const& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield_view<UnderlyingType, field_lengths...>&& t) noexcept {
		return t.template get<I>();
	}
	template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
	constexpr auto get(bitfield_view<UnderlyingType, field_lengths...> const&& t) noexcept {
		return t.template get<I>();
	}
}

// Specializations of std::tuple_size and std::tuple_element
template <class UnderlyingType, std::size_t... field_lengths>
class std::tuple_size<jkj::bitfield<UnderlyingType, field_lengths...>> :
	public std::integral_constant<std::size_t, sizeof...(field_lengths)> {};

template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
struct std::tuple_element<I, jkj::bitfield<UnderlyingType, field_lengths...>> {
	using type = typename jkj::bitfield<UnderlyingType, field_lengths...>::template reference<I>;
};
template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
struct std::tuple_element<I, jkj::bitfield<UnderlyingType, field_lengths...> const> {
	using type = typename jkj::bitfield<UnderlyingType, field_lengths...>::template const_reference<I>;
};

template <class UnderlyingType, std::size_t... field_lengths>
class std::tuple_size<jkj::bitfield_view<UnderlyingType, field_lengths...>> :
	public std::integral_constant<std::size_t, sizeof...(field_lengths)> {};

template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
struct std::tuple_element<I, jkj::bitfield_view<UnderlyingType, field_lengths...>> {
	using type = typename jkj::bitfield<UnderlyingType, field_lengths...>::template reference<I>;
};
template <std::size_t I, class UnderlyingType, std::size_t... field_lengths>
struct std::tuple_element<I, jkj::bitfield_view<UnderlyingType, field_lengths...> const> {
	using type = typename jkj::bitfield<UnderlyingType, field_lengths...>::template const_reference<I>;
};