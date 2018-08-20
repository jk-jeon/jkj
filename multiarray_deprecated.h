///////////////////////////////////////////////////////////////////////////////////////////////////////
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
#include <cassert>
#include <memory>
#include "tmp.h"

namespace jkl {
	/// Tensor view class
	/// A proxy class providing an ability to view a chunk of memory as a multi-dimensional array.
	/// This class is not responsible for allocating/releasing memory.

	template <std::size_t rank>
	struct extent_type;

	template <class... Args>
	FORCEINLINE constexpr extent_type<sizeof...(Args)> make_extent(Args&&... args) noexcept;

	// First, define "extent_type" which is nothing but a tuple of size_t's that are representing the 
	// length of the array on each dimension.
	template <std::size_t rank_>
	struct extent_type : private extent_type<rank_ - 1> {
	private:
		std::size_t			first;

	public:
		static constexpr auto rank = rank_;
		using subextent_type = extent_type<rank - 1>;

		// Constructors
		constexpr extent_type(std::size_t first) noexcept : first{ first } {
			static_assert(rank == 1, "extent_type initializer size mismatch!");
		}
		template <class Second, class... Remainings>
		constexpr extent_type(std::size_t first, Second&& second, Remainings&&... remainings) noexcept :
			subextent_type{ std::forward<Second>(second), std::forward<Remainings>(remainings)... },
			first{ first } {}


		// Accessors
		subextent_type& subextent() noexcept {
			return static_cast<subextent_type&>(*this);
		}
		constexpr subextent_type const& subextent() const noexcept {
			return static_cast<subextent_type const&>(*this);
		}
		template <std::size_t from, std::size_t subrank>
		decltype(auto) subextent() noexcept {
			return subextent_helper<from, rank - subrank>::get(*this);
		}
		template <std::size_t from, std::size_t subrank>
		constexpr decltype(auto) subextent() const noexcept {
			return subextent_helper<from, rank - subrank>::get(*this);
		}

		template <std::size_t I>
		std::size_t& get() noexcept {
			return get_helper<I>::get(*this);
		}
		template <std::size_t I>
		constexpr std::size_t const& get() const noexcept {
			return get_helper<I>::get(*this);
		}

		// Direct sum
		constexpr extent_type operator+(extent_type const& amount) const noexcept {
			return binary_op_helper(sum{}, amount, std::make_index_sequence<rank>());
		}

		// Tensor product
		constexpr extent_type operator*(extent_type const& amount) const noexcept {
			return binary_op_helper(product{}, amount, std::make_index_sequence<rank>());
		}

		constexpr std::size_t size() const noexcept {
			return first * subextent_size();
		}
		constexpr std::size_t subextent_size() const noexcept {
			return static_cast<subextent_type const&>(*this).size();
		}

		static constexpr extent_type zero() noexcept {
			return zero_helper(std::make_index_sequence<rank>());
		}

		// Comparisons (component-wise partial order)
		constexpr bool operator==(extent_type const& that) const noexcept {
			return first == that.first && subextent() == that.subextent();
		}
		constexpr bool operator!=(extent_type const& that) const noexcept {
			return !(*this == that);
		}
		constexpr bool operator<=(extent_type const& that) const noexcept {
			return first <= that.first && subextent() <= that.subextent();
		}
		constexpr bool operator>=(extent_type const& that) const noexcept {
			return first >= that.first && subextent() <= that.subextent();
		}
		constexpr bool operator<(extent_type const& that) const noexcept {
			return *this <= that && *this != that;
		}
		constexpr bool operator>(extent_type const& that) const noexcept {
			return *this >= that && *this != that;
		}

	private:
		template <std::size_t... I>
		static constexpr auto zero_helper(std::index_sequence<I...>) noexcept {
			return extent_type{ (I * 0)... };
		}

		template <std::size_t I, class = void>
		struct get_helper {
			static_assert(I < rank, "jkl::extent_type: index out of range!");

			template <class ExtentType>
			static constexpr auto&& get(ExtentType&& ext) noexcept {
				return ext.subextent().template get<I - 1>();
			}
		};

		template <class dummy>
		struct get_helper<0, dummy> {
			template <class ExtentType>
			static constexpr auto&& get(ExtentType&& ext) noexcept {
				return ext.first;
			}
		};

		template <std::size_t from, std::size_t complement_rank, class = void>
		struct subextent_helper {
			static constexpr std::size_t subrank = rank - complement_rank;
			static_assert(from + subrank < rank, "jkl::extent_type: index out of range!");

			template <class ExtentType, std::size_t... I>
			static constexpr extent_type<subrank> get(ExtentType&& ext, std::index_sequence<I...>) noexcept {
				return make_extent(std::forward<ExtentType>(ext).template get<from + I>()...);
			}

			template <class ExtentType>
			static constexpr extent_type<subrank> get(ExtentType&& ext) noexcept {
				return get(std::forward<ExtentType>(ext), std::make_index_sequence<subrank>{});
			}
		};

		template <class dummy>
		struct subextent_helper<0, 0, dummy> {
			template <class ExtentType>
			static constexpr ExtentType&& get(ExtentType&& ext) noexcept {
				return std::forward<ExtentType>(ext);
			}
		};

		template <class dummy>
		struct subextent_helper<1, 1, dummy> {
			template <class ExtentType>
			using return_t = jkl::tmp::propagate_qual_t<ExtentType, extent_type<rank - 1>>;

			template <class ExtentType>
			static constexpr return_t<ExtentType> get(ExtentType&& ext) noexcept {
				return static_cast<return_t<ExtentType>>(std::forward<ExtentType>(ext));
			}
		};

		// constexpr lambda's are not yet supported
		struct sum {
			FORCEINLINE constexpr auto operator()(std::size_t i, std::size_t j) const noexcept {
				return i + j;
			}
		};
		struct product {
			FORCEINLINE constexpr auto operator()(std::size_t i, std::size_t j) const noexcept {
				return i * j;
			}
		};

		template <class Op, std::size_t... I>
		FORCEINLINE constexpr extent_type binary_op_helper(Op op, extent_type const& amount,
			std::index_sequence<I...>) const noexcept
		{
			return{ op(get<I>(), amount.template get<I>())... };
		}
	};

	template <std::size_t I, std::size_t rank>
	std::size_t& get(extent_type<rank>& ext) noexcept {
		return ext.get<I>();
	}
	template <std::size_t I, std::size_t rank>
	constexpr std::size_t const& get(extent_type<rank> const& ext) noexcept {
		return ext.get<I>();
	}

	template <>
	struct extent_type<0> {
		static constexpr auto rank = 0;

		// Constructors
		constexpr extent_type() = default;

		// Accessors
		template <std::size_t from, std::size_t subrank>
		extent_type& subextent() noexcept {
			static_assert(from == 0 && subrank == 0, "jkl::extent_type: index out of range!");
			return *this;
		}
		template <std::size_t from, std::size_t subrank>
		constexpr extent_type const& subextent() const noexcept {
			static_assert(from == 0 && subrank == 0, "jkl::extent_type: index out of range!");
			return *this;
		}

		// Direct sum
		constexpr extent_type operator+(extent_type const&) const noexcept {
			return{};
		}

		// Tensor product
		constexpr extent_type operator*(extent_type const&) const noexcept {
			return{};
		}

		constexpr std::size_t size() const noexcept {
			return 1;
		}

		static constexpr extent_type zero() noexcept {
			return{};
		}

		// Comparisons
		constexpr bool operator==(extent_type const&) const noexcept {
			return true;
		}
		constexpr bool operator!=(extent_type const&) const noexcept {
			return false;
		}
		constexpr bool operator<=(extent_type const&) const noexcept {
			return true;
		}
		constexpr bool operator>=(extent_type const&) const noexcept {
			return true;
		}
		constexpr bool operator<(extent_type const&) const noexcept {
			return false;
		}
		constexpr bool operator>(extent_type const&) const noexcept {
			return false;
		}
	};
}

namespace std {
	template <std::size_t I, std::size_t rank>
	class tuple_element<I, ::jkl::extent_type<rank>> {
	public:
		static_assert(I < rank, "jkl::extent_type: index out of range!");
		using type = std::size_t;
	};

	template <std::size_t rank>
	class tuple_size< ::jkl::extent_type<rank>> : public std::integral_constant<std::size_t, rank> {};
}

namespace jkl {
	template <class... Args>
	FORCEINLINE constexpr extent_type<sizeof...(Args)> make_extent(Args&&... args) noexcept {
		return{ std::size_t(std::forward<Args>(args))... };
	}

	// Concatenation of extent_type
	namespace detail {
		template <std::size_t rank1, std::size_t rank2>
		struct join_extent_helper {
			FORCEINLINE static constexpr extent_type<rank1 + rank2> join(extent_type<rank1> const& e1, 
				extent_type<rank2> const& e2) noexcept
			{
				return join_extent_helper<1, rank1 + rank2 - 1>::join({ e1.template get<0>() },
					join_extent_helper<rank1 - 1, rank2>::join(e1.subextent(), e2));
			}
		};

		template <std::size_t rank2>
		struct join_extent_helper<0, rank2> {
			FORCEINLINE static constexpr extent_type<rank2> join(extent_type<0> const&,
				extent_type<rank2> const& e2) noexcept
			{
				return e2;
			}
		};

		template <std::size_t rank2>
		struct join_extent_helper<1, rank2> {
			template <std::size_t... I>
			static constexpr extent_type<rank2 + 1> join_aux(std::size_t e1,
				extent_type<rank2> const& e2, std::index_sequence<I...>) noexcept
			{
				return{ e1, e2.template get<I>()... };
			}

			FORCEINLINE static constexpr extent_type<rank2 + 1> join(extent_type<1> const& e1,
				extent_type<rank2> const& e2) noexcept
			{
				return join_aux(e1.template get<0>(), e2, std::make_index_sequence<rank2>());
			}
		};
	}

	template <std::size_t rank1, std::size_t rank2>
	FORCEINLINE constexpr extent_type<rank1 + rank2> join_extent(
		extent_type<rank1> const& e1, extent_type<rank2> const& e2) noexcept
	{
		return detail::join_extent_helper<rank1, rank2>::join(e1, e2);
	}

	// Next, define "multirange" struct, which represents a multi-fold range
	template <std::size_t rank_>
	struct multirange {
		using extent_type = jkl::extent_type<rank_>;
		using subextent_type = jkl::extent_type<rank_ - 1>;
		using subrange_type = multirange<rank_ - 1>;

		constexpr multirange(extent_type const& range) noexcept : range(range) {}

		extent_type	const		range;

		auto length() const noexcept {
			return range.size();
		}

		subrange_type subrange() const noexcept {
			return{ range.subextent() };
		}

		extent_type centinel() const noexcept {
			return join_extent<1>({ get<0>(range) }, subextent_type::zero());
		}
		extent_type last() const noexcept {
			return join_extent<1>({ get<0>(range) - 1 }, subrange().last());
		}

		extent_type& increment(extent_type& idx) const noexcept {
			if( idx == centinel() )
				return idx;

			auto s = idx.subextent();
			subrange().increment(s);

			if( s == subrange().centinel() )
				idx = join_extent<1>({ get<0>(idx) + 1 }, subextent_type::zero());
			else
				idx = join_extent<1>({ get<0>(idx) }, s);
			return idx;
		}

		extent_type& decrement(extent_type& idx) const noexcept {
			auto s = idx.subextent();
			if( s == subextent_type::zero() ) {
				if( get<0>(idx) == 0 )
					return idx;
				else
					idx = join_extent<1>({ get<0>(idx) - 1 }, subrange().last());
			}
			else
				idx = join_extent<1>({ get<0>(idx) }, subrange().decrement(s));
			return idx;
		}

		extent_type& increment(extent_type& idx, std::size_t n) const noexcept {
			auto d = n / subrange().length();
			auto r = n % subrange().length();

			if( d >= get<0>(range) - get<0>(idx) )
				idx = centinel();
			else {
				get<0>(idx) += d;

				auto s = idx.subextent();
				auto remaining = std::size_t(subrange().distance(s, subrange().centinel()));
				if( remaining > r )
					idx = join_extent<1>({ get<0>(idx) }, subrange().increment(s, r));
				else {
					if( ++get<0>(idx) == get<0>(range) )
						idx = centinel();
					else
						idx = join_extent<1>({ get<0>(idx) }, subrange().increment(s, r - remaining));
				}
			}
			return idx;
		}

		extent_type& decrement(extent_type& idx, std::size_t n) const noexcept {
			auto d = n / subrange().length();
			auto r = n % subrange().length();

			if( d > get<0>(idx) )
				idx = extent_type::zero();
			else {
				get<0>(idx) -= d;

				auto s = idx.subextent();
				auto remaining = std::size_t(subrange().distance(subextent_type::zero(), s));

				if( remaining >= r )
					idx = join_extent<1>({ get<0>(idx) }, subrange().decrement(s, r));
				else {
					if( --get<0>(idx) == 0 )
						idx = extent_type::zero();
					else {
						idx = join_extent<1>({ get<0>(idx) }, 
							subrange().decrement(subrange().centinel(), r - remaining));
					}
				}
			}
			return idx;
		}

		std::ptrdiff_t distance(extent_type const& from, extent_type const& to) const noexcept {
			if( get<0>(from) < get<0>(to) ) {
				auto r = subrange().distance(from.subextent(), subrange().centinel())
					+ subrange().distance(subextent_type::zero(), to.subextent());
				return (get<0>(to) - get<0>(from) - 1) * subrange().length() + r;
			} else if( get<0>(to) < get<0>(from) ) {
				auto r = subrange().distance(to.subextent(), subrange().centinel())
					+ subrange().distance(subextent_type::zero(), from.subextent());
				return -std::ptrdiff_t((get<0>(from) - get<0>(to) - 1) * subrange().length() + r);
			} else
				return subrange().distance(from.subextent(), to.subextent());
		}

		bool prior_to(extent_type const& candidate, extent_type const& reference) const noexcept {
			if( get<0>(candidate) < get<0>(reference) )
				return true;
			else if( get<0>(candidate) > get<0>(reference) )
				return false;
			else
				return subrange().prior_to(candidate.subextent(), reference.subextent());
		}
	};

	template <>
	struct multirange<1> {
		using extent_type = jkl::extent_type<1>;
		using subextent_type = jkl::extent_type<0>;
		using subrange_type = multirange<0>;

		constexpr multirange(extent_type const& range) noexcept : range(range) {}

		extent_type	const		range;

		constexpr auto length() const noexcept {
			return range.size();
		}

		constexpr subrange_type subrange() const noexcept {
			return{ range.subextent() };
		}

		constexpr extent_type centinel() const noexcept {
			return{ get<0>(range) };
		}
		constexpr extent_type last() const noexcept {
			return{ get<0>(range) - 1 };
		}

		extent_type& increment(extent_type& idx) noexcept {
			if( idx != centinel() )
				++get<0>(idx);
			return idx;
		}

		extent_type& decrement(extent_type& idx) noexcept {
			if( idx != extent_type::zero() )
				--get<0>(idx);
			return idx;
		}

		extent_type& increment(extent_type& idx, std::size_t n) noexcept {
			if( n >= get<0>(range) - get<0>(idx) )
				get<0>(idx) = get<0>(range);
			else
				get<0>(idx) += n;
			return idx;
		}

		extent_type& decrement(extent_type& idx, std::size_t n) noexcept {
			if( n >= get<0>(idx) )
				get<0>(idx) = 0;
			else
				get<0>(idx) -= n;
			return idx;
		}

		std::ptrdiff_t distance(extent_type const& from, extent_type const& to) noexcept {
			return get<0>(to) - get<0>(from);
		}

		bool prior_to(extent_type const& candidate, extent_type const& reference) noexcept {
			return get<0>(candidate) < get<0>(reference);
		}
	};

	template <std::size_t rank>
	constexpr extent_type<rank> index_to_position(std::size_t idx, extent_type<rank> const& ext) noexcept {
		return join_extent<1>({ idx / ext.subextent_size() },
			index_to_position(idx % ext.subextent_size(), ext.subextent()));
	}

	constexpr extent_type<1> index_to_position(std::size_t idx, extent_type<1> const&) noexcept {
		return{ idx };
	}

	// A random access iterator built over multirange
	template <class MultiDimContainer, std::size_t rank, bool is_const>
	class multi_iterator {
		MultiDimContainer*	m_container_ptr;
		std::size_t			m_idx;
		extent_type<rank>	m_extent;

		template <class dummy>
		struct is_full_rank : std::integral_constant<bool, rank == MultiDimContainer::rank> {};

	public:
		// Special utility functions
		auto pos() const noexcept {
			return index_to_position(m_idx, m_extent);
		}
		template <std::size_t i>
		auto pos() const noexcept {
			return get<i>(pos());
		}
		auto const& container() const noexcept {
			return *m_container_ptr;
		}

		using iterator_category = std::random_access_iterator_tag;
		using value_type = std::remove_cv_t<std::remove_reference_t<decltype((*m_container_ptr)[m_extent])>>;
		using difference_type = typename MultiDimContainer::difference_type;
		using reference = std::conditional_t<is_const,
			decltype((*static_cast<MultiDimContainer const*>(m_container_ptr))[m_extent]),
			decltype((*m_container_ptr)[m_extent])
		>;
		using pointer = std::remove_reference_t<reference>*;

		friend class multi_iterator<MultiDimContainer, rank, !is_const>;
		constexpr multi_iterator(MultiDimContainer& container, std::size_t idx) noexcept 
			: m_container_ptr{ &container }, m_idx{ idx },
			m_extent{ m_container_ptr->extent().template subextent<0, rank>() } {}

		// Conversion from iterator to const_iterator
		constexpr multi_iterator(multi_iterator<MultiDimContainer, rank, false> const& that) noexcept
			: m_container_ptr{ that.m_container_ptr }, m_idx{ that.m_idx }, m_extent{ that.m_extent } {}

		// Requirements from Iterator
		reference operator*() const noexcept {
			return (*m_container_ptr)[pos()];
		}
		multi_iterator& operator++() noexcept {
			++m_idx;
			return *this;
		}

		// Requirements from InputIterator
		template <bool that_is_const>
		constexpr bool operator==(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return m_idx == that.m_idx;
		}
		template <bool that_is_const>
		constexpr bool operator!=(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return !(*this == that);
		}
		// This operation cannot be defined when rank != MultiDimContainer::rank;
		// otherwise, the result of dereference is a temporary value, thus taking the address is meaningless
		template <class dummy = void, class = std::enable_if_t<is_full_rank<dummy>::value>>
		pointer operator->() const noexcept {
			return &(*m_container_ptr)[pos()];
		}
		multi_iterator& operator++(int) noexcept {
			multi_iterator ret_value = *this;
			++(*this);
			return ret_value;
		}

		// Requirements from ForwardIterator
		constexpr multi_iterator() noexcept : m_container_ptr{ nullptr }, m_idx{ 0 },
			m_extent{ extent_type<rank>::zero() } {}

		// Requirements from BidirectionalIterator
		multi_iterator& operator--() noexcept {
			--m_idx;
			return *this;
		}
		multi_iterator& operator--(int) noexcept {
			multi_iterator ret_value = *this;
			--(*this);
			return ret_value;
		}

		// Requirements from RandomAccessIterator
		multi_iterator& operator+=(difference_type n) noexcept {
			m_idx += n;
			return *this;
		}
		multi_iterator operator+(difference_type n) const noexcept {
			multi_iterator ret_value = *this;
			return ret_value += n;
		}
		friend multi_iterator operator+(difference_type n, multi_iterator const& itr) noexcept {
			return itr + n;
		}
		multi_iterator& operator-=(difference_type n) noexcept {
			m_idx -= n;
			return *this;
		}
		multi_iterator operator-(difference_type n) const noexcept {
			multi_iterator ret_value = *this;
			return ret_value -= n;
		}
		template <bool that_is_const>
		typename difference_type operator-(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return m_idx - that.m_idx;
		}
		typename reference operator[](std::size_t n) const noexcept {
			return *((*this) + n);
		}

		template <bool that_is_const>
		bool operator<(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return m_idx < that.m_idx;
		}
		template <bool that_is_const>
		bool operator>(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return that < *this;
		}
		template <bool that_is_const>
		bool operator<=(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return *this == that || *this < that;
		}
		template <bool that_is_const>
		bool operator>=(multi_iterator<MultiDimContainer, rank, that_is_const> const& that) const noexcept {
			return that <= *this;
		}
	};

	// The multiarray_ref class
	// T must not be a reference.
	template <class T, std::size_t rank_, std::size_t alignment_ = alignof(T)>
	class multiarray_ref
	{
		using storage_elmt_type = jkl::tmp::propagate_cv_t<T, std::aligned_storage_t<sizeof(T), alignment_>>;
	public:
		static constexpr auto rank = rank_;
		static constexpr auto alignment = alignment_;

		using extent_type = jkl::extent_type<rank>;
		using subextent_type = jkl::extent_type<rank - 1>;
		using const_multiarray_ref = multiarray_ref<std::add_const_t<T>, rank, alignment>;
		using volatile_multiarray_ref = multiarray_ref<std::add_volatile_t<T>, rank, alignment>;
		using const_volatile_multiarray_ref = multiarray_ref<std::add_cv_t<T>, rank, alignment>;

		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using const_reference = T const&;
		using pointer = T*;
		using const_pointer = T const*;
		using iterator = multi_iterator<multiarray_ref const, rank, false>;
		using const_iterator = multi_iterator<multiarray_ref const, rank, true>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		friend iterator;
		friend const_iterator;

		constexpr multiarray_ref(jkl::tmp::propagate_cv_t<T, void>* data, extent_type const& extent) noexcept
			: multiarray_ref(data, extent, extent.subextent()) {}
		constexpr multiarray_ref(jkl::tmp::propagate_cv_t<T, void>* data, 
			extent_type const& extent, subextent_type const& stride) noexcept
			: m_data(reinterpret_cast<storage_elmt_type*>(data)),
			m_extent(extent), m_stride(stride) {}
		constexpr multiarray_ref(multiarray_ref const& that, extent_type const& extent, 
			extent_type const& origin = extent_type::zero()) noexcept
			: multiarray_ref(that.m_data + get<0>(origin) * that.stride().size(), 
				extent, that.m_stride + origin.subextent()) {}

		// Copy semantics

		template <class U = T, class = std::enable_if_t<!std::is_const<U>::value>>
		constexpr operator const_multiarray_ref() const noexcept {
			return{ m_data, m_extent, m_stride };
		}
		template <class U = T, class = void, class = std::enable_if_t<!std::is_volatile<U>::value>>
		constexpr operator volatile_multiarray_ref() const noexcept {
			return{ m_data, m_extent, m_stride };
		}
		template <class U = T, class = void, class = void,
			class = std::enable_if_t<!std::is_const<U>::value && !std::is_volatile<U>::value>>
		constexpr operator const_volatile_multiarray_ref() const noexcept {
			return{ m_data, m_extent, m_stride };
		}

		void rebind(jkl::tmp::propagate_cv_t<T, void>* chunk, extent_type const& extent) noexcept {
			rebind(chunk, extent, extent.subextent());
		}
		void rebind(jkl::tmp::propagate_cv_t<T, void>* chunk, 
			extent_type const& extent, subextent_type const& stride) noexcept
		{
			std::tie(m_data, m_extent, m_stride) = std::forward_as_tuple(chunk, extent, stride);
		}
		void rebind(multiarray_ref const& that, extent_type const& extent,
			extent_type const& origin = extent_type::zero()) noexcept
		{
			rebind(that.m_data + get<0>(origin) * that.stride().size(), 
				extent, that.m_stride + origin.subextent());
		}
			
		jkl::tmp::propagate_cv_t<T, void>* data() const noexcept {
			return m_data;
		}
		constexpr auto const& extent() const noexcept{ 
			return m_extent;
		}
		template <std::size_t i>
		constexpr auto extent() const noexcept {
			return get<i>(m_extent);
		}
		constexpr auto const& stride() const noexcept {
			return m_stride;
		}
		constexpr auto size() const noexcept {
			return m_extent.size();
		}
		constexpr auto length() const noexcept {
			return get<0>(m_extent);
		}

		FORCEINLINE constexpr multiarray_ref<T, rank - 1, alignment> operator[](std::size_t idx) const noexcept {
			return{ m_data + idx * m_stride.size(), m_extent.subextent(), m_stride.subextent() };
		}

		template <std::size_t other_rank>
		constexpr auto operator[](jkl::extent_type<other_rank> const& idx) const noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}

		constexpr T& operator[](extent_type const& idx) const noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}

		FORCEINLINE constexpr auto& operator[](jkl::extent_type<0> const& idx) const noexcept {
			return *this;
		}
			
		constexpr multiarray_ref subview(extent_type const& extent, extent_type const& origin) const noexcept {
			return{ m_data + get<0>(origin) * m_stride.size(), extent, m_stride + origin.subextent() };
		}

		// Although multiarray_ref is not a container, container-like functionalities are provided
		// Remember that multiarray_ref follows pointer-like semantics

		// Here, if subrank is smaller than rank, the deference operator of the returned iterator
		// gives a multiarray_ref of smaller size (or pointer, when subrank = rank - 1).
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, false> begin() const noexcept {
			static_assert(subrank != 0, "jkl::multiarray_ref: Iterator must have positive rank!");
			static_assert(subrank <= rank, "jkl::multiarray_ref: Index out of range!");
			return{ *this, 0 };
		}
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, false> end() const noexcept {
			return{ *this, m_extent.template subextent<0, subrank>().size() };
		}
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, true> cbegin() const noexcept { return begin<subrank>(); }
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, true> cend() const noexcept { return end<subrank>(); }

		template <std::size_t subrank = rank>
		auto rbegin() const noexcept { return std::make_reverse_iterator(end<subrank>()); }
		template <std::size_t subrank = rank>
		auto rend() const noexcept { return std::make_reverse_iterator(begin<subrank>()); }
		template <std::size_t subrank = rank>
		auto crbegin() const noexcept { return std::make_reverse_iterator(cend<subrank>()); }
		template <std::size_t subrank = rank>
		auto crend() const noexcept { return std::make_reverse_iterator(cbegin<subrank>()); }

		bool operator==(multiarray_ref const& that) noexcept {
			return m_data == that.m_data && m_extent == that.m_extent && m_stride == that.m_stride;
		}

		bool operator!=(multiarray_ref const& that) noexcept {
			return !(*this == that);
		}

		bool empty() const noexcept {
			return size() == 0;
		}

	private:
		storage_elmt_type*					m_data;
		extent_type							m_extent;
		subextent_type						m_stride;
	};

	template <class T, std::size_t alignment_>
	class multiarray_ref<T, 1, alignment_>
	{
		using storage_elmt_type = jkl::tmp::propagate_cv_t<T, std::aligned_storage_t<sizeof(T), alignment_>>;
	public:
		static constexpr auto rank = 1;
		static constexpr auto alignment = alignment_;

		using extent_type = jkl::extent_type<1>;
		using subextent_type = jkl::extent_type<0>;
		using const_multiarray_ref = multiarray_ref<std::add_const_t<T>, 1, alignment>;
		using volatile_multiarray_ref = multiarray_ref<std::add_volatile_t<T>, 1, alignment>;
		using const_volatile_multiarray_ref = multiarray_ref<std::add_cv_t<T>, 1, alignment>;

		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using const_reference = T const&;
		using pointer = T*;
		using const_pointer = T const*;
		using iterator = multi_iterator<multiarray_ref const, 1, false>;
		using const_iterator = multi_iterator<multiarray_ref const, 1, true>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		constexpr multiarray_ref(jkl::tmp::propagate_cv_t<T, void>* data, 
			extent_type const& extent, subextent_type const& = {}) noexcept
			: m_data(reinterpret_cast<storage_elmt_type*>(data)), m_extent(extent) {}
		constexpr multiarray_ref(multiarray_ref const& that, extent_type const& extent, 
			extent_type const& origin = extent_type::zero()) noexcept
			: multiarray_ref(that.m_data + get<0>(that.m_origin), extent) {}
		
		template <class U = T, class = std::enable_if_t<!std::is_const<U>::value>>
		constexpr operator const_multiarray_ref() const noexcept {
			return{ m_data, m_extent };
		}
		template <class U = T, class = void, class = std::enable_if_t<!std::is_volatile<U>::value>>
		constexpr operator volatile_multiarray_ref() const noexcept {
			return{ m_data, m_extent };
		}
		template <class U = T, class = void, class = void, 
			class = std::enable_if_t<!std::is_const<U>::value && !std::is_volatile<U>::value>>
		constexpr operator const_volatile_multiarray_ref() const noexcept {
			return{ m_data, m_extent };
		}

		void rebind(jkl::tmp::propagate_cv_t<T, void>* data, extent_type const& extent, 
			extent_type const& origin = extent_type::zero(), subextent_type const& dummy = {}) noexcept {
			std::tie(m_data, m_extent) = std::forward_as_tuple(data, extent);
		}
		void rebind(multiarray_ref const& that, extent_type const& extent,
			extent_type const& origin = extent_type::zero()) noexcept {
			rebind(that.m_data + get<0>(origin), extent);
		}

		jkl::tmp::propagate_cv_t<T, void>* data() const noexcept {
			return m_data;
		}
		constexpr auto const& extent() const noexcept {
			return m_extent;
		}
		template <std::size_t i>
		constexpr auto extent() const noexcept {
			return get<i>(m_extent);
		}
		constexpr auto stride() const noexcept {
			return subextent_type{};
		}
		constexpr auto size() const noexcept {
			return m_extent.size();
		}
		constexpr auto length() const noexcept {
			return get<0>(m_extent);
		}

		FORCEINLINE constexpr T& operator[](std::size_t idx) const noexcept {
			return *reinterpret_cast<T*>(m_data + idx);
		}

		FORCEINLINE constexpr T& operator[](extent_type const& idx) const noexcept {
			return (*this)[get<0>(idx)];
		}

		FORCEINLINE constexpr auto& operator[](subextent_type const& idx) const noexcept {
			return *this;
		}

		constexpr multiarray_ref subview(extent_type const& extent, extent_type const& origin) const noexcept {
			return{ m_data + get<0>(origin), extent };
		}

		// Although multiarray_ref is not a container, container-like functionalities are provided
		// Remember that multiarray_ref follows pointer-like semantics

		// Here, subrank must be equal to rank, as rank == 1
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, false> begin() const noexcept {
			static_assert(subrank != 0, "jkl::multiarray_ref: Iterator must have positive rank!");
			static_assert(subrank <= rank, "jkl::multiarray_ref: Index out of range!");
			return{ *this, 0 };
		}
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, false> end() const noexcept {
			return{ *this, m_extent.template subextent<0, subrank>().size() };
		}
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, true> cbegin() const noexcept { return begin<subrank>(); }
		template <std::size_t subrank = rank>
		multi_iterator<multiarray_ref const, subrank, true> cend() const noexcept { return end<subrank>(); }

		template <std::size_t subrank = rank>
		auto rbegin() const noexcept { return std::make_reverse_iterator(end<subrank>()); }
		template <std::size_t subrank = rank>
		auto rend() const noexcept { return std::make_reverse_iterator(begin<subrank>()); }
		template <std::size_t subrank = rank>
		auto crbegin() const noexcept { return std::make_reverse_iterator(cend<subrank>()); }
		template <std::size_t subrank = rank>
		auto crend() const noexcept { return std::make_reverse_iterator(cbegin<subrank>()); }

		bool operator==(multiarray_ref const& that) noexcept {
			return m_data == that.m_data && m_extent == that.m_extent;
		}

		bool operator!=(multiarray_ref const& that) noexcept {
			return !(*this == that);
		}

		bool empty() const noexcept {
			return size() == 0;
		}

	private:
		storage_elmt_type*					m_data;
		extent_type							m_extent;
	};

	template <class T, std::size_t rank_, std::size_t alignment_ = alignof(T)>
	using const_multiarray_ref = multiarray_ref<std::add_const_t<T>, rank_, alignment_>;
	template <class T, std::size_t rank_, std::size_t alignment_ = alignof(T)>
	using volatile_multiarray_ref = multiarray_ref<std::add_volatile_t<T>, rank_, alignment_>;
	template <class T, std::size_t rank_, std::size_t alignment_ = alignof(T)>
	using const_volatile_multiarray_ref = multiarray_ref<std::add_cv_t<T>, rank_, alignment_>;


	/// Multi-dimensional dynamic array class
	/// Not like multiarray_ref, this class allocates/deallocates itself.
	/// Not like std::vector, this class has no ability to grow/shrink
		
	// A helper type alias representing nested initializer-list
	namespace detail {
		template <class T, std::size_t rank>
		struct nested_initializer_list_helper {
			using type = std::initializer_list<typename nested_initializer_list_helper<T, rank - 1>::type>;
		};

		template <class T>
		struct nested_initializer_list_helper<T, 0> {
			using type = T;
		};
	}
	template <class T, std::size_t rank>
	using nested_initializer_list = typename detail::nested_initializer_list_helper<T, rank>::type;

	// Tags for constructor
	struct copy_raw_buffer_tag {};
	struct takes_ownership_tag {};

	// A helper type for initializing the array by invoking a functor
	template <std::size_t rank, class Functor>
	struct multiarray_initializer {
		extent_type<rank>	extent;
		Functor				functor;

		multiarray_initializer(extent_type<rank> const& extent, Functor&& functor)
			noexcept(std::is_nothrow_move_constructible<Functor>::value)
			: extent{ extent }, functor{ std::forward<Functor>(functor) } {}
	};

	template <std::size_t rank, class Functor>
	multiarray_initializer<rank, Functor> make_multiarray_initializer(extent_type<rank> const& extent, Functor&& functor)
		noexcept(std::is_nothrow_move_constructible<Functor>::value)
	{
		return{ extent, std::forward<Functor>(functor) };
	}


	// The multiarray class
	// T must not be a reference.
	template <class T, std::size_t rank_, std::size_t alignment_ = alignof(T)>
	class multiarray
	{
		using storage_elmt_type = std::aligned_storage_t<sizeof(T), alignment_>;

		// Call destructors for all elements up to bound
		static void destroy_up_to(storage_elmt_type* ptr, std::size_t bound) noexcept {
			for( std::size_t i = 0; i < bound; ++i )
				reinterpret_cast<T*>(ptr + i)->~T();
		}
		void destroy_all() noexcept {
			destroy_up_to(m_data.get(), size());
		}

		// Call constructors for all elements
		// If a constructor throws, all the previously contructed elements are destroyed
		template <bool is_memset_possible = std::is_trivially_constructible<T>::value && 
			sizeof(T) == 1 && alignment_ == 1, class = void>
		struct initialize_helper {};

		template <class dummy>
		struct initialize_helper<false, dummy> {
			template <class... Args>
			static void initialize(storage_elmt_type* dst, std::size_t size, Args&&... args) {
				for( std::size_t i = 0; i < size; ++i ) {
					try {
						new(dst + i) T(args...);
					}
					catch( ... ) {
						destroy_up_to(dst, i);
						throw;
					}
				}
			}
		};

		template <class dummy>
		struct initialize_helper<true, dummy> {
			template <class... Args>
			FORCEINLINE static void initialize(storage_elmt_type* dst, std::size_t size, Args&&... args) {
				int temp;
				new(&temp) T(std::forward<Args>(args)...);
				std::memset(dst, temp, size);
			}
		};

		template <class Functor, class ExtentType>
		static void functor_initialize(storage_elmt_type* dst, ExtentType const& ext, Functor&& functor) {
			for( std::size_t idx = 0; idx < ext.size(); ++idx ) {
				try {
					new(dst + idx) T(functor(index_to_position(idx, ext)));
				}
				catch( ... ) {
					destroy_up_to(dst, idx);
					throw;
				}
			}
		}

		// Copy-construct elements from a buffer to another buffer
		// If a constructor throws, all the previously contructed elements are destroyed
		template <bool is_trivial = std::is_trivially_constructible<T>::value, class = void>
		struct copy_helper {};

		template <class dummy>
		struct copy_helper<false, dummy> {
			static void copy(storage_elmt_type* dst, storage_elmt_type const* src, std::size_t size) {
				for( std::size_t i = 0; i < size; ++i ) {
					try {
						new(dst + i) T(*reinterpret_cast<T const*>(&src[i]));
					}
					catch( ... ) {
						destroy_up_to(dst, i);
						throw;
					}
				}
			}

			template <class View>
			static void copy_view(storage_elmt_type* dst, View&& view) {
				std::size_t i = 0;
				for( auto const& elmt : view ) {
					try {
						new(dst + i) T(elmt);
					}
					catch( ... ) {
						destroy_up_to(dst, i);
						throw;
					}
					++i;
				}
			}
		};

		template <class dummy>
		struct copy_helper<true, dummy> {
			FORCEINLINE static void copy(storage_elmt_type* dst, storage_elmt_type const* src, std::size_t size) {
				std::memcpy(dst, src, alignment_ * size);
			}

			template <class View, class = std::enable_if_t<rank_ >= 2>>
			static void copy_view(storage_elmt_type* dst, View&& view) {
				if( view.extent().subextent() == view.stride() )
					std::memcpy(dst, view.data(), alignment_ * view.size());
				else {
					for( std::size_t i = 0; i < view.length(); ++i, dst += view.extent().subextent().size() )
						multiarray<T, rank_ - 1, alignment_>::template copy_helper<>::copy_view(dst, view[i]);
				}
			}

			template <class View, class = std::enable_if_t<rank_ == 1>, class = void>
			static void copy_view(storage_elmt_type* dst, View&& view) {
				std::memcpy(dst, view.data(), alignment_ * view.size());
			}
		};

		template <class Initializer>
		void resize_helper(jkl::extent_type<rank_> const& new_extent, Initializer&& initializer) {
			// Destroy previous elements
			destroy_all();

			// Make the extent to be zero
			auto prev_size = m_extent.size();
			m_extent = extent_type::zero();

			decltype(m_data) new_buffer;

			// If resizing is required
			if( new_extent.size() > prev_size ) {
				// Destroy the previous buffer and then make a new buffer
				m_data.reset();
				new_buffer = std::make_unique<storage_elmt_type[]>(new_extent.size());
			}
			// Otherwise, set new_buffer as m_data
			else {
				std::swap(new_buffer, m_data);
			}

			// Initialize the new buffer
			initializer(new_buffer.get(), new_extent);

			// Set the newly created buffer to this
			std::swap(m_data, new_buffer);
			m_extent = new_extent;
		}

	public:
		static constexpr auto rank = rank_;
		static constexpr auto alignment = alignment_;

		using extent_type = jkl::extent_type<rank>;
		using subextent_type = jkl::extent_type<rank - 1>;
		template <std::size_t subrank>
		using index_type = jkl::extent_type<subrank>;

		using view_type = multiarray_ref<T, rank, alignment>;
		using const_multiarray_ref = multiarray_ref<std::add_const_t<T>, rank, alignment>;
		using volatile_multiarray_ref = multiarray_ref<std::add_volatile_t<T>, rank, alignment>;
		using const_volatile_multiarray_ref = multiarray_ref<std::add_cv_t<T>, rank, alignment>;

		using subview_type = std::conditional_t<rank >= 2, 
			multiarray_ref<T, rank - 1, alignment>, T&>;
		using const_subview_type = std::conditional_t<rank >= 2, 
			multiarray_ref<std::add_const_t<T>, rank - 1, alignment>, T const&>;
		using volatile_subview_type = std::conditional_t<rank >= 2, 
			multiarray_ref<std::add_volatile_t<T>, rank - 1, alignment>, T volatile&>;
		using const_volatile_subview_type = std::conditional_t<rank >= 2, 
			multiarray_ref<std::add_cv_t<T>, rank - 1, alignment>, T const volatile&>;

		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using const_reference = T const&;
		using pointer = T*;
		using const_pointer = T const*;
		using iterator = multi_iterator<multiarray, rank, false>;
		using const_iterator = multi_iterator<multiarray const, rank, true>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		friend class multiarray<T, rank_ + 1, alignment_>;

		// Default constructor does nothing
		multiarray() = default;

		// Construct the array, call the default constructor for each element
		multiarray(extent_type const& extent) : m_extent(extent),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			initialize_helper<>::initialize(m_data.get(), m_extent.size());
		}

		// Construct the array, fill it with a certain value
		multiarray(extent_type const& extent, T const& value) : m_extent(extent),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			initialize_helper<>::initialize(m_data.get(), m_extent.size(), value);
		}

		// Construct the array, initialize all the elements using a certain set of arguments
		template <class FirstArg, class... RemainingArgs, class = jkl::tmp::prevent_too_perfect_fwd<T, FirstArg>>
		multiarray(extent_type const& extent, FirstArg&& first_arg, RemainingArgs&&... remaining_args) 
			: m_extent(extent), m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			initialize_helper<>::initialize(m_data.get(), m_extent.size(),
				std::forward<FirstArg>(first_arg), std::forward<RemainingArgs>(remaining_args)...);
		}

		// Initialize by invoking a functor
		// The functor must take one argument of type index_type<rank>
		template <class Functor>
		multiarray(multiarray_initializer<rank, Functor> const& initializer) : m_extent(initializer.extent),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			functor_initialize(m_data.get(), initializer.extent, initializer.functor);
		}
		template <class Functor>
		multiarray(multiarray_initializer<rank, Functor>&& initializer) : m_extent(initializer.extent),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			functor_initialize(m_data.get(), initializer.extent, std::move(initializer.functor));
		}

		// Copy from a raw buffer of already initialized elements
		multiarray(void const* data, extent_type const& extent, copy_raw_buffer_tag = {}) : m_extent(extent), 
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			copy_helper<>::copy(m_data.get(), reinterpret_cast<storage_elmt_type*>(data), m_extent.size());
		}

		// Does not construct the array, but takes the ownership of a chunk of memory with 
		// already initialized elements; on destruction, the memory is being released
		multiarray(void* data, extent_type const& extent, takes_ownership_tag) : m_extent(extent), 
			m_data(reinterpret_cast<storage_elmt_type*>(data)) {}

		// Copy constructor
		multiarray(multiarray const& that) : m_extent(that.m_extent),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			copy_helper<>::copy(m_data.get(), that.m_data.get(), m_extent.size());
		}

		// Move constructor
		multiarray(multiarray&& that) = default;

		// Copy from view
		multiarray(view_type const& view) : m_extent(view.extent()),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			copy_helper<>::copy_view(m_data.get(), view);
		}
		multiarray(const_multiarray_ref const& view) : m_extent(view.extent()),
			m_data(std::make_unique<storage_elmt_type[]>(m_extent.size())) {
			copy_helper<>::copy_view(m_data.get(), view);
		}


		// Copy assignment operator
		multiarray& operator=(multiarray const& that) {
			if( this != &that ) {
				multiarray temp = that;
				swap(temp);
			}
			return *this;
		}

		// Move assignment operator
		multiarray& operator=(multiarray&& that) = default;

		// Copy assign from view
		multiarray& operator=(view_type const& view) {
			multiarray temp = view;
			swap(temp);
			return *this;
		}
		multiarray& operator=(const_multiarray_ref const& view) {
			multiarray temp = view;
			swap(temp);
			return *this;
		}

		// Assign from multiarray_initializer
		// The procedure just the same as that of resize() happens
		template <class Functor>
		multiarray& operator=(multiarray_initializer<rank, Functor> const& initializer) {
			resize_helper(initializer.extent, [&initializer]
			(storage_elmt_type* dst, extent_type const& new_extent) {
				functor_initialize(dst, new_extent, initializer.functor);
			});
			return *this;
		}
		template <class Functor>
		multiarray& operator=(multiarray_initializer<rank, Functor>&& initializer) {
			resize_helper(initializer.extent, [functor = std::move(initializer.functor)]
			(storage_elmt_type* dst, extent_type const& new_extent) {
				functor_initialize(dst, new_extent, std::move(functor));
			});
			return *this;
		}

		// Destructor
		~multiarray() {
			destroy_all();
		}

		// Construct from a nested initializer-list
		// Each element of the list must be initializer-list of the equal size
		multiarray(nested_initializer_list<T, rank> list)
			: multiarray(construct_from_initializer_list_tag{}, jkl::extent_type<0>{}, list, list) {}
			
		// Reshape / resize

		// reshape doesn't reallocate the buffer
		// If the new extent matches the current extent's size, reshaping is done, 
		// and if not, it just returns as reallocation might be needed.
		bool reshape(extent_type const& new_extent) {
			// For the case of size mismatch, returns false
			// Otherwise, reshaping is done
			if( m_extent.size() == new_extent.size() ) {
				m_extent = new_extent;
				return true;
			}
			return false;
		}
		// When resize() is called,
		// (1) First, every previous element is destroyed.
		// (2) Next, a new buffer is allocated if necessary.
		// (3) Next, the new buffer is initialized, and then
		// (4) The new buffer is set as the buffer.
		// The member function resize() should not be expected to be called too often
		// Exception guarantees:
		// (1) If reallocation of the new buffer fails, resize throws std::bad_alloc.
		//     After that, the multiarray instance becomes empty.
		// (2) If one of constructors throws during initialization of the new buffer, 
		//     the corresponding exception is thrown. Every previously constructed element is 
		//     destroyed, and then the new buffer is deallocated.
		//     After that, the multiarray instance becomes empty
		// (3) Otherwise, resizing never fails, assuming the destructor of T does not throw.
		void resize(extent_type const& new_extent) {
			resize_helper(new_extent, [](storage_elmt_type* dst, extent_type const& new_extent) {
				initialize_helper<>::initialize(dst, new_extent.size());
			});
		}
		void resize(extent_type const& new_extent, T const& value) {
			resize_helper(new_extent, [&value](storage_elmt_type* dst, extent_type const& new_extent) {
				initialize_helper<>::initialize(dst, new_extent.size(), value);
			});
		}
		template <class FirstArg, class... RemainingArgs, class = jkl::tmp::prevent_too_perfect_fwd<T, FirstArg>>
		void resize(extent_type const& new_extent, FirstArg&& first_arg, RemainingArgs&&... remaining_args)
		{
			resize_helper(new_extent, [arg_tuple = std::forward_as_tuple(
			std::forward<FirstArg>(first_arg), std::forward<RemainingArgs>(remaining_args)...)]
			(storage_elmt_type* dst, extent_type const& new_extent) mutable {
				jkl::tmp::unpack_and_apply([dst, &new_extent](auto&&... args) {
					initialize_helper<>::initialize(dst, new_extent.size(), std::forward<decltype(args)>(args)...);
				}, std::move(arg_tuple));
			});
		}
			
		operator view_type() noexcept {
			return{ m_data.get(), m_extent };
		}
		operator const_multiarray_ref() const noexcept {
			return{ m_data.get(), m_extent };
		}
		operator volatile_multiarray_ref() volatile noexcept {
			return{ m_data.get(), m_extent };
		}
		operator const_volatile_multiarray_ref() const volatile noexcept {
			return{ m_data.get(), m_extent };
		}

		void* data() noexcept {
			return m_data.get();
		}
		void const* data() const noexcept {
			return m_data.get();
		}
		auto const& extent() const noexcept {
			return m_extent;
		}
		template <std::size_t i>
		auto extent() const noexcept {
			return get<i>(m_extent);
		}
		auto stride() const noexcept {
			return m_extent.subextent();
		}
		auto size() const noexcept {
			return m_extent.size();
		}
		auto length() const noexcept {
			return get<0>(m_extent);
		}

		FORCEINLINE subview_type operator[](std::size_t idx) noexcept {
			return{ m_data.get() + idx * stride().size(), m_extent.subextent() };
		}
		FORCEINLINE const_subview_type operator[](std::size_t idx) const noexcept {
			return{ m_data.get() + idx * stride().size(), m_extent.subextent() };
		}

		template <std::size_t subrank>
		auto operator[](index_type<subrank> const& idx) noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}
		template <std::size_t subrank>
		auto operator[](index_type<subrank> const& idx) const noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}

		T& operator[](index_type<rank> const& idx) noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}
		T const& operator[](index_type<rank> const& idx) const noexcept {
			return (*this)[get<0>(idx)][idx.subextent()];
		}

		FORCEINLINE auto& operator[](index_type<0> const& idx) noexcept {
			return *this;
		}
		FORCEINLINE auto const& operator[](index_type<0> const& idx) const noexcept {
			return *this;
		}

		view_type subview(extent_type const& extent, index_type<rank> const& origin) noexcept {
			return{ m_data.get(), extent, origin, m_extent.subextent() };
		}
		const_multiarray_ref subview(extent_type const& extent, index_type<rank> const& origin) const noexcept {
			return{ m_data.get(), extent, origin, m_extent.subextent() };
		}

		view_type view() noexcept {
			return subview(m_extent, extent_type::zero());
		}
		const_multiarray_ref view() const noexcept {
			return subview(m_extent, extent_type::zero());
		}

		// STL Container requirements

		// Here, if subrank is smaller than rank, the deference operator of the returned iterator
		// gives a multiarray_ref of smaller size (or pointer, when subrank = rank - 1).
		template <std::size_t subrank = rank>
		auto begin() noexcept {
			return iterator_helper<rank - subrank>::get(*this, 0);
		}
		template <std::size_t subrank = rank>
		auto end() noexcept {
			return iterator_helper<rank - subrank>::get(*this, m_extent.template subextent<0, subrank>().size());
		}

		template <std::size_t subrank = rank>
		multi_iterator<multiarray const, subrank, true> begin() const noexcept {
			return iterator_helper<rank - subrank>::get(*this, 0);
		}
		template <std::size_t subrank = rank>
		multi_iterator<multiarray const, subrank, true> end() const noexcept {
			return iterator_helper<rank - subrank>::get(*this, m_extent.template subextent<0, subrank>().size());
		}

		template <std::size_t subrank = rank>
		multi_iterator<multiarray const, subrank, true> cbegin() const noexcept { return begin<subrank>(); }
		template <std::size_t subrank = rank>
		multi_iterator<multiarray const, subrank, true> cend() const noexcept { return end<subrank>(); }

		template <std::size_t subrank = rank>
		auto rbegin() noexcept { return std::make_reverse_iterator(end<subrank>()); }
		template <std::size_t subrank = rank>
		auto rend() noexcept { return std::make_reverse_iterator(begin<subrank>()); }
		template <std::size_t subrank = rank>
		auto rbegin() const noexcept { return std::make_reverse_iterator(cend<subrank>()); }
		template <std::size_t subrank = rank>
		auto rend() const noexcept { return std::make_reverse_iterator(cbegin<subrank>()); }
		template <std::size_t subrank = rank>
		auto crbegin() const noexcept { return std::make_reverse_iterator(cend<subrank>()); }
		template <std::size_t subrank = rank>
		auto crend() const noexcept { return std::make_reverse_iterator(cbegin<subrank>()); }

		template <class MultiDimContainer>
		bool operator==(MultiDimContainer&& that) const noexcept {
			if( extent() != that.extent() )
				return false;

			auto this_itr = begin();
			auto that_itr = that.begin();
			for( ; this_itr != end(); ++this_itr, ++that_itr )
				if( *this_itr != *that_itr )
					return false;

			return true;
		}

		template <class MultiDimContainer>
		bool operator!=(MultiDimContainer&& that) const noexcept {
			return !(*this == std::forward<MultiDimContainer>(that));
		}

		void swap(multiarray& that) noexcept {
			std::swap(m_extent, that.m_extent);
			std::swap(m_data, that.m_data);
		}

		static std::size_t max_size() noexcept {
			return std::numeric_limits<std::size_t>::max();
		}

		bool empty() const noexcept {
			return m_extent.size() == 0;
		}

	private:
		extent_type										m_extent = extent_type::zero();
		std::unique_ptr<storage_elmt_type[]>			m_data;

		// Extract the extent from the initializer list and then copy
		struct construct_from_initializer_list_tag {};
		template <std::size_t r, class OriginalInitializerList, class FirstList, class... RemainingLists>
		FORCEINLINE multiarray(construct_from_initializer_list_tag tag, jkl::extent_type<r> const& extent,
			OriginalInitializerList&& original_list, FirstList&& first_list, RemainingLists&&... remaining_lists) :
			multiarray(tag, join_extent<r, 1>(extent, { first_list.size() }), 
			std::forward<OriginalInitializerList>(original_list), *first_list.begin(),
			std::forward<FirstList>(first_list), std::forward<RemainingLists>(remaining_lists)...) {}

		template <class OriginalInitializerList, class... InitializerLists>
		FORCEINLINE multiarray(construct_from_initializer_list_tag, extent_type const& extent,
			OriginalInitializerList&& original_list, InitializerLists&&...) :
			m_extent{ extent }, m_data{ std::make_unique<storage_elmt_type[]>(m_extent.size()) }
		{
			storage_elmt_type* ptr = m_data.get();
			copy_from_initializer_list(m_extent, original_list, ptr);
		}

		template <std::size_t r, class InitializerList, bool is_trivial>
		struct copy_from_initializer_list_helper {
			FORCEINLINE static void copy(jkl::extent_type<r> const& extent, 
				InitializerList&& list, storage_elmt_type*& base_address)
			{
				for( std::size_t i = 0; i < get<0>(extent); ++i ) {
					assert(list.begin()[i].size() == get<1>(extent) && 
						"[jkl::multiarray] Initializer list size mismatch");
					copy_from_initializer_list(extent.subextent(), list.begin()[i], base_address);
				}
			}
		};

		template <class InitializerList>
		struct copy_from_initializer_list_helper<1, InitializerList, false> {
			FORCEINLINE static void copy(jkl::extent_type<1> const& extent, 
				InitializerList&& list, storage_elmt_type*& base_address)
			{
				for( std::size_t i = 0; i < get<0>(extent); ++i, ++base_address )
					new(base_address) T(list.begin()[i]);
			}
		};

		template <class InitializerList>
		struct copy_from_initializer_list_helper<1, InitializerList, true> {
			FORCEINLINE static void copy(jkl::extent_type<1> const& extent,
				InitializerList&& list, storage_elmt_type*& base_address)
			{
				std::memcpy(base_address, list.begin(), alignment * get<0>(extent));
				base_address += get<0>(extent);
			}
		};

		template <std::size_t r, class InitializerList, bool is_trivial = std::is_trivially_constructible<T>::value>
		static void copy_from_initializer_list(jkl::extent_type<r> const& extent, 
			InitializerList&& list, storage_elmt_type*& base_address)
		{
			copy_from_initializer_list_helper<r, InitializerList, is_trivial>
				::copy(extent, std::forward<InitializerList>(list), base_address);
		}

		// Calculate the appropriate multi_iterator
		template <std::size_t rank_complement, class = void>
		struct iterator_helper {
			template <class Container>
			static auto get(Container&& c, std::size_t idx) noexcept {
				static_assert(rank_complement != rank, "jkl::multiarray: Iterator must have positive rank!");
				static_assert(rank_complement <= rank, "jkl::multiarray: Index out of range!");

				using return_type = multi_iterator<std::remove_reference_t<Container>,
					rank - rank_complement,
					std::is_const<std::remove_reference_t<Container>>::value>;
				return return_type{ std::forward<Container>(c), idx };
			}
		};

		// When subrank == rank, we can use raw pointers, which are faster to work with
		template <class dummy>
		struct iterator_helper<0, dummy> {
			template <class Container>
			static auto get(Container&& c, std::size_t idx) noexcept {
				using return_type = std::conditional_t<std::is_const<std::remove_reference_t<Container>>::value,
					const_pointer, pointer>;
				return static_cast<return_type>(std::forward<Container>(c).data()) + idx;
			}
		};
	};

	template <class T, std::size_t rank_, std::size_t alignment>
	class multiarray<T&, rank_, alignment> {
		static_assert(jkl::tmp::assert_helper<T>::value,
			"multiarray must not be instantiated with lvalue reference types!");
	};

	template <class T, std::size_t rank_, std::size_t alignment>
	class multiarray<T&&, rank_, alignment> {
		static_assert(jkl::tmp::assert_helper<T>::value,
			"multiarray must not be instantiated with rvalue reference types!");
	};
}