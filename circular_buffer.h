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
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <memory>
#include "tmp.h"

namespace jkl {
	/// A simple circular queue with a given size
	/// Just like other STL containers, T must not be reference types.
	/// This implementation wastes one storage in the buffer to save the implementer from a lot of headaches 
	/// to resolve ambiguity between empty and full state.
	/// Following behaviours can be fine-tunned using template parameters:
	///   - If a new element is pushed while the buffer is full, it is dropped.
	///   - If you want this to be notified, set check_push flag. On that case, push() & emplace() returns true iff 
	///     it succeeded pushing. If the flag is not set, push() & emplace() does not return anything. 
	///     The default of this flag is true.
	///   - If check_pop flag is set, pop() returns true iff the buffer is not empty. 
	///     If the flag is not set, pop() does not return anything. The default of this flag is false.
	///   - EmptyQueryPolicy paramter controls what to do when front()/back() is called while the buffer is empty.
	///     The handle_error() member function is called when this happens. The default is to do nothing.
	/// circular_buffer's with the same element type can be freely copied/moved/assigned/compared/swapped, even when 
	/// policy parameters (check_push, check_pop, and EmptyQueryPolicy) are not matched.
	/// circular_buffer detects trivial copyability/destructibility of the element type and 
	/// optimize operations according to them; e.g., it utilizes std::memcpy if the element type is trivially copyable.

	struct circular_buffer_do_nothing_on_empty_query {
		void handle_error() const noexcept {}
	};
	class empty_query_exception : public std::runtime_error {
		using std::runtime_error::runtime_error;
	};
	struct circular_buffer_throw_on_empty_query {
		void handle_error() const {
			throw empty_query_exception{ "[jkl::circular_buffer] The circular_buffer is empty!" };
		}
	};

	template <
		class T,
		// If the following flag is set, push() and emplace() returns true when the buffer is not full, and false when the buffer is full.
		bool check_push = true,
		// If the following flag is set, pop() returns true when the buffer is not empty, and false when the buffer is empty.
		bool check_pop = false,
		// When either front() or back() is called while the buffer is empty, then handle_error() member function of the 
		// following class is invoked.
		class EmptyQueryPolicy = circular_buffer_do_nothing_on_empty_query
	>
	class circular_buffer : private EmptyQueryPolicy
	{
		using storage_elmt_type = std::aligned_storage_t<sizeof(T), alignof(T)>;
		using empty_query_policy = EmptyQueryPolicy;

		template <bool flag, class = void>
		struct check_helper_t {};
			
		template <class dummy>
		struct check_helper_t<false, dummy> {
			static void propagate(bool) noexcept {}
		};
		template <class dummy>
		struct check_helper_t<true, dummy> {
			static bool propagate(bool value) noexcept {
				return value;
			}
		};

		template <bool is_const, class = void>
		class iterator_impl {
			using container_type = std::conditional_t<is_const, circular_buffer const, circular_buffer>;
			container_type*	m_container_ptr;
			std::size_t		m_idx;

			constexpr iterator_impl(container_type* container_ptr, std::size_t idx) noexcept
				: m_container_ptr{ container_ptr }, m_idx{ idx } {}

		public:
			friend circular_buffer;
			template <bool, class>
			friend class iterator_impl;

			using iterator_category = std::random_access_iterator_tag;
			using value_type = T;
			using reference = std::conditional_t<is_const, std::add_const_t<T>, T>&;
			using pointer = std::remove_reference_t<reference>*;
			using difference_type = std::ptrdiff_t;

			// Conversion from iterator to const_iterator
			constexpr iterator_impl(iterator_impl<false> const& that) noexcept
				: m_container_ptr{ that.m_container_ptr }, m_idx{ that.m_idx } {}

			// Requirements from Iterator
			reference operator*() const noexcept {
				return *(reinterpret_cast<pointer>(m_container_ptr->m_buffer.get() + m_idx));
			}
			iterator_impl& operator++() noexcept {
				if( m_idx != m_container_ptr->m_back_idx ) {
					if( ++m_idx == m_container_ptr->m_capacity )
						m_idx = 0;
				}
				return *this;
			}

			// Requirements from InputIterator
			template <bool that_are_const>
			constexpr bool operator==(iterator_impl<that_are_const> const& that) const noexcept {
				return m_idx == that.m_idx;
			}
			template <bool that_are_const>
			constexpr bool operator!=(iterator_impl<that_are_const> const& that) const noexcept {
				return !(*this == that);
			}
			pointer operator->() const noexcept {
				return reinterpret_cast<pointer>(m_container_ptr->m_buffer.get() + m_idx);
			}
			iterator_impl operator++(int) noexcept {
				iterator_impl ret_value = *this;
				++(*this);
				return ret_value;
			}

			// Requirements from ForwardIterator
			constexpr iterator_impl() noexcept : m_container_ptr{ nullptr }, m_idx{ 0 } {}

			// Requirements from BidirectionalIterator
			iterator_impl& operator--() noexcept {
				if( m_idx != m_container_ptr->m_front_idx ) {
					if( m_idx == 0 )
						m_idx = m_container_ptr->m_capacity - 1;
					else
						--m_idx;
				}
				return *this;
			}
			iterator_impl& operator--(int) noexcept {
				iterator_impl ret_value = *this;
				--(*this);
				return ret_value;
			}

			// Requirements from RandomAccessIterator
			iterator_impl& operator+=(difference_type n) noexcept {
				if( n > 0 ) {
					auto const unsigned_n = std::make_unsigned_t<difference_type>(n);
					if( m_idx != m_container_ptr->m_back_idx ) {
						// -----f----i----b----
						if( m_container_ptr->m_front_idx < m_container_ptr->m_back_idx ) {
							if( unsigned_n >= m_container_ptr->m_back_idx - m_idx )
								m_idx = m_container_ptr->m_back_idx;
							else
								m_idx += unsigned_n;
						} else {
							// -----b----f----i----
							if( m_idx >= m_container_ptr->m_front_idx ) {
								if( unsigned_n < m_container_ptr->m_capacity - m_idx )
									m_idx += unsigned_n;
								else if( unsigned_n < m_container_ptr->m_capacity - m_idx + m_container_ptr->m_back_idx )
									m_idx = (m_idx + unsigned_n) - m_container_ptr->m_capacity;
								else
									m_idx = m_container_ptr->m_back_idx;
							}
							// ----i----b----f----
							else {
								if( unsigned_n >= m_container_ptr->m_back_idx - m_idx )
									m_idx = m_container_ptr->m_back_idx;
								else
									m_idx += unsigned_n;
							}
						}
					}
				} else if( n < 0 ) {
					auto const unsigned_n = std::make_unsigned_t<difference_type>(-n);
					if( m_idx != m_container_ptr->m_front_idx ) {
						// ----f----i----b----
						if( m_container_ptr->m_front_idx < m_container_ptr->m_back_idx ) {
							if( unsigned_n >= m_idx - m_container_ptr->m_front_idx )
								m_idx = m_container_ptr->m_front_idx;
							else
								m_idx -= unsigned_n;
						} else {
							// -----b----f----i----
							if( m_idx >= m_container_ptr->m_front_idx ) {
								if( unsigned_n >= m_idx - m_container_ptr->m_front_idx )
									m_idx = m_container_ptr->m_front_idx;
								else
									m_idx -= unsigned_n;
							}
							// ----i----b----f----
							else {
								if( unsigned_n <= m_idx )
									m_idx -= unsigned_n;
								else if( unsigned_n <= m_idx + m_container_ptr->m_capacity - m_container_ptr->m_front_idx )
									m_idx = m_container_ptr->m_capacity - unsigned_n + m_idx;
								else
									m_idx = m_container_ptr->m_front_idx;
							}
						}
					}
				}

				return *this;
			}
			iterator_impl operator+(difference_type n) const noexcept {
				iterator_impl ret_value = *this;
				return ret_value += n;
			}
			friend iterator_impl operator+(difference_type n, iterator_impl const& itr) noexcept {
				return itr + n;
			}
			iterator_impl& operator-=(difference_type n) noexcept {
				return (*this) += -n;
			}
			iterator_impl operator-(difference_type n) const noexcept {
				iterator_impl ret_value = *this;
				return ret_value -= n;
			}
			template <bool that_are_const>
			difference_type operator-(iterator_impl<that_are_const> const& that) const {
				assert(m_container_ptr == that.m_container_ptr && "jkl::circular_buffer: Incomparable iterators");

				// ----f----i1----i2----b----
				// ----f----i2----i1----b----
				if( m_container_ptr->m_front_idx <= m_container_ptr->m_back_idx ) {
					return difference_type(m_idx - that.m_idx);
				} else {
					if( m_idx >= m_container_ptr->m_front_idx ) {
						// ----b----f----i1----i2----
						// ----b----f----i2----i1----
						if( that.m_idx >= m_container_ptr->m_front_idx )
							return difference_type(m_idx - that.m_idx);

						// ----i2----b----f----i1----
						else
							return -difference_type(m_container_ptr->m_capacity - m_idx + that.m_idx);
					} else {
						// ----i1----b----f----i2----
						if( that.m_idx >= m_container_ptr->m_front_idx )
							return difference_type(m_container_ptr->m_capacity - that.m_idx + m_idx);

						// ----i1----i2----b----f----
						// ----i2----i1----b----f----
						else
							return difference_type(m_idx - that.m_idx);
					}
				}
			}
			reference operator[](std::size_t n) const noexcept {
				return *((*this) + n);
			}
			template <bool that_are_const>
			bool operator<(iterator_impl<that_are_const> const& that) const noexcept {
				// ----f----i1----i2----b----
				// ----f----i2----i1----b----
				if( m_container_ptr->m_front_idx <= m_container_ptr->m_back_idx ) {
					return m_idx < that.m_idx;
				} else {
					if( m_idx >= m_container_ptr->m_front_idx ) {
						// ----b----f----i1----i2----
						// ----b----f----i2----i1----
						if( that.m_idx >= m_container_ptr->m_front_idx )
							return m_idx < that.m_idx;

						// ----i2----b----f----i1----
						else
							return true;
					} else {
						// ----i1----b----f----i2----
						if( that.m_idx >= m_container_ptr->m_front_idx )
							return false;

						// ----i1----i2----b----f----
						// ----i2----i1----b----f----
						else
							return m_idx < that.m_idx;
					}
				}
			}
			template <bool that_are_const>
			bool operator>(iterator_impl<that_are_const> const& that) const noexcept {
				return that < *this;
			}
			template <bool that_are_const>
			bool operator<=(iterator_impl<that_are_const> const& that) const noexcept {
				return *this == that || *this < that;
			}
			template <bool that_are_const>
			bool operator>=(iterator_impl<that_are_const> const& that) const noexcept {
				return that <= *this;
			}
		};

		// Copy helper
		// Realign to make m_front_idx = 0
		template <bool trivial_copy, class = void>
		struct copy_helper_t {};

		template <class dummy>
		struct copy_helper_t<true, dummy> {
			template <class CircularBuffer>
			static std::size_t copy(storage_elmt_type* dst, CircularBuffer const& src) {
				if( src.m_front_idx <= src.m_back_idx ) {
					auto const size = src.m_back_idx - src.m_front_idx;
					std::memcpy(dst, src.m_buffer.get() + src.m_front_idx, sizeof(storage_elmt_type) * size);
					return size;
				} else {
					auto const offset = src.m_capacity - src.m_front_idx;
					std::memcpy(dst, src.m_buffer.get() + src.m_front_idx, sizeof(storage_elmt_type) * offset);
					std::memcpy(dst + offset, src.m_buffer.get(), sizeof(storage_elmt_type) * src.m_back_idx);
					return offset + src.m_back_idx;
				}
			}

			template <class CircularBuffer>
			static std::size_t move(storage_elmt_type* dst, CircularBuffer const& src) {
				return copy(dst, src);
			}
		};

		template <class dummy>
		struct copy_helper_t<false, dummy> {
			template <class CircularBuffer, class Constructor>
			static std::size_t copy_impl(storage_elmt_type* dst, CircularBuffer const& src, Constructor&& constructor) {
				if( src.m_front_idx <= src.m_back_idx ) {
					auto const size = src.m_back_idx - src.m_front_idx;
					for( std::size_t idx = 0; idx < size; ++idx )
						constructor(dst + idx, src.m_buffer.get() + src.m_front_idx + idx);
					return size;
				} else {
					auto const offset = src.m_capacity - src.m_front_idx;
					for( std::size_t idx = 0; idx < src.m_capacity - src.m_front_idx; ++idx )
						constructor(dst + idx, src.m_buffer.get() + src.m_front_idx + idx);
					for( std::size_t idx = 0; idx < src.m_back_idx; ++idx )
						constructor(dst + offset + idx, src.m_buffer.get() + idx);
					return offset + src.m_back_idx;
				}
			}

			template <class CircularBuffer>
			static std::size_t copy(storage_elmt_type* dst, CircularBuffer const& src) {
				return copy_impl(dst, src, [](void* dest_ptr, void* source_ptr) {
					new(dest_ptr) T(*reinterpret_cast<T const*>(source_ptr));
				});
			}

			template <class CircularBuffer>
			static std::size_t move(storage_elmt_type* dst, CircularBuffer const& src) {
				return copy_impl(dst, src, [](void* dest_ptr, void* source_ptr) {
					new(dest_ptr) T(std::move(*reinterpret_cast<T*>(source_ptr)));
				});
			}
		};

		template <class CircularBuffer, bool trivial_copy>
		circular_buffer(CircularBuffer const& that, copy_helper_t<trivial_copy>)
			: m_capacity{ that.m_capacity }, m_front_idx{ 0 },
			m_buffer{ std::make_unique<storage_elmt_type[]>(m_capacity) },
			m_back_idx{ copy_helper_t<trivial_copy>::copy(m_buffer.get(), that) } {}

		template <bool trivial_destruction, class = void>
		struct destroy_helper_t {};

		template <class dummy>
		struct destroy_helper_t<true, dummy> {
			static void destroy_all(circular_buffer&) noexcept {}
		};

		template <class dummy>
		struct destroy_helper_t<false, dummy> {
			static void destroy_all(circular_buffer& buffer) noexcept(std::is_nothrow_destructible<T>::value) {
				// Destruct all the elements
				if( buffer.m_front_idx <= buffer.m_back_idx ) {
					for( auto i = buffer.m_front_idx; i < buffer.m_back_idx; ++i )
						reinterpret_cast<T*>(buffer.m_buffer.get() + i)->~T();
				} else {
					for( auto i = buffer.m_front_idx; i < buffer.m_capacity; ++i )
						reinterpret_cast<T*>(buffer.m_buffer.get() + i)->~T();
					for( auto i = 0; i < buffer.m_back_idx; ++i )
						reinterpret_cast<T*>(buffer.m_buffer.get() + i)->~T();
				}
			}
		};

		void destroy_all() noexcept(std::is_nothrow_destructible<T>::value) {
			destroy_helper_t<std::is_trivially_destructible<T>::value>::destroy_all(*this);
		}

		// Copy from initializer-list helper
		// Realign to make m_front_idx = 0
		template <bool trivial_copy, class = void>
		struct copy_list_helper_t {};

		template <class dummy>
		struct copy_list_helper_t<true, dummy> {
			static void copy(storage_elmt_type* dst, std::initializer_list<T> list) {
				std::memcpy(dst, list.begin(), sizeof(storage_elmt_type) * list.size());
			}
		};

		template <class dummy>
		struct copy_list_helper_t<false, dummy> {
			static void copy(storage_elmt_type* dst, std::initializer_list<T> list) {
				for( std::size_t idx = 0; idx < list.size(); ++idx )
					new(dst + idx) T(list.begin()[idx]);
			}
		};

		template <bool trivial_copy>
		circular_buffer(std::initializer_list<T> list, copy_list_helper_t<trivial_copy>)
			: m_capacity{ list.size() + 1 }, m_front_idx{ 0 },
			m_buffer{ std::make_unique<storage_elmt_type[]>(list.size() + 1) },
			m_back_idx{ list.size() }
		{
			copy_list_helper_t<trivial_copy>::copy(m_buffer.get(), list);
		}

		// Implementation of push() and emplace()
		template <class Maker>
		std::conditional_t<check_push, bool, void> emplace_impl(Maker&& maker) {
			if( full() )
				return check_helper_t<check_push>::propagate(false);

			std::forward<Maker>(maker)();

			if( ++m_back_idx == m_capacity )
				m_back_idx = 0;
			return check_helper_t<check_push>::propagate(true);
		}

	public:
		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using const_reference = T const&;
		using pointer = T*;
		using const_pointer = T const*;
		using iterator = iterator_impl<false>;
		using const_iterator = iterator_impl<true>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		friend iterator;
		friend const_iterator;
		template <class T_that, bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		friend class circular_buffer;

		// Default constructor
		circular_buffer() = default;

		// Standard constructor
		circular_buffer(std::size_t capacity) : m_capacity{ capacity + 1 }, m_front_idx{ 0 }, m_back_idx{ 0 },
			m_buffer{ std::make_unique<storage_elmt_type[]>(m_capacity) } {}

		// Initializer-list constructor
		circular_buffer(std::initializer_list<T> list) 
			: circular_buffer(list, copy_list_helper_t<std::is_trivially_copy_constructible<T>::value>{}) {}

		// Copy constructor
		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		circular_buffer(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat> const& that)
			: circular_buffer(that, copy_helper_t<std::is_trivially_copyable<T>::value>{}) {}

		// Move constructor
		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		circular_buffer(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat>&& that)
			: m_capacity{ that.m_capacity }, m_front_idx{ that.m_front_idx },
			m_back_idx{ that.m_back_idx }, m_buffer{ std::move(that.m_buffer) }
		{
			that.m_capacity = 0;
			that.m_front_idx = 0;
			that.m_back_idx = 0;
		}

		// Destructor
		~circular_buffer() {
			destroy_all();
		}

		// Copy assignment operator
		// Destroy all elements and reconstruct using copy construction
		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		circular_buffer& operator=(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat> const& that) {
			if( this != reinterpret_cast<circular_buffer const*>(&that) ) {
				destroy_all();

				// Check if reallocation is needed
				// Do not use the already existing buffer even when the size must shrink
				if( m_capacity != that.m_capacity )
					m_buffer = std::make_unique<storage_elmt_type[]>(that.m_capacity);

				m_capacity = that.m_capacity;
				m_front_idx = 0;

				m_back_idx = copy_helper_t<std::is_trivially_copyable<T>::value>::copy(m_buffer.get(), that);
			}
			return *this;
		}

		// Move assignment operator
		// Destroy all elements and steal from another buffer
		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		circular_buffer& operator=(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat>&& that) {
			if( this != reinterpret_cast<circular_buffer const*>(&that) ) {
				destroy_all();
				m_capacity = that.m_capacity;
				m_front_idx = that.m_front_idx;
				m_back_idx = that.m_back_idx;
				m_buffer = std::move(that.m_buffer);
				that.m_capacity = 0;
				that.m_front_idx = 0;
				that.m_back_idx = 0;
			}
			return *this;
		}

		// STL Container requirements

		iterator begin() noexcept { return{ this, m_front_idx }; }
		iterator end() noexcept { return{ this, m_back_idx }; }
		const_iterator begin() const noexcept { return{ this, m_front_idx }; }
		const_iterator end() const noexcept { return{ this, m_back_idx }; }
		const_iterator cbegin() const noexcept { return{ this, m_front_idx }; }
		const_iterator cend() const noexcept { return{ this, m_back_idx }; }

		reverse_iterator rbegin() noexcept { return std::make_reverse_iterator(end()); }
		reverse_iterator rend() noexcept { return std::make_reverse_iterator(begin()); }
		const_reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
		const_reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
		const_reverse_iterator crbegin() const noexcept { return std::make_reverse_iterator(cend()); }
		const_reverse_iterator crend() const noexcept { return std::make_reverse_iterator(cbegin()); }

		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		bool operator==(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat> const& that) const noexcept {
			if( m_capacity != that.m_capacity )
				return false;

			auto itr = begin();
			auto that_itr = that.begin();
				
			for( ; itr != end(); ++itr, ++that_itr )
				if( !(*itr == *that_itr) )
					return false;
			return true;
		}

		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		bool operator!=(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat> const& that) const noexcept {
			return !(*this == that);
		}

		template <bool check_push_that, bool check_pop_that, class EmptyQueryPolicyThat>
		void swap(circular_buffer<T, check_push_that, check_pop_that, EmptyQueryPolicyThat>& that) noexcept {
			circular_buffer temp = std::move(*this);
			*this = std::move(that);
			that = std::move(temp);
		}

		std::size_t size() const noexcept {
			if( m_front_idx <= m_back_idx )
				return m_back_idx - m_front_idx;
			else
				return m_capacity - m_front_idx + m_back_idx;
		}

		static std::size_t max_size() noexcept {
			return std::numeric_limits<std::size_t>::max();
		}

		bool empty() const noexcept {
			return m_front_idx == m_back_idx;
		}

		// As a circular queue

		bool full() const noexcept {
			return m_back_idx + 1 == m_front_idx || (m_front_idx == 0 && m_back_idx == m_capacity - 1);
		}

		T& front() noexcept(noexcept(empty_query_policy::handle_error())) {
			if( empty() )
				empty_query_policy::handle_error();
			return begin()[0];
		}

		T const& front() const noexcept(noexcept(empty_query_policy::handle_error())) {
			if( empty() )
				empty_query_policy::handle_error();
			return begin()[0];
		}

		T& back() noexcept(noexcept(empty_query_policy::handle_error())) {
			if( empty() )
				empty_query_policy::handle_error();
			return rbegin()[0];
		}

		T const& back() const noexcept(noexcept(empty_query_policy::handle_error())) {
			if( empty() )
				empty_query_policy::handle_error();
			return rbegin()[0];
		}

		std::conditional_t<check_push, bool, void> push(T const& value) {
			return emplace_impl([this, &value] {
				new(reinterpret_cast<T*>(m_buffer.get() + m_back_idx)) T(value);
			});
		}

		std::conditional_t<check_push, bool, void> push(T&& value) {
			return emplace_impl([this, &value] {
				new(reinterpret_cast<T*>(m_buffer.get() + m_back_idx)) T(std::move(value));
			});
		}

		template <class... Args>
		std::conditional_t<check_push, bool, void> emplace(Args&&... args) {
			return emplace_impl([this, &args...] {
				new(reinterpret_cast<T*>(m_buffer.get() + m_back_idx)) T(std::forward<Args>(args)...);
			});
		}

		std::conditional_t<check_pop, bool, void> pop() {
			if( empty() )
				return check_helper_t<check_pop>::propagate(false);
			else {
				reinterpret_cast<T*>(m_buffer.get() + m_front_idx)->~T();
				if( ++m_front_idx == m_capacity )
					m_front_idx = 0;
				return check_helper_t<check_pop>::propagate(true);
			}
		}

		void resize_capacity(std::size_t capacity) {
			// Do not use the already existing buffer even when the size must shrink
			if( m_capacity == capacity + 1 )
				return;

			auto new_buffer = std::make_unique<storage_elmt_type[]>(capacity + 1);
			auto size = copy_helper_t<std::is_trivially_copyable<T>::value>::move(new_buffer.get(), *this);
			destroy_all();

			std::swap(m_buffer, new_buffer);

			m_capacity = capacity + 1;
			m_front_idx = 0;
			m_back_idx = size;
		}

		std::size_t capacity() const noexcept {
			return m_capacity - 1;
		}

	private:
		std::size_t								m_capacity = 0;			// Size of the buffer + 1 (counting the empty tail)
		std::size_t								m_front_idx = 0;		// The current push position
		std::unique_ptr<storage_elmt_type[]>	m_buffer = nullptr;		// Pointer to the buffer
		std::size_t								m_back_idx = 0;			// The current pop position
	};

	template <class T, bool check_push, bool check_pop, class EmptyQueryPolicy>
	class circular_buffer<T&, check_push, check_pop, EmptyQueryPolicy> {
		static_assert(jkl::tmp::assert_helper<T>::value, 
			"circular_buffer must not be instantiated with lvalue reference types!");
	};

	template <class T, bool check_push, bool check_pop, class EmptyQueryPolicy>
	class circular_buffer<T&&, check_push, check_pop, EmptyQueryPolicy> {
		static_assert(jkl::tmp::assert_helper<T>::value, 
			"circular_buffer must not be instantiated with rvalue reference types!");
	};
}

namespace std {
	template <
		class T1, bool check_push1, bool check_pop1, class EmptyQueryPolicy1,
		class T2, bool check_push2, bool check_pop2, class EmptyQueryPolicy2
	>
	void swap(::jkl::circular_buffer<T1, check_push1, check_pop1, EmptyQueryPolicy1>& x,
		::jkl::circular_buffer<T2, check_push2, check_pop2, EmptyQueryPolicy2>& y) {
		x.swap(y);
	}
}