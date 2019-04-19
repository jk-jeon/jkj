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
#include <array>
#include <cassert>
#include <utility>
#include "pseudo_ptr.h"
#include "tmp.h"

namespace jkj {
	namespace array_detail {
		// The main value class
		template <class T, std::size_t N1, std::size_t... Nr>
		class array;

		// Proxy reference-like view to arrays
		// It abstracts away the actual way how to access a particular element,
		// but assumes that no matter how such a way is given, the result of element access
		// should give something that can be implicitly converted to TargetElmtRef.
		template <class TargetElmtRef, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref;

		template <std::size_t... N>
		struct get_array_type;

		template <>
		struct get_array_type<> {
			template <class T>
			using array_type = T;

			template <class TargetElmtRef, class SourceArrayRef>
			using array_ref_type = TargetElmtRef;
		};

		template <std::size_t N1, std::size_t... Nr>
		struct get_array_type<N1, Nr...> {
			template <class T>
			using array_type = array<T, N1, Nr...>;

			template <class TargetElmtRef, class SourceArrayRef>
			using array_ref_type = array_ref<TargetElmtRef, SourceArrayRef, N1, Nr...>;
		};
	}

	template <class T, std::size_t... N>
	using array = typename array_detail::get_array_type<N...>::template array_type<T>;

	template <class TargetElmtRef, class SourceArrayRef, std::size_t... N>
	using array_ref = typename array_detail::get_array_type<N...>::template array_ref_type<TargetElmtRef, SourceArrayRef>;

	namespace array_detail {
		// Transform (N1, N2, ... Nd) into (N1 x N2 x ... x Nd, N2 x ... x Nd, ... , Nd)
		template <std::size_t... N>
		struct get_strides;
		
		template <std::size_t N1, std::size_t... Nr>
		struct get_strides<N1, Nr...> {
		private:
			template <std::size_t... prev_strides>
			static constexpr auto calculate(std::index_sequence<prev_strides...>) noexcept {
				return std::index_sequence<N1 * get_strides<Nr...>::total_size,
					prev_strides...>{};
			}

		public:
			static constexpr std::size_t total_size = N1 * get_strides<Nr...>::total_size;
			using type = decltype(calculate(std::declval<typename get_strides<Nr...>::type>()));
			using substride = get_strides<Nr...>;
		};

		template <>
		struct get_strides<> {
			static constexpr std::size_t total_size = 1;
			using type = std::index_sequence<>;
		};
		
		// When SourceArrayRef = T*, it is assumed that there is a consecutive linear memory
		// and array_ref accesses that memory with a multi-dimensional fashion.
		// This is the base case for every other constructions.
		// No bound check or other safe guard features are provided.

		// Common things for 4 combinations of const-qualifier and reference qualifier
		template <class TargetElmtRef, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref_base;

		// Common things for lvalue and rvalue mutable reference qualifier
		template <class TargetElmtRef, class SourceArrayRef, std::size_t... N>
		class array_mutable_ref_base;

		template <class TargetElmtRef, class T, std::size_t N1, std::size_t... Nr>
		class linear_ptr_array_iterator {
			T* ptr_;

		public:
			static_assert(std::is_reference_v<TargetElmtRef>,
				"jkj::array: the first template argument into array_ref must be "
				"a reference type");
			static_assert(std::is_convertible_v<T*, std::remove_reference_t<TargetElmtRef>*>,
				"jkj::array: the target reference type and the underlying pointer type "
				"of array_ref is not compatiable");

			template <class, std::size_t, std::size_t...>
			friend class array;
			template <class, class, std::size_t, std::size_t...>
			friend class array_ref;
			template <class, class, std::size_t, std::size_t...>
			friend class array_ref_base;
			template <class, class, std::size_t...>
			friend class array_mutable_ref_base;
			template <class, class T, std::size_t, std::size_t...>
			friend class linear_ptr_array_iterator;

			// As this iterator dereferences to a proxy, rigorously speaking,
			// this iterator is not a RandomAccessIterator; it is only an InputIterator.
			using iterator_category = std::random_access_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using element = std::remove_cv_t<std::remove_reference_t<TargetElmtRef>>;
			using value_type = jkj::array<element, Nr...>;
			using reference = jkj::array_ref<TargetElmtRef, T*, Nr...>;
			using pointer = std::conditional_t<sizeof...(Nr) == 0,
				std::remove_reference_t<TargetElmtRef>*,
				pseudo_ptr<reference>>;
			
			linear_ptr_array_iterator() = default;
			constexpr explicit linear_ptr_array_iterator(T* ptr) noexcept : ptr_{ ptr } {}

			// Enable nonconst iterator to const iterator.
			// Note that not lvalue to rvalue nor rvalue to lvalue conversion are allowed.
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<
				// First, the target reference must be const
				std::is_const_v<std::remove_reference_t<TargetElmtRef>> &&
				// Second, the two references must be the same except constness
				std::is_same_v<
					std::conditional_t<std::is_lvalue_reference_v<TargetElmtRef>,
						element&, element&&>,
					OtherElmtRef>
				> &&
				// Third, T and U should be the same type except cv-qualifiers
				std::is_convertible_v<U*, T*> &&
				std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>
			> constexpr linear_ptr_array_iterator(
				linear_ptr_array_iterator<OtherElmtRef, T, N1, Nr...> itr) noexcept
				: ptr_{ itr.ptr_ } {}

			// Dereference
			constexpr auto operator*() const noexcept {
				return reference{ *ptr_ };
			}
			constexpr auto operator[](difference_type n) const noexcept {
				return reference{ *(ptr_ + n * get_strides<Nr...>::total_size) };
			}

			// Member access
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) == 0, dummy>>
			constexpr pointer operator->() const noexcept {
				return &*ptr_;
			}
			template <class dummy = void, class = std::enable_if_t<sizeof...(Nr) != 0, dummy>, class = void>
			constexpr pointer operator->() const noexcept {
				return pointer{ **this };
			}

			// Increment
			NONCONST_CONSTEXPR linear_ptr_array_iterator& operator++() noexcept {
				ptr_ += get_strides<Nr...>::total_size;
				return *this;
			}
			NONCONST_CONSTEXPR linear_ptr_array_iterator operator++(int) noexcept {
				auto prev = *this;
				ptr_ += get_strides<Nr...>::total_size;
				return prev;
			}
			NONCONST_CONSTEXPR linear_ptr_array_iterator& operator+=(difference_type n) noexcept {
				ptr_ += n * get_strides<Nr...>::total_size;
				return *this;
			}
			constexpr linear_ptr_array_iterator operator+(difference_type n) const noexcept {
				return linear_ptr_array_iterator{ ptr_ + n * get_strides<Nr...>::total_size };
			}
			friend constexpr linear_ptr_array_iterator operator+(difference_type n,
				linear_ptr_array_iterator itr) noexcept
			{
				return itr + n;
			}

			// Decrement
			NONCONST_CONSTEXPR linear_ptr_array_iterator& operator--() noexcept {
				ptr_ -= get_strides<Nr...>::total_size;
				return *this;
			}
			NONCONST_CONSTEXPR linear_ptr_array_iterator operator--(int) noexcept {
				auto prev = *this;
				ptr_ -= get_strides<Nr...>::total_size;
				return prev;
			}
			NONCONST_CONSTEXPR linear_ptr_array_iterator& operator-=(difference_type n) noexcept {
				ptr_ -= n * get_strides<Nr...>::total_size;
				return *this;
			}
			constexpr linear_ptr_array_iterator operator-(difference_type n) const noexcept {
				return linear_ptr_array_iterator{ ptr_ - n * get_strides<Nr...>::total_size };
			}

			// Distance
		private:
			template <class OtherElmtRef, class U>
			static constexpr bool is_comparable =
				std::is_reference_v<OtherElmtRef> &&
				std::is_same_v<element, std::remove_cv_t<std::remove_reference_t<OtherElmtRef>>> &&
				std::is_convertible_v<U*, T*> &&
				std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>;

		public:
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr difference_type operator-(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return difference_type((ptr_ - itr.ptr_) / get_strides<Nr...>::total_size);
			}

			// Relations
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator==(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ == itr.ptr_;
			}
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator!=(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ != itr.ptr_;
			}
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator<(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ < itr.ptr_;
			}
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator<=(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ <= itr.ptr_;
			}
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator>(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ > itr.ptr_;
			}
			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_comparable<OtherElmtRef, U>>
			> constexpr bool operator>=(
				linear_ptr_array_iterator<OtherElmtRef, U, N1, Nr...> itr) const noexcept
			{
				return ptr_ >= itr.ptr_;
			}
		};

		template <class TargetElmtRef, class T, std::size_t N1, std::size_t... Nr>
		class array_ref_base<TargetElmtRef, T*, N1, Nr...> {
			T* ptr_;

		public:
			static_assert(std::is_reference_v<TargetElmtRef>,
				"jkj::array: the first template argument into array_ref must be "
				"a reference type");
			static_assert(std::is_convertible_v<T*, std::remove_reference_t<TargetElmtRef>*>,
				"jkj::array: the target reference type and the underlying pointer type "
				"of array_ref is not compatiable");
			
		public:
			using element = std::remove_cv_t<std::remove_reference_t<TargetElmtRef>>;

			using element_reference = std::add_lvalue_reference_t<TargetElmtRef>;
			using element_const_reference = std::add_const_t<std::remove_reference_t<TargetElmtRef>>&;
			using element_rvalue_reference = TargetElmtRef;
			using element_const_rvalue_reference = std::conditional_t<std::is_lvalue_reference_v<TargetElmtRef>,
				element_const_reference,
				std::add_const_t<std::remove_reference_t<TargetElmtRef>>&&
			>;

			using iterator = linear_ptr_array_iterator<element_reference, T, N1, Nr...>;
			using const_iterator = linear_ptr_array_iterator<element_const_reference, T, N1, Nr...>;
			using rvalue_iterator = linear_ptr_array_iterator<element_rvalue_reference, T, N1, Nr...>;
			using const_rvalue_iterator = linear_ptr_array_iterator<element_const_rvalue_reference, T, N1, Nr...>;

			using reverse_iterator = std::reverse_iterator<iterator>;
			using const_reverse_iterator = std::reverse_iterator<const_iterator>;
			using rvalue_reverse_iterator = std::reverse_iterator<rvalue_iterator>;
			using const_rvalue_reverse_iterator = std::reverse_iterator<const_rvalue_iterator>;
			
			using reference = typename iterator::reference;
			using const_reference = typename const_iterator::reference;
			using rvalue_reference = typename rvalue_iterator::reference;
			using const_rvalue_reference = typename const_rvalue_iterator::reference;
			
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using value_type = typename iterator::value_type;

			static constexpr size_type rank = sizeof...(Nr)+1;

			// Be careful of object slicing!
			// ptr MUST be the address of the first element
			// of a linear array of objects of type T, not any other type.
			constexpr explicit array_ref_base(T* ptr) noexcept
				: ptr_{ ptr } {}

			// Can be constructed from other array_ref with a different target element reference
		//private:
			template <class OtherElmtRef, class U, bool is_rvalue, bool is_const, template <class, class> class helper_template>
			static constexpr bool is_convertible_impl() noexcept {
				// First of all, check if OtherElmtRef is indeed a reference
				if constexpr(std::is_reference_v<OtherElmtRef>) {
					// Obtain properly const-qualified other element type
					using other_element = std::conditional_t<is_const,
						std::add_const_t<std::remove_reference_t<OtherElmtRef>>,
						std::remove_reference_t<OtherElmtRef>
					>;

					// Obtain properly ref-qualified other element type
					// NOTE: std::is_rvalue_reference_v couldn't be used since it
					//       had generated a compile error. I reported that error so it will be fixed soon.
					using other_element_ref = std::conditional_t<
						is_rvalue && std::is_rvalue_reference<OtherElmtRef>::value,
						other_element&&,
						other_element&
					>;
					
					// T and U should be the same except cv-qualifiers
					return std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>> &&
						helper_template<other_element_ref, U>::value;
				}
				else {
					return false;
				}
			}

			template <class OtherElmtRef, class U>
			struct is_convertible_from_helper : std::integral_constant<bool,
				std::is_convertible_v<OtherElmtRef, TargetElmtRef> &&
				std::is_convertible_v<U*, T*>> {};

			template <class OtherElmtRef, class U>
			struct is_explicitly_convertible_to_helper : std::integral_constant<bool,
				tmp::is_explicitly_convertible_v<TargetElmtRef, OtherElmtRef> &&
				std::is_convertible_v<T*, U*>> {};

			template <class OtherElmtRef, class U, bool is_rvalue, bool is_const>
			static constexpr bool is_convertible_from() noexcept {
				return is_convertible_impl<OtherElmtRef, U, is_rvalue, is_const, is_convertible_from_helper>();
			}

			template <class OtherElmtRef, class U, bool is_rvalue, bool is_const>
			static constexpr bool is_explicitly_convertible_to() noexcept {
				return is_convertible_impl<OtherElmtRef, U, is_rvalue, is_const, is_explicitly_convertible_to_helper>();
			}

		public:
			/* Implicit conversions from */

			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_convertible_from<OtherElmtRef, U, false, false>()>
			>
			constexpr array_ref_base(array_ref<OtherElmtRef, U*, N1, Nr...>& arr) noexcept
				: array_ref_base{ arr.data() } {}

			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_convertible_from<OtherElmtRef, U, false, true>()>
			>
			constexpr array_ref_base(array_ref<OtherElmtRef, U*, N1, Nr...> const& arr) noexcept
				: array_ref_base{ arr.data() } {}

			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_convertible_from<OtherElmtRef, U, true, false>()>
			>
			constexpr array_ref_base(array_ref<OtherElmtRef, U*, N1, Nr...>&& arr) noexcept
				: array_ref_base{ std::move(arr).data() } {}

			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_convertible_from<OtherElmtRef, U, true, true>()>
			>
			constexpr array_ref_base(array_ref<OtherElmtRef, U*, N1, Nr...> const&& arr) noexcept
				: array_ref_base{ std::move(arr).data() } {}


			/* Explicit conversions into */
			// NOTE: These are necessary to convert an lvalue reference into an rvalue reference.

			template <class OtherElmtRef, class U,
				class = std::enable_if_t<is_explicitly_convertible_to<OtherElmtRef, U, true, false>()>
			>
			explicit constexpr operator array_ref<OtherElmtRef, U*, N1, Nr...>() noexcept {
				return array_ref<OtherElmtRef, U*, N1, Nr...>{ ptr_ };
			}


			// Disable copy/move constructors and assignments
			array_ref_base(array_ref_base const&) = delete;
			array_ref_base(array_ref_base&&) = delete;
			array_ref_base& operator=(array_ref_base const&) = delete;
			array_ref_base& operator=(array_ref_base&&) = delete;


			constexpr T* data() noexcept {
				return ptr_;
			}
			constexpr std::add_const_t<T>* data() const noexcept {
				return ptr_;
			}

			constexpr std::size_t stride() const noexcept {
				return get_strides<Nr...>::total_size;
			}
			constexpr std::size_t size() const noexcept {
				return N1;
			}
			constexpr std::size_t num_elements() const noexcept {
				return get_strides<N1, Nr...>::total_size;
			}
			constexpr bool empty() const noexcept {
				return size() == 0;
			}
			constexpr size_type max_size() const noexcept {
				return size();
			}

			// Iterators
			constexpr auto begin() & noexcept {
				return iterator{ data() };
			}
			constexpr auto begin() const& noexcept {
				return const_iterator{ data() };
			}
			constexpr auto begin() && noexcept {
				return rvalue_iterator{ data() };
			}
			constexpr auto begin() const&& noexcept {
				return const_rvalue_iterator{ data() };
			}
			constexpr auto cbegin() const& noexcept {
				return begin();
			}
			constexpr auto cbegin() const&& noexcept {
				return std::move(*this).begin();
			}

			constexpr auto end() & noexcept {
				return iterator{ data() + size() };
			}
			constexpr auto end() const& noexcept {
				return const_iterator{ data() + size() };
			}
			constexpr auto end() && noexcept {
				return rvalue_iterator{ data() + size() };
			}
			constexpr auto end() const&& noexcept {
				return const_rvalue_iterator{ data() + size() };
			}
			constexpr auto cend() const& noexcept {
				return end();
			}
			constexpr auto cend() const&& noexcept {
				return std::move(*this).end();
			}

			constexpr auto rbegin() & noexcept {
				return std::make_reverse_iterator(end());
			}
			constexpr auto rbegin() && noexcept {
				return std::make_reverse_iterator(std::move(*this).end());
			}
			constexpr auto rbegin() const& noexcept {
				return std::make_reverse_iterator(end());
			}
			constexpr auto rbegin() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).end());
			}
			constexpr auto crbegin() const& noexcept {
				return rbegin();
			}
			constexpr auto crbegin() const&& noexcept {
				return std::move(*this).rbegin();
			}

			constexpr auto rend() & noexcept {
				return std::make_reverse_iterator(begin());
			}
			constexpr auto rend() && noexcept {
				return std::make_reverse_iterator(std::move(*this).begin());
			}
			constexpr auto rend() const& noexcept {
				return std::make_reverse_iterator(begin());
			}
			constexpr auto rend() const&& noexcept {
				return std::make_reverse_iterator(std::move(*this).begin());
			}
			constexpr auto crend() const& noexcept {
				return rend();
			}
			constexpr auto crend() const&& noexcept {
				return std::move(*this).rend();
			}

			// Access elements
		private:
			template <class ArrayRef>
			static constexpr decltype(auto) access_impl(ArrayRef&& arr, size_type pos) noexcept {
				assert(pos < N1);
				return std::forward<ArrayRef>(arr).begin()[pos];
			}
			template <class ArrayRef>
			static constexpr decltype(auto) at_impl(ArrayRef&& arr, size_type pos) {
				if( pos >= N1 )
					throw std::out_of_range{ "jkj::array: out of range" };
				return access_impl(std::forward<ArrayRef>(arr), pos);
			}
			template <class ArrayRef>
			static constexpr decltype(auto) front_impl(ArrayRef&& arr) noexcept {
				assert(N1 > 0);
				return access_impl(std::forward<ArrayRef>(arr), 0);
			}
			template <class ArrayRef>
			static constexpr decltype(auto) back_impl(ArrayRef&& arr) noexcept {
				assert(N1 > 0);
				access_impl(std::forward<ArrayRef>(arr), N1 - 1);
			}

		public:
			constexpr decltype(auto) operator[](size_type pos) & {
				return access_impl(*this, pos);
			}
			constexpr decltype(auto) operator[](size_type pos) const& {
				return access_impl(*this, pos);
			}
			constexpr decltype(auto) operator[](size_type pos) && {
				return access_impl(std::move(*this), pos);
			}
			constexpr decltype(auto) operator[](size_type pos) const&& {
				return access_impl(std::move(*this), pos);
			}

			constexpr decltype(auto) at(size_type pos) & {
				return at_impl(*this, pos);
			}
			constexpr decltype(auto) at(size_type pos) const& {
				return at_impl(*this, pos);
			}
			constexpr decltype(auto) at(size_type pos) && {
				return at_impl(std::move(*this), pos);
			}
			constexpr decltype(auto) at(size_type pos) const&& {
				return at_impl(std::move(*this), pos);
			}

			constexpr decltype(auto) front() & noexcept {
				return front_impl(*this);
			}
			constexpr decltype(auto) front() const& noexcept {
				return front_impl(*this);
			}
			constexpr decltype(auto) front() && noexcept {
				return front_impl(std::move(*this));
			}
			constexpr decltype(auto) front() const&& noexcept {
				return front_impl(std::move(*this));
			}

			constexpr decltype(auto) back() & noexcept {
				return back_impl(*this);
			}
			constexpr decltype(auto) back() const& noexcept {
				return back_impl(*this);
			}
			constexpr decltype(auto) back() && noexcept {
				return back_impl(std::move(*this));
			}
			constexpr decltype(auto) back() const&& noexcept {
				return back_impl(std::move(*this));
			}
		};


		template <class TargetElmtRef, class T, std::size_t... N>
		class array_mutable_ref_base<TargetElmtRef, T*, N...> :
			public array_ref_base<TargetElmtRef, T*, N...>
		{
			using base_type = array_ref_base<TargetElmtRef, T*, N...>;

		public:
			using base_type::base_type;
			using base_type::data;
			using base_type::num_elements;
			using element = typename base_type::element;

		private:
			template <class ArrayRef>
			static constexpr void fill_impl(ArrayRef&& arr, element const& x)
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				// std::fill is constexpr since C++20; before that, we role our own
				for( std::size_t i = 0; i < arr.num_elements(); ++i )
					std::forward<ArrayRef>(arr).data()[i] = x;
			}

		public:
			constexpr void fill(element const& x) &
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				fill_impl(*this, x);
			}
			constexpr void fill(element const& x) &&
				noexcept(std::is_nothrow_copy_assignable_v<element>)
			{
				fill_impl(std::move(*this), x);
			}

			// TBD: Swap
		};

		template <class TargetElmt, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref<TargetElmt&, SourceArrayRef, N1, Nr...> :
			public array_mutable_ref_base<TargetElmt&, SourceArrayRef, N1, Nr...>
		{
			using base_type = array_mutable_ref_base<TargetElmt&, SourceArrayRef, N1, Nr...>;

		public:
			using base_type::base_type;

			// Copy constructor
			// NOTE: cannot be copy constructed from a const-reference
			constexpr array_ref(array_ref& that) noexcept
				: base_type{ that } {}

			// Move constructor
			constexpr array_ref(array_ref&& that) noexcept
				: base_type{ std::move(that) } {}

			// TBD: assignment
		};

		template <class TargetElmt, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref<TargetElmt const&, SourceArrayRef, N1, Nr...> :
			public array_ref_base<TargetElmt const&, SourceArrayRef, N1, Nr...>
		{
			using base_type = array_ref_base<TargetElmt const&, SourceArrayRef, N1, Nr...>;

		public:
			using base_type::base_type;

			// Copy constructor
			constexpr array_ref(array_ref const& that) noexcept
				: base_type{ that } {}

			// Move constructor
			constexpr array_ref(array_ref const&& that) noexcept
				: base_type{ std::move(that) } {}
		};

		template <class TargetElmt, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref<TargetElmt&&, SourceArrayRef, N1, Nr...> :
			public array_mutable_ref_base<TargetElmt&&, SourceArrayRef, N1, Nr...>
		{
			using base_type = array_mutable_ref_base<TargetElmt&&, SourceArrayRef, N1, Nr...>;

		public:
			using base_type::base_type;
			
			// Move constructor
			constexpr array_ref(array_ref&& that) noexcept
				: base_type{ std::move(that) } {}

			// TBD: assignment
		};

		template <class TargetElmt, class SourceArrayRef, std::size_t N1, std::size_t... Nr>
		class array_ref<TargetElmt const&&, SourceArrayRef, N1, Nr...> :
			public array_ref_base<TargetElmt const&&, SourceArrayRef, N1, Nr...>
		{
			using base_type = array_ref_base<TargetElmt const&&, SourceArrayRef, N1, Nr...>;

		public:
			using base_type::base_type;
			
			// Move constructor
			constexpr array_ref(array_ref const&& that) noexcept
				: base_type{ std::move(that) } {}
		};
	}
}