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
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "tmp/forward.h"

namespace jkl {
	/// A minimal variant of std::optional of C++17
	/// This template will be deprecated after std::optional becomes available
	class bad_optional_access : public std::logic_error {
		using std::logic_error::logic_error;
	};
	template <class T>
	class optional {
		std::aligned_storage_t<sizeof(std::pair<T, bool>), alignof(std::pair<T, bool>)> body;
		// Mark the "is-constructed-flag" as value
		void mark_flag(bool value) noexcept {
			reinterpret_cast<std::pair<T, bool>&>(body).second = value;
		}

		template <class Optional>
		optional& assignment(Optional&& that) {
			if( *this ) {
				// If both are constructed, do copy/move assignment
				if( that ) {
					**this = *std::forward<Optional>(that);
				}
				// If only this is constructed, release it and mark this as unconstructed
				else {
					(**this).~T();
					mark_flag(false);
				}
			} else {
				// If only that is constructed, copy/move construct from it
				if( that ) {
					new(this) T(*std::forward<Optional>(that));
					mark_flag(true);
				}
				// If both are not constructed, do nothing
			}
			return *this;
		}

	public:
		// Contextual conversion into bool
		explicit operator bool() const noexcept {
			return reinterpret_cast<std::pair<T, bool> const&>(body).second;
		}

		// Default constructor
		optional() noexcept {
			mark_flag(false);
		}

		// Perfect forwarding constructor for single argument
		template <class Arg, class = ::jkl::tmp::prevent_too_perfect_fwd<optional, Arg>>
		optional(Arg&& arg) noexcept(std::is_nothrow_constructible<T, Arg&&>::value)
		{
			new(this) T(std::forward<Arg>(arg));
			mark_flag(true);
		}
		// Perfect forwarding constructor for multiple arguments
		template <class FirstArg, class SecondArg, class... RemainingArgs>
		optional(FirstArg&& first_arg, SecondArg&& second_arg, RemainingArgs&&... remaining_args)
			noexcept(std::is_nothrow_constructible<T, FirstArg&&, SecondArg&&, RemainingArgs&&...>::value)
		{
			new(this) T(std::forward<FirstArg>(first_arg), std::forward<SecondArg>(second_arg),
				std::forward<RemainingArgs>(remaining_args)...);
			mark_flag(true);
		}

		// Destructor
		~optional() {
			reset();
		}

		// Move constructor
		optional(optional&& that) noexcept(std::is_nothrow_move_constructible<T>::value) {
			if( that ) {
				new(this) T(std::move(*that));
				mark_flag(true);
			} else
				mark_flag(false);
		}

		// Move assignment
		optional& operator=(optional&& that) noexcept(std::is_nothrow_move_assignable<T>::value
			&& std::is_nothrow_move_constructible<T>::value)
		{
			return assignment(std::move(that));
		}

		// Copy constructor
		optional(optional const& that) noexcept(std::is_nothrow_copy_constructible<T>::value) {
			if( that ) {
				new(this) T(*that);
				mark_flag(true);
			} else
				mark_flag(false);
		}

		// Copy assignment operator
		optional& operator=(optional const& that) noexcept(std::is_nothrow_copy_assignable<T>::value
			&& std::is_nothrow_copy_constructible<T>::value)
		{
			return assignment(that);
		}

		// Accessors
		T* operator->() noexcept { return reinterpret_cast<T*>(this); }
		T const* operator->() const noexcept { return reinterpret_cast<T const*>(this); }
		T& operator*() & noexcept { return *reinterpret_cast<T*>(this); }
		T const& operator*() const& noexcept { return *reinterpret_cast<T const*>(this); }
		T&& operator*() && noexcept { return std::move(**this); }
		T const&& operator*() const&& noexcept { return std::move(**this); }
		T& value() & {
			return const_cast<T&>(static_cast<optional const*>(this)->value());
		}
		T const& value() const& {
			if( *this )
				return **this;
			else
				throw bad_optional_access{ "optional<T>::value: not engaged" };
		}

		// Assignment from U
		// See http://en.cppreference.com/w/cpp/utility/optional/operator%3D
		template <class U, class = std::enable_if_t<
			!std::is_same<std::decay_t<U>, optional>::value && 
			std::is_constructible<T, U>::value &&
			std::is_assignable<T&, U>::value &&
			(!std::is_scalar<T>::value || !std::is_same<std::decay_t<U>, T>::value)>
		>
		optional& operator=(U&& that) noexcept(std::is_nothrow_assignable<T&, U>::value
				&& std::is_nothrow_constructible<T, U>::value)
		{
			// If this is already constructed, do assignment
			if( *this )
				**this = std::forward<U>(that);
			// Otherwise, do copy construction
			else {
				new(this) T(std::forward<U>(that));
				mark_flag(true);
			}
			return *this;
		}

		// Modifiers
		void reset() noexcept(std::is_nothrow_destructible<T>::value) {
			if( *this ) {
				(**this).~T();
				mark_flag(false);
			}
		}

		template <typename... Args>
		void unsafe_emplace(Args&&... args) noexcept(std::is_nothrow_constructible<T, Args&&...>::value &&
			std::is_nothrow_destructible<T>::value)
		{
			new(this) T(std::forward<Args>(args)...);
			mark_flag(true);
		}

		template <typename U, typename... Args,
			class = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>, Args&&...>::value>>
		void unsafe_emplace(std::initializer_list<U> list, Args&&... args)
			noexcept(std::is_nothrow_constructible<T, std::initializer_list<U>, Args&&...>::value)
		{
			new(this) T(list, std::forward<Args>(args)...);
			mark_flag(true);
		}

		template <typename... Args>
		void emplace(Args&&... args) noexcept(std::is_nothrow_constructible<T, Args&&...>::value &&
			std::is_nothrow_destructible<T>::value)
		{
			reset();
			unsafe_emplace(std::forward<Args>(args)...);
		}

		template <typename U, typename... Args,
			class = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>, Args&&...>::value>>
		void emplace(std::initializer_list<U> list, Args&&... args)
			noexcept(std::is_nothrow_constructible<T, std::initializer_list<U>, Args&&...>::value)
		{
			reset();
			unsafe_emplace(list, std::forward<Args>(args)...);
		}

		void swap(optional& that) noexcept(noexcept(std::swap(**this, *that)) &&
			std::is_nothrow_move_constructible<T>::value && std::is_nothrow_destructible<T>::value)
		{
			if( *this ) {
				// If both are constructed, do usual swap
				if( that ) {
					std::swap(**this, *that);
				}
				// If only this is constructed, move it to that and mark this as released
				else {
					that.unsafe_emplace(*std::move(*this));
					(**this).~T();
					mark_flag(false);
				}
			} else {
				// If only that is constructed, move it the this and mark that as released
				if( that ) {
					unsafe_emplace(*std::move(that));
					(*that).~T();
					that.mark_flag(false);
				}
				// If both are not constructed, do nothing
			}
		}
	};

	// ADL swap for jkl::util::optional
	template <typename T>
	void swap(optional<T>& x, optional<T>& y) noexcept(noexcept(x.swap(y))) {
		x.swap(y);
	}
}