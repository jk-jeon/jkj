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
#include <atomic>
#include <mutex>

#ifdef _MSC_VER

# if _MSC_VER > 1900
#include <optional>
# else
#include "optional.h"
#define USE_JKL_OPTIONAL
# endif

#else

#include <optional>

#endif

#include <shared_mutex>
#include <tuple>
#include "utilities.h"
#include "tmp/forward.h"
#include "tmp/is_convertible.h"
#include "tmp/unpack_and_apply.h"
#include "shared_mutex.h"

namespace jkj {
	/// A thread-safe data cache
	/// Data cannot be a reference type

	template <typename... InitArgs>
	struct update_cache_as_arg_t : public std::tuple<InitArgs...> {
		using std::tuple<InitArgs...>::tuple;
	};
	template <typename... InitArgs>
	static update_cache_as_arg_t<InitArgs&&...> update_cache_as_arg(InitArgs&&... args) noexcept
	{
		return{ std::forward<InitArgs>(args)... };
	}

	template <typename Data>
	class cache {
	#ifdef USE_JKL_OPTIONAL
		template <typename Data>
		using optional = jkj::optional<Data>;
	#else
		template <typename Data>
		using optional = std::optional<Data>;
	#endif

		mutable shared_mutex					m_mutex;
		bool									m_available;
		optional<Data>							m_cache;
		
		struct guarded_construction {};
		
		static bool mark_moved_cache_as_invalidated(cache const& that) {
			return that.m_available;
		}
		static bool mark_moved_cache_as_invalidated(cache&& that) {
			auto r = that.available;
			that.m_available = false;
			return r;
		}

		template <typename OtherCache>
		cache(guarded_construction, OtherCache&& that, bool available) :
			m_available{ available },
			m_cache{ available ? std::forward<OtherCache>(that).m_cache : optional<Data>{} } {}

		template <typename OtherCache, typename LockGuard>
		cache(guarded_construction, OtherCache&& that, LockGuard const&) :
			cache(guarded_construction{}, std::forward<OtherCache>(that),
				mark_moved_cache_as_invalidated(std::forward<OtherCache>(that))) {}

		template <typename OtherCache>
		void assignment_impl(OtherCache&& that) {
			if( that.m_available ) {
				m_cache = std::forward<OtherCache>(that).m_cache;
				m_available = mark_moved_cache_as_invalidated(std::forward<OtherCache>(that));
			}
			else
				m_available = false;
		}

		struct upgrade_lock_guard {
			shared_mutex& mtx;
			explicit upgrade_lock_guard(shared_mutex& mtx) : mtx{ mtx } {
				mtx.unlock_shared_and_lock();
			}
			~upgrade_lock_guard() {
				mtx.unlock_and_lock_shared();
			}
		};

		template <typename Lambda, class = std::enable_if_t<
			std::is_same<decltype(std::declval<Lambda>()()), void>::value>>
		bool get_update_result(Lambda&& l) {
			std::forward<Lambda>(l)();
			return true;
		}
		template <typename Lambda, class = void, class = std::enable_if_t<
			!std::is_same<decltype(std::declval<Lambda>()()), void>::value>>
		bool get_update_result(Lambda&& l) {
			return std::forward<Lambda>(l)();
		}

		struct update_if_return_ref {
			template <class ThisType>
			util::auto_locked_data<std::shared_lock<shared_mutex>, Data const&>
				operator()(std::shared_lock<shared_mutex>&& lg, ThisType* me)
			{
				return{ std::move(lg), *(me->m_cache) };
			}
		};
		struct update_if_return_ptr {
			template <class ThisType>
			util::auto_locked_ptr<std::shared_lock<shared_mutex>, Data const>
				operator()(std::shared_lock<shared_mutex>&& lg, ThisType* me)
			{
				if( me->m_available )
					return{ std::move(lg), &*(me->m_cache) };
				else
					return{ std::move(lg), nullptr };
			}
		};

	public:
		// Default constructor
		cache() : m_available{ false } {}

		// Perfect forwarding constructor for single argument (implicit)
		template <typename Arg, typename = ::jkj::tmp::prevent_too_perfect_fwd<cache, Arg>,
			typename = std::enable_if_t<std::is_constructible<optional<Data>, Arg>::value &&
			std::is_convertible<Arg, optional<Data>>::value>>
		cache(Arg&& arg) : m_available{ true }, m_cache(std::forward<Arg>(arg)) {}

		// Perfect forwarding constructor for single argument (explicit)
		template <typename Arg, typename = ::jkj::tmp::prevent_too_perfect_fwd<cache, Arg>,
			typename = std::enable_if_t<std::is_constructible<optional<Data>, Arg>::value &&
			!std::is_convertible<Arg, optional<Data>>::value>, class = void>
		explicit cache(Arg&& arg) : m_available{ true }, m_cache(std::forward<Arg>(arg)) {}

		// Perfect forwarding constructor for multiple arguments (implicit)
		template <typename FirstArg, typename SecondArg, typename... RemainingArgs,
			typename = std::enable_if_t<
			std::is_constructible<optional<Data>, FirstArg, SecondArg, RemainingArgs...>::value &&
			jkj::tmp::is_convertible<optional<Data>, FirstArg, SecondArg, RemainingArgs...>::value>>
		cache(FirstArg&& first_arg, SecondArg&& second_arg, RemainingArgs&&... remaining_args)
			: m_available{ true }, m_cache(std::forward<FirstArg>(first_arg),
			std::forward<SecondArg>(second_arg), std::forward<RemainingArgs>(remaining_args)...) {}

		// Perfect forwarding constructor for multiple arguments (explicit)
		template <typename FirstArg, typename SecondArg, typename... RemainingArgs,
			typename = void, typename = std::enable_if_t<
			std::is_constructible<optional<Data>, FirstArg, SecondArg, RemainingArgs...>::value &&
			!jkj::tmp::is_convertible<optional<Data>, FirstArg, SecondArg, RemainingArgs...>::value>>
		explicit cache(FirstArg&& first_arg, SecondArg&& second_arg, RemainingArgs&&... remaining_args)
			: m_available{ true }, m_cache(std::forward<FirstArg>(first_arg),
			std::forward<SecondArg>(second_arg), std::forward<RemainingArgs>(remaining_args)...) {}

		// Destructor
		~cache() = default;

		// Move constructor (do not share the mutex)
		cache(cache&& that) noexcept(false)
			: cache{ guarded_construction{}, std::move(that), std::lock_guard<shared_mutex>{ that.m_mutex } } {}

		// Move assignment (do not share the mutex)
		cache& operator=(cache&& that) noexcept(false) {
			if( this != &that ) {
				std::lock(m_mutex, that.m_mutex);
				std::lock_guard<shared_mutex> lg1{ m_mutex, std::adopt_lock };
				std::lock_guard<shared_mutex> lg2{ that.m_mutex, std::adopt_lock };
				assignment_impl(std::move(that));
			}
			return *this;
		}

		// Move assignment from Data
		cache& operator=(Data&& that) {
			update([&that]() {
				return std::move(that);
			});
			return *this;
		}

		// Copy constructor (do not share the mutex)
		cache(cache const& that)
			: cache{ guarded_construction{}, that, std::shared_lock<shared_mutex>{ that.m_mutex } } {}

		// Copy assignment (do not share the mutex)
		cache& operator=(cache const& that) {
			if( this != &that ) {
				std::shared_lock<shared_mutex> lg2{ that.m_mutex, std::defer_lock };
				std::lock(m_mutex, lg2);
				std::lock_guard<shared_mutex> lg1{ m_mutex, std::adopt_lock };
				assignment_impl(that);
			}
			return *this;
		}

		// Copy assignment from Data
		cache& operator=(Data const& that) {
			update([&that]() {
				return that;
			});
			return *this;
		}

		// Mark the current cache as out-dated.
		void invalidate() {
			std::lock_guard<shared_mutex> lg{ m_mutex };
			m_available = false;
		}
		// Mark the current cache as out-dated if a predicate evaluates to true.
		// If m_available is already false, it doesn't evaluate predicate.
		// Otherwise, the current cache value will be given as the argument to the predicate.
		template <typename Predicate>
		void invalidate_if(Predicate&& pred) {
			std::shared_lock<shared_mutex> lg{ m_mutex };
			if( m_available ) {
				upgrade_lock_guard ulg{ m_mutex };

				// It is possible that some other thread have written
				// a data during acquiring the exclusive ownership,
				// thus we should check m_available again.
				if( m_available ) {
					if( pred(*m_cache) )
						m_available = false;
				}
			}
		}
		// Check if the current cache is updated.
		bool available() const {
			std::shared_lock<shared_mutex> lg{ m_mutex };
			return m_available;
		}

		// Update of the cache using the return value of updator() and mark the current cache as updated.
		template <typename Updator, typename... Args>
		void update(Updator&& updator, Args&&... args) {
			std::lock_guard<shared_mutex> lg{ m_mutex };
			m_cache.emplace(std::forward<Updator>(updator)(std::forward<Args>(args)...));
			m_available = true;
		}
		// Update the cache by passing it as the first argument to updator().
		// If the member m_cache is not yet constructed, construct it using arguments stored in the update_cache_as_arg_t object.
		// The updator() may return either void or bool; for the first case, the current cache is marked as updated,
		// and for the second case, the current cache is marked as updated only when the return value is true.
		template <typename Updator, typename... Args, typename... InitArgs>
		void update(update_cache_as_arg_t<InitArgs...> init_args_pack, Updator&& updator, Args&&... args) {
			std::lock_guard<shared_mutex> lg{ m_mutex };
			if( !m_cache ) {
				jkj::tmp::unpack_and_apply([this](auto&&... init_args) {
					new(&m_cache) optional<Data>(std::forward<decltype(init_args)>(init_args)...);
				}, static_cast<std::tuple<InitArgs...>&&>(init_args_pack));
			}
			m_available = get_update_result([&]() {
				return std::forward<Updator>(updator)(*m_cache, std::forward<Args>(args)...);
			});
		}

		// Update the cache using the return value of updator(),
		// mark the current cache as updated, and get the result.
		template <typename Updator, typename... Args>
		util::auto_locked_data<std::shared_lock<shared_mutex>, Data const&>
			update_and_get(Updator&& updator, Args&&... args)
		{
			std::unique_lock<shared_mutex> lg{ m_mutex };
			m_cache.emplace(std::forward<Updator>(updator)(std::forward<Args>(args)...));
			m_available = true;

			lg.release();
			m_mutex.unlock_and_lock_shared();
			std::shared_lock<shared_mutex> shared_lg{ m_mutex, std::adopt_lock };

			return{ std::move(shared_lg), *m_cache };
		}
		// Update the cache by passing it as the first argument to updator(), mark the current cache as updated
		// according to the return value of updator(), and then return the pointer to the current cache if
		// the result was successful. If updator() returns false, nullptr is returned instead.
		// If the member m_cache is not yet constructed, construct it using arguments stored in the update_cache_as_arg_t object.
		// The updator() may return either void or bool; for the first case, the current cache is marked as updated,
		// and for the second case, the current cache is marked as updated only when the return value is true.
		template <typename Updator, typename... Args, typename... InitArgs>
		util::auto_locked_ptr<std::shared_lock<shared_mutex>, Data const>
			update_and_get(update_cache_as_arg_t<InitArgs...> init_args_pack, Updator&& updator, Args&&... args)
		{
			std::unique_lock<shared_mutex> lg{ m_mutex };
			if( !m_cache ) {
				jkj::tmp::unpack_and_apply([this](auto&&... init_args) {
					new(&m_cache) optional<Data>(std::forward<decltype(init_args)>(init_args)...);
				}, static_cast<std::tuple<InitArgs...>&&>(init_args_pack));
			}
			m_available = get_update_result([&]() {
				return std::forward<Updator>(updator)(*m_cache, std::forward<Args>(args)...);
			});

			lg.release();
			m_mutex.unlock_and_lock_shared();
			std::shared_lock<shared_mutex> shared_lg{ m_mutex, std::adopt_lock };

			if( m_available )
				return{ std::move(shared_lg), &*m_cache };
			else
				return{ std::move(shared_lg), nullptr };
		}
		
		// If the current cache is out-dated, then
		// update the cache using the return value of updator(), 
		// mark the current cache as updated, and get the result. If the current cache is already updated, then get it.
		template <typename Updator, typename... Args>
		FORCEINLINE util::auto_locked_data<std::shared_lock<shared_mutex>, Data const&>
			get(Updator&& updator, Args&&... args)
		{
			return update_if_impl(update_if_return_ref{}, [](auto&&) { return false; },
				[this, &updator](Args&&... args) {
				m_cache.emplace(std::forward<Updator>(updator)(std::forward<Args>(args)...));
			}, std::forward<Args>(args)...);
		}
		// If the current cache is out-dated, then
		// update the cache by passing it as the first argument to updator(), mark the current cache as updated
		// according to the return value of updator(), and then return the pointer to the current cache if
		// the result was successful. If updator() returns false, nullptr is returned instead.
		// If the member m_cache is not yet constructed, construct it using arguments stored in the update_cache_as_arg_t object.
		template <typename Updator, typename... Args, typename... InitArgs>
		FORCEINLINE util::auto_locked_ptr<std::shared_lock<shared_mutex>, Data const>
			get(update_cache_as_arg_t<InitArgs...> init_args_pack, Updator&& updator, Args&&... args)
		{
			return update_if_impl(update_if_return_ptr{}, [](auto&&) { return false; },
				[this, &updator, &init_args_pack](Args&&... args) {
				if( !m_cache ) {
					jkj::tmp::unpack_and_apply([this, &init_args_pack](auto&&... init_args) {
						new(&m_cache) optional<Data>(std::forward<decltype(init_args)>(init_args)...);
					}, static_cast<std::tuple<InitArgs...>&&>(init_args_pack));
				}
				return std::forward<Updator>(updator)(*m_cache, std::forward<Args>(args)...);
			}, std::forward<Args>(args)...);
		}

		// If the current cache is out-dated or the predicate evaluates to true, then
		// update the cache using the return value of updator(), 
		// mark the current cache as updated, and get the result. If the current cache is already updated, then get it.
		template <typename Predicate, typename Updator, typename... Args>
		FORCEINLINE util::auto_locked_data<std::shared_lock<shared_mutex>, Data const&>
			update_if(Predicate&& pred, Updator&& updator, Args&&... args)
		{
			return update_if_impl(update_if_return_ref{}, std::forward<Predicate>(pred),
				[this, &updator](Args&&... args) {
				m_cache.emplace(std::forward<Updator>(updator)(std::forward<Args>(args)...));
			}, std::forward<Args>(args)...);
		}
		// If the current cache is out-dated or the predicate evaluates to true, then
		// update the cache by passing it as the first argument to updator(), 
		// mark the current cache as updated, and get the result. If the current cache is already updated, then get it.
		// If the member m_cache is not yet constructed, construct it using arguments stored in the update_cache_as_arg_t object.
		template <typename Predicate, typename Updator, typename... Args, typename... InitArgs>
		FORCEINLINE util::auto_locked_ptr<std::shared_lock<shared_mutex>, Data const>
			update_if(Predicate&& pred, update_cache_as_arg_t<InitArgs...> init_args_pack, Updator&& updator, Args&&... args)
		{
			return update_if_impl(update_if_return_ptr{}, std::forward<Predicate>(pred),
				[this, &updator, &init_args_pack](Args&&... args) {
				if( !m_cache ) {
					jkj::tmp::unpack_and_apply([this, &init_args_pack](auto&&... init_args) {
						new(&m_cache) optional<Data>(std::forward<decltype(init_args)>(init_args)...);
					}, static_cast<std::tuple<InitArgs...>&&>(init_args_pack));
				}
				return std::forward<Updator>(updator)(*m_cache, std::forward<Args>(args)...);
			}, std::forward<Args>(args)...);
		}

		// If the current cache is out-dated, then return nullptr;
		// otherwise, return the address of current cache
		util::auto_locked_ptr<std::shared_lock<shared_mutex>, Data const> get_if_available() const {
			std::shared_lock<shared_mutex> lg{ m_mutex };
			return update_if_return_ptr{}(std::move(lg), this);
		}

		// Get the current value of the cache, without the concern of optional-ness of m_cache, 
		// availability of the current cache, and thread-safety.
		// If m_cache is not yet constructed, the behavior is undefined
		Data const& unsafe_get() const {
			return *m_cache;
		}
		// Same as above, but throws an exception if m_cache is not yet constructed
		Data const& unsynchronized_get() const {
			return m_cache.value();
		}
		// Set the current value of the cahe, without the concern of thread-safety
		void unsynchronized_set(Data const& new_data) {
			m_cache = new_data;
		}
		void unsynchronized_set(Data&& new_data) {
			m_cache = std::move(new_data);
		}

	private:
		template <typename RetFunctor, typename Predicate, typename Updator, typename... Args>
		auto update_if_impl(RetFunctor&& ret_functor, Predicate&& pred, Updator&& updator, Args&&... args)
		{
			std::shared_lock<shared_mutex> lg{ m_mutex };
			if( !m_available || pred(*m_cache) ) {
				upgrade_lock_guard ulg{ m_mutex };

				// It is possible that some other thread have written
				// a data during acquiring the exclusive ownership,
				// thus we should check m_available again.
				if( !m_available || pred(*m_cache) ) {
					m_available = get_update_result([&]() {
						return updator(std::forward<Args>(args)...);
					});
				}
			}
			return ret_functor(std::move(lg), this);
		}
	};
}

#ifdef USE_JKL_OPTIONAL
#undef USE_JKL_OPTIONAL
#endif