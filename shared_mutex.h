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
#include <cassert>
#include "semaphore.h"
#include "bitfield.h"

namespace jkj {
	// A modification of https://github.com/preshing/cpp11-on-multicore/blob/master/common/rwlock.h
	// This RW-lock has following properties:
	//   1. There is no locking at all if there is no contention, especially, when either
	//       (1) there is only one thread, or
	//       (2) there are multiple threads but all of them are readers, not writers,
	//      provided that std::atomic<CounterType> is truely lock-free.
	//   2. If there is a thread waiting to write while there already are reading threads,
	//      any other threads attempting to read should wait until the waiting writer to complete;
	//      that is, "readers do not starve writers."
	//   3. If there are threads waiting to read while there already is a writing thread,
	//      any other threads attempting to write should wait until the waiting readers to complete;
	//      that is, "writers do not starve readers."
	//   4. Read-lock (shared ownership) is recursive; that is, without any problems
	//      (e.g., a deadlock), a thread can acquire a shared ownership while it already has one.
	//      However, all of the followings will cause a deadlock:
	//       (1) A thread attempts to acquire a shared ownership
	//           while it already has an exclusive ownership. Use unlock_and_lock_shared() instead.
	//       (2) A thread attempts to acquire an exclusive ownership
	//           while it already has a shared ownership. Use unlock_shared_and_lock() instead.
	//       (3) A thread attempts to acquire an exclusive ownership while it already has one.
	// Things modified from the original source code:
	//   1. Platform-specific semaphores were replaced with jkj::semaphore.
	//      Note that jkj::semaphore might be slower than platform-specific semaphores.
	//   2. Bitfields were replaced with jkj::bitfield, and
	//      the default reader_bits was increased from 10 to 11.
	//   3. Some more assert's were added.
	//   4. Member function names were changed to make them usable along with
	//      std::lock_guard, std::unique_lock, std::shared_lock, or any others.
	//   5. Added try_lock() and try_lock_shared().
	//   6. Added unlock_shared_and_lock() and unlock_and_lock_shared().
	// Anyone using this class must be careful about following pitfalls:
	//   1. I'm not completely sure of correctness of memory orders.
	//   2. There is no atomic upgrade.
	//   3. std::shared_mutex is guaranteed not to fail when implementation-defined maximum number
	//      of threads that can have the shared ownership is reached; it is guaranteed that
	//      in such a case, any further threads that try to obtain the shared ownership will be
	//      blocked. This shared_mutex doesn't follow that semantics,
	//      and simply fails (asserts) on that situation.
	//   4. I'm not sure how to deal with exception safety. Currently nothing is done.
	//      It might be truly a disaster if semaphore::signal() throws.
	//      Indeed semaphore::signal() may throw because it involves
	//      locking a mutex, which can fail. However, exceptions on that operation
	//      almost certainly indicates program bugs. Proper handling of those exceptions
	//      inside this class would help finding out those bugs, but would have little effect
	//      on correctness itself.

	template <class CounterType = std::uint32_t, std::size_t reader_bits = 11>
	class basic_shared_mutex {
		std::atomic<CounterType>	status_{ 0 };
		semaphore					reader_semaphore;
		semaphore					writer_semaphore;

		static constexpr std::size_t total_bits = sizeof(CounterType) * 8;
		static_assert(reader_bits * 2 < total_bits, "jkj::basic_shared_mutex: you have allocated "
			"too many bits for reader-counters");
		static constexpr std::size_t writer_bits = total_bits - 2 * reader_bits;

		using bitfield_type = bitfield<CounterType, reader_bits, reader_bits, writer_bits>;
		using bitfield_ref = bitfield_view<CounterType, reader_bits, reader_bits, writer_bits>;

		static constexpr decltype(auto) readers(CounterType& x) noexcept {
			using jkj::get;
			return get<0>(bitfield_ref{ x });
		}
		static constexpr decltype(auto) waiting_readers(CounterType& x) noexcept {
			using jkj::get;
			return get<1>(bitfield_ref{ x });
		}
		static constexpr decltype(auto) writers(CounterType& x) noexcept {
			using jkj::get;
			return get<2>(bitfield_ref{ x });
		}

	public:
		// Reader lock
		void lock_shared() {
			auto prev_status = status_.load(std::memory_order_relaxed);
			CounterType new_status;
			do {
				new_status = prev_status;
				if( writers(prev_status) != 0 ) {
					assert(waiting_readers(new_status) + 1 <= bitfield_ref::template maximum<1>());
					++waiting_readers(new_status);
				}
				else {
					assert(readers(new_status) + 1 <= bitfield_ref::template maximum<0>());
					++readers(new_status);
				}

				// Since we are to see data written by writers,
				// memory_order_acquire should be used for the success case.
			} while( !status_.compare_exchange_weak(prev_status, new_status,
				std::memory_order_acquire, std::memory_order_relaxed) );

			// If there is a writer, wait
			if( writers(prev_status) != 0 ) {
				reader_semaphore.wait();
			}
		}

		// Reader try_lock
		// This function is allowed to spuriously fail and returns false.
		bool try_lock_shared() {
			// If currently there is no writer,
			auto prev_status = status_.load(std::memory_order_relaxed);
			if( writers(prev_status) == 0 ) {
				assert(readers(prev_status) + 1 <= bitfield_ref::template maximum<0>());
				auto new_status = prev_status;
				++readers(new_status);

				// Then try to publish new_status.
				// If this succeeds, then this_thread have obtained a shared ownership.
				return status_.compare_exchange_strong(prev_status, new_status,
					std::memory_order_acquire, std::memory_order_relaxed);
			}
			return false;
		}

		// Reader unlock
		void unlock_shared() {
			// Since we may possibly publish a new data, memory_order_release should be used.
			auto prev_status = status_.fetch_sub(bitfield_type(1, 0, 0).get_packed(),
				std::memory_order_release);
			assert(readers(prev_status) > 0);

			// Wake a writer up
			if( readers(prev_status) == 1 && writers(prev_status) != 0 ) {
				writer_semaphore.signal();
			}
		}
		
		// Writer lock
		void lock() {
			// Since we are to see data written by other writers,
			// memory_order_acquire should be used here.
			auto prev_status = status_.fetch_add(bitfield_type(0, 0, 1).get_packed(),
				std::memory_order_acquire);
			assert(writers(prev_status) + 1 <= bitfield_ref::template maximum<2>());

			// Wait if there are other readers/writers
			if( readers(prev_status) != 0 || writers(prev_status) != 0 ) {
				writer_semaphore.wait();
			}
		}

		// Writer try_lock
		// This function is allowed to spuriously fail and returns false.
		bool try_lock() {
			// If currently there is no writer as well as no reader,
			auto prev_status = status_.load(std::memory_order_relaxed);
			if( readers(prev_status) == 0 || writers(prev_status) == 0 ) {
				assert(writers(prev_status) + 1 <= bitfield_ref::template maximum<2>());
				auto new_status = prev_status;
				++writers(new_status);

				// Then try to publish new_status.
				// If this succeeds, then this_thread have obtained an exclusive ownership.
				return status_.compare_exchange_strong(prev_status, new_status,
					std::memory_order_acquire, std::memory_order_relaxed);
			}
			return false;
		}

		// Writer unlock
		void unlock() {
			auto prev_status = status_.load(std::memory_order_relaxed);
			CounterType new_status;
			do {
				assert(writers(prev_status) > 0);
				assert(readers(prev_status) == 0);

				new_status = prev_status;
				--writers(new_status);
				readers(new_status) = waiting_readers(new_status);
				waiting_readers(new_status) = 0;

				// Since we will publish a new data, memory_order_release should be used.
			} while( !status_.compare_exchange_weak(prev_status, new_status,
				std::memory_order_release) );
			
			// If there are readers waiting
			if( waiting_readers(prev_status) != 0 ) {
				// Signal to all of them
				reader_semaphore.signal(waiting_readers(prev_status));
			}
			// Otherwise, if there are writers waiting
			else if( writers(prev_status) > 1 ) {
				// Signal to one of them
				writer_semaphore.signal();
			}
		}

		// Obtain exclusive ownership from shared ownership
		// Basically unlock_shared() and then lock(), but the intermediate phase is not introduced.
		// NOTE: this operation is not immediate; if there are other readers or writers waiting,
		//       the operation waits for those waiting readers/writers to complete.
		void unlock_shared_and_lock() {
			auto prev_status = status_.load(std::memory_order_relaxed);
			CounterType new_status;
			do {
				assert(readers(prev_status) > 0);
				assert(writers(prev_status) + 1 <= bitfield_ref::template maximum<2>());

				new_status = prev_status;
				--readers(new_status);
				++writers(new_status);

				// We may possibly publish a new data, so the memory order for the
				// success case should be stronger than or equal to memory_order_release.
				// However, I'm not sure if memory_order_acquire is really needed here.
			} while( !status_.compare_exchange_weak(prev_status, new_status,
				std::memory_order_acq_rel, std::memory_order_relaxed) );

			// If there are other waiting readers or writers
			if( readers(prev_status) > 1 || writers(prev_status) != 0 ) {
				// If this was the last reader, wake up a writer
				if( readers(prev_status) == 1 )
					writer_semaphore.signal();
				writer_semaphore.wait();
			}
		}

		// Obtain shared ownership from exclusive ownership
		// Basically unlock() and then lock_shared(), but the intermediate phase is not introduced.
		// NOTE: this operation is immediate; any other waiting readers/writers can't
		//       proceed before the this operation completes.
		void unlock_and_lock_shared() {
			auto prev_status = status_.load(std::memory_order_relaxed);
			CounterType new_status;
			do {
				assert(writers(prev_status) > 0);
				assert(readers(prev_status) == 0);
				assert(waiting_readers(prev_status) + 1 <= bitfield_ref::template maximum<0>());

				new_status = prev_status;
				--writers(new_status);
				readers(new_status) = waiting_readers(new_status) + 1;
				waiting_readers(new_status) = 0;

				// Since we will publish a new data, memory_order_release should be used.
				// There is no need to specify memory_order_acquire, because this thread will
				// immediately proceed without interferences from other threads.
			} while( !status_.compare_exchange_weak(prev_status, new_status,
				std::memory_order_release, std::memory_order_relaxed) );

			// If there are readers waiting
			if( waiting_readers(prev_status) != 0 ) {
				// Signal to all of them
				reader_semaphore.signal(waiting_readers(prev_status));
			}
		}
	};

	using shared_mutex = basic_shared_mutex<>;
	using shared_mutex64 = basic_shared_mutex<unsigned long long, 22>;
}
