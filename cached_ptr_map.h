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

// A typical cached factory using std::shared_ptr and std::weak_ptr

#pragma once
#include <condition_variable>
#include <future>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include "shared_mutex.h"

namespace jkl {
	template <class Key, class Value,
		template <class, class, class...> class Map = std::unordered_map,
		class... AdditionalArgs>
	class cached_ptr_map {
		using map_type = Map<Key, std::weak_ptr<Value>, AdditionalArgs...>;

		mutable jkl::shared_mutex				mtx_;
		mutable std::condition_variable_any		cv_;
		mutable bool							terminate_ = false;
		mutable bool							something_to_clean_ = false;
		std::future<void>						cleaner_;
		map_type								map_;


		class upgrade_lock_guard {
			jkl::shared_mutex& mtx;

		public:
			upgrade_lock_guard(jkl::shared_mutex& mtx) : mtx{ mtx } {
				mtx.unlock_shared_and_lock();
			}
			~upgrade_lock_guard() {
				mtx.unlock();
			}
		};

	public:
		// Launch the cleaner thread
		cached_ptr_map()
		{
			cleaner_ = std::async(std::launch::async, [this]() {
				cleaner_thread();
			});
		}

		// Terminate the cleaner thread
		~cached_ptr_map()
		{
			// Signal to the cleaner thread
			{
				std::lock_guard<jkl::shared_mutex> lg{ mtx_ };
				terminate_ = true;
			}
			cv_.notify_one();

			// Wait
			cleaner_.get();	// should not throw
		}

		// Perform a sort of double-checked locking
		template <class Query, class... InitialArgs>
		std::shared_ptr<Value> get(Query&& q, InitialArgs&&... args)
		{
			// Called when a new object is to be created
			auto make_new_object = [&]() {
				// The deleter tells the cleaner thread to erase the weak_ptr from the map
				return std::shared_ptr<Value>(new Value(std::forward<InitialArgs>(args)...),
					[this](Value* ptr)
				{
					// Signal to the cleaner thread
					{
						std::lock_guard<jkl::shared_mutex> lg{ mtx_ };
						something_to_clean_ = true;
					}
					cv_.notify_one();

					// Delete the object
					delete ptr;
				});
			};

			// Return value
			// This object should be cleared after the lock is released,
			// because the destructor might perform locking.
			std::shared_ptr<Value> ptr;

			// This is called if there is no cache found in the map
			auto emplace_new = [&](auto& lg) {
				// Obtain write lock
				upgrade_lock_guard lg2{ *lg.release() };

				// Find the key again, because the map might have been modified
				auto itr = map_.find(q);

				// If the key still does not exist,
				if( itr == map_.end() ) {
					// Make a new object and put it in the map
					ptr = make_new_object();
					map_.emplace(q, ptr);
					return ptr;
				}
				// Otherwise
				else {
					// If the pointer is expired, recreate the entry
					ptr = itr->second.lock();
					if( !ptr ) {
						ptr = make_new_object();
						itr->second = ptr;
					}
					return ptr;
				}
			};

			// Obtain read-lock
			std::shared_lock<jkl::shared_mutex>	lg{ mtx_ };

			// If the key is not in the map
			auto itr = map_.find(q);
			if( itr == map_.end() ) {
				return emplace_new(lg);
			}
			// Otherwise
			else {
				// If the pointer is expired
				ptr = itr->second.lock();
				if( !ptr )
					return emplace_new(lg);
				// Otherwise
				else
					return ptr;
			}
		}

		Map<Key, std::shared_ptr<Value>> get_all() const {
			// Obtain read lock
			std::shared_lock<jkl::shared_mutex> lg{ mtx_ };

			// Return value
			Map<Key, std::shared_ptr<Value>> ret;

			// Iterate over the map
			for( auto const& p : map_ ) {
				// If the pointer is not expired, push it to the ret
				auto ptr = p.second.lock();
				if( ptr )
					ret.emplace(p.first, std::move(ptr));
			}

			return ret;
		}

	private:
		// Erase expired weak_ptr's from the map
		void cleaner_thread() {
			while( true ) {
				// Wait for signal
				std::unique_lock<jkl::shared_mutex> lg{ mtx_ };
				cv_.wait(lg, [this]() { return something_to_clean_ || terminate_; });

				if( terminate_ )
					break;

				// Erase expired weak_ptr's
				for( auto itr = map_.begin(); itr != map_.end(); ) {
					if( itr->second.expired() )
						map_.erase(itr++);
					else
						++itr;
				}
			}
		}
	};
}