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
#include <mutex>
#include <limits>
#include <condition_variable>

namespace jkl {
	// [Notes on exception safety]
	// According to https://en.cppreference.com/w/cpp/named_req/Mutex,
	// std::mutex::lock() can fail (and throws an exception) only on following situations:
	//  1. The thread does not have a sufficient privillege; in this case,
	//     std::system_error with std::errc::operation_not_permitted is thrown, or
	//  2. The implementation detects that this operation would lead to deadlock; in this case,
	//     std::system_error with std::errc::resource_deadlock_would_occur is thrown, or
	//  3. The thread already owns the mutex; in this case, the behavior is undefined.
	// Since member functions of this semaphore class (wait() and signal()) contains
	// complete transaction of the critical section, the second and the third cases
	// are guaranteed not to occur. According to the same webpage, std::mutex::unlock()
	// should not throw, and in addition, according to
	// https://en.cppreference.com/w/cpp/thread/condition_variable,
	// std::condition_variable::notify_one/all() are noexcept while
	// std::condition_variable::wait() in the wait() member function
	// can throw only when std::mutex::lock() fails. As a consequence,
	// member functions of semaphore can throw only when the calling thread does not
	// have a sufficient privillege.
	
	class semaphore {
		std::mutex				mtx_;
		std::condition_variable	cv_;
		unsigned int			counter_;

	public:
		semaphore(unsigned int init_count = 0) : counter_{ init_count } {}

		void wait() {
			std::unique_lock<std::mutex> lg{ mtx_ };
			cv_.wait(lg, [this]() { return counter_ != 0; });
			assert(counter_ != 0);
			--counter_;
		}

		void signal() {
			{
				std::lock_guard<std::mutex> lg{ mtx_ };
				assert(counter_ <= std::numeric_limits<decltype(counter_)>::max() - 1);
				++counter_;
			}
			cv_.notify_one();
		}

		void signal(unsigned int count) {
			{
				std::lock_guard<std::mutex> lg{ mtx_ };
				assert(counter_ <= std::numeric_limits<decltype(counter_)>::max() - count);
				counter_ += count;
			}
			cv_.notify_all();
		}
	};
}