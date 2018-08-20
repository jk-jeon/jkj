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
			std::lock_guard<std::mutex> lg{ mtx_ };
			assert(counter_ <= std::numeric_limits<decltype(counter_)>::max() - 1);
			++counter_;
			cv_.notify_one();
		}

		void signal(unsigned int count) {
			std::lock_guard<std::mutex> lg{ mtx_ };
			assert(counter_ <= std::numeric_limits<decltype(counter_)>::max() - count);
			counter_ += count;
			cv_.notify_all();
		}
	};
}
