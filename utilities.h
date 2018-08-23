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
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>

namespace jkl {
	namespace util {
		/// Self-updating frame rate counter
		/// Counter is reset to zero under one of the following conditions:
		///   1. More than max_period of time has passed since the last update
		///   2. Counter becomes greater than or eqeual to max_count

		class fps_counter
		{
			std::atomic<unsigned int>	counter;
			std::atomic<float>			fps_;

			std::chrono::high_resolution_clock::time_point		last;
			std::chrono::high_resolution_clock::duration		duration;

			std::chrono::high_resolution_clock::duration const	update_period;
			std::chrono::high_resolution_clock::duration const	max_period;
			unsigned int const									max_count;

			std::atomic<bool>			terminate;
			std::condition_variable		cv;
			std::mutex					wait_mutex;
			std::thread					update_thread;

			void update_routine() {
				while( true ) {
					{
						std::unique_lock<std::mutex> lock{ wait_mutex };
						cv.wait_for(lock, update_period);
						if( terminate.load(std::memory_order_relaxed) )
							break;
					}

					auto now = std::chrono::high_resolution_clock::now();
					duration += now - last;

					auto current_counter = counter.load(std::memory_order_relaxed);
					fps_.store(float(current_counter) * 1E9f
						/ std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count(),
						std::memory_order_relaxed);

					if( current_counter >= max_count || duration >= max_period ) {
						counter.store(0, std::memory_order_relaxed);
						duration = std::chrono::seconds(0);
					}
					last = now;
				}
			}

		public:
			fps_counter(
				std::chrono::high_resolution_clock::duration update_period = std::chrono::milliseconds(100),
				std::chrono::high_resolution_clock::duration max_period = std::chrono::seconds(30),
				unsigned int max_count = 10000) noexcept
				: counter{ 0 }, fps_{ 0 }, last{ std::chrono::high_resolution_clock::now() },
				duration{ std::chrono::seconds(0) },
				update_period{ update_period }, max_period{ max_period }, max_count{ max_count },
				terminate{ false }, update_thread{ &fps_counter::update_routine, this } {}
			~fps_counter() {
				{
					std::lock_guard<std::mutex> lg{ wait_mutex };
					terminate.store(true, std::memory_order_relaxed);
				}
				cv.notify_one();
				update_thread.join();
			}

			void count() noexcept { counter.fetch_add(1, std::memory_order_relaxed); }			
			float fps() const noexcept { return fps_.load(std::memory_order_relaxed); }
		};


		/// A proxy object holding a data together with a lock
		/// This proxy object locks on its construction, and unlocks on its destruction
		/// Interfaces for accessing the data resemble that of std::optional,
		/// but T usually has a reference semantics.

		template <typename MovableLock, typename T>
		class auto_locked_data {
			MovableLock		lg;
			T				data_;

		public:
			template <typename... Args,
				typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
			auto_locked_data(MovableLock&& lg, Args&&... args) :
				lg(std::forward<MovableLock>(lg)), data_(std::forward<Args>(args)...) {}

			// Move only (lg prevents copy)
			auto_locked_data(auto_locked_data&&) = default;
			auto_locked_data& operator=(auto_locked_data&&) = default;

			// Get the stored data
			T& operator*() & noexcept { return data_; }
			T const& operator*() const& noexcept { return data_; }
			T& value() & noexcept { return data_; }
			T const& value() const& noexcept { return data_; }

			T&& operator*() && noexcept { return std::move(data_); }
			T const&& operator*() const&& noexcept { return std::move(data_); }
			T&& value() && noexcept { return std::move(data_); }
			T const&& value() const&& noexcept { return std::move(data_); }
			
			std::remove_reference_t<T>* operator->() & noexcept { return &data_; }
			std::remove_reference_t<T> const* operator->() const& noexcept { return &data_; }
			
			// Move lock
			MovableLock release() && noexcept { return std::move(lg); }
		};

		template <typename MovableLock, typename T>
		auto_locked_data<MovableLock, T> make_auto_locked_data(MovableLock&& lg, T&& data) {
			return{ std::forward<MovableLock>(lg), std::forward<T>(data) };
		}

		template <typename MovableLock, typename... T>
		using auto_locked_tuple = auto_locked_data<MovableLock, std::tuple<T...>>;

		template <typename MovableLock, typename... T>
		auto_locked_tuple<MovableLock, T...> make_auto_locked_tuple(MovableLock& lg, T&&... data) {
			return{ std::forward<MovableLock>(lg), std::forward<T>(data)... };
		}

		// Provides std::get()-like function for auto_locked_tuple
		// The result is of auto_locked_data corresponding to the tuple_element_t
		template <std::size_t I, typename MovableLock, typename... T>
		auto get(auto_locked_tuple<MovableLock, T...>&& t)
			noexcept(std::is_nothrow_move_constructible<std::tuple_element_t<I, std::tuple<T...>>>::value)
		{
			using std::get;
			return make_auto_locked_data(std::move(t).release(), get<I>(std::move(t).value()));
		}
		
		// An extension of std::get(); extracts multiple elements from auto_locked_tuple
		namespace detail {
			template <typename... T>
			struct extract_noexcept;

			template <typename FirstType, typename... RemainingType>
			struct extract_noexcept<FirstType, RemainingType...> {
				constexpr static bool value = std::is_nothrow_move_constructible<FirstType>::value
					&& extract_noexcept<RemainingType...>::value;
			};

			template <>
			struct extract_noexcept<> {
				constexpr static bool value = true;
			};
		}
		template <size_t... I, typename MovableLock, typename... T>
		auto extract(auto_locked_tuple<MovableLock, T...>&& t)
			noexcept(detail::extract_noexcept<T...>::value)
		{
			using std::get;
			return make_auto_locked_tuple(std::move(t).release(),
				get<I>(std::move(t).value())...);
		}

		// Pointer version of auto_locked_data
		template <typename MovableLock, typename T>
		class auto_locked_ptr {
			MovableLock		lg;
			T*				ptr_;

		public:
			template <class U, class = std::enable_if_t<std::is_convertible<U*, T*>::value>>
			auto_locked_ptr(MovableLock&& lg, U* ptr) :
				lg(std::forward<MovableLock>(lg)), ptr_{ ptr } {}

			auto_locked_ptr(MovableLock&& lg, std::nullptr_t) :
				lg(std::forward<MovableLock>(lg)), ptr_{ nullptr } {}

			// Move only (lg prevents copy)
			auto_locked_ptr(auto_locked_ptr&&) = default;
			auto_locked_ptr& operator=(auto_locked_ptr&&) = default;

			// Get the stored pointer
			T& operator*() const noexcept { return *ptr_; }			
			T* operator->() const noexcept { return ptr_; }
			T* get() const noexcept { return ptr_; }
			operator bool() const noexcept {
				return ptr_ != nullptr;
			}

			// Move lock
			MovableLock release() && noexcept { return std::move(lg); }
		};

		template <typename MovableLock, typename T>
		auto_locked_ptr<MovableLock, T> make_auto_locked_ptr(MovableLock&& lg, T* ptr) {
			return{ std::forward<MovableLock>(lg), ptr };
		}


		/// Add several copies to a vector

		template <class Vector, typename Instance>
		void push_back(Vector& cont, std::size_t num_of_elements, Instance const& inst) {
			cont.reserve(cont.size() + num_of_elements);
			for( std::size_t i = 0; i < num_of_elements; i++ )
				cont.push_back(inst);
		}

		/// Add several elements created by a common set of parameters to a vector

		template <class Vector, typename... Args>
		void emplace_back(Vector& cont, std::size_t num_of_elements, Args&&... args) {
			cont.reserve(cont.size() + num_of_elements);
			for( std::size_t i = 0; i < num_of_elements; i++ )
				cont.emplace_back(std::forward<Args>(args)...);
		}

		/// Add several elements created by invoking a functor to a vector

		template <class ConstructFunctor, class Vector, typename... Args>
		void create_back(Vector& cont, std::size_t num_of_elements,
			ConstructFunctor&& functor, Args&&... args) {
			cont.reserve(cont.size() + num_of_elements);
			for( std::size_t i = 0; i < num_of_elements; i++ )
				cont.emplace_back(functor(std::forward<Args>(args)...));
		}


		/// Measure the time consumed to do a job; for benchmarking

		template <class TimeUnit, class Functor, typename... Args>
		auto elapsed(Functor&& job, Args&&... args) {
			auto from = std::chrono::high_resolution_clock::now();
			job(std::forward<Args>(args)...);
			return std::chrono::duration_cast<TimeUnit>(std::chrono::high_resolution_clock::now() - from);
		}
		template <class Functor, typename... Args>
		auto elapsed_ms(Functor&& job, Args&&... args) {
			return elapsed<std::chrono::milliseconds>(std::forward<Functor>(job), std::forward<Args>(args)...);
		}
		template <class Functor, typename... Args>
		auto elapsed_us(Functor&& job, Args&&... args) {
			return elapsed<std::chrono::microseconds>(std::forward<Functor>(job), std::forward<Args>(args)...);
		}
		template <class Functor, typename... Args>
		auto elapsed_ns(Functor&& job, Args&&... args) {
			return elapsed<std::chrono::nanoseconds>(std::forward<Functor>(job), std::forward<Args>(args)...);
		}


		/// Uniformly randomly sample k-permutations from [0:n)
		/// It holds a linear array consisting of 0, 1, ... , n - 1 and shuffles it to generate a k-permutation.
		/// After returning the generated permutation, it recovers the shuffled array back to
		/// the original sequence 0, 1, ... , n - 1. This can possibly slow down the sampling procedure,
		/// but it makes the class essentially stateless.

		template <class UIntType>
		class permutation_sampler {
		public:
			static_assert(std::is_integral<UIntType>::value && std::is_unsigned<UIntType>::value,
				"jkl::permutation_sampler must be instantiated with an unsigned integral type");

			permutation_sampler() noexcept = default;
			explicit permutation_sampler(UIntType n) :
				m_seq{ std::make_unique<UIntType[]>(n) }, m_capacity{ n }, m_size{ n }
			{
				if( n == 0 )
					throw std::invalid_argument{ "jkl::permutation_sampler: n should be positive" };

				for( UIntType i = 0; i < n; ++i )
					m_seq[i] = i;
			}

			auto size() const noexcept {
				return m_size;
			}
			auto capacity() const noexcept {
				return m_capacity;
			}

			void reserve(UIntType n) {
				if( m_capacity < n ) {
					std::unique_ptr<UIntType[]> new_seq = std::make_unique<UIntType[]>(n);
					std::copy_n(m_seq.get(), m_capacity, new_seq.get());
					for( UIntType i = m_capacity; i < n; ++i )
						new_seq[i] = i;
					std::swap(m_seq, new_seq);
					m_capacity = n;
				}
			}

			void resize(UIntType n) {
				reserve(n);
				m_size = n;
			}

			void shrink_to_fit() {
				if( m_capacity > m_size ) {
					std::unique_ptr<UIntType[]> new_seq = std::make_unique<UIntType[]>(m_size);
					std::copy_n(m_seq.get(), m_size, new_seq.get());
					std::swap(m_seq, new_seq);
					m_capacity = m_size;
				}
			}

			template <class Generator, class OutputIterator>
			void operator()(Generator& g, UIntType k, OutputIterator itr) const {
				if( m_size < k )
					throw std::invalid_argument{ "permutation_sampler: k should be smaller than or equal to n" };

				struct auto_undo {
					UIntType				i;
					permutation_sampler&	self;

					auto& operator++() noexcept {
						++i;
						return *this;
					}

					operator UIntType() const noexcept {
						return i;
					}

					~auto_undo() {
						// Recover original sequence 0, 1, 2, ... , n - 1
						// This procedure makes permutation_sampler essentially stateless.
						// Even in presence of an exception thrown by the generator,
						// this procedure recovers the original sequence thanks to the RAII idiom,
						// though I don't know if any exception could be thrown by standard
						// random number generators.
						
						for( UIntType j = self.m_size - i; j < self.m_size; ++j ) {
							auto& r = self.m_seq[j];
							if( r < self.m_size - i )
								self.m_seq[r] = r;
							r = j;
						}
					}
				};
				
				// Generate a k-permutation
				auto_undo idx{ 0, *this };
				for( ; idx < k; ++idx ) {
					auto r = std::uniform_int_distribution<UIntType>{ 0, m_size - idx - 1 }(g);
					std::swap(m_seq[r], m_seq[m_size - idx - 1]);
				}
				// Copy generated permutation
				std::copy_n(m_seq.get() + (m_size - k), k, itr);
			}

		private:
			std::unique_ptr<UIntType[]>		m_seq = nullptr;
			UIntType						m_capacity = 0;
			UIntType						m_size = 0;
		};
	}
}