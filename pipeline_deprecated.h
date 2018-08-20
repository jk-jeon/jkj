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

// This library is deprecated; use strmnet instead.

#pragma once
#include <atomic>
#include <future>
#include <cstdint>
#include <cstddef>
#include "tmp.h"
#include "utilities.h"

namespace jkl {
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Pipelines for processing stream data
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	namespace pipeline {
		/// The main class
		template <class...>
		class bundle;

		/// Policies for processing data
		enum class process_policy { 
			none,				// Invalid; reserved for internal use
			single,				// Process the input using a single thread
			multiple,			// Process the input using multiple threads
			matched				// Number of threads are determined to match 
								// the number of outputs from the previous stage
		};
		/// Policies for providing inputs
		enum class input_policy {
			none,				// No input is provided
			direct,				// Inputs are provided only from the outside function call
			passed_from,		// Inputs are generated from the previous stage or from the outside function call
			blocking_call		// Inputs are generated from a blocking function call
		};
		/// Policies for making outputs
		enum class output_policy {
			none,				// Do not produce any output
			single,				// Produce a single output
			multiple			// Produce multiple outputs
		};

		namespace detail {
			// Implemenations of policies
			template <class Impl, class CrtpBaseType, process_policy pp>
			struct process_policy_base;
			template <class Impl, class CrtpBaseType, process_policy pp, input_policy ip>
			struct input_policy_base;
			template <class Impl, class CrtpBaseType, process_policy pp, output_policy op>
			struct output_policy_base;

			// Hiding details from the aboves
			template <class Impl, class CrtpBaseType, process_policy pp>
			class process_policy_class;
			template <class Impl, class CrtpBaseType, process_policy pp, input_policy ip>
			class input_policy_class;
			template <class Impl, class CrtpBaseType, process_policy pp, output_policy op>
			class output_policy_class;
		}

	#pragma region stage class definition
		/// CRTP base class for pipeline stages
		template <class Impl,
			input_policy ip = input_policy::passed_from,
			output_policy op = output_policy::single,
			process_policy pp = process_policy::single,
			bool check_input_flag_ = false,		// If check_input_flag = true, the return type of input() and feed() methods are bool,
												// which are otherwise void. When check_input_flag is on, process() and notify() methods 
												// are called only when input() didn't returned false.
			bool check_process_flag_ = false	// If check_process_flag = true, the return type of process() method is bool,
												// which is otherwise void. When check_process_flag is on, notify() method is called 
												// only when process() didn't returned false.
		>
		class stage : 
			public detail::process_policy_class<Impl, stage<Impl, ip, op, pp, check_input_flag_, check_process_flag_>, pp>,
			public detail::input_policy_class<Impl, stage<Impl, ip, op, pp, check_input_flag_, check_process_flag_>, pp, ip>,
			public detail::output_policy_class<Impl, stage<Impl, ip, op, pp, check_input_flag_, check_process_flag_>, pp, op>
		{
		protected:
			// This is a CRTP base class
			stage() = default;
			~stage() = default;

		public:
			using crtp_base_type = stage;
			stage& get_crtp_base() noexcept { return *this; }
			stage const& get_crtp_base() const noexcept { return *this; }

			template <typename...>
			friend class bundle;
			template <class, class, process_policy>
			friend struct detail::process_policy_base;
			template <class, class, process_policy, input_policy>
			friend struct detail::input_policy_base;
			template <class, class, process_policy, output_policy>
			friend struct detail::output_policy_base;

			static constexpr process_policy process_policy = pp;
			static constexpr input_policy input_policy = ip;
			static constexpr output_policy output_policy = op;

			static constexpr bool check_input_flag = check_input_flag_;
			static constexpr bool check_process_flag = check_process_flag_;

			// Default implementations of before/after_start/stop/work, begin/end_cycle() : do nothing
			template <typename SourcePtr, typename SinkPtr>
			void before_start(SourcePtr, SinkPtr) {}
			template <typename SourcePtr, typename SinkPtr>
			void after_start(SourcePtr, SinkPtr) {}
			template <typename... T>
			void before_work(T&&...) {}
			template <typename... T>
			void begin_cycle(T&&...) {}
			template <typename... T>
			void end_cycle(T&&...) {}
			template <typename... T>
			void after_work(T&&...) {}
			void before_stop() {}
			void after_stop() {}

		private:
			auto& get_process_policy() noexcept {
				return static_cast<detail::process_policy_base<Impl, stage, pp>&>(*this);
			}
			auto& get_process_policy() const noexcept {
				return static_cast<detail::process_policy_base<Impl, stage, pp> const&>(*this);
			}
			auto& get_input_policy() noexcept {
				return static_cast<detail::input_policy_base<Impl, stage, pp, ip>&>(*this);
			}
			auto& get_input_policy() const noexcept {
				return static_cast<detail::input_policy_base<Impl, stage, pp, ip> const&>(*this);
			}
			auto& get_output_policy() noexcept {
				return static_cast<detail::output_policy_base<Impl, stage, pp, op>&>(*this);
			}
			auto& get_output_policy() const noexcept {
				return static_cast<detail::output_policy_base<Impl, stage, pp, op> const&>(*this);
			}

			template <class dummy = std::integral_constant<bool, check_input_flag>, class = std::enable_if_t<dummy::value>>
			static constexpr bool default_return_for_input() noexcept { return true; }
			template <class dummy = std::integral_constant<bool, !check_input_flag>, class = std::enable_if_t<dummy::value>>
			static void default_return_for_input() noexcept {}

			template <class dummy = std::integral_constant<bool, check_process_flag>, class = std::enable_if_t<dummy::value>>
			static constexpr bool default_return_for_process() noexcept { return true; }
			template <class dummy = std::integral_constant<bool, !check_process_flag>, class = std::enable_if_t<dummy::value>>
			static void default_return_for_process() noexcept {}
		};
	#pragma endregion

		// ~~_policy_base provides fully open access to all members
		// ~~_policy_class hides all members of ~~_policy_base except those necessarily accessible outside stage
		// This macro provides a convenient way to do that
	#define BEGIN_PIPELINE_POLICY_CLASS(deco, ...)	template <class Impl, class CrtpBaseType> \
		class deco##_policy_class<Impl, CrtpBaseType, __VA_ARGS__> : private deco##_policy_base<Impl, CrtpBaseType, __VA_ARGS__> { \
			using policy_base = deco##_policy_base<Impl, CrtpBaseType, __VA_ARGS__>; \
		public: \
			template <class, input_policy, output_policy, process_policy, bool, bool> \
			friend class stage;
	#define END_PIPELINE_POLICY_CLASS				}

		namespace detail {
		#pragma region For the case of single worker
			template <class Impl, class CrtpBaseType>
			struct process_policy_base<Impl, CrtpBaseType, process_policy::single> {
				std::future<void>						m_worker_thread;
				std::atomic<bool>						m_terminate = { false };

				// Prepare start
				template <class SourcePtr, class SinkPtr>
				void before_start(SourcePtr source_ptr, SinkPtr sink_ptr) {
					static_cast<Impl*>(this)->before_start(source_ptr, sink_ptr);
				}

				// Launch a worker thread
				template <class SourceBinding, class SinkBinding>
				void start(SourceBinding&& source, SinkBinding&& sink) {
					m_terminate.store(false, std::memory_order_relaxed);

					bool success_flag = false;
					std::mutex mutex;
					std::condition_variable cv;
					std::unique_lock<std::mutex> lg{ mutex };

					m_worker_thread = std::async(std::launch::async,
						&process_policy_base::worker_proc<SourceBinding, SinkBinding>,
						this, std::ref(success_flag), std::ref(mutex), std::ref(cv),
						jkl::tmp::ref_if_lvalue(std::forward<SourceBinding>(source)),
						jkl::tmp::ref_if_lvalue(std::forward<SinkBinding>(sink)));

					// Wait for before_work() function to return
					cv.wait(lg);
					// If before_work() hasn't been successfully returned due to an exception, call m_worker_thread.get()
					// This will effectively rethrow the exception
					if( !success_flag )
						m_worker_thread.get();
				}

				// Postprocessing
				template <class SourcePtr, class SinkPtr>
				void after_start(SourcePtr source_ptr, SinkPtr sink_ptr) {
					static_cast<Impl*>(this)->after_start(source_ptr, sink_ptr);
				}

				// Prepare stop
				void before_stop() {
					static_cast<Impl*>(this)->before_stop();
				}

				// Stop the worker thread
				void stop() {
					static_cast<Impl*>(this)->get_crtp_base().notify([this]() {
						m_terminate.store(true, std::memory_order_relaxed);
					});

					if( m_worker_thread.valid() )
						m_worker_thread.get();
				}

				// Postprocessing
				void after_stop() {
					static_cast<Impl*>(this)->after_stop();
				}

				// The routine that will be repeatedly called inside the worker thread
				template <class SourceBinding, class SinkBinding, bool check_input>
				struct inner_cycle;

				template <class SourceBinding, class SinkBinding>
				struct inner_cycle<SourceBinding, SinkBinding, false> {
					FORCEINLINE static void do_work(SourceBinding& source, SinkBinding& sink, process_policy_base* p) {
						// Deal with inputs
						static_cast<Impl*>(p)->get_crtp_base().get_input_policy().do_input(source);

						if( p->m_terminate.load(std::memory_order_relaxed) )
							return;

						// Deal with outputs
						static_cast<Impl*>(p)->get_crtp_base().get_output_policy().do_output(sink);
					}
				};

				template <class SourceBinding, class SinkBinding>
				struct inner_cycle<SourceBinding, SinkBinding, true> {
					FORCEINLINE static void do_work(SourceBinding& source, SinkBinding& sink, process_policy_base* p) {
						// Deal with inputs
						bool result = static_cast<Impl*>(p)->get_crtp_base().get_input_policy().do_input(source);

						if( p->m_terminate.load(std::memory_order_relaxed) )
							return;

						// Deal with outputs
						if( result )
							static_cast<Impl*>(p)->get_crtp_base().get_output_policy().do_output(sink);
					}
				};

				// Thread routine
				template <class SourceBinding, class SinkBinding>
				void worker_proc(bool& success_flag, std::mutex& mutex, std::condition_variable& cv,
					SourceBinding source, SinkBinding sink) {

					auto source_ptr = source.get_source_ptr();
					using source_ptr_type = decltype(source_ptr);
					auto sink_ptr = sink.get_sink_ptr();
					using sink_ptr_type = decltype(sink_ptr);

					struct raii {
						// Pointer to the stage implementation
						Impl*				p;
						// Pointer to the source & sink stages connected to the current stage
						source_ptr_type		source_ptr;
						sink_ptr_type		sink_ptr;
						// Error indicator passed into after_work()
						// The worker thread is not properly terminated if this flag is set when after_work() is called
						bool				working;
						
						raii(Impl* p, source_ptr_type source_ptr, sink_ptr_type sink_ptr,
							bool& success_flag, std::mutex& mutex, std::condition_variable& cv) :
							p(p), source_ptr(source_ptr), sink_ptr(sink_ptr), working(true) {
							try {
								std::lock_guard<std::mutex> lg{ mutex };
								p->before_work(source_ptr, sink_ptr);
								success_flag = true;
							} catch( ... ) {
								cv.notify_one();
								throw;
							}
							cv.notify_one();
						}

						~raii() {
							p->after_work(source_ptr, sink_ptr, working);
						}
					} raii_obj{ static_cast<Impl*>(this), source_ptr, sink_ptr, success_flag, mutex, cv };

					while( !m_terminate.load(std::memory_order_relaxed) ) {
						static_cast<Impl*>(this)->begin_cycle();

						inner_cycle<SourceBinding, SinkBinding, CrtpBaseType::check_input_flag>::do_work(source, sink, this);
						if( m_terminate.load(std::memory_order_relaxed) )
							break;

						static_cast<Impl*>(this)->end_cycle();
					}

					raii_obj.working = false;
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(process, process_policy::single)
			END_PIPELINE_POLICY_CLASS;

		#pragma region input policies
			// none input policy for single worker
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::single, input_policy::none> {
				// The method called from the thread routine
				template <class SourceBinding>
				auto do_input(SourceBinding&) noexcept { return CrtpBaseType::default_return_for_input(); }

				// No need to notify
				template <class Functor, class... Args>
				void notify(Functor&& functor, Args&&... args) const {
					functor(std::forward<Args>(args)...);
				}
				void notify() const {}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::single, input_policy::none)
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// direct input policy for single worker
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::single, input_policy::direct> {
				mutable std::mutex						m_input_mutex;
				mutable std::condition_variable			m_notifier;
				mutable bool							m_notified = false;

				// The method called from the thread routine
				template <class SourceBinding>
				auto do_input(SourceBinding& source) {
					// Wait for notification
					std::unique_lock<std::mutex> lg{ m_input_mutex };
					if( !m_notified )
						m_notifier.wait(lg);
					m_notified = false;

					// If there are inputs to read, read them
					return source.read();
				}
												
				// Feed an input: calls input() and notifies
				template <typename... Args>
				auto feed(Args&&... args) {
					::jkl::tmp::result_holder<decltype(static_cast<Impl*>(this)
						->input(std::forward<Args>(args)...))> result;
					{
						std::lock_guard<std::mutex> lg{ m_input_mutex };
						result = ::jkl::tmp::hold_result([this](auto&&... args) {
							std::lock_guard<std::mutex> lg{ static_cast<Impl*>(this)->get_mutex() };
							return static_cast<Impl*>(this)->input(std::forward<decltype(args)>(args)...);
						}, std::forward<Args>(args)...);
						m_notified = true;
					}
					m_notifier.notify_one();
					return result.result();
				}
				
				// Wake up the worker after doing the job assigned
				template <class Functor, class... Args>
				void notify(Functor&& functor, Args&&... args) const {
					{
						std::lock_guard<std::mutex> lg{ m_input_mutex };
						functor(std::forward<Args>(args)...);
						m_notified = true;
					}
					m_notifier.notify_one();
				}
				void notify() const {
					notify([]() {});
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::single, input_policy::direct)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// passed_from input policy for single worker
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::single, input_policy::passed_from>
				: input_policy_base<Impl, CrtpBaseType, process_policy::single, input_policy::direct> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::single, input_policy::passed_from)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// blocking_call input policy for single worker
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::single, input_policy::blocking_call> {
				// The method called from the thread routine
				template <class SourceBinding>
				auto do_input(SourceBinding&) {
					return static_cast<Impl*>(this)->wait_for_input();
				}

				// Not possible to notify
				template <class Functor, class... Args>
				void notify(Functor&& functor, Args&&... args) const {
					functor(std::forward<Args>(args)...);
				}
				void notify() const {}

				// Implementation class must provide wait_for_input()
				auto wait_for_input() noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having input_policy::blocking_call must provide the "
						"implementation of wait_for_input() method!");
					return CrtpBaseType::default_return_for_input();
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::single, input_policy::blocking_call)
				using policy_base::wait_for_input;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma region output policies
			// none output policy for single worker
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::none> {
				mutable std::mutex						m_output_mutex;

				// The method called from the thread routine
				template <class SinkBinding>
				void do_output(SinkBinding& sink) {
					::jkl::tmp::static_if_else<CrtpBaseType::check_process_flag>(
						[&sink](auto* ptr) {
						bool result = false;
						{
							std::lock_guard<std::mutex> lg{ ptr->get_mutex() };
							result = ptr->process();
						}
						if( result )
							sink.notify();
					}, [&sink](auto* ptr) {
						{
							std::lock_guard<std::mutex> lg{ ptr->get_mutex() };
							ptr->process();
						}
						sink.notify();
					}, static_cast<Impl*>(this));
				}

				auto& get_mutex() const noexcept { return m_output_mutex; }

				// Default implementations of process() : do nothing
				auto process() noexcept { return CrtpBaseType::default_return_for_process(); }
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::single, output_policy::none)
				using policy_base::get_mutex;
				using policy_base::process;
			END_PIPELINE_POLICY_CLASS;


			// single output policy for single worker
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::single>
				: output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::none> {
				// Implementation class must provide output()
				constexpr auto output() const noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having output_policy::single must provide the "
						"implementation of output() method!");
					return std::make_tuple();
				}

				// Get the output
				// - If the return value of output() is not a tuple, then wrap it into an instance of auto_locked_data.
				//   - If the return value is a temporary value, then move it into auto_locked_data.
				//   - If the return value is an lvalue reference, then keep that reference inside auto_locked_data.
				// - If the return value of output() is a tuple, then wrap it into an instance of auto_locked_tuple.
				//   - If the return value is a temporary tuple, then each element 
				//     - of lvalue reference type is kept inside auto_locked_tuple.
				//     - of temporary value type is moved inside auto_locked_tuple.
				//   - If the return value is an lvalue reference to a tuple, then lvalue references to all of 
				//     the elements are kept inside auto_locked_tuple
				// The proxy instance of auto_locked_data/tuple is created as the return value.
				// Substituting this proxy instance into a variable 
				//   - copies data held by variables refered from the elements of lvalue reference types, and
				//   - moves data held by the elements of temporary value types
				// while locking the m_output_mutex. This mutex is unlocked instantly after the destruction of the proxy.

				// Helper class for get() method
				// Depending on the situation, provides the appropriate functor making the return value
				template <class Output, bool = ::jkl::tmp::is_tuple<std::decay_t<Output>>::value>
				struct get_helper {
					static auto get(output_policy_base const* p) {
						return[p](auto&&... outputs) -> auto {
							return ::jkl::util::make_auto_locked_tuple(p->m_output_mutex,
								std::forward<decltype(outputs)>(outputs)...);
						};
					}
				};

				template <class Output>
				struct get_helper<Output, false> {
					static auto get(output_policy_base const* p) {
						return[p](auto&& output) -> auto {
							return ::jkl::util::make_auto_locked_data(p->m_output_mutex,
								std::forward<decltype(output)>(output));
						};
					}
				};

				FORCEINLINE auto get() const {
					return ::jkl::tmp::unpack_and_apply(
						get_helper<decltype(static_cast<Impl const*>(this)->output())>::get(this),
						static_cast<Impl const*>(this)->output());
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::single, output_policy::single)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;


			// multiple output policy for single worker
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::multiple>
				: output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::single>
			{
				using parent_type = output_policy_base<Impl, CrtpBaseType, process_policy::single, output_policy::single>;

				// Implementations class must provide number_of_outputs()
				constexpr std::size_t number_of_outputs() const noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having both "
						"process_policy::none and output_policy::multiple "
						"must provide the implementation of number_of_outputs() method!");
					return 0;
				}

				// Implementation class must provide output()
				constexpr auto output(std::size_t) const noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having output_policy::multiple must provide the "
						"implementation of output() method!");
					return std::make_tuple();
				}
				
				FORCEINLINE auto get(std::size_t i) const {
					return ::jkl::tmp::unpack_and_apply(
						parent_type::template get_helper<decltype(static_cast<Impl const*>(this)->output(i))>::get(this),
						static_cast<Impl const*>(this)->output(i));
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::single, output_policy::multiple)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::number_of_outputs;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma endregion

		#pragma region For the case of multiple workers
			template <class Impl, class CrtpBaseType>
			struct process_policy_base<Impl, CrtpBaseType, process_policy::multiple> {
				std::unique_ptr<std::future<void>[]>	m_worker_threads;
				std::size_t								m_number_of_workers = 0;
				std::atomic<bool>						m_terminate = { false };
				
				void terminate_threads(std::size_t up_to) {
					m_terminate.store(true, std::memory_order_relaxed);

					for( std::size_t i = 0; i < up_to; i++ ) {
						static_cast<Impl*>(this)->get_crtp_base().notify(i);
						if( m_worker_threads[i].valid() )
							m_worker_threads[i].get();
					}
				}

				void release_resources() noexcept {
					static_cast<Impl*>(this)->get_crtp_base().get_output_policy().finalize();
					static_cast<Impl*>(this)->get_crtp_base().get_input_policy().finalize();					
					m_worker_threads.release();
					m_number_of_workers = 0;
				}

				// Prepare start
				template <class SourcePtr, class SinkPtr>
				void before_start(std::size_t workers, SourcePtr source_ptr, SinkPtr sink_ptr) {
					m_number_of_workers = workers;
					m_worker_threads = std::make_unique<std::future<void>[]>(workers);

					try {
						static_cast<Impl*>(this)->get_crtp_base().get_input_policy().initialize(workers);
					} catch( ... ) {
						m_worker_threads.release();
						throw;
					}

					try {
						static_cast<Impl*>(this)->get_crtp_base().get_output_policy().initialize(workers);
					} catch( ... ) {
						m_worker_threads.release();
						static_cast<Impl*>(this)->get_crtp_base().get_input_policy().finalize();
						throw;
					}

					try {
						static_cast<Impl*>(this)->before_start(source_ptr, sink_ptr);
					} catch( ... ) {
						release_resources();
						throw;
					}
				}
				
				// Launch worker threads
				template <class SourceBinding, class SinkBinding>
				void start(SourceBinding&& source, SinkBinding&& sink) {
					m_terminate.store(false, std::memory_order_relaxed);

					for( std::size_t i = 0; i < m_number_of_workers; i++ ) {
						bool success_flag = false;
						std::mutex mutex;
						std::condition_variable cv;
						std::unique_lock<std::mutex> lg{ mutex };

						try {
							m_worker_threads[i] = std::async(std::launch::async,
								&process_policy_base::worker_proc<SourceBinding, SinkBinding>, this,
								i, std::ref(success_flag), std::ref(mutex), std::ref(cv),
								::jkl::tmp::ref_if_lvalue(std::forward<SourceBinding>(source)),
								::jkl::tmp::ref_if_lvalue(std::forward<SinkBinding>(sink)));

							// Wait for before_work() function to return
							cv.wait(lg);
							// If before_work() hasn't been successfully returned due to an exception, call m_worker_threads[i].get()
							// This will effectively rethrow the exception
							if( !success_flag )
								m_worker_threads[i].get();
						} catch( ... ) {
							terminate_threads(i);
							throw;
						}
					}
				}

				// Postprocessing
				template <class SourcePtr, class SinkPtr>
				void after_start(SourcePtr source_ptr, SinkPtr sink_ptr) {
					static_cast<Impl*>(this)->after_start(source_ptr, sink_ptr);
				}

				// Prepare stop
				void before_stop() {
					static_cast<Impl*>(this)->before_stop();
				}

				// Stop the worker threads
				void stop() {
					terminate_threads(m_number_of_workers);
				}

				// Postprocessing
				void after_stop() {
					static_cast<Impl*>(this)->after_stop();
					release_resources();
				}

				// The routine that will be repeatedly called inside the worker thread
				template <class SourceBinding, class SinkBinding, bool check_input>
				struct inner_cycle;

				template <class SourceBinding, class SinkBinding>
				struct inner_cycle<SourceBinding, SinkBinding, false> {
					FORCEINLINE static void do_work(std::size_t i, SourceBinding& source, SinkBinding& sink, process_policy_base* p) {
						// Deal with inputs
						static_cast<Impl*>(p)->get_crtp_base().get_input_policy().do_input(i, source);

						if( p->m_terminate.load(std::memory_order_relaxed) )
							return;

						// Deal with outputs
						static_cast<Impl*>(p)->get_crtp_base().get_output_policy().do_output(i, sink);
					}
				};

				template <class SourceBinding, class SinkBinding>
				struct inner_cycle<SourceBinding, SinkBinding, true> {
					FORCEINLINE static void do_work(std::size_t i, SourceBinding& source, SinkBinding& sink, process_policy_base* p) {
						// Deal with inputs
						bool result = static_cast<Impl*>(p)->get_crtp_base().get_input_policy().do_input(i, source);

						if( p->m_terminate.load(std::memory_order_relaxed) )
							return;

						// Deal with outputs
						if( result )
							static_cast<Impl*>(p)->get_crtp_base().get_output_policy().do_output(i, sink);
					}
				};

				// Thread routine
				template <class SourceBinding, class SinkBinding>
				void worker_proc(std::size_t i, bool& success_flag, std::mutex& mutex, std::condition_variable& cv,
					SourceBinding source, SinkBinding sink) {

					auto source_ptr = source.get_source_ptr();
					using source_ptr_type = decltype(source_ptr);
					auto sink_ptr = sink.get_sink_ptr();
					using sink_ptr_type = decltype(sink_ptr);

					struct raii {
						// Pointer to the stage implementation
						Impl*				p;
						// Worker thread index
						std::size_t			i;
						// Pointer to the source & sink stages connected to the current stage
						source_ptr_type		source_ptr;
						sink_ptr_type		sink_ptr;
						// Error indicator passed into after_work()
						// The worker thread is not properly terminated if this flag is set when after_work() is called
						bool				working;
						

						raii(Impl* p, std::size_t i, 
							source_ptr_type source_ptr, sink_ptr_type sink_ptr,
							bool& success_flag, std::mutex& mutex, std::condition_variable& cv) : 
							p(p), i(i), source_ptr(source_ptr), sink_ptr(sink_ptr), working(true) {
							try {
								std::lock_guard<std::mutex> lg{ mutex };
								p->before_work(i, source_ptr, sink_ptr);
								success_flag = true;
							} catch( ... ) {
								cv.notify_one();
								throw;
							}
							cv.notify_one();
						}

						~raii() {
							p->after_work(i, source_ptr, sink_ptr, working);
						}
					} raii_obj{ static_cast<Impl*>(this), i, source_ptr, sink_ptr, success_flag, mutex, cv };

					while( !m_terminate.load(std::memory_order_relaxed) ) {
						static_cast<Impl*>(this)->begin_cycle(i);

						inner_cycle<SourceBinding, SinkBinding, CrtpBaseType::check_input_flag>::do_work(i, source, sink, this);
						if( m_terminate.load(std::memory_order_relaxed) )
							break;

						static_cast<Impl*>(this)->end_cycle(i);
					}

					raii_obj.working = false;
				}

				std::size_t number_of_workers() noexcept { return m_number_of_workers; }
			};

			BEGIN_PIPELINE_POLICY_CLASS(process, process_policy::multiple)
				using policy_base::number_of_workers;
			END_PIPELINE_POLICY_CLASS;

		#pragma region input policies
			// none input policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::none> {
				void initialize(std::size_t) const {}
				void finalize() const noexcept {}

				// The method called from the thread routine
				template <class SourceBinding>
				bool do_input(std::size_t, SourceBinding&) noexcept { return CrtpBaseType::default_return_for_input(); }

				// No need to notify
				template <class Functor, class... Args>
				void notify(std::size_t i, Functor&& functor, Args&&... args) const {
					functor(i, std::forward<Args>(args)...);
				}
				void notify(std::size_t) const {}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::multiple, input_policy::none)
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// direct input policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::direct> {
				struct resource_t {
					std::mutex										input_mutex;
					std::condition_variable							notifier;
					bool											notified;
				};
				mutable std::unique_ptr<resource_t[]>				m_resources;

				void initialize(std::size_t workers) {
					m_resources = std::make_unique<resource_t[]>(workers);
				}

				void finalize() noexcept {
					m_resources.release();
				}

				// The method called from the thread routine
				template <class SourceBinding>
				auto do_input(std::size_t i, SourceBinding& source) {
					// Wait for notification
					std::unique_lock<std::mutex> lg{ m_resources[i].input_mutex };

					if( !m_resources[i].notified )
						m_resources[i].notifier.wait(lg);
					m_resources[i].notified = false;

					// If there are inputs to read, read them
					return source.read(i);
				}

				// Helper class for feed() method
				// Depending on the output policy, provide the appropriate method to do locked input
				template <output_policy, class = void>
				struct feed_helper {
					template <typename... Args>
					static auto locked_feed(Impl* p, std::size_t i, Args&&... args) {
						std::lock_guard<std::mutex> lg{ p->get_mutex() };
						return p->input(i, std::forward<Args>(args)...);
					}
				};
				template <class dummy>
				struct feed_helper<output_policy::multiple, dummy> {
					template <typename... Args>
					static auto locked_feed(Impl* p, std::size_t i, Args&&... args) {
						std::lock_guard<std::mutex> lg{ p->get_mutex(i) };
						return p->input(i, std::forward<Args>(args)...);
					}
				};

				// Feed an input: calls input() and notifies
				template <typename... Args>
				auto feed(std::size_t i, Args&&... args) {
					::jkl::tmp::result_holder<decltype(feed_helper<CrtpBaseType::output_policy>::locked_feed(
						static_cast<Impl*>(this), i, std::forward<Args>(args)...))> result;
					{
						std::lock_guard<std::mutex> lg{ m_resources[i].input_mutex };
						result = ::jkl::tmp::hold_result([this, i](auto&&... args) {
							return feed_helper<CrtpBaseType::output_policy>::locked_feed(
								static_cast<Impl*>(this), i, std::forward<decltype(args)>(args)...);
						}, std::forward<Args>(args)...);
						m_resources[i].notified = true;
					}
					m_resources[i].notifier.notify_one();
					return result.result();
				}

				// Wake up the worker after doing the job assigned
				template <class Functor, class... Args>
				void notify(std::size_t i, Functor&& functor, Args&&... args) const {
					{
						std::lock_guard<std::mutex> lg{ m_resources[i].input_mutex };
						functor(i, std::forward<Args>(args)...);
						m_resources[i].notified = true;
					}
					m_resources[i].notifier.notify_one();
				}
				void notify(std::size_t i) const {
					notify(i, [](std::size_t) {});
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::multiple, input_policy::direct)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// passed_from input policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::passed_from>
				: public input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::direct> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::multiple, input_policy::passed_from)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// blocking_call for multiple workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::blocking_call> {
				void initialize(std::size_t) const noexcept {}
				void finalize() const noexcept {}

				// The method called from the thread routine
				template <class SourceBinding>
				auto do_input(std::size_t i, SourceBinding&) {
					return static_cast<Impl*>(this)->wait_for_input(i);
				}

				// Not possible to notify
				template <class Functor, class... Args>
				void notify(std::size_t i, Functor&& functor, Args&&... args) const {
					functor(i, std::forward<Args>(args)...);
				}
				void notify(std::size_t) const {}

				// Implementation class must provide wait_for_input()
				auto wait_for_input(std::size_t) noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having input_policy::blocking_call must provide the "
						"implementation of wait_for_input() method!");
					return CrtpBaseType::default_return_for_input();
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::multiple, input_policy::blocking_call)
				using policy_base::notify;
				using policy_base::wait_for_input;
			END_PIPELINE_POLICY_CLASS;


			// none output policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::none> {
				mutable std::mutex					m_output_mutex;

				void initialize(std::size_t) const noexcept {}
				void finalize() const noexcept {}

				// The method called from the thread routine
				template <class SinkBinding>
				void do_output(std::size_t i, SinkBinding& sink) {
					::jkl::tmp::static_if_else<CrtpBaseType::check_process_flag>(
						[i, &sink](auto* ptr) {
						bool result = false;
						{
							std::lock_guard<std::mutex> lg{ ptr->get_mutex() };
							result = ptr->process(i);
						}
						if( result )
							sink.notify();
					}, [i, &sink](auto* ptr) {
						{
							std::lock_guard<std::mutex> lg{ ptr->get_mutex() };
							ptr->process(i);
						}
						sink.notify();
					}, static_cast<Impl*>(this));
				}

				auto& get_mutex() const noexcept { return m_output_mutex; }

				// Default implementations of process() : do nothing
				auto process(std::size_t) noexcept { return CrtpBaseType::default_return_for_process(); }
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::multiple, output_policy::none)
				using policy_base::get_mutex;
				using policy_base::process;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma region output policies
			// single output policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::single>
				: output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::none> {
				// Implementation class must provide output()
				constexpr auto output() const noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having output_policy::singke must provide the "
						"implementation of output() method!");
					return std::make_tuple();
				}

				// Get the output
				// - If the return value of output() is not a tuple, then wrap it into an instance of auto_locked_data.
				//   - If the return value is a temporary value, then move it into auto_locked_data.
				//   - If the return value is an lvalue reference, then keep that reference inside auto_locked_data.
				// - If the return value of output() is a tuple, then wrap it into an instance of auto_locked_tuple.
				//   - If the return value is a temporary tuple, then each element 
				//     - of lvalue reference type is kept inside auto_locked_tuple.
				//     - of temporary value type is moved inside auto_locked_tuple.
				//   - If the return value is an lvalue reference to a tuple, then lvalue references to all of 
				//     the elements are kept inside auto_locked_tuple
				// The proxy instance of auto_locked_data/tuple is created as the return value.
				// Substituting this proxy instance into a variable 
				//   - copies data held by variables refered from the elements of lvalue reference types, and
				//   - moves data held by the elements of temporary value types
				// while locking the m_output_mutex. This mutex is unlocked instantly after the destruction of the proxy.

				// Helper class for get() method
				// Depending on the situation, provides the appropriate functor making the return value
				template <class Output, bool = ::jkl::tmp::is_tuple<std::decay_t<Output>>::value>
				struct get_helper {
					static auto get(output_policy_base const* p) {
						return[p](auto&&... outputs) -> auto {
							return ::jkl::util::make_auto_locked_tuple(p->m_output_mutex,
								std::forward<decltype(outputs)>(outputs)...);
						};
					}
				};

				template <class Output>
				struct get_helper<Output, false> {
					static auto get(output_policy_base const* p) {
						return[p](auto&& output) -> auto {
							return ::jkl::util::make_auto_locked_data(p->m_output_mutex,
								std::forward<decltype(output)>(output));
						};
					}
				};

				FORCEINLINE auto get() const {
					return ::jkl::tmp::unpack_and_apply(
						get_helper<decltype(static_cast<Impl const*>(this)->output())>::get(this),
						static_cast<Impl const*>(this)->output());
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::multiple, output_policy::single)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;


			// multiple output policy for multiple workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::multiple> {
				mutable std::unique_ptr<std::mutex[]>		m_output_mutices;

				void initialize(std::size_t workers) {
					m_output_mutices = std::make_unique<std::mutex[]>(workers);
				}

				void finalize() noexcept {
					m_output_mutices.release();
				}

				// The method called from the thread routine
				template <class SinkBinding, class = std::enable_if_t<CrtpBaseType::check_process_flag>>
				void do_output(std::size_t i, SinkBinding& sink) {
					bool result = false;
					{
						std::lock_guard<std::mutex> lg{ m_output_mutices[i] };
						result = static_cast<Impl*>(this)->process(i);
					}
					if( result )
						sink.notify(i);
				}
				template <class SinkBinding, class = std::enable_if_t<!CrtpBaseType::check_process_flag>, class = void>
				void do_output(std::size_t i, SinkBinding& sink) {
					{
						std::lock_guard<std::mutex> lg{ m_output_mutices[i] };
						static_cast<Impl*>(this)->process(i);
					}
					sink.notify(i);
				}

				auto& get_mutex(std::size_t i) const noexcept { return m_output_mutices[i]; }

				// Default implementations of process() : do nothing
				auto process(std::size_t) noexcept { return CrtpBaseType::default_return_for_process(); }

				// Implementation class must provide output()
				constexpr auto output(std::size_t) const noexcept {
					static_assert(::jkl::tmp::assert_helper<Impl>::value,
						"Any implementation class having output_policy::multiple must provide the "
						"implementation of output() method!");
					return std::make_tuple();
				}

				// Get the output
				// - If the return value of output() is not a tuple, then wrap it into an instance of auto_locked_data.
				//   - If the return value is a temporary value, then move it into auto_locked_data.
				//   - If the return value is an lvalue reference, then keep that reference inside auto_locked_data.
				// - If the return value of output() is a tuple, then wrap it into an instance of auto_locked_tuple.
				//   - If the return value is a temporary tuple, then each element 
				//     - of lvalue reference type is kept inside auto_locked_tuple.
				//     - of temporary value type is moved inside auto_locked_tuple.
				//   - If the return value is an lvalue reference to a tuple, then lvalue references to all of 
				//     the elements are kept inside auto_locked_tuple
				// The proxy instance of auto_locked_data/tuple is created as the return value.
				// Substituting this proxy instance into a variable 
				//   - copies data held by variables refered from the elements of lvalue reference types, and
				//   - moves data held by the elements of temporary value types
				// while locking the m_output_mutex. This mutex is unlocked instantly after the destruction of the proxy.

				// Helper class for get() method
				// Depending on the situation, provides the appropriate functor making the return value
				template <class Output, bool = ::jkl::tmp::is_tuple<std::decay_t<Output>>::value>
				struct get_helper {
					static auto get(std::size_t i, output_policy_base const* p) {
						return[i, p](auto&&... outputs) -> auto {
							return ::jkl::util::make_auto_locked_tuple(p->m_output_mutices[i],
								std::forward<decltype(outputs)>(outputs)...);
						};
					}
				};

				template <class Output>
				struct get_helper<Output, false> {
					static auto get(std::size_t i, output_policy_base const* p) {
						return[i, p](auto&& output) -> auto {
							return ::jkl::util::make_auto_locked_data(p->m_output_mutices[i],
								std::forward<decltype(output)>(output));
						};
					}
				};

				FORCEINLINE auto get(std::size_t i) const {
					return ::jkl::tmp::unpack_and_apply(
						get_helper<decltype(static_cast<Impl const*>(this)->output(i))>::get(i, this),
						static_cast<Impl const*>(this)->output(i));
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::multiple, output_policy::multiple)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma endregion

		#pragma region For the case of matched workers
			template <class Impl, class CrtpBaseType>
			struct process_policy_base<Impl, CrtpBaseType, process_policy::matched>
				: public process_policy_base<Impl, CrtpBaseType, process_policy::multiple> {
				using base_policy = process_policy_base<Impl, CrtpBaseType, process_policy::multiple>;

				// Launch worker threads
				template <class SourcePtr, class SinkPtr, output_policy op>
				struct before_start_helper {
					static void before_start(process_policy_base* p,
						SourcePtr source_ptr, SinkPtr sink_ptr) {
						p->base_policy::before_start(source_ptr->number_of_outputs(), source_ptr, sink_ptr);
					}
				};
				
				template <class SourcePtr, class SinkPtr>
				struct before_start_helper<SourcePtr, SinkPtr, output_policy::single> {
					static void before_start(process_policy_base* p,
						SourcePtr source_ptr, SinkPtr sink_ptr) {
						p->base_policy::before_start(1, source_ptr, sink_ptr);
					}
				};

				template <class SourcePtr, class SinkPtr>
				void before_start(SourcePtr source_ptr, SinkPtr sink_ptr) {
					before_start_helper<SourcePtr, SinkPtr, std::remove_pointer_t<SourcePtr>::output_policy>::
						before_start(this, source_ptr, sink_ptr);
				}

				std::size_t number_of_workers() noexcept { 
					return process_policy_base<Impl, CrtpBaseType, process_policy::multiple>::m_number_of_workers;
				}
			};

			BEGIN_PIPELINE_POLICY_CLASS(process, process_policy::matched)
				using policy_base::number_of_workers;
			END_PIPELINE_POLICY_CLASS;

		#pragma region input policies
			// none input policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::matched, input_policy::none>
				: public input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::none> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::matched, input_policy::none)
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// direct input policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::matched, input_policy::direct>
				: public input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::direct> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::matched, input_policy::direct)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;
			

			// passed_from input policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::matched, input_policy::passed_from>
				: public input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::passed_from> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::matched, input_policy::passed_from)
				using policy_base::feed;
				using policy_base::notify;
			END_PIPELINE_POLICY_CLASS;


			// blocking_call for matched workers
			template <class Impl, class CrtpBaseType>
			struct input_policy_base<Impl, CrtpBaseType, process_policy::matched, input_policy::blocking_call>
				: public input_policy_base<Impl, CrtpBaseType, process_policy::multiple, input_policy::blocking_call> {};

			BEGIN_PIPELINE_POLICY_CLASS(input, process_policy::matched, input_policy::blocking_call)
				using policy_base::notify;
				using policy_base::wait_for_inputs;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma region output policies
			// none output policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::matched, output_policy::none>
				: public output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::none> {};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::matched, output_policy::none)
				using policy_base::process;
			END_PIPELINE_POLICY_CLASS;


			// single output policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::matched, output_policy::single>
				: public output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::single> {};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::matched, output_policy::single)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;


			// multiple output policy for matched workers
			template <class Impl, class CrtpBaseType>
			struct output_policy_base<Impl, CrtpBaseType, process_policy::matched, output_policy::multiple>
				: public output_policy_base<Impl, CrtpBaseType, process_policy::multiple, output_policy::multiple> {};

			BEGIN_PIPELINE_POLICY_CLASS(output, process_policy::matched, output_policy::multiple)
				using policy_base::get_mutex;
				using policy_base::process;
				using policy_base::output;
				using policy_base::get;
			END_PIPELINE_POLICY_CLASS;
		#pragma endregion

		#pragma endregion

			// Binding from a stage to the rest of the bundle
			template <class Source, class Sink>
			struct binding {
				struct base_type {
					using source_type	= Source;
					using sink_type		= Sink;
					Source*				m_source_ptr;
					Sink*				m_sink_ptr;
					base_type(Source* source_ptr, Sink* sink_ptr) :
						m_source_ptr(source_ptr), m_sink_ptr(sink_ptr) {}
					Source* get_source_ptr() { return m_source_ptr; }
					Sink* get_sink_ptr() { return m_sink_ptr; }
				};

				// Unless input_policy of Sink is passed_from, the interface should be trivial
				struct default_type : base_type {
					using base_type::base_type;
					void initialize() {}
					void initialize(std::size_t) {}
					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read() { return true; }
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read() {}
					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read(std::size_t) { return true; }
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read(std::size_t) {}
					void notify() const {}
					void notify(std::size_t) const {}
					void finalize() noexcept {}
				};

				// Convenient utility function
				template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
				static constexpr bool default_return() noexcept { return false; }
				template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>, class = void>
				static void default_return() noexcept {}

				// When source: single output and sink: single worker
				struct single_single : base_type {
					mutable bool m_has_new_data = false;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {}

					auto read() {
						if( m_has_new_data ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							auto result = ::jkl::tmp::hold_result([this] {
								return ::jkl::tmp::unpack_and_apply([this](auto&&... args) {
									return m_sink_ptr->input(std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output());
							});
							m_has_new_data = false;
							return result.result();
						}
						return default_return();
					}

					void notify() const {
						m_sink_ptr->get_crtp_base().notify([this]() { m_has_new_data = true; });
					}

					void notify(std::size_t) const {
						notify();
					}

					void finalize() noexcept {}
				};

				// When source: single worker & multiple outputs and sink: single worker
				struct single_multiple_single : base_type {
					mutable bool m_has_new_data = false;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {}

					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read() {
						if( m_has_new_data ) {
							bool result = false;
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							for( std::size_t i = 0; i < m_source_ptr->number_of_outputs(); i++ ) {
								result |= ::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
							}
							m_has_new_data = false;
							return result;
						}
						return false;
					}
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read() {
						if( m_has_new_data ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							for( std::size_t i = 0; i < m_source_ptr->number_of_outputs(); i++ ) {
								::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
							}
							m_has_new_data = false;
						}
					}

					void notify() const {
						m_sink_ptr->get_crtp_base().notify([this]() { m_has_new_data = true; });
					}

					void finalize() noexcept {}
				};

				// When source: multiple workers & multiple outputs and sink: single worker
				struct multiple_multiple_single : base_type {
					std::size_t						m_number_of_outputs = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {
						m_number_of_outputs = m_source_ptr->number_of_workers();
						m_has_new_data = std::make_unique<std::atomic<bool>[]>(m_number_of_outputs);
						for( std::size_t i = 0; i < m_number_of_outputs; i++ )
							m_has_new_data[i] = false;
					}

					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read() {
						bool result = false;
						for( std::size_t i = 0; i < m_number_of_outputs; i++ ) {							
							if( m_has_new_data[i] ) {
								std::lock_guard<std::mutex> lg{ m_source_ptr->get_mutex(i) };
								result |= ::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
								m_has_new_data[i] = false;
							}
						}
						return result;
					}
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read() {
						for( std::size_t i = 0; i < m_number_of_outputs; i++ ) {
							if( m_has_new_data[i] ) {
								std::lock_guard<std::mutex> lg{ m_source_ptr->get_mutex(i) };
								::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
								m_has_new_data[i] = false;
							}
						}
					}

					void notify(std::size_t i) const {
						m_sink_ptr->get_crtp_base().notify([=]() { m_has_new_data[i] = true; });
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};

				// When source: single output and sink: multiple workers
				struct single_multiple : base_type {
					std::size_t						m_number_of_workers = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize(std::size_t number_of_workers) {
						m_number_of_workers = number_of_workers;
						m_has_new_data = std::make_unique<bool[]>(number_of_workers);
						for( std::size_t i = 0; i < number_of_workers; i++ )
							m_has_new_data[i] = false;
					}

					auto read(std::size_t i) {
						if( m_has_new_data[i] ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							auto result = ::jkl::tmp::hold_result([this, i] {
								return ::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output());
							});
							m_has_new_data[i] = false;
							return result.result();
						}
						return default_return();
					}

					void notify() const {
						for( std::size_t i = 0; i < m_number_of_workers; i++ )
							m_sink_ptr->get_crtp_base().notify(i, [=]() { m_has_new_data[i] = true; });
					}

					void notify(std::size_t) const {
						notify();
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};

				// When source: single worker & multiple outputs and sink: multiple workers
				struct single_multiple_multiple : base_type {
					std::size_t						m_number_of_workers = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize(std::size_t number_of_workers) {
						m_number_of_workers = number_of_workers;
						m_has_new_data = std::make_unique<std::atomic<bool>[]>(m_number_of_workers);
						for( std::size_t i = 0; i < m_number_of_workers; i++ )
							m_has_new_data[i] = false;
					}

					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read(std::size_t worker_index) {
						if( m_has_new_data[worker_index] ) {
							bool result = false;
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							for( std::size_t output_index = 0; output_index < m_source_ptr->number_of_outputs(); output_index++ ) {
								result |= ::jkl::tmp::unpack_and_apply([this, worker_index, output_index](auto&&... args) {
									return m_sink_ptr->input(worker_index, output_index, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(output_index));
							}
							m_has_new_data[worker_index] = false;
							return result;
						}
						return false;
					}
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read(std::size_t worker_index) {
						if( m_has_new_data[worker_index] ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							for( std::size_t output_index = 0; output_index < m_source_ptr->number_of_outputs(); output_index++ ) {
								::jkl::tmp::unpack_and_apply([this, worker_index, output_index](auto&&... args) {
									return m_sink_ptr->input(worker_index, output_index, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(output_index));
							}
							m_has_new_data[worker_index] = false;
						}
					}

					void notify() const {
						for( std::size_t worker_index = 0; worker_index < m_number_of_workers; worker_index++ )
							m_sink_ptr->get_crtp_base().notify(worker_index, [=]() { m_has_new_data[worker_index] = true; });
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};

				// When source: multiple workers & multiple outputs and sink: multiple workers
				struct multiple_multiple_multiple : base_type {
					std::size_t						m_number_of_workers = 0;
					std::size_t						m_number_of_outputs = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize(std::size_t number_of_workers) {
						m_number_of_outputs = m_source_ptr->number_of_workers();
						m_number_of_workers = number_of_workers;
						m_has_new_data = std::make_unique<std::atomic<bool>[]>(number_of_workers * m_number_of_outputs);
						for( std::size_t i = 0; i < number_of_workers * m_number_of_outputs; i++ )
							m_has_new_data[i] = false;
					}

					template <class = std::enable_if_t<Sink::crtp_base_type::check_input_flag>>
					bool read(std::size_t worker_index) {
						bool result = false;
						std::size_t index = worker_index;
						for( std::size_t output_index = 0; output_index < m_number_of_outputs; output_index++, index += m_number_of_workers ) {
							if( m_has_new_data[index] ) {
								std::lock_guard<std::mutex> lg{ m_source_ptr->get_mutex(output_index) };
								result |= ::jkl::tmp::unpack_and_apply([this, worker_index, output_index](auto&&... args) {
									return m_sink_ptr->input(worker_index, output_index, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(output_index));
								m_has_new_data[index] = false;
							}
						}
						return result;
					}
					template <class = std::enable_if_t<!Sink::crtp_base_type::check_input_flag>>
					void read(std::size_t worker_index) {
						std::size_t index = worker_index;
						for( std::size_t output_index = 0; output_index < m_number_of_outputs; output_index++, index += m_number_of_workers ) {
							if( m_has_new_data[index] ) {
								std::lock_guard<std::mutex> lg{ m_source_ptr->get_mutex(output_index) };
								::jkl::tmp::unpack_and_apply([this, worker_index, output_index](auto&&... args) {
									return m_sink_ptr->input(worker_index, output_index, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(output_index));
								m_has_new_data[index] = false;
							}
						}
					}

					void notify(std::size_t output_index) const {
						std::size_t base_index = output_index * m_number_of_workers;
						for( std::size_t worker_index = 0; worker_index < m_number_of_workers; worker_index++ ) {
							m_sink_ptr->get_crtp_base().notify(worker_index, [=]() {
								m_has_new_data[worker_index + base_index] = true;
							});
						}
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};

				// When source: single output and sink: matched workers
				struct single_matched : base_type {
					mutable bool m_has_new_data = false;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {}

					auto read(std::size_t) {
						if( m_has_new_data ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							auto result = ::jkl::tmp::hold_result([this] {
								return ::jkl::tmp::unpack_and_apply([this](auto&&... args) {
									return m_sink_ptr->input(0, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output());
							});
							m_has_new_data = false;
							return result.result();
						}
						return default_return();
					}

					void notify() const {
						m_sink_ptr->get_crtp_base().notify(0, [this]() { m_has_new_data = true; });
					}

					void notify(std::size_t) const {
						notify();
					}

					void finalize() noexcept {}
				};

				// When source: single worker & multiple outputs and sink: matched workers
				struct single_multiple_matched : base_type {
					std::size_t						m_number_of_workers = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {
						m_number_of_workers = m_source_ptr->number_of_outputs();
						m_has_new_data = std::make_unique<std::atomic<bool>[]>(m_number_of_workers);
						for( std::size_t i = 0; i < m_number_of_workers; i++ )
							m_has_new_data[i] = false;
					}

					auto read(std::size_t i) {
						if( m_has_new_data[i] ) {
							std::lock_guard<std::mutex> lg{ m_source_ptr->get_crtp_base().get_mutex() };
							auto result = ::jkl::tmp::hold_result([this, i] {
								return ::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
							});
							m_has_new_data[i] = false;
							return result.result();
						}
						return default_return();
					}

					void notify() const {
						for( std::size_t i = 0; i < m_number_of_workers; i++ ) {
							m_sink_ptr->get_crtp_base().notify(i, [=]() { m_has_new_data[i] = true; });
						}
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};

				// When source: multiple workers & multiple outputs and sink: matched workers
				struct multiple_multiple_matched : base_type {
					std::size_t						m_number_of_workers = 0;
					mutable std::unique_ptr<bool[]>	m_has_new_data;

					using base_type::base_type;
					using base_type::m_source_ptr;
					using base_type::m_sink_ptr;

					void initialize() {
						m_number_of_workers = m_source_ptr->number_of_workers();
						m_has_new_data = std::make_unique<std::atomic<bool>[]>(m_number_of_workers);
						for( std::size_t i = 0; i < m_number_of_workers; i++ )
							m_has_new_data[i] = false;
					}

					auto read(std::size_t i) {
						if( m_has_new_data[i] ) {
							std::lock_guard<std::mutex> lg(m_source_ptr->get_mutex(i));
							auto result = ::jkl::tmp::hold_result([this, i] {
								return ::jkl::tmp::unpack_and_apply([this, i](auto&&... args) {
									return m_sink_ptr->input(i, std::forward<decltype(args)>(args)...);
								}, m_source_ptr->output(i));
							});
							m_has_new_data[i] = false;
							return result.result();
						}
						return default_return();
					}

					void notify(std::size_t i) const {
						m_sink_ptr->get_crtp_base().notify(i, [=]() { m_has_new_data[i] = true; });
					}

					void finalize() noexcept {
						m_has_new_data.release();
					}
				};
				
				/// Select one of the interfaces defined above

				static constexpr process_policy source_pp = Source::process_policy;
				static constexpr output_policy source_op = Source::output_policy;
				static constexpr process_policy sink_pp = Sink::process_policy;
				static constexpr input_policy sink_ip = Sink::input_policy;

				// If the sink has single worker
				struct sink_pp_single {					
					// If source has single output
					struct source_op_single {
						using type = std::tuple<single_single, single_single>;
					};
					// If source has multiple output
					struct source_op_multiple {
						using type = std::tuple<single_multiple_single, multiple_multiple_single>;
					};
					// If the source has no output
					struct source_op_default {
						using type = std::tuple<default_type, default_type>;
					};
				};
				// If the sink has multiple workers
				struct sink_pp_multiple {					
					// If source has single output
					struct source_op_single {
						using type = std::tuple<single_multiple, single_multiple>;
					};
					// If source has multiple outputs
					struct source_op_multiple {
						using type = std::tuple<single_multiple_multiple, multiple_multiple_multiple>;
					};
					// If the source has no output
					struct source_op_default {
						using type = std::tuple<default_type, default_type>;
					};
				};
				// If the sink has matched worker
				struct sink_pp_matched {					
					// If source has single output
					struct source_op_single {
						using type = std::tuple<single_matched, single_matched>;
					};
					// If source has multiple outputs
					struct source_op_multiple {
						using type = std::tuple<single_multiple_matched, multiple_multiple_matched>;
					};
					// If the source has no output
					struct source_op_default {
						using type = std::tuple<default_type, default_type>;
					};
				};
				// Otherwise
				struct sink_pp_default {
					struct source_op_single {
						using type = std::tuple<default_type, default_type>;
					};
					struct source_op_multiple {
						using type = std::tuple<default_type, default_type>;
					};
					struct source_op_default {
						using type = std::tuple<default_type, default_type>;
					};
				};

				using select_sink_pp = std::conditional_t<
					sink_pp == process_policy::single,
					sink_pp_single,
					std::conditional_t<
						sink_pp == process_policy::multiple,
						sink_pp_multiple,
						std::conditional_t<
							sink_pp == process_policy::matched,
							sink_pp_matched,
							sink_pp_default
						>
					>
				>;
				
				using select_source_op = std::conditional_t<
					source_op == output_policy::single,
					typename select_sink_pp::source_op_single,
					std::conditional_t<
						source_op == output_policy::multiple,
						typename select_sink_pp::source_op_multiple,
						typename select_sink_pp::source_op_default
					>
				>;

				using type = std::conditional_t<
					sink_ip == input_policy::passed_from,
					std::conditional_t<
						source_pp == process_policy::single,
						std::tuple_element_t<0, typename select_source_op::type>,
						std::conditional_t<
							source_pp == process_policy::multiple || 
								source_pp == process_policy::matched,
							std::tuple_element_t<1, typename select_source_op::type>,
							default_type
						>
					>,
					default_type
				>;
			};
			template <class Source, class Sink>
			using stage_binding = typename binding<Source, Sink>::type;

			struct empty_stage {
				using crtp_base_type = empty_stage;
				static constexpr process_policy process_policy = process_policy::none;
				static constexpr input_policy input_policy = input_policy::none;
				static constexpr output_policy output_policy = output_policy::none;
			};
		}

		template <class First, class... Remainings>
		class bundle<First, Remainings...> : public bundle<Remainings...> {

			First					m_first_stage;
			using parent_type = bundle<Remainings...>;
			using sink_binding_type = typename detail::stage_binding<First, typename parent_type::first_stage_type>;
			sink_binding_type		m_sink_binding;

			using empty_source_binding = typename detail::stage_binding<detail::empty_stage, First>;

			// Unpack a tuple using std::index_sequence
			template <std::size_t... I, typename... FirstArgs, typename... Others>
			bundle(std::tuple<FirstArgs...> const& arg_pack, std::index_sequence<I...>, Others&&... others)
				: m_first_stage(std::get<I>(arg_pack)...), parent_type(std::forward<Others>(others)...),
				m_sink_binding(&m_first_stage, &static_cast<parent_type*>(this)->first_stage()) {}
			template <std::size_t... I, typename... FirstArgs, typename... Others>
			bundle(std::tuple<FirstArgs...>&& arg_pack, std::index_sequence<I...>, Others&&... others)
				: m_first_stage(std::get<I>(std::move(arg_pack))...), parent_type(std::forward<Others>(others)...),
				m_sink_binding(&m_first_stage, &static_cast<parent_type*>(this)->first_stage()) {}

		protected:
			template <typename SourceBinding, typename... SizeType,
				class = std::enable_if_t<First::process_policy != process_policy::multiple>>
			void before_start(SourceBinding& source_binding, SizeType... number_of_workers) {
				source_binding.initialize();
				try {
					m_first_stage.get_crtp_base().get_process_policy().before_start(
						source_binding.get_source_ptr(), m_sink_binding.get_sink_ptr());
				} catch( ... ) {
					source_binding.finalize();
					throw;
				}

				try {
					parent_type::before_start(m_sink_binding, number_of_workers...);
				} catch( ... ) {
					m_first_stage.get_crtp_base().get_process_policy().after_stop();
					source_binding.finalize();
					throw;
				}
			}

			template <typename SourceBinding, typename... SizeType,
				class = std::enable_if_t<First::process_policy == process_policy::multiple>>
			void before_start(SourceBinding& source_binding,
				std::size_t number_of_workers_first, SizeType... number_of_workers_remainings) {
				source_binding.initialize(number_of_workers_first);
				try {
					m_first_stage.get_crtp_base().get_process_policy().before_start(number_of_workers_first,
						source_binding.get_source_ptr(), m_sink_binding.get_sink_ptr());
				} catch( ... ) {
					source_binding.finalize();
					throw;
				}

				try {
					parent_type::before_start(m_sink_binding, number_of_workers_remainings...);
				} catch( ... ) {
					m_first_stage.get_crtp_base().get_process_policy().after_stop();
					source_binding.finalize();
					throw;
				}
			}

			template <typename SourceBinding>
			void start_stages(SourceBinding&& source_binding) {
				m_first_stage.get_crtp_base().get_process_policy().start(
					std::forward<SourceBinding>(source_binding), m_sink_binding);

				try {
					parent_type::start_stages(m_sink_binding);
				} catch( ... ) {
					m_first_stage.get_crtp_base().get_process_policy().stop();
					throw;
				}
			}

			template <typename SourcePtr>
			void after_start(SourcePtr source_ptr) {
				m_first_stage.get_crtp_base().get_process_policy().after_start(
					source_ptr, m_sink_binding.get_sink_ptr());

				try {
					parent_type::after_start(&m_first_stage);
				} catch( ... ) {
					m_first_stage.get_crtp_base().get_process_policy().before_stop();
					throw;
				}
			}

			template <typename SourceBinding, typename... SizeType>
			void start_impl(SourceBinding&& source_binding, SizeType... number_of_workers) {
				auto source_ptr = source_binding.get_source_ptr();
				before_start(source_binding, number_of_workers...);

				try {
					start_stages(std::forward<SourceBinding>(source_binding));
				} catch( ... ) {
					after_stop();
					throw;
				}

				try {
					after_start(source_ptr);
				} catch( ... ) {
					stop_stages();
					after_stop();
					throw;
				}
			}

			void before_stop() {
				parent_type::before_stop();
				m_first_stage.get_crtp_base().get_process_policy().before_stop();
			}

			void stop_stages() {
				parent_type::stop_stages();
				m_first_stage.get_crtp_base().get_process_policy().stop();
			}

			void after_stop() {
				parent_type::after_stop();
				m_sink_binding.finalize();
				m_first_stage.get_crtp_base().get_process_policy().after_stop();
			}

			void stop_impl() {
				before_stop();
				stop_stages();
				after_stop();
			}

			// Access a stage specified as the stage implementation type
			template <std::size_t index, bool is_const>
			struct stage_by_index_impl {
				static_assert(index < bundle::length, "Index out of range!");
				using this_type = std::conditional_t<is_const, bundle const, bundle>;
				using qualified_parent_type_ = std::conditional_t<is_const, parent_type const, parent_type>;
				static auto& get(this_type& p) noexcept {
					return static_cast<qualified_parent_type_&>(p).template stage<index - 1>();
				}
			};
			template <bool is_const>
			struct stage_by_index_impl<0, is_const> {
				using this_type = std::conditional_t<is_const, bundle const, bundle>;
				static auto& get(this_type& p) noexcept {
					return p.m_first_stage;
				}
			};
			template <class StageType>
			struct stage_by_type_impl {
				static constexpr std::size_t index = 
					std::conditional_t<
					std::is_same<StageType, First>::value, 
					std::integral_constant<std::size_t, 0>, 
					std::integral_constant<std::size_t, parent_type::template stage_by_type_impl<StageType>::index + 1>
					>::value;

				static_assert(parent_type::template stage_by_type_impl<StageType>::index == sizeof...(Remainings) ||
					parent_type::template stage_by_type_impl<StageType>::index == stage_by_type_impl<StageType>::index - 1,
					"There are multiple number of stages of the specified type in this bundle!");
			};

		public:
			static constexpr std::size_t length = sizeof...(Remainings) + 1;

			// Each stage should be emplaced from constructor arguments that are packed into tuples
			template <typename... FirstArgs, typename... Others>
			bundle(std::tuple<FirstArgs...> const& arg_pack, Others&&... others)
				: bundle(arg_pack, std::make_index_sequence<sizeof...(FirstArgs)>(),
				std::forward<Others>(others)...) {}
			template <typename... FirstArgs, typename... Others>
			bundle(std::tuple<FirstArgs...>&& arg_pack, Others&&... others)
				: bundle(std::move(arg_pack), std::make_index_sequence<sizeof...(FirstArgs)>(),
				std::forward<Others>(others)...) {}

			// If no argument is passed, default-construct all the stages
			bundle() : m_sink_binding(&m_first_stage, &static_cast<parent_type*>(this)->first_stage()) {};

			~bundle() {
				stop();
			}

			using first_stage_type = First;

			// Start the pipeline
			// Numbers of workers in the stages having multiple workers are specified as arguments.
			template <typename... SizeType>
			void start(SizeType... number_of_workers) {
				if( !is_working() ) {
					start_impl(empty_source_binding{ nullptr, &m_first_stage }, number_of_workers...);
				}
			}

			// Stop the pipeline
			void stop() {
				if( is_working() ) {
					stop_impl();
				}
			}

			// Is the pipeline working now?
			using parent_type::is_working;

			// Access a specified stage using a 0-based static index
			template <std::size_t index>
			auto& stage() noexcept {
				return stage_by_index_impl<index, false>::get(*this);
			}
			// Access a stage specified as the stage implementation type
			// Provide a compile error when there is no such type or there are multiple number of such types
			template <class StageType>
			auto& stage() noexcept {
				static_assert(stage_by_type_impl<StageType>::index < length,
					"There is no stage of the specified type in this bundle!");
				return stage<stage_by_type_impl<StageType>::index>();
			}
			// Access the first or the last stage
			auto& first_stage() noexcept { return m_first_stage; }
			auto& last_stage() noexcept { return stage<sizeof...(Remainings)>(); }

			// Const versions of the aboves
			template <std::size_t index>
			auto const& stage() const noexcept {				
				return stage_by_index_impl<index, true>::get(*this);
			}
			template <class StageType>
			auto const& stage() const noexcept {
				static_assert(stage_by_type_impl<StageType>::index < length,
					"There is no stage of the specified type in this bundle!");
				return stage<stage_by_type_impl<StageType>::index>();
			}
			auto const& first_stage() const noexcept { return m_first_stage; }
			auto const& last_stage() const noexcept { return stage<sizeof...(Remainings)>(); }

			// Feed an input (the first stage must not have blocking_call policy)
			template <typename... Args, class = std::enable_if_t<First::input_policy != input_policy::blocking_call>>
			auto feed(Args&&... args) {
				return m_first_stage.get_crtp_base().feed(std::forward<Args>(args)...);
			}

			// Get the output
			template <class = std::enable_if_t<
				std::tuple_element_t<length - 1, std::tuple<First, Remainings...>>::output_policy == output_policy::single>
			>
			auto get() const {
				return last_stage().get_crtp_base().get();
			}
			template <class = std::enable_if_t<
				std::tuple_element_t<length - 1, std::tuple<First, Remainings...>>::output_policy == output_policy::multiple>
			>
			auto get(std::size_t i) const {
				return last_stage().get_crtp_base().get(i);
			}

			// Notification
			template <typename... Args>
			void notify(Args&&... args) const {
				m_first_stage.get_crtp_base().notify(std::forward<Args>(args)...);
			}
		};

		template <>
		class bundle<> : detail::empty_stage {
			bool		is_working_ = false;

		protected:
			template <typename SourceBinding>
			void before_start(SourceBinding&) noexcept {}

			template <typename SourceBinding>
			void start_stages(SourceBinding&&) noexcept { is_working_ = true; }

			template <typename SourcePtr>
			void after_start(SourcePtr) noexcept {}

			void before_stop() noexcept {}

			void stop_stages() noexcept { is_working_ = false; }

			void after_stop() noexcept {}

			template <class StageType>
			struct stage_by_type_impl {
				static constexpr std::size_t index = 0;
			};

		public:
			static constexpr std::size_t length = 0;
			using first_stage_type = detail::empty_stage;

			auto& first_stage() noexcept { 
				return static_cast<detail::empty_stage&>(*this);
			}
			const auto& first_stage() const noexcept {
				return static_cast<const detail::empty_stage&>(*this);
			}

			void start() noexcept { start_stages(*this); }
			void stop() noexcept { stop_stages(); }
			bool is_working() const noexcept {
				return is_working_;
			}

			template <class... Args>
			bool feed(Args&&... args) const noexcept { return false; }
			void notify() const noexcept {}
		};
	}
}