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
#include <queue>
#include <future>
#include "../tmp/assert_helper.h"
#include "../tmp/forward.h"
#include "../tmp/is_tuple.h"
#include "../tmp/static_for.h"
#include "../tmp/unpack_and_apply.h"
#include "../utilities.h"

namespace jkj {
	namespace strmnet {
		/// Proceed to the process phase as often as possible
		template <class Node>
		struct update_policy_always {
		protected:
			// This is a policy class
			update_policy_always() = default;
			~update_policy_always() = default;

			bool update_condition() const noexcept {
				return true;
			}
			void before_work() {}
			void after_work() noexcept {}
			void stop() noexcept {}
		};

		/// Proceed to the process phase when a new input is read
		template <class Node>
		struct update_policy_on_new_input {
		protected:
			// This is a policy class
			update_policy_on_new_input() = default;
			~update_policy_on_new_input() = default;

			bool update_condition() const noexcept {
				return static_cast<Node const*>(this)->read_new_data();
			}
			void before_work() {}
			void after_work() noexcept {}
			void stop() noexcept {}
		};

		/// Proceed to the process phase regularly
		template <class Node>
		struct update_policy_fps {
			// This function can be called without any synchronization
			template <class FpsType>
			void set_fps(FpsType const& fps) {
				m_target_delay.store(std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
					std::chrono::duration<FpsType>(1 / fps)), std::memory_order_relaxed);
			}

			// Returns true if succeeded; returns false if the thread is already running.
			// Never call this in the worker thread!
			bool start_update_timer() {
				if( !m_signal_thread.valid() ) {
					m_terminate = false;
					m_next_update = std::chrono::high_resolution_clock::now();
					m_signal_thread_should_wait = true;
					m_signal_thread_waiting = false;

					m_signal_thread = std::async(std::launch::async,
						&update_policy_fps::signal_thread_routine, this);
					return true;
				}
				else
					return false;
			}

			// Returns true if succeeded; returns false if the thread is not running.
			// Never call this in the worker thread!
			bool stop_update_timer() {
				if( m_signal_thread.valid() ) {
					{
						std::lock(static_cast<Node*>(this)->input_mutex(), m_timer_mutex);
						std::lock_guard<std::mutex> lg1{ static_cast<Node*>(this)->input_mutex(), std::adopt_lock };
						std::lock_guard<std::mutex> lg2{ m_timer_mutex, std::adopt_lock };
						m_terminate = true;
					}
					m_wait_process_cv.notify_one();
					m_wait_update_time_cv.notify_one();

					m_signal_thread.get();
					return true;
				}
				else
					return false;
			}

		protected:
			// This is a policy class
			update_policy_fps() = default;
			~update_policy_fps() = default;

			bool update_condition() const {
				// If a sufficient amount of time has been passed, return true.
				// In this case, the signal thread should sleep until the next call to update_condition().
				auto now = std::chrono::high_resolution_clock::now();
				if( now >= m_next_update ) {
					m_next_update += m_target_delay.load(std::memory_order_relaxed);
					m_signal_thread_should_wait = true;
					return true;
				}
				// Otherwise, the signal thread should be activated since the worker thread is going to sleep.
				// If the signal thread has not yet entered the critical section, do not enter.
				m_signal_thread_should_wait = false;
				// If the signal thread has entered the critical section (so it is waiting), then stop waiting.
				if( m_signal_thread_waiting )
					m_wait_process_cv.notify_one();
				return false;
			}

			void before_work() {
				start_update_timer();
			}
			void after_work() noexcept {}
			void stop() noexcept {
				stop_update_timer();
			}

		private:
			bool								m_terminate = false;
			mutable std::condition_variable		m_wait_process_cv;
			mutable std::mutex					m_timer_mutex;
			std::condition_variable				m_wait_update_time_cv;
			std::future<void>					m_signal_thread;

			mutable std::chrono::high_resolution_clock::time_point		m_next_update;
			std::atomic<std::chrono::high_resolution_clock::duration>	m_target_delay;
			mutable bool												m_signal_thread_should_wait = true;
			bool														m_signal_thread_waiting = false;

			// Send a signal to the worker if the next update time has been passed
			// The goal is to make the worker thread run approximately at the target FPS,
			// while at the same time to keep this thread run not faster than the target FPS.
			void signal_thread_routine() {
				while( true ) {
					// Claim the input mutex of the worker; when this returns, there are two possibilities:
					// 1. The worker thread is waiting for the signal.
					// 2. The worker thread has done with the input phase and doing other stuffs.
					{
						std::unique_lock<std::mutex> lg{ static_cast<Node*>(this)->input_mutex() };

						// The flag m_signal_thread_should_wait is the indicator of the case 2;
						// if this flag is set, we should wait until update_condition() is called by the worker thread.
						if( m_signal_thread_should_wait ) {
							m_signal_thread_waiting = true;
							// Here we don't need to gaurd the condition variable from spurious awakening.
							m_wait_process_cv.wait(lg);
							m_signal_thread_waiting = false;
							if( m_terminate )
								break;
							continue;
						}
					}
					// If the flag is cleared, this means that the worker thread is waiting for the signal,
					// because we have claimed the input mutex of the worker thread.
					// In this case, wait until the next update time and signal the worker.
					{
						std::unique_lock<std::mutex> lock{ m_timer_mutex };
						m_wait_update_time_cv.wait_until(lock, m_next_update,
							[this] { return m_terminate || std::chrono::high_resolution_clock::now() >= m_next_update; });
						if( m_terminate )
							break;
						m_signal_thread_should_wait = true;
					}
					// The control reaches this point only if any immediate call to update_condition()
					// is guaranteed to return true.
					static_cast<Node*>(this)->notify();
				}
			}
		};

		/// Proceed to the process phase when a new input is read while trying to keep
		/// waiting time below a certain amount (don't wait too long)
		template <class Node>
		struct update_policy_min_fps : update_policy_fps<Node> {
		protected:
			// This is a policy class
			update_policy_min_fps() = default;
			~update_policy_min_fps() = default;

			bool update_condition() const {
				return update_policy_fps<Node>::update_condition() ||
					static_cast<Node const*>(this)->read_new_data();
			}
		};

		/// Proceed to the process phase when a new input is read while trying to keep
		/// waiting time above a certain amount (don't rush too fast)
		template <class Node>
		struct update_policy_max_fps : public update_policy_fps<Node> {
		protected:
			// This is a policy class
			update_policy_max_fps() = default;
			~update_policy_max_fps() = default;

			bool update_condition() const {
				if( static_cast<Node const*>(this)->read_new_data() ) {
					m_read_new_data_cache = true;
				}
				if( update_policy_fps<Node>::update_condition() && m_read_new_data_cache ) {
					m_read_new_data_cache = false;
					return true;
				}
				return false;
			}

			mutable bool						m_read_new_data_cache = false;
		};

		/// Do not create a thread
		/// input(), process(), produce(), followed by notifications into
		/// nodes expecting output from this node, are all done inside feed().
		/// A node having this update policy cannot have any incoming link,
		/// because notify() is not defined.
		template <class Node>
		struct update_policy_self {
		protected:
			// This is a policy class
			update_policy_self() = default;
			~update_policy_self() = default;

			bool update_condition() const {
				return true;
			}
		};

		/// Provide pause()/resume() member functions
		template <class Node>
		struct pause_policy_default {
		protected:
			// This is a policy class
			pause_policy_default() = default;
			~pause_policy_default() = default;

			void stop() {
				m_is_paused.store(0, std::memory_order_relaxed);
			}

		public:
			void pause() {
				std::lock_guard<std::mutex> lg{ static_cast<Node*>(this)->input_mutex() };
				m_is_paused.store(1, std::memory_order_relaxed);
			}

			void resume() {
				std::lock_guard<std::mutex> lg{ static_cast<Node*>(this)->input_mutex() };
				m_is_paused.store(0, std::memory_order_relaxed);
			}

			bool toggle_pause() {
				std::lock_guard<std::mutex> lg{ static_cast<Node*>(this)->input_mutex() };
				return m_is_paused.fetch_xor(1, std::memory_order_relaxed);
			}

			bool is_paused() const noexcept {
				return m_is_paused.load(std::memory_order_relaxed) == 1;
			}

		private:
			std::atomic<unsigned char> m_is_paused;
		};

		/// Pauseless
		template <class Node>
		struct pause_policy_pauseless {
		protected:
			// This is a policy class
			pause_policy_pauseless() = default;
			~pause_policy_pauseless() = default;

			void stop() noexcept {}

		public:
			bool is_paused() const noexcept {
				return false;
			}
		};

		namespace detail {
			// Helpers for node::get()
			template <class Node, class... Args>
			struct get_helper_impl {
				using output_type = decltype(std::declval<Node*>()->call_output(std::declval<Args>()...));

				static constexpr bool is_void = std::is_same<output_type, void>::value;
				static constexpr bool is_tuple = tmp::is_tuple<output_type>::value;

				template <bool is_void, bool is_tuple, class dummy>
				struct impl;

				template <bool is_tuple, class dummy>
				struct impl<true, is_tuple, dummy> {
					static auto get(Node* p, Args&&... args) {
						p->call_output(std::forward<Args>(args)...);
						return util::make_auto_locked_tuple(std::unique_lock<std::mutex>{ p->output_mutex() });
					}
				};

				template <class dummy>
				struct impl<false, true, dummy> {
					static auto get(Node* p, Args&&... args) {
						return tmp::unpack_and_apply([p](auto&&... outputs) -> auto {
							return util::make_auto_locked_tuple(std::unique_lock<std::mutex>{ p->output_mutex() },
								std::forward<decltype(outputs)>(outputs)...);
						}, p->call_output(std::forward<Args>(args)...));
					}
				};

				template <class dummy>
				struct impl<false, false, dummy> {
					static auto get(Node* p, Args&&... args) {
						return jkj::util::make_auto_locked_data(std::unique_lock<std::mutex>{ p->output_mutex() },
							p->call_output(std::forward<Args>(args)...));
					}
				};

				static auto get(Node* p, Args&&... args) {
					return impl<is_void, is_tuple, void>::get(p, std::forward<Args>(args)...);
				}
			};

			template <class Node, class... Args>
			auto get_helper(Node* p, Args&&... args) {
				return get_helper_impl<Node, Args...>::get(p, std::forward<Args>(args)...);
			}
		}

		/// The node class
		template <class Impl,
			template <class...> class UpdatePolicy = update_policy_on_new_input,
			template <class...> class PausePolicy = pause_policy_default
		>
		class node :
			public UpdatePolicy<node<Impl, UpdatePolicy, PausePolicy>>,
			public PausePolicy<node<Impl, UpdatePolicy, PausePolicy>>
		{
		protected:
			/// This is a CRTP base
			node() = default;
			~node() = default;

			using update_policy = UpdatePolicy<node>;
			using pause_policy = PausePolicy<node>;

		public:
			friend update_policy;
			friend pause_policy;

			template <class Node, class... Args>
			friend struct detail::get_helper_impl;

			template <class SourceNode, class TargetNode>
			friend class mutable_link;

			template <class NodeList, class LinkList>
			friend class network;

			using crtp_base_type = node;
			auto& crtp_base() noexcept { return *this; }
			auto& crtp_base() const noexcept { return *this; }


			/// Following routines may be overriden
		private:
			/// Parameters passed to output() from incoming links
			/// Default: nothing
			/// This function is supposed to be very simple; e.g., returns a tuple of references.
			template <class Link>
			void input_params(Link& link) const noexcept {}

			/// Get input data from incoming nodes
			/// Default: generates a compile error
			template <class dummy = void>
			void input(...) {
				static_assert(jkj::tmp::assert_helper<dummy>::value,
					"jkj::strmnet::node: input() is not implemented");
			}
			
			/// Process data
			/// Default: do nothing
			void process() {}

			/// Generates outputs
			/// Default: do nothing
			void produce() {}

			/// Get output
			/// Default: generates a compile error
			/// This function is supposed to be very simple; e.g., returns a tuple of references.
			template <class dummy = void>
			void output(...) const noexcept {
				static_assert(jkj::tmp::assert_helper<dummy>::value,
					"jkj::strmnet::node: output() is not implemented");
			}
			template <class dummy = void>
			void output(...) noexcept {
				static_assert(jkj::tmp::assert_helper<dummy>::value,
					"jkj::strmnet::node: output() is not implemented");
			}

			/// Initialize/Cleanup routines
			/// Default: do nothing

			// Called before the worker thread is launched
			template <class... StartParams>
			void prepare(StartParams&&... sp) {}

			// Called after the worker thread has been terminated
			// Should be noexcept
			void finish() noexcept {}

			// Called inside the worker thread before start working
			void before_work() {}

			// Called inside the worker thread after finish working
			// Should be noexcept
			void after_work() noexcept {}

			// Called after the worker thread has been successfully launched
			void after_start() {}

			// Called before notifying the worker thread to terminate
			// Should be noexcept
			void before_stop() noexcept {}

			// Called at the begining of the worker loop
			void begin_cycle() {}

			// Called at the end of the worker loop
			// Should be noexcept
			void end_cycle() noexcept {}

			/// Error handling routines
			/// Default: rethrow
			/// This will silently kill the worker thread, and the exception is rethrown only after stop() is called on the network.
			/// Although the exception is rethrown after stop() is called, finish() and before_stop(), and other
			/// cleanup routines for other nodes may not be properly called, so the network will be laid on undefined state.
			/// In this case, you can repeatedely call stop() until it does not throw, to make the network in the normal state;
			/// for example, you may do like this:
			/// while( true ) {
			///   try {
			///     network.stop();
			///     break;
			///   }
			///   catch( std::exception& e ) {
			///     // .. deal with the error ..
			///   }
			/// }
			/// Some may see this ugly, but I can't think of any other ways to gracefully deal with errors occured in worker threads.
			/// It is highly recommended to reimplement these functions;
			/// indeed, it is better not to throw inside begin_cycle(), input(), process(), and produce().

			// Called when an exception is thrown inside the worker loop
			void on_worker_error(std::exception_ptr eptr) {
				std::rethrow_exception(eptr);
			}
			// Called when an exception is thrown during the input phase
			void on_input_error(std::exception_ptr eptr) {
				std::rethrow_exception(eptr);
			}
			// Called when process() throws
			void on_process_error(std::exception_ptr eptr) {
				std::rethrow_exception(eptr);
			}
			// Called when an exception is thrown during the produce phase
			void on_produce_error(std::exception_ptr eptr) {
				std::rethrow_exception(eptr);
			}


			/// Following functions should not be overriden
		public:
			// Check if a new data from other nodes have been read
			// Mainly useful for update policy implementations
			bool read_new_data() const noexcept {
				return m_read_new_data;
			}

			// Direct access to mutices for low-level synchronization
			auto& input_mutex() const noexcept {
				return m_input_mutex;
			}
			auto& process_mutex() const noexcept {
				return m_process_mutex;
			}
			auto& output_mutex() const noexcept {
				return m_output_mutex;
			}

			// Provide an input directly with a proper synchronization
			// This function is synchronous. Since feeding an input may incur data race,
			// this function locks the process mutex.
			// Hence, the calling thread should wait the current call to process() to finish.
			// Therefore, you should not use this function too often.
			// node class is intended to take its necessary inputs by itself when needed.
			// Direct feeding is not a "normal" behavior, and should be limited.
			template <class... Args>
			auto feed(Args&&... args) {
				using result_type = decltype(static_cast<Impl*>(this)->input(std::forward<Args>(args)...));
				jkj::tmp::result_holder<result_type> result;
				{
					std::lock_guard<std::mutex> process_lg{ m_process_mutex };
					std::lock_guard<std::mutex> input_lg{ m_input_mutex };
					result = jkj::tmp::hold_result([this](auto&&... args) {
						return static_cast<Impl*>(this)->input(std::forward<decltype(args)>(args)...);
					}, std::forward<Args>(args)...);

					m_read_new_data = true;
				}
				m_wait_cv.notify_one();
				return std::move(result).result();
			}

			// Provide an input through an input queue
			// Provided inputs will be copied to temporary buffers, which will be moved into the node
			// with the call to input(). The calling thread cannot get any return or exception thrown inside input().
			// However, it should deal with exceptions thrown while copying data into the temporary buffers.
			// Since data is transferred across multiple threads, reference arguments are not stored as references;
			// they are copied as values, just like std::make_tuple() and std::async(). Like them, you should use
			// std::ref() or std::cref() if you want to force not to copy values.
			// This function does not lock the process mutex, but it does lock the input mutex.
			// Hence, the function may be blocked if there is an input contention.
			// If this occurs, you should review your design, because input() is supposed to be a simple function.
			template <class... Args>
			void feed_async(Args&&... args) {
				notify_after([this](auto&&... args) {
					m_input_queue.feed(std::forward<decltype(args)>(args)...);
				}, std::forward<Args>(args)...);
			}			


			// Wake up the worker if waiting
			// Used for some low-level synchronizations.
			void notify() const noexcept {
				m_wait_cv.notify_one();
			}

			// Wake up the worker after doing an additional job with the input locking
			// Used for some low-level tasks.
			template <class Functor, class... Args>
			void notify_after(Functor&& f, Args&&... args) const
				noexcept(noexcept(std::forward<Functor>(f)(std::forward<Args>(args)...)))
			{
				{
					std::lock_guard<std::mutex> lg{ m_input_mutex };
					std::forward<Functor>(f)(std::forward<Args>(args)...);
				}
				m_wait_cv.notify_one();
			}

			// Get the output with a proper synchronization
			// If output() returns void, the return value is jkj::auto_locked_tuple with an empty tuple; otherwise,
			// If output() returns a tuple, the return value is jkj::auto_locked_tuple; otherwise,
			// The return value is jkj::auto_locked_data.
			// Hence, the output locking is automatically released when the lifetime of the return value ends.
			template <class... Args>
			auto get(Args&&... args) const {
				return detail::get_helper(this, std::forward<Args>(args)...);
			}
			template <class... Args>
			auto get(Args&&... args) {
				return detail::get_helper(this, std::forward<Args>(args)...);
			}

		private:
			std::atomic<bool>					m_terminate;
			std::future<void>					m_worker_thread;
			mutable std::mutex					m_process_mutex;
			mutable std::mutex					m_input_mutex;
			mutable std::mutex					m_output_mutex;
			mutable std::condition_variable		m_wait_cv;
			bool								m_read_new_data = false;

			// Input queue for asynchronous feed
			// Use type-erasure to store any type of input arguments
			class input_queue {
			public:
				struct input_forwarder {
					virtual ~input_forwarder() {};
					virtual void consume_input(node& obj) = 0;
				};

				template <class... Args>
				struct input_forwarder_impl : input_forwarder {
					using tuple_type = decltype(std::make_tuple(std::declval<Args>()...));
					tuple_type args_tuple;

					template <class... InitArgs>
					input_forwarder_impl(InitArgs&&... args) : args_tuple{ std::forward_as_tuple(std::forward<InitArgs>(args)...) } {}

					void consume_input(node& n) override {
						jkj::tmp::unpack_and_apply([&](auto&&... args) {
							static_cast<Impl&>(n).input(std::forward<decltype(args)>(args)...);
						}, std::move(args_tuple));
					}
				};

				template <class... Args>
				static auto make_input_forwarder(Args&&... args) {
					return std::make_unique<input_forwarder_impl<Args...>>(std::forward<Args>(args)...);
				}

				std::queue<std::unique_ptr<input_forwarder>>	m_queue;

			public:
				template <class... Args>
				void feed(Args&&... args) {
					m_queue.push(make_input_forwarder(std::forward<Args>(args)...));
				}

				// Returns false if the queue is empty; otherwise, returns true
				bool consume(node& n) {
					if( m_queue.empty() )
						return false;

					do {
						auto ptr = std::move(m_queue.front());
						m_queue.pop();
						ptr->consume_input(n);
					} while( !m_queue.empty() );

					return true;
				}
			};
			input_queue					m_input_queue;

			
			// This makes possible to call Impl::output() from friend classes
			template <class... Args>
			decltype(auto) call_output(Args&&... args) const {
				return static_cast<Impl const&>(*this).output(std::forward<Args>(args)...);
			}
			template <class... Args>
			decltype(auto) call_output(Args&&... args) {
				return static_cast<Impl&>(*this).output(std::forward<Args>(args)...);
			}

			enum class launch_status { not_finished, succeeded, failed };

			// Launch a worker thread
			template <class InLinks, class OutLinks>
			void start(InLinks&& in_links, OutLinks&& out_links) {
				m_terminate.store(false, std::memory_order_relaxed);

				auto success_flag = launch_status::not_finished;
				std::mutex mtx;
				std::condition_variable cv;

				std::unique_lock<std::mutex> lg{ mtx };

				m_worker_thread = std::async(std::launch::async,
					&node::worker_proc<InLinks, OutLinks>,
					this, std::ref(success_flag), std::ref(mtx), std::ref(cv),
					jkj::tmp::ref_if_lvalue(std::forward<InLinks>(in_links)),
					jkj::tmp::ref_if_lvalue(std::forward<OutLinks>(out_links)));

				// Wait for before_work() function to return
				cv.wait(lg, [&success_flag] { return success_flag != launch_status::not_finished; });
				// If before_work() hasn't been successfully returned due to an exception, call m_worker_thread.get()
				// This will effectively rethrow the exception
				if( success_flag == launch_status::failed )
					m_worker_thread.get();
			}

			// Stop the worker thread
			void stop() {
				update_policy::stop();
				{
					std::lock_guard<std::mutex> lg{ m_input_mutex };
					pause_policy::stop();
					m_terminate.store(true, std::memory_order_relaxed);
				}
				m_wait_cv.notify_one();

				if( m_worker_thread.valid() )
					m_worker_thread.get();
			}

			template <class InLinks, class OutLinks>
			void worker_proc(launch_status& success_flag, std::mutex& mtx, std::condition_variable& cv,
				InLinks in_links, OutLinks out_links)
			{
				// Make sure after_work() to be called even when an exception is thrown
				struct worker_raii {
					node* p;
					worker_raii(node* p, launch_status& success_flag,
						std::mutex& mtx, std::condition_variable& cv) : p{ p } {
						try {
							p->update_policy::before_work();
							static_cast<Impl*>(p)->before_work();

							std::lock_guard<std::mutex> lg{ mtx };
							success_flag = launch_status::succeeded;
						}
						catch( ... ) {
							{
								std::lock_guard<std::mutex> lg{ mtx };
								success_flag = launch_status::failed;
							}
							cv.notify_one();
							throw;
						}
						cv.notify_one();
					}

					~worker_raii() {
						static_cast<Impl*>(p)->after_work();
						p->update_policy::after_work();
					}
				} worker_raii_obj{ this, success_flag, mtx, cv };

				// Main worker loop
				while( !m_terminate.load(std::memory_order_relaxed) ) {
					try {
						// Make sure end_cycle() to be called even when an exception is thrown
						struct cycle_raii {
							Impl* p;
							cycle_raii(Impl* p) : p{ p } {
								p->begin_cycle();
							}

							~cycle_raii() {
								p->end_cycle();
							}
						} cycle_raii_obj{ static_cast<Impl*>(this) };

						/// Input phase
						{
							std::unique_lock<std::mutex> input_lg{ m_input_mutex };
							try {
								// See if a new data is available.
								while( !m_terminate.load(std::memory_order_relaxed) ) {
									// Read data from incoming nodes
									jkj::tmp::static_for_each(in_links, [this](auto& link) {
										// If there is a new data, read the data
										if( link.has_new_data() ) {
											// Lock output mutex of the incoming link
											std::lock_guard<std::mutex> lg{ link.output_mutex() };
											// First, apply input_params() to output() of the link, and then
											// apply the result to input().
											jkj::tmp::chain_call([this](auto&&... outputs) {
												return static_cast<Impl*>(this)->input(std::forward<decltype(outputs)>(outputs)...);
											}, [this, &link]() {
												return jkj::tmp::chain_call([&link](auto&&... input_args) {
													return link.output(std::forward<decltype(input_args)>(input_args)...);
												}, [this, &link]() { return static_cast<Impl*>(this)->input_params(link); });
											});

											// Mark there is no new input data
											link.clear();

											m_read_new_data = true;
										}
									});

									// Read data from input queue
									if( m_input_queue.consume(*this) )
										m_read_new_data = true;

									if( static_cast<Impl*>(this)->update_condition() && !pause_policy::is_paused() )
										break;

									// Wait for signals if update_condition() has been not satisfied
									// Here, there is no need to gaurd the condition variable from spurious awakening
									do {
										m_wait_cv.wait(input_lg);
									} while( pause_policy::is_paused() );
								}	// while
							}	// try
							catch( ... ) {
								m_read_new_data = false;
								static_cast<Impl*>(this)->on_input_error(std::current_exception());
								continue;
							}
						}

						if( m_terminate.load(std::memory_order_relaxed) )
							break;


						/// Process phase
						{
							std::lock_guard<std::mutex> process_lg{ m_process_mutex };
							try {
								m_read_new_data = false;
								static_cast<Impl*>(this)->process();

								/// Produce phase
								std::lock_guard<std::mutex> output_lg{ m_output_mutex };
								try {
									static_cast<Impl*>(this)->produce();
								}
								catch( ... ) {
									static_cast<Impl*>(this)->on_produce_error(std::current_exception());
									continue;
								}
							}
							catch( ... ) {
								static_cast<Impl*>(this)->on_process_error(std::current_exception());
								continue;
							}
						}

						if( m_terminate.load(std::memory_order_relaxed) )
							break;


						/// Broadcast phase
						jkj::tmp::static_for_each(out_links, [this](auto& link) {
							// Notify to each outcoming link there is a new data
							link.notify();
						});
					}	// try
					catch( ... ) {
						static_cast<Impl*>(this)->on_worker_error(std::current_exception());
					}
				}	// main loop
			}	// worker_proc

		};

		/// Specialization for the case when UpdatePolicy = update_policy_self
		template <class Impl>
		class node<Impl, update_policy_self> : public update_policy_self<Impl> {
		protected:
			/// This is a CRTP base
			node() = default;
			~node() = default;

			using update_policy = update_policy_self<Impl>;

		public:
			friend update_policy;

			template <class SourceNode, class TargetNode>
			friend class mutable_link;

			template <class NodeList, class LinkList>
			friend class network;

			template <class Node, class... Args>
			friend struct detail::get_helper_impl;

			using crtp_base_type = node;
			auto& crtp_base() noexcept { return *this; }
			auto& crtp_base() const noexcept { return *this; }


			/// Following routines may be overriden
		private:
			/// There is no input_params()

			/// Get input data; called inside feed()
			/// Default: generates a compile error
			template <class... Args>
			void input(Args&&...) {
				static_assert(jkj::tmp::assert_helper<Args...>::value,
					"jkj::strmnet::node: input() is not implemented");
			}

			/// Process data
			/// Default: do nothing
			void process() {}

			/// Generates outputs
			/// Default: do nothing
			void produce() {}

			/// Get output
			/// Default: generates a compile error
			/// This function is supposed to be very simple; e.g., returns a tuple of references.
			template <class... Args>
			void output(Args&&...) const noexcept {
				static_assert(jkj::tmp::assert_helper<Args...>::value,
					"jkj::strmnet::node: output() is not implemented");
			}

			/// Initialize/Cleanup routines
			/// Default: do nothing

			// Called inside start() of the network, before before_work() is called
			template <class... StartParams>
			void prepare(StartParams&&... sp) {}

			// Called inside stop() of the network, after after_work() is called
			// Should be noexcept
			void finish() noexcept {}

			// Called inside start() of the network, after prepare() is called, before after_start() is called
			void before_work() {}

			// Called inside stop() of the network, before finish() is called, after before_stop() is called
			// Should be noexcept
			void after_work() noexcept {}

			// Called inside start() of the network, after before_work() is called
			void after_start() {}

			// Called inside stop() of the network, before after_work() is called
			// Should be noexcept
			void before_stop() noexcept {}

			// There is no begin_cycle() and end_cycle()


			/// There is no error handling routines, as error will be immediately reported to the thread calling feed()


			/// Following functions should not be overriden
		public:
			// There is no read_new_data() and input_mutex(), as they are not necessary

			// Direct access to mutices for low-level synchronization
			auto& output_mutex() const noexcept {
				return m_output_mutex;
			}

			// Provide an input, call process() and produce(), and broadcast
			template <class... Args>
			auto feed(Args&&... args) {
				// Input
				auto input_result = jkj::tmp::hold_result([this](auto&&... args) {
					return static_cast<Impl*>(this)->input(std::forward<decltype(args)>(args)...);
				}, std::forward<Args>(args)...);

				if( static_cast<Impl*>(this)->update_condition() ) {
					// Process
					static_cast<Impl*>(this)->process();

					// Produce
					{
						std::lock_guard<std::mutex> lg{ m_output_mutex };
						static_cast<Impl*>(this)->produce();
					}

					// Notify
					m_output_notifier();
				}

				return std::move(input_result).result();
			}

			// There is no feed_async(), notify(), and notify_after()

			// Get the output with a proper synchronization
			// If output() returns void, the return value is jkj::auto_locked_tuple with an empty tuple; otherwise,
			// If output() returns a tuple, the return value is jkj::auto_locked_tuple; otherwise,
			// The return value is jkj::auto_locked_data.
			// Hence, the output locking is automatically released when the lifetime of the return value ends.
			template <class... Args>
			auto get(Args&&... args) const {
				return detail::get_helper(this, std::forward<Args>(args)...);
			}
			template <class... Args>
			auto get(Args&&... args) {
				return detail::get_helper(this, std::forward<Args>(args)...);
			}

		private:
			mutable std::mutex			m_output_mutex;
			std::function<void()>		m_output_notifier;

			// This makes possible to call Impl::output() from friend classes
			template <class... Args>
			decltype(auto) call_output(Args&&... args) const {
				return static_cast<Impl const&>(*this).output(std::forward<Args>(args)...);
			}
			template <class... Args>
			decltype(auto) call_output(Args&&... args) {
				return static_cast<Impl&>(*this).output(std::forward<Args>(args)...);
			}

			// Store out-links and call before_work()
			template <class InLinks, class OutLinks>
			void start(InLinks&&, OutLinks&& out_links) {
				m_output_notifier = output_notifier_functor<OutLinks>{ std::forward<OutLinks>(out_links) };
				static_cast<Impl*>(this)->before_work();
			}

			// Call after_work() and nullify m_output_notifier
			void stop() {
				static_cast<Impl*>(this)->after_work();
				m_output_notifier = [] {};
			}

			template <class OutLinks>
			struct output_notifier_functor {
				OutLinks out_links;
				void operator()() {
					jkj::tmp::static_for_each(out_links, [](auto& link) {
						// Notify to each outcoming link there is a new data
						link.notify();
					});
				}
			};
		};
	}
}