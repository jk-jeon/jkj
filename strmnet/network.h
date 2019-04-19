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
#include "node.h"
#include "link.h"

namespace jkj {
	namespace strmnet {
		/// The whole stream network
		/// NodeList must be an instance of jkj::strmnet::static_list,
		/// but LinkSpecifierList can be any std::tuple-like class, including
		/// std::tuple, std::pair, std::array, and jkj::strmnet::static_list.
		/// Note that while NodeList is instantiated as a member variable,
		/// LinkSpecifierList is only a type information; it is never instantiated.
		/// Rather, a concrete type of links are inferred from it and then instantiated.
		/// The initialization of nodes is guaranteed to be in-order, because NodeList
		/// must be a jkj::strmnnet::static_list.
		/// This is not the case for links. Regardless of the actual type of LinkSpecifierList,
		/// actual link list will be instantiated as an std::tuple, whose initialization order
		/// is not guaranteed (the order is usually backward).
		template <class NodeList, class LinkSpecifierList>
		class network {
			NodeList			m_node_list;

		public:
			using node_list = NodeList;
			using link_specifier_list = LinkSpecifierList;

			static constexpr std::size_t number_of_nodes = std::tuple_size<NodeList>::value;
			template <std::size_t node_idx>
			using node_type = std::tuple_element_t<node_idx, NodeList>;

			/// Stuffs needed to derive link_list_type
		private:
			template <std::size_t link_idx>
			using link_specifier_type = std::tuple_element_t<link_idx, LinkSpecifierList>;

			template <std::size_t I>
			static auto make_link(NodeList& node_list) noexcept {
				return link_specifier_type<I>::make_link(
					get_source<link_specifier_type<I>>(node_list),
					get_target<link_specifier_type<I>>(node_list));
			}

		public:
			static constexpr std::size_t number_of_links = std::tuple_size<LinkSpecifierList>::value;
			template <std::size_t link_idx>
			using link_type = decltype(make_link<link_idx>(m_node_list));

		private:
			template <std::size_t... I>
			static auto make_link_list(NodeList& node_list, std::index_sequence<I...>) noexcept {
				return std::make_tuple(make_link<I>(node_list)...);
			}

			using link_list_type = decltype(make_link_list(m_node_list,
				std::make_index_sequence<std::tuple_size<LinkSpecifierList>::value>{}));

			link_list_type		m_link_list;

		public:
			/// Default constructor
			template <class = std::enable_if_t<std::is_default_constructible<NodeList>::value>>
			network() : m_link_list{ make_link_list(m_node_list, std::make_index_sequence<number_of_links>{}) } {}

			/// Constructor with one argument
			template <class Arg, class = jkj::tmp::prevent_too_perfect_fwd<network, Arg>,
				class = std::enable_if_t<std::is_constructible<NodeList, Arg>::value>>
			network(Arg&& arg) : m_node_list{ std::forward<Arg>(arg) },
				m_link_list{ make_link_list(m_node_list, std::make_index_sequence<number_of_links>{}) } {}

			/// Constructor with multiple arguments
			template <class... Args, class = std::enable_if_t<
				std::is_constructible<NodeList, Args...>::value && (sizeof...(Args) >= 2)>>
			network(Args&&... args) : m_node_list { std::forward<Args>(args)... },
				m_link_list{ make_link_list(m_node_list, std::make_index_sequence<number_of_links>{}) } {}

			/// Destructor
			~network() {
				// Try to stop the network
				while( true ) {
					try {
						stop();
						break;
					}
					// Silently drop all exceptions thrown
					catch( ... ) {}
				}
			}


			/// Access node by index
			template <std::size_t node_idx>
			auto& node() {
				using jkj::strmnet::get;
				return get<node_idx>(m_node_list);
			}
			template <std::size_t node_idx>
			auto& node() const {
				using jkj::strmnet::get;
				return get<node_idx>(m_node_list);
			}
			
			/// Access node by type
			template <class QueryType>
			auto& node() {
				using jkj::strmnet::get;
				return get<QueryType>(m_node_list);
			}
			template <class QueryType>
			auto& node() const {
				using jkj::strmnet::get;
				return get<QueryType>(m_node_list);
			}

			/// Access link by index
			template <std::size_t link_idx>
			auto& link() {
				using jkj::strmnet::get;
				return get<link_idx>(m_link_list);
			}
			template <std::size_t link_idx>
			auto& link() const {
				using jkj::strmnet::get;
				return get<link_idx>(m_link_list);
			}

			/// Access link by specifier type
			template <class QueryType>
			auto& link() {
				using std::get;
				return get<LinkSpecifierList::template type_to_index<QueryType>::value>(m_link_list);
			}
			template <class QueryType>
			auto& link() const {
				using std::get;
				return get<LinkSpecifierList::template type_to_index<QueryType>::value>(m_link_list);
			}

			/// Access link by source node index & target node index
			/// Fails if there are multiple links between the specified nodes
			template <std::size_t source_idx, std::size_t target_idx>
			auto& link() {
				return link<link_idx_by_source_target<source_idx, target_idx>::value>();
			}
			template <std::size_t source_idx, std::size_t target_idx>
			auto& link() const {
				return link<link_idx_by_source_target<source_idx, target_idx>::value>();
			}

			/// Access link by source node index & target node type
			/// Fails if there are multiple links between the specified nodes
			template <std::size_t source_idx, class TargetQueryType>
			auto& link() {
				return link<source_idx, node_list::template type_to_index<TargetQueryType>::value>();
			}
			template <std::size_t source_idx, class TargetQueryType>
			auto& link() const {
				return link<source_idx, node_list::template type_to_index<TargetQueryType>::value>();
			}

			/// Access link by source node type & target node index
			/// Fails if there are multiple links between the specified nodes
			template <class SourceQueryType, std::size_t target_idx>
			auto& link() {
				return link<node_list::template type_to_index<SourceQueryType>::value, target_idx>();
			}
			template <class SourceQueryType, std::size_t target_idx>
			auto& link() const {
				return link<node_list::template type_to_index<SourceQueryType>::value, target_idx>();
			}

			/// Access link by source node type & target node index
			/// Fails if there are multiple links between the specified nodes
			template <class SourceQueryType, class TargetQueryType>
			auto& link() {
				return link<
					node_list::template type_to_index<SourceQueryType>::value,
					node_list::template type_to_index<TargetQueryType>::value
				>();
			}
			template <class SourceQueryType, class TargetQueryType>
			auto& link() const {
				return link<
					node_list::template type_to_index<SourceQueryType>::value,
					node_list::template type_to_index<TargetQueryType>::value
				>();
			}


			/// Get the reference tuple of all incoming links for a given node
			template <std::size_t node_idx>
			auto in_links() {
				return get_link_ref_tuple(typename in_out_link_finder<node_idx>::in_links{});
			}
			template <std::size_t node_idx>
			auto in_links() const {
				return get_link_ref_tuple(typename in_out_link_finder<node_idx>::in_links{});
			}
			template <class QueryType>
			auto in_links() {
				return in_links<NodeList::template type_to_index<QueryType>::value>();
			}
			template <class QueryType>
			auto in_links() const {
				return in_links<NodeList::template type_to_index<QueryType>::value>();
			}

			/// Get the reference tuple of all outcoming links for a given node
			template <std::size_t node_idx>
			auto out_links() {
				return get_link_ref_tuple(typename in_out_link_finder<node_idx>::out_links{});
			}
			template <std::size_t node_idx>
			auto out_links() const {
				return get_link_ref_tuple(typename in_out_link_finder<node_idx>::out_links{});
			}
			template <class QueryType>
			auto out_links() {
				return out_links<NodeList::template type_to_index<QueryType>::value>();
			}
			template <class QueryType>
			auto out_links() const {
				return out_links<NodeList::template type_to_index<QueryType>::value>();
			}


			/// Start / stop

			enum class state {
				rest,
				running,
				stop_not_finished,
			};

			// Thrown on an attempt to start the network when the network has been
			// failed to stop properly
			class cannot_start : public std::runtime_error {
				std::runtime_error::runtime_error;
			};

		private:
			state				m_current_state;

		public:
			state current_state() const noexcept {
				return m_current_state;
			}

			template <class... StartParams>
			void start(StartParams&&... sp) {
				switch( m_current_state ) {
					// Don't do anything if the network is already running
				case state::running:
					return;

					// Throw an exception if the network is in an invalid state
				case state::stop_not_finished:
					throw cannot_start{ "Cannot start the network: the network has failed to stop properly" };
				}

				/// Phase 1 - call prepare() on each node
				m_node_list.prepare_each(
					[&sp...](auto& node) {
					node.prepare(std::forward<StartParams>(sp)...);
				}, [](auto& node) {
					node.finish();
				});

				/// Phase 2 - call start() on each node
				try {
					start_impl<0, number_of_nodes == 0>::start(this);
				}
				catch( ... ) {
					// Rollback preparation and rethrow
					m_node_list.for_each_backward([](auto& node) { node.finish(); });
					throw;
				}

				/// Phase 3 - call after_start() on each node
				try {
					m_node_list.prepare_each(
						[](auto& node) { node.after_start(); },
						[](auto& node) { node.before_stop(); });
				}
				catch( ... ) {
					// Stop nodes
					m_node_list.for_each_backward([](auto& node) {
						try {
							node.stop();
						}
						// Just drop the exception thrown inside the worker thread
						catch( ... ) {}
					});
					// Rollback preparation and rethrow
					m_node_list.for_each_backward([](auto& node) { node.finish(); });
					throw;
				}

				m_current_state = state::running;
			}

			void stop() {
				/// Phase 1 - call before_stop() on each node (assumed to be noexcept)
				// Do only when the network is running
				if( m_current_state == state::running ) {
					m_node_list.for_each_backward([](auto& node) { node.before_stop(); });
				}

				// Don't do anything if the network is already in rest
				if( m_current_state != state::rest ) {
					/// Phase 2 - call stop() on each node (may throw)
					m_node_list.for_each_backward([](auto& node) { node.stop(); });

					/// Phase 3 - call finish() on each node (assumed to be noexcept)
					m_node_list.for_each_backward([](auto& node) { node.finish(); });

					m_current_state = state::rest;
				}
			}

		private:
			/// Search a link by source & target indices
			template <std::size_t source_idx, std::size_t target_idx>
			struct link_idx_by_source_target {
				template <std::size_t link_idx, bool end = (link_idx == number_of_links)>
				struct impl {
					static constexpr bool found_here =
						find_source_index<link_specifier_type<link_idx>, node_list>::value == source_idx &&
						find_target_index<link_specifier_type<link_idx>, node_list>::value == target_idx;
					static constexpr bool found_before = impl<link_idx + 1>::found;
					static constexpr bool found = found_here || found_before;
					static constexpr bool not_unique = (found_here && found_before) ||
						impl<link_idx + 1>::not_unique;
					static constexpr std::size_t value = found_here ? link_idx
						: impl<link_idx + 1>::value;
				};

				template <std::size_t link_idx>
				struct impl<link_idx, true> {
					static constexpr bool found = false;
					static constexpr bool not_unique = false;
					static constexpr std::size_t value = number_of_links;
				};

				static constexpr bool found = impl<0>::found;
				static constexpr bool not_unique = impl<0>::not_unique;
				static constexpr std::size_t value = impl<0>::value;

				static_assert(found, "jkj::strmnet::network: there is no link connecting the specified nodes");
				static_assert(!not_unique, "jkj::strmnet::network: there are multiple links connecting the specified nodes");
			};

			/// in_links/out_links implementation

			template <std::size_t I, class IndexSequence>
			struct push_front_impl;

			template <std::size_t I, std::size_t... J>
			struct push_front_impl<I, std::index_sequence<J...>> {
				using type = std::index_sequence<I, J...>;
			};

			template <std::size_t I, class IndexSequence>
			using push_front = typename push_front_impl<I, IndexSequence>::type;

			template <std::size_t node_idx>
			struct in_out_link_finder {
				template <std::size_t link_idx, bool end>
				struct impl;
				
				template <std::size_t link_idx>
				struct impl<link_idx, false>
				{
					using next_impl = typename impl<link_idx + 1, link_idx + 1 == number_of_links>;

					using in_links = std::conditional_t<
						find_target_index<link_specifier_type<link_idx>, NodeList>::value == node_idx,
						push_front<link_idx, typename next_impl::in_links>,
						typename next_impl::in_links
					>;
					using out_links = std::conditional_t<
						find_source_index<link_specifier_type<link_idx>, NodeList>::value == node_idx,
						push_front<link_idx, typename next_impl::out_links>,
						typename next_impl::out_links
					>;
				};

				template <std::size_t link_idx>
				struct impl<link_idx, true> {
					using in_links = std::index_sequence<>;
					using out_links = std::index_sequence<>;
				};

				using in_links = typename impl<0, number_of_links == 0>::in_links;
				using out_links = typename impl<0, number_of_links == 0>::out_links;
			};

			template <std::size_t... link_indices>
			auto get_link_ref_tuple(std::index_sequence<link_indices...>) {
				using std::get;
				return std::make_tuple(std::ref(get<link_indices>(m_link_list))...);
			}


			/// start() implementation

			template <std::size_t node_idx, bool end>
			struct start_impl;
			
			template <std::size_t node_idx>
			struct start_impl<node_idx, false> {
				static void start(network* p) {
					// Call start() on the current node
					p->node<node_idx>().start(p->in_links<node_idx>(), p->out_links<node_idx>());
					try {
						// Call start() on other nodes
						start_impl<node_idx + 1, node_idx + 1 == number_of_nodes>::start(p);
					}
					catch( ... ) {
						// If a call to start() on some node have failed
						try {
							// Stop the current node
							p->node<node_idx>().stop();
						}
						// Exceptions thrown in the worker thread of the current node are ignored
						catch( ... ) {}
						// Rethrow the exception
						throw;
					}
				}
			};

			template <std::size_t node_idx>
			struct start_impl<node_idx, true> {
				static void start(network*) {}
			};
		};
	}
}