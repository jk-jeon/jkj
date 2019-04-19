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
#include <future>
#include "network.h"

namespace jkj {
	namespace strmnet {
		/// Add new nodes or links to a network
		/// Added items will be pushed to the end of the lists.

		namespace detail {
			template <class List, class... NewItems>
			struct add_to_static_list {
			private:
				static constexpr std::size_t original_length = std::tuple_size<List>::value;

				template <std::size_t idx, bool end = (idx == original_length)>
				struct convert_to_static_list {
					using type = typename push_front_static_list<
						std::tuple_element_t<idx, List>,
						typename convert_to_static_list<idx + 1>::type>::type;
				};

				template <std::size_t idx>
				struct convert_to_static_list<idx, true> {
					using type = static_list<>;
				};

			public:
				using type = typename add_to_static_list<
					typename convert_to_static_list<0>::type, NewItems...>::type;
			};

			template <class... ExistingItems, class... NewItems>
			struct add_to_static_list<static_list<ExistingItems...>, NewItems...> {
				using type = static_list<ExistingItems..., NewItems...>;
			};
		}

		template <class Network, class... NodeDescriptors>
		using add_nodes = network<
			typename detail::add_to_static_list<typename Network::node_list, NodeDescriptors...>::type,
			typename Network::link_specifier_list>;

		template <class Network, class... LinkSpecifiers>
		using add_links = network<typename Network::node_list,
			typename detail::add_to_static_list<typename Network::link_specifier_list, LinkSpecifiers...>::type>;


		/// Add a node together with a link from it

		template <class Network, class NodeDescriptor,
			std::size_t target_idx, template <class...> class LinkTemplate = link>
		using attach_source_i = network<
			typename detail::add_to_static_list<typename Network::node_list, NodeDescriptor>::type,
			typename detail::add_to_static_list<typename Network::link_specifier_list,
				link_ii<Network::number_of_nodes, target_idx, LinkTemplate>>::type>;

		template <class Network, class NodeDescriptor,
			class TargetType, template <class...> class LinkTemplate = link>
		using attach_source_t = network<
			typename detail::add_to_static_list<typename Network::node_list, NodeDescriptor>::type,
			typename detail::add_to_static_list<typename Network::link_specifier_list,
				link_it<Network::number_of_nodes, TargetType, LinkTemplate>>::type>;


		/// Add a node together with a link into it

		template <class Network, class NodeDescriptor,
			std::size_t source_idx, template <class...> class LinkTemplate = link>
		using attach_sink_i = network<
			typename detail::add_to_static_list<typename Network::node_list, NodeDescriptor>::type,
			typename detail::add_to_static_list<typename Network::link_specifier_list,
				link_ii<source_idx, Network::number_of_nodes, LinkTemplate>>::type>;

		template <class Network, class NodeDescriptor,
			class SourceType, template <class...> class LinkTemplate = link>
		using attach_sink_t = network<
			typename detail::add_to_static_list<typename Network::node_list, NodeDescriptor>::type,
			typename detail::add_to_static_list<typename Network::link_specifier_list,
				link_ti<SourceType, Network::number_of_nodes, LinkTemplate>>::type>;


		/// Replace the link template from a network

		namespace detail {
			template <class Network, std::size_t link_idx>
			struct find_source_target_idx {
				using link_specifier = std::tuple_element_t<link_idx, typename Network::link_specifier_list>;
				using node_list = typename Network::node_list;

				static constexpr std::size_t source_idx = find_source_index<link_specifier, node_list>::value;
				static constexpr std::size_t target_idx = find_target_index<link_specifier, node_list>::value;
			};
		}

		template <class Network, std::size_t link_idx, template <class...> class LinkTemplate>
		using replace_link = network<typename Network::node_list,
			typename detail::replace_item<typename Network::link_specifier_list, link_idx, link_ii<
				detail::find_source_target_idx<Network, link_idx>::source_idx,
				detail::find_source_target_idx<Network, link_idx>::target_idx, LinkTemplate>
			>::type
		>;
	}
}