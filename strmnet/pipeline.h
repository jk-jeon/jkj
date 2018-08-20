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
#include "network.h"

namespace jkl {
	namespace strmnet {
		/// Convenient interfaces for linear networks
		namespace detail {
			template <class Type, class TypeContainer>
			struct push_front_tuple;

			template <class Type, class... Types>
			struct push_front_tuple<Type, std::tuple<Types...>> {
				using type = std::tuple<Type, Types...>;
			};

			template <std::size_t start_idx, std::size_t end_idx, bool = (start_idx == end_idx)>
			struct pipeline_link_list_generator {
				using type = typename push_front_tuple<link_ii<start_idx, start_idx + 1>,
					typename pipeline_link_list_generator<start_idx + 1, end_idx>::type>::type;
			};

			template <std::size_t start_idx, std::size_t end_idx>
			struct pipeline_link_list_generator<start_idx, end_idx, true> {
				using type = std::tuple<>;
			};
		}

		template <class FirstNodeDescriptor, class... RemainingNodeDescriptors>
		using pipeline = network<static_list<FirstNodeDescriptor, RemainingNodeDescriptors...>,
			typename detail::pipeline_link_list_generator<0, sizeof...(RemainingNodeDescriptors)>::type>;

		
		template <template <class...> class LinkTemplate, class NodeDescriptor>
		struct link_node_pair {
			template <class SourceNode, class TargetNode>
			using link_template = LinkTemplate<SourceNode, TargetNode>;
			using node_descriptor = NodeDescriptor;

			template <std::size_t idx>
			using link_type = link_ii<idx, idx + 1, link_template>;
		};

		namespace detail {
			template <std::size_t start_idx, std::size_t end_idx, class LinkNodePairList, bool = (start_idx == end_idx)>
			struct transform_pipeline_impl {
				using node_list = typename push_front_static_list<
					typename std::tuple_element_t<start_idx, LinkNodePairList>::node_descriptor,
					typename transform_pipeline_impl<start_idx + 1, end_idx, LinkNodePairList>::node_list>::type;

				using link_list = typename push_front_tuple<
					typename std::tuple_element_t<start_idx, LinkNodePairList>::template link_type<start_idx>,
					typename transform_pipeline_impl<start_idx + 1, end_idx, LinkNodePairList>::link_list>::type;
			};

			template <std::size_t start_idx, std::size_t end_idx, class LinkNodePairList>
			struct transform_pipeline_impl<start_idx, end_idx, LinkNodePairList, true> {
				using node_list = static_list<>;
				using link_list = std::tuple<>;
			};
		}

		template <class FirstNodeDescriptor, class... LinkNodePairs>
		using transform_pipeline = network<typename detail::push_front_static_list<FirstNodeDescriptor,
			typename detail::transform_pipeline_impl<0, sizeof...(LinkNodePairs), std::tuple<LinkNodePairs...>>::node_list>::type,
			typename detail::transform_pipeline_impl<0, sizeof...(LinkNodePairs), std::tuple<LinkNodePairs...>>::link_list>;
	}
}