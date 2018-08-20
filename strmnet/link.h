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
#include "static_list.h"

namespace jkl {
	namespace strmnet {
		/// Link
		/// Provide an interface between two nodes

		template <class SourceNode, class TargetNode>
		class mutable_link {
			SourceNode&			m_source;
			TargetNode const&	m_target;
			bool				m_has_new_data = false;

		public:
			mutable_link(SourceNode& source, TargetNode const& target) noexcept
				: m_source{ source }, m_target{ target } {}

			mutable_link(mutable_link const& other) noexcept
				: m_source{ other.m_source }, m_target{ other.m_target } {}

			auto& source() const noexcept {
				return m_source;
			}
			auto& target() const noexcept {
				return m_target;
			}

			// Check if a new data is available in the source node
			bool has_new_data() const noexcept {
				return m_has_new_data;
			}

			// Get output mutex of the source node
			auto& output_mutex() const noexcept {
				return m_source.output_mutex();
			}

			// Clear new data flag
			void clear() noexcept {
				m_has_new_data = false;
			}

			// Set new data flag and send a signal to the target node
			void notify() noexcept {
				m_target.notify_after([this] { m_has_new_data = true; });
			}

			// Get output from the source node
			// One may override this function to perform a sort of transformation.
			// This will be useful if interfaces of two nodes are almost but not perfectly compatible.
			// However, one should note that there is no way to specify constructor arguments to
			// link class; they are always fixed as (SourceNode&, TargetNode&).
			template <class... Args>
			auto output(Args&&... args) const {
				using crtp_base_type = std::conditional_t<std::is_const<SourceNode>::value,
					typename std::remove_const_t<SourceNode>::crtp_base_type const,
					typename std::remove_const_t<SourceNode>::crtp_base_type>;

				return static_cast<crtp_base_type&>(m_source).call_output(std::forward<Args>(args)...);
			}
		};

		template <class SourceNode, class TargetNode>
		using link = mutable_link<SourceNode const, TargetNode>;

		/// Link specification
		/// A link is a ordered pair of nodes in a directed graph.
		/// Since nodes can be specified in two ways
		/// (using index or type; type can be either alias or the node's actual type),
		/// there are four ways to specify a link:
		/// (index - index), (index - type), (type - index), and (type - type).

		enum class link_category_t {
			index_index,
			index_type,
			type_index,
			type_type
		};

		// Index-Index link specifier
		template <std::size_t source, std::size_t target, template <class...> class LinkTemplate = link>
		struct link_ii {
			static constexpr auto link_category = link_category_t::index_index;
			static constexpr std::size_t source_idx = source;
			static constexpr std::size_t target_idx = target;

			template <class SourceNode, class TargetNode>
			using link_type = LinkTemplate<SourceNode, TargetNode>;

			template <class SourceNode, class TargetNode>
			static auto make_link(SourceNode& source, TargetNode& target) {
				return link_type<SourceNode, TargetNode>(source, target);
			}
		};

		// Index-Type link specifier
		template <std::size_t source, class Target, template <class...> class LinkTemplate = link>
		struct link_it {
			static constexpr auto link_category = link_category_t::index_type;
			static constexpr std::size_t source_idx = source;
			using target_type = Target;

			template <class SourceNode, class TargetNode>
			using link_type = LinkTemplate<SourceNode, TargetNode>;

			template <class SourceNode, class TargetNode>
			static auto make_link(SourceNode& source, TargetNode& target) {
				return link_type<SourceNode, TargetNode>(source, target);
			}
		};

		// Type-Index link specifier
		template <class Source, std::size_t target, template <class...> class LinkTemplate = link>
		struct link_ti {
			static constexpr auto link_category = link_category_t::type_index;
			using source_type = Source;
			static constexpr std::size_t target_idx = target;

			template <class SourceNode, class TargetNode>
			using link_type = LinkTemplate<SourceNode, TargetNode>;

			template <class SourceNode, class TargetNode>
			static auto make_link(SourceNode& source, TargetNode& target) {
				return link_type<SourceNode, TargetNode>(source, target);
			}
		};

		// Type-Type link specifier
		template <class Source, class Target, template <class...> class LinkTemplate = link>
		struct link_tt {
			static constexpr auto link_category = link_category_t::type_type;
			using source_type = Source;
			using target_type = Target;

			template <class SourceNode, class TargetNode>
			using link_type = LinkTemplate<SourceNode, TargetNode>;

			template <class SourceNode, class TargetNode>
			static auto make_link(SourceNode& source, TargetNode& target) {
				return link_type<SourceNode, TargetNode>(source, target);
			}
		};

		// Access the source/target node specified in a link
		// Link can be specified either as a type (derived from one of the above four) or
		// a variable (using template deduction).
		namespace detail {
			template <link_category_t lt>
			struct get_source_target_impl;

			template <>
			struct get_source_target_impl<link_category_t::index_index> {
				template <class LinkSpecifier, class NodeList>
				static auto&& get_source(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<LinkSpecifier::source_idx>(std::forward<NodeList>(nl));
				}

				template <class LinkSpecifier, class NodeList>
				static auto&& get_target(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<LinkSpecifier::target_idx>(std::forward<NodeList>(nl));
				}
			};

			template <>
			struct get_source_target_impl<link_category_t::index_type> {
				template <class LinkSpecifier, class NodeList>
				static auto&& get_source(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<LinkSpecifier::source_idx>(std::forward<NodeList>(nl));
				}

				template <class LinkSpecifier, class NodeList>
				static auto&& get_target(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<typename LinkSpecifier::target_type>(std::forward<NodeList>(nl));
				}
			};

			template <>
			struct get_source_target_impl<link_category_t::type_index> {
				template <class LinkSpecifier, class NodeList>
				static auto&& get_source(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<typename LinkSpecifier::source_type>(std::forward<NodeList>(nl));
				}

				template <class LinkSpecifier, class NodeList>
				static auto&& get_target(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<LinkSpecifier::target_idx>(std::forward<NodeList>(nl));
				}
			};

			template <>
			struct get_source_target_impl<link_category_t::type_type> {
				template <class LinkSpecifier, class NodeList>
				static auto&& get_source(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<typename LinkSpecifier::source_type>(std::forward<NodeList>(nl));
				}

				template <class LinkSpecifier, class NodeList>
				static auto&& get_target(NodeList&& nl) noexcept {
					using jkl::strmnet::get;
					return get<typename LinkSpecifier::target_type>(std::forward<NodeList>(nl));
				}
			};
		}

		template <class LinkSpecifier, class NodeList>
		auto&& get_source(LinkSpecifier const& lk, NodeList&& nl) noexcept {
			return detail::get_source_target_impl<LinkSpecifier::link_category>::
				template get_source<LinkSpecifier>(std::forward<NodeList>(nl));
		}
		template <class LinkSpecifier, class NodeList>
		auto&& get_source(NodeList&& nl) noexcept {
			return detail::get_source_target_impl<LinkSpecifier::link_category>::
				template get_source<LinkSpecifier>(std::forward<NodeList>(nl));
		}
		template <class LinkSpecifier, class NodeList>
		auto&& get_target(LinkSpecifier const& lk, NodeList&& nl) noexcept {
			return detail::get_source_target_impl<LinkSpecifier::link_category>::
				template get_target<LinkSpecifier>(std::forward<NodeList>(nl));
		}
		template <class LinkSpecifier, class NodeList>
		auto&& get_target(NodeList&& nl) noexcept {
			return detail::get_source_target_impl<LinkSpecifier::link_category>::
				template get_target<LinkSpecifier>(std::forward<NodeList>(nl));
		}

		// Find node indices of source/target nodes of a given link specifier
		namespace detail {
			template <class LinkSpecifier, class NodeList, link_category_t c>
			struct find_source_target_index_impl;
			
			template <class LinkSpecifier, class NodeList>
			struct find_source_target_index_impl<LinkSpecifier, NodeList, link_category_t::index_index> {
				static constexpr std::size_t source_idx = LinkSpecifier::source_idx;
				static constexpr std::size_t target_idx = LinkSpecifier::target_idx;
			};

			template <class LinkSpecifier, class NodeList>
			struct find_source_target_index_impl<LinkSpecifier, NodeList, link_category_t::index_type> {
				static constexpr std::size_t source_idx = LinkSpecifier::source_idx;
				static constexpr std::size_t target_idx =
					NodeList::template type_to_index<typename LinkSpecifier::target_type>::value;
			};

			template <class LinkSpecifier, class NodeList>
			struct find_source_target_index_impl<LinkSpecifier, NodeList, link_category_t::type_index> {
				static constexpr std::size_t source_idx =
					NodeList::template type_to_index<typename LinkSpecifier::source_type>::value;
				static constexpr std::size_t target_idx = LinkSpecifier::target_idx;
			};

			template <class LinkSpecifier, class NodeList>
			struct find_source_target_index_impl<LinkSpecifier, NodeList, link_category_t::type_type> {
				static constexpr std::size_t source_idx =
					NodeList::template type_to_index<typename LinkSpecifier::source_type>::value;
				static constexpr std::size_t target_idx =
					NodeList::template type_to_index<typename LinkSpecifier::target_type>::value;
			};
		}
		template <class LinkSpecifier, class NodeList>
		struct find_source_index : std::integral_constant<std::size_t,
			detail::find_source_target_index_impl<LinkSpecifier, NodeList, LinkSpecifier::link_category>::source_idx> {};

		template <class LinkSpecifier, class NodeList>
		struct find_target_index : std::integral_constant<std::size_t,
			detail::find_source_target_index_impl<LinkSpecifier, NodeList, LinkSpecifier::link_category>::target_idx> {};
	}
}