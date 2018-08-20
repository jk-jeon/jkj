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
#include "../tmp/assert_helper.h"
#include "../tmp/forward.h"
#include "../tmp/is_complete.h"

namespace jkl {
	namespace strmnet {
		/// Static list
		/// static_list is basically a std::tuple with some more features.
		///  - In-order initialization is guaranteed.
		///  - "get" function is provided as a member function.
		///  - "for_each" feature is provided.
		///  - Alias can be set to each item.
		/// Either the direct class name or std::pair<Alias, Type> (or any other class template X providing
		/// std::tuple_element<0, X> and std::tuple_element<1, X>) can be specified as Descriptors.
		/// Alias is never really constructed; only Type part is constructed.
		/// It is only used as a name tag, that would be useful
		///  - when there are multiple items having the same underlying type, or
		///  - when typenames of underlying types of some items are too complicated.
		/// Hence, Alias can be, for example, an incomplete type.
		/// std::pair<Alias, Type> form and Type form can be freely mixed; for example, you may instantiate:
		///   static_list<item1_type, std::pair<item2_alias, item2_type>, std::pair<item3_alias, item2_type>, item4_type>
		/// That will produce a type with memory layout approximately look something like:
		///   [item1_type]
		///   [item2_type]
		///   [item2_type]
		///   [item4_type]
		/// with some possible alignment paddings.
		template <class... Descriptors>
		class static_list;

		namespace detail {
			template <class Descriptor>
			struct has_alias : std::integral_constant<bool,
				jkl::tmp::is_complete<std::tuple_element<0, Descriptor>>::value &&
				jkl::tmp::is_complete<std::tuple_element<1, Descriptor>>::value> {};

			template <class Descriptor>
			struct item_alias_type_impl {
				template <bool has_tuple_elmt, class>
				struct impl;

				template <class dummy>
				struct impl<false, dummy> {
					using item_type = Descriptor;
					using alias_type = Descriptor;
				};

				template <class dummy>
				struct impl<true, dummy> {
					using item_type = std::tuple_element_t<1, Descriptor>;
					using alias_type = std::tuple_element_t<0, Descriptor>;
				};

				using item_type = typename impl<has_alias<Descriptor>::value, void>::item_type;
				using alias_type = typename impl<has_alias<Descriptor>::value, void>::alias_type;
			};
		}
		template <class Descriptor>
		using get_item_type = typename detail::item_alias_type_impl<Descriptor>::item_type;
		template <class Descriptor>
		using get_alias_type = typename detail::item_alias_type_impl<Descriptor>::alias_type;
		

		/// For the case of empty list
		template <>
		class static_list<> {
			template <std::size_t idx>
			struct element_type_impl {
				static_assert(jkl::tmp::assert_helper<std::integral_constant<std::size_t, idx>>::value,
					"jkl::strmnet::static_list: element_type index out of range");
				using type = void;
			};

		public:
			static constexpr std::size_t number_of_items = 0;

			template <std::size_t idx>
			using element_type = typename element_type_impl<idx>::type;

			/// Perform f(item, additional_args) for each item in-order
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&&, AdditionalArgs&&...) {}
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&&, AdditionalArgs&&...) const {}

			/// Same, but in reverse-order
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&&, AdditionalArgs&&...) {}
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&&, AdditionalArgs&&...) const {}

			/// Similar to for_each, but calls cleanup on already prepared items when exceptions are thrown
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&&, Cleanup&&, PrepareArgs&&...) {}
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&&, Cleanup&&, PrepareArgs&&...) const {}
		};

		/// For the case of only one item
		template <class Descriptor>
		class static_list<Descriptor> {
			using first_item_type = get_item_type<Descriptor>;
			first_item_type							first_item_;

			template <std::size_t idx>
			struct element_type_impl {
				static_assert(idx == 0, "jkl::strmnet::static_list: element_type index out of range");
				using type = first_item_type;
			};

			template <class QueryType>
			struct type_to_index_impl;
			
		public:
			static constexpr std::size_t number_of_items = 1;
			template <std::size_t idx>
			using element_type = typename element_type_impl<idx>::type;

			template <class... OtherDescriptors>
			friend class strmnet::static_list;

			/// Default constructor
			template <class = std::enable_if_t<std::is_default_constructible<first_item_type>::value>>
			static_list() {}

			/// Unpack and forward arguments packed in a tuple
			template <class Tuple, class = jkl::tmp::prevent_too_perfect_fwd<static_list, Tuple>>
			explicit static_list(Tuple&& arg_pack)
				: static_list(std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{},
					std::forward<Tuple>(arg_pack)) {}

			/// Access the first item
			auto&& first_item() & noexcept { return first_item_; }
			auto&& first_item() const& noexcept { return first_item_; }
			auto&& first_item() && noexcept { return std::move(first_item_); }
			auto&& first_item() const&& noexcept { return std::move(first_item_); }

			/// Access the last item
			auto&& last_item() & noexcept { return first_item_; }
			auto&& last_item() const& noexcept { return first_item_; }
			auto&& last_item() && noexcept { return std::move(first_item_); }
			auto&& last_item() const&& noexcept { return std::move(first_item_); }

			/// Access an item by index
			template <std::size_t idx>
			auto&& get() & noexcept { return item_by_idx<idx>::get(this); }
			template <std::size_t idx>
			auto&& get() const& noexcept { return item_by_idx<idx>::get(this); }
			template <std::size_t idx>
			auto&& get() && noexcept { return std::move(item_by_idx<idx>::get(this)); }
			template <std::size_t idx>
			auto&& get() const&& noexcept { return std::move(item_by_idx<idx>::get(this)); }

			/// Access an item by type
			/// First, find the specified type among aliases.
			///   If only one alias is found, return the corresponding item.
			///   If multiple aliases are found, generate an error message.
			/// (Classes with no alias specified are treated themselves as their aliases)
			/// If the specified type is not found among aliases, find it among item types themselves.
			///   If only one item type is found, return the corresponding item.
			///   Otherwise, generate an error message.
			template <class QueryType>
			auto&& get() & noexcept { return get<type_to_index<QueryType>::value>(); }
			template <class QueryType>
			auto&& get() const& noexcept { return get<type_to_index<QueryType>::value>(); }
			template <class QueryType>
			auto&& get() && noexcept { return std::move(get<type_to_index<QueryType>::value>()); }
			template <class QueryType>
			auto&& get() const&& noexcept { return std::move(get<type_to_index<QueryType>::value>()); }

			/// Perform f(item, additional_args) for each item in-order
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&& f, AdditionalArgs&&... additional_args) {
				f(first_item_, std::forward<AdditionalArgs>(additional_args)...);
			}
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&& f, AdditionalArgs&&... additional_args) const {
				f(first_item_, std::forward<AdditionalArgs>(additional_args)...);
			}

			/// Same, but in reverse-order
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&& f, AdditionalArgs&&... additional_args) {
				f(first_item_, std::forward<AdditionalArgs>(additional_args)...);
			}
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&& f, AdditionalArgs&&... additional_args) const {
				f(first_item_, std::forward<AdditionalArgs>(additional_args)...);
			}

			/// Similar to for_each, but calls cleanup on already prepared items when exceptions are thrown
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&& p, Cleanup&& c, PrepareArgs&&... prepare_args) {
				p(first_item_, std::forward<PrepareArgs>(prepare_args)...);
			}
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&& p, Cleanup&& c, PrepareArgs&&... prepare_args) const {
				p(first_item_, std::forward<PrepareArgs>(prepare_args)...);
			}

			template <class QueryType>
			struct type_to_index {
				static constexpr bool found_as_alias = type_to_index_impl<QueryType>::found_as_alias;
				static constexpr bool nonunique_as_alias = type_to_index_impl<QueryType>::nonunique_as_alias;
				static constexpr std::size_t alias_idx = type_to_index_impl<QueryType>::alias_idx;
				static constexpr bool found_directly = type_to_index_impl<QueryType>::found_directly;
				static constexpr bool nonunique_directly = type_to_index_impl<QueryType>::nonunique_directly;
				static constexpr std::size_t direct_idx = type_to_index_impl<QueryType>::direct_idx;

				static_assert(type_to_index_impl<QueryType>::found, "jkl::strmnet::static_list: cannot find the requested item");

				static constexpr std::size_t value = found_as_alias ? alias_idx : direct_idx;
			};

		private:
			/// Tuple unpacking constructor implementation
			template <class Tuple, std::size_t... I>
			static_list(std::index_sequence<I...>, Tuple&& arg_pack)
				: first_item_(std::get<I>(std::forward<Tuple>(arg_pack))...) {}

			/// Access an item by index implementation
			template <std::size_t idx, class = void>
			struct item_by_idx {
				static_assert(idx == 0, "jkl::strmnet::static_list: item access index out of range");
			};

			template <class dummy>
			struct item_by_idx<0, dummy> {
				template <class StaticList>
				static auto& get(StaticList* t) noexcept {
					return t->first_item_;
				}
			};

			/// Access an item by type implementation
			template <class QueryType>
			struct type_to_index_impl {
				template <bool has_tuple_elmt, class>
				struct impl;

				template <class Alias, class Type>
				struct impl_base {
					static constexpr bool found_as_alias = std::is_same<QueryType, Alias>::value;
					static constexpr bool found_directly = std::is_same<QueryType, Type>::value;
				};

				// If Descriptor is an item class
				template <class dummy>
				struct impl<false, dummy> :
					impl_base<first_item_type, first_item_type> {};

				// If Descriptor is a std::pair<Alias, Type> or similar
				template <class dummy>
				struct impl<true, dummy> :
					impl_base<std::tuple_element_t<0, Descriptor>, std::tuple_element_t<1, Descriptor>> {};

				static constexpr bool found_as_alias =
					impl<detail::has_alias<Descriptor>::value, void>::found_as_alias;
				static constexpr bool nonunique_as_alias = false;
				static constexpr std::size_t alias_idx = found_as_alias ? 0 : 1;
				static constexpr bool found_directly =
					impl<detail::has_alias<Descriptor>::value, void>::found_directly;
				static constexpr bool nonunique_directly = false;
				static constexpr std::size_t direct_idx = found_directly ? 0 : 1;

				static constexpr bool found = found_as_alias || found_directly;
			};
		};

		/// For the case of multiple items
		template <class FirstDescriptor, class... RemainingDescriptors>
		class static_list<FirstDescriptor, RemainingDescriptors...> {
			using first_item_type = get_item_type<FirstDescriptor>;
			first_item_type							first_item_;
			static_list<RemainingDescriptors...>	remaining_items_;

			template <std::size_t idx, class = void>
			struct element_type_impl {
				static_assert(idx <= sizeof...(RemainingDescriptors),
					"jkl::strmnet::static_list: element_type index out of range");
				using type = typename static_list<RemainingDescriptors...>::template element_type<idx - 1>;
			};

			template <class dummy>
			struct element_type_impl<0, dummy> {
				using type = first_item_type;
			};

			template <class QueryType>
			struct type_to_index_impl;

			struct tuple_unpack_tag {};

		public:
			static constexpr std::size_t number_of_items = sizeof...(RemainingDescriptors)+1;
			template <std::size_t idx>
			using element_type = typename element_type_impl<idx>::type;

			template <class... OtherDescriptors>
			friend class strmnet::static_list;

			/// Default constructor
			template <class = std::enable_if_t<
				std::is_default_constructible<first_item_type>::value &&
				std::is_default_constructible<static_list<RemainingDescriptors...>>::value>>
			static_list() {}

			/// Unpack and forward arguments packed in tuples
			template <class FirstTuple, class... OtherTuples,
				class = jkl::tmp::prevent_too_perfect_fwd<static_list, FirstTuple>,
				class = std::enable_if_t<!std::is_same<FirstTuple, tuple_unpack_tag>::value>>
			explicit static_list(FirstTuple&& first_arg_pack, OtherTuples&&... remaining_arg_packs)
				: static_list(tuple_unpack_tag{},
					std::make_index_sequence<std::tuple_size<std::remove_reference_t<FirstTuple>>::value>{},
					std::forward<FirstTuple>(first_arg_pack),
					std::forward<OtherTuples>(remaining_arg_packs)...) {}

			/// Access the first item
			auto&& first_item() & noexcept { return first_item_; }
			auto&& first_item() const& noexcept { return first_item_; }
			auto&& first_item() && noexcept { return std::move(first_item_); }
			auto&& first_item() const&& noexcept { return std::move(first_item_); }

			/// Access the last item
			auto&& last_item() & noexcept { return remaining_items_.last_item(); }
			auto&& last_item() const& noexcept { return remaining_items_.last_item(); }
			auto&& last_item() && noexcept { return std::move(remaining_items_.last_item()); }
			auto&& last_item() const&& noexcept { return std::move(remaining_items_.last_item()); }

			/// Access an item by index
			template <std::size_t idx>
			auto&& get() & noexcept { return item_by_idx<idx>::get(this); }
			template <std::size_t idx>
			auto&& get() const& noexcept { return item_by_idx<idx>::get(this); }
			template <std::size_t idx>
			auto&& get() && noexcept { return std::move(item_by_idx<idx>::get(this)); }
			template <std::size_t idx>
			auto&& get() const&& noexcept { return std::move(item_by_idx<idx>::get(this)); }

			/// Access an item by type
			/// First, find the specified type among aliases.
			///   If only one alias is found, return the corresponding item.
			///   If multiple aliases are found, generate an error message.
			/// (Classes with no alias specified are treated themselves as their aliases)
			/// If the specified type is not found among aliases, find it among item types themselves.
			///   If only one item type is found, return the corresponding item.
			///   Otherwise, generate an error message.
			template <class QueryType>
			auto&& get() & noexcept { return get<type_to_index<QueryType>::value>(); }
			template <class QueryType>
			auto&& get() const& noexcept { return get<type_to_index<QueryType>::value>(); }
			template <class QueryType>
			auto&& get() && noexcept { return std::move(get<type_to_index<QueryType>::value>()); }
			template <class QueryType>
			auto&& get() const&& noexcept { return std::move(get<type_to_index<QueryType>::value>()); }

			/// Perform f(item, additional_args) for each item in-order
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&& f, AdditionalArgs&&... additional_args) {
				for_each_impl(this, std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
			}
			template <class Functor, class... AdditionalArgs>
			void for_each(Functor&& f, AdditionalArgs&&... additional_args) const {
				for_each_impl(this, std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
			}

			/// Same, but in reverse-order
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&& f, AdditionalArgs&&... additional_args) {
				for_each_backward_impl(this, std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
			}
			template <class Functor, class... AdditionalArgs>
			void for_each_backward(Functor&& f, AdditionalArgs&&... additional_args) const {
				for_each_backward_impl(this, std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
			}

			/// Similar to for_each, but calls cleanup on already prepared items when exceptions are thrown
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&& p, Cleanup&& c, PrepareArgs&&... prepare_args) {
				prepare_each_impl(this, std::forward<Prepare>(p), std::forward<Cleanup>(c),
					std::forward<PrepareArgs>(prepare_args)...);
			}
			template <class Prepare, class Cleanup, class... PrepareArgs>
			void prepare_each(Prepare&& p, Cleanup&& c, PrepareArgs&&... prepare_args) const {
				prepare_each_impl(this, std::forward<Prepare>(p), std::forward<Cleanup>(c),
					std::forward<PrepareArgs>(prepare_args)...);
			}

			template <class QueryType>
			struct type_to_index {
				static constexpr bool found_as_alias = type_to_index_impl<QueryType>::found_as_alias;
				static constexpr bool nonunique_as_alias = type_to_index_impl<QueryType>::nonunique_as_alias;
				static constexpr std::size_t alias_idx = type_to_index_impl<QueryType>::alias_idx;
				static constexpr bool found_directly = type_to_index_impl<QueryType>::found_directly;
				static constexpr bool nonunique_directly = type_to_index_impl<QueryType>::nonunique_directly;
				static constexpr std::size_t direct_idx = type_to_index_impl<QueryType>::direct_idx;

				// Error case #1 - non-unique alias
				static_assert(!nonunique_as_alias, "jkl::strmnet::static_list: "
					"there are multiple items having the requested type as their aliases");

				// Error case #2 - not found as alias & non-unique item class
				static_assert(found_as_alias || !nonunique_directly, "jkl::strmnet::static_list: "
					"the requested type is not an alias, but there are multiple items of that type");

				// Error case #3 - no such an item
				static_assert(found_as_alias || found_directly, "jkl::strmnet::static_list: "
					"cannot find the requested item");

				static constexpr std::size_t value = found_as_alias ? alias_idx : direct_idx;
			};

		private:
			/// Tuple unpacking constructor implementation
			template <class FirstTuple, class... OtherTuples, std::size_t... I>
			static_list(tuple_unpack_tag, std::index_sequence<I...>,
				FirstTuple&& first_arg_pack, OtherTuples&&... remaining_arg_packs)
				: first_item_(std::get<I>(std::forward<FirstTuple>(first_arg_pack))...),
				remaining_items_(std::forward<OtherTuples>(remaining_arg_packs)...) {}

			/// Access an item by index implementation
			template <std::size_t idx, class = void>
			struct item_by_idx {
				static_assert(idx < number_of_items, "jkl::strmnet::static_list: item access index out of range");
				template <class StaticList>
				static auto& get(StaticList* t) noexcept {
					return t->remaining_items_.template get<idx - 1>();
				}
			};

			template <class dummy>
			struct item_by_idx<0, dummy> {
				template <class StaticList>
				static auto& get(StaticList* t) noexcept {
					return t->first_item_;
				}
			};

			/// Access an item by type implementation
			template <class QueryType>
			struct type_to_index_impl {
				template <bool has_tuple_elmt, class>
				struct impl;

				template <class Alias, class Type>
				struct impl_base {
					static constexpr bool found_as_alias_here = std::is_same<QueryType, Alias>::value;

					static constexpr bool found_as_alias_before =
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::found_as_alias;

					static constexpr bool found_as_alias = found_as_alias_here || found_as_alias_before;

					static constexpr bool nonunique_as_alias =
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::nonunique_as_alias ||
						(found_as_alias_here && found_as_alias_before);

					static constexpr std::size_t alias_idx = found_as_alias_here ? 0 :
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::alias_idx + 1;

					static constexpr bool found_directly_here = std::is_same<QueryType, Type>::value;

					static constexpr bool found_directly_before =
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::found_directly;

					static constexpr bool found_directly = found_directly_here || found_directly_before;

					static constexpr bool nonunique_directly =
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::nonunique_directly ||
						(found_directly_here && found_directly_before);

					static constexpr std::size_t direct_idx = found_directly_here ? 0 :
						static_list<RemainingDescriptors...>::template type_to_index_impl<QueryType>::direct_idx + 1;
				};

				// If FirstDescriptor is an item class
				template <class dummy>
				struct impl<false, dummy> :
					impl_base<first_item_type, first_item_type> {};

				// If FirstDescriptor is a std::pair<Alias, Type> or similar
				template <class dummy>
				struct impl<true, dummy> :
					impl_base<std::tuple_element_t<0, FirstDescriptor>, std::tuple_element_t<1, FirstDescriptor>> {};

				static constexpr bool found_as_alias =
					impl<detail::has_alias<FirstDescriptor>::value, void>::found_as_alias;
				static constexpr bool nonunique_as_alias =
					impl<detail::has_alias<FirstDescriptor>::value, void>::nonunique_as_alias;
				static constexpr std::size_t alias_idx =
					impl<detail::has_alias<FirstDescriptor>::value, void>::alias_idx;

				static constexpr bool found_directly =
					impl<detail::has_alias<FirstDescriptor>::value, void>::found_directly;
				static constexpr bool nonunique_directly =
					impl<detail::has_alias<FirstDescriptor>::value, void>::nonunique_directly;
				static constexpr std::size_t direct_idx =
					impl<detail::has_alias<FirstDescriptor>::value, void>::direct_idx;
			};

			template <class StaticList, class Functor, class... AdditionalArgs>
			static void for_each_impl(StaticList* t, Functor&& f, AdditionalArgs&&... additional_args) {
				f(t->first_item_, additional_args...);
				t->remaining_items_.for_each(std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
			}

			template <class StaticList, class Functor, class... AdditionalArgs>
			static void for_each_backward_impl(StaticList* t, Functor&& f, AdditionalArgs&&... additional_args) {
				t->remaining_items_.for_each_backward(std::forward<Functor>(f), std::forward<AdditionalArgs>(additional_args)...);
				f(t->first_item_, additional_args...);
			}

			template <class StaticList, class Prepare, class Cleanup, class... PrepareArgs>
			static void prepare_each_impl(StaticList* t, Prepare&& p, Cleanup&& c, PrepareArgs&&... prepare_args) {
				p(t->first_item_, std::forward<PrepareArgs>(prepare_args)...);				
				try {
					t->remaining_items_.prepare_each(std::forward<Prepare>(p), std::forward<Cleanup>(c),
						std::forward<PrepareArgs>(prepare_args)...);
				}
				catch( ... ) {
					c(t->first_item_);
					throw;
				}
			}
		};

		/// Stand-alone versions of get()
		template <std::size_t idx, class... Descriptors>
		auto&& get(static_list<Descriptors...>& t) {
			return t.template get<idx>();
		}
		template <std::size_t idx, class... Descriptors>
		auto&& get(static_list<Descriptors...> const& t) {
			return t.template get<idx>();
		}
		template <std::size_t idx, class... Descriptors>
		auto&& get(static_list<Descriptors...>&& t) {
			return std::move(t).template get<idx>();
		}
		template <std::size_t idx, class... Descriptors>
		auto&& get(static_list<Descriptors...> const&& t) {
			return std::move(t).template get<idx>();
		}
		/// Ditto
		template <class QueryType, class... Descriptors>
		auto&& get(static_list<Descriptors...>& t) {
			return t.template get<QueryType>();
		}
		template <class QueryType, class... Descriptors>
		auto&& get(static_list<Descriptors...> const& t) {
			return t.template get<QueryType>();
		}
		template <class QueryType, class... Descriptors>
		auto&& get(static_list<Descriptors...>&& t) {
			return std::move(t).template get<QueryType>();
		}
		template <class QueryType, class... Descriptors>
		auto&& get(static_list<Descriptors...> const&& t) {
			return std::move(t).template get<QueryType>();
		}
	}
}

/// Specialization of std::tuple_size and std::tuple_elment_t for jkl::strmnet::static_list
namespace std {
	template <class... Descriptors>
	class tuple_size<jkl::strmnet::static_list<Descriptors...>>
		: public std::integral_constant<std::size_t, sizeof...(Descriptors)> {};

	template <std::size_t idx, class... Descriptors>
	struct tuple_element<idx, jkl::strmnet::static_list<Descriptors...>> {
		using type = typename jkl::strmnet::static_list<Descriptors...>::template element_type<idx>;
	};
}

/// Tempalte utilities both for implementation & interface
namespace jkl {
	namespace strmnet {
		namespace detail {
			template <class Type, class TypeContainer>
			struct push_front_static_list;

			template <class Type, class... Types>
			struct push_front_static_list<Type, static_list<Types...>> {
				using type = static_list<Type, Types...>;
			};

			template <class List, std::size_t target_idx, class NewItem>
			struct replace_item {
			private:
				static constexpr std::size_t length = std::tuple_size<List>::value;

				template <std::size_t idx, bool target = (idx == target_idx), bool end = (idx == length)>
				struct impl {
					using type = typename push_front_static_list<
						std::tuple_element_t<idx, List>,
						typename impl<idx + 1>::type>::type;
				};

				template <std::size_t idx>
				struct impl<idx, true, false> {
					using type = typename push_front_static_list<
						NewItem,
						typename impl<idx + 1>::type>::type;
				};

				template <std::size_t idx>
				struct impl<idx, false, true> {
					using type = static_list<>;
				};

			public:
				using type = typename impl<0>::type;
			};
		}

		template <class StaticList, std::size_t idx, class NewAlias>
		using replace_alias = typename detail::replace_item<StaticList, idx,
			std::pair<NewAlias, get_item_type<std::tuple_element_t<idx, StaticList>>>>::type;
	}
}