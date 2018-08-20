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
#include "polar_decomposition.h"

namespace jkl {
	namespace math {
		namespace detail {
			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_elmt_base {
				using storage_type = Storage;
				using storage_traits = StorageTraits;

			protected:
				using storage_wrapper = typename StorageTraits::template storage_wrapper<
					Storage, sym3_elmt<ComponentType, Storage, StorageTraits>>;
				storage_wrapper r_;

			public:
				sym3_elmt_base() = default;

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr sym3_elmt_base(direct_construction, Args&&... args)
					noexcept(std::is_nothrow_constructible<storage_wrapper, Args...>::value) :
					r_(std::forward<Args>(args)...) {}

			protected:
				template <class... Args>
				struct is_nothrow_constructible {
					static constexpr bool value =
						tmp::is_nothrow_braces_constructible<storage_wrapper, Args...>::value;
				};

				template <class... Args>
				JKL_GPU_EXECUTABLE constexpr sym3_elmt_base(forward_to_storage_tag, Args&&... args)
					noexcept(is_nothrow_constructible<Args...>::value) :
					r_{ std::forward<Args>(args)... } {}
			};
		}

		// 3x3 symmetric matrix
		template <class ComponentType, class Storage, class StorageTraits>
		class sym3_elmt :
			public detail::constructor_provider<6, ComponentType,
			detail::sym3_elmt_base<ComponentType, Storage, StorageTraits>>
		{
			using constructor_provider = detail::constructor_provider<6, ComponentType,
				detail::sym3_elmt_base<ComponentType, Storage, StorageTraits>>;

		public:
			using constructor_provider::storage;
			using constructor_provider::constructor_provider;

			// Component access requirements
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<0, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the first component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<1, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the second component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<2, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the third component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<3, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the fourth component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<4, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the fifth component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");
			static_assert(detail::storage_traits_inspector<StorageTraits>::
				template can_get<5, Storage const&>::value,
				"jkl::math: sym3_elmt requires access to the sixth component from the storage; "
				"the given storage traits cannot find any way to make such an access from the given storage");

			// xx-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xx() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xx() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xx() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xx() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<0>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<0>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the first component from the storage; "
				"the first component deduced from the given storage traits cannot be converted to the component type");

			// xy-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xy() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xy() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xy() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xy() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<1>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<1>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the second component from the storage; "
				"the second component deduced from the given storage traits cannot be converted to the component type");

			// yx-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) yx() & noexcept(noexcept(xy()))
			{
				return xy();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) yx() const& noexcept(noexcept(xy()))
			{
				return xy();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) yx() && noexcept(noexcept(std::move(*this).xy()))
			{
				return std::move(*this).xy();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) yx() const&& noexcept(noexcept(std::move(*this).xy()))
			{
				return std::move(*this).xy();
			}

			// xz-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xz() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xz() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto xz() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto xz() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<2>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<2>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the third component from the storage; "
				"the third component deduced from the given storage traits cannot be converted to the component type");

			// zx-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zx() & noexcept(noexcept(xz()))
			{
				return xz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zx() const& noexcept(noexcept(xz()))
			{
				return xz();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zx() && noexcept(noexcept(std::move(*this).xz()))
			{
				return std::move(*this).xz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zx() const&& noexcept(noexcept(std::move(*this).xz()))
			{
				return std::move(*this).xz();
			}

			// yy-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto yy() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto yy() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto yy() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto yy() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<3>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<3>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the fourth component from the storage; "
				"the fourth component deduced from the given storage traits cannot be converted to the component type");

			// yz-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto yz() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<4>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<4>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<4>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto yz() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<4>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<4>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<4>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto yz() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto yz() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<4>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<4>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the fifth component from the storage; "
				"the fifth component deduced from the given storage traits cannot be converted to the component type");

			// zy-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zy() & noexcept(noexcept(yz()))
			{
				return yz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zy() const& noexcept(noexcept(xz()))
			{
				return yz();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zy() && noexcept(noexcept(std::move(*this).yz()))
			{
				return std::move(*this).yz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zy() const&& noexcept(noexcept(std::move(*this).yz()))
			{
				return std::move(*this).yz();
			}

			// zz-component accessors
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto zz() & noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<5>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<5>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<5>(storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto zz() const& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<5>(storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<5>(storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<5>(storage());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto zz() && noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage());
			}
			JKL_GPU_EXECUTABLE constexpr auto zz() const&& noexcept(noexcept(
				detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage())))
				-> decltype(detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage()))
			{
				return detail::storage_traits_inspector<StorageTraits>::template get<5>(std::move(*this).storage());
			}

			static_assert(std::is_convertible<
				decltype(detail::storage_traits_inspector<StorageTraits>::template get<5>(std::declval<Storage const&>())),
				ComponentType const&>::value, "jkl::math: sym3_elmt requires access to the sixth component from the storage; "
				"the sixth component deduced from the given storage traits cannot be converted to the component type");


			// Row accessors
		private:
			// Deduce row type as tuple
			// Store rvalues references as values, store lvalue references as references.
			template <class T>
			struct type_wrapper {
				using type = T;
			};
			template <class T>
			JKL_GPU_EXECUTABLE static constexpr auto get_component_type(T&&) {
				return type_wrapper<T>{};
			}

			template <class ThisType>
			struct row_type_base {
				using xx_wrapper = decltype(get_component_type(std::declval<ThisType>().xx()));
				using xy_wrapper = decltype(get_component_type(std::declval<ThisType>().xy()));
				using xz_wrapper = decltype(get_component_type(std::declval<ThisType>().xz()));
				using yy_wrapper = decltype(get_component_type(std::declval<ThisType>().yy()));
				using yz_wrapper = decltype(get_component_type(std::declval<ThisType>().yz()));
				using zz_wrapper = decltype(get_component_type(std::declval<ThisType>().zz()));

				using xx_type = typename xx_wrapper::type;
				using xy_type = typename xy_wrapper::type;
				using xz_type = typename xz_wrapper::type;
				using yy_type = typename yy_wrapper::type;
				using yz_type = typename yz_wrapper::type;
				using zz_type = typename zz_wrapper::type;
			};

			template <std::size_t I, class ThisType>
			struct row_type_helper;

			template <class ThisType>
			struct row_type_helper<0, ThisType> {
				using storage = detail::tuple<
					typename row_type_base<ThisType>::xx_type,
					typename row_type_base<ThisType>::xy_type,
					typename row_type_base<ThisType>::xz_type>;
				
				static constexpr bool is_noexcept =
					noexcept(std::declval<ThisType>().xx()) &&
					noexcept(std::declval<ThisType>().xy()) &&
					noexcept(std::declval<ThisType>().xz()) &&
					std::is_nothrow_constructible<storage,
					decltype(std::declval<ThisType>().xx()),
					decltype(std::declval<ThisType>().xy()),
					decltype(std::declval<ThisType>().xz())>::value;

				using type = R3_elmt<ComponentType, storage>;

				JKL_GPU_EXECUTABLE static constexpr type make(ThisType&& m) noexcept(is_noexcept) {
					return{ direct_construction{},
						std::forward<ThisType>(m).xx(),
						std::forward<ThisType>(m).xy(),
						std::forward<ThisType>(m).xz() };
				}
			};

			template <class ThisType>
			struct row_type_helper<1, ThisType> {
				using storage = detail::tuple<
					typename row_type_base<ThisType>::xy_type,
					typename row_type_base<ThisType>::yy_type,
					typename row_type_base<ThisType>::yz_type>;

				static constexpr bool is_noexcept =
					noexcept(std::declval<ThisType>().xy()) &&
					noexcept(std::declval<ThisType>().yy()) &&
					noexcept(std::declval<ThisType>().yz()) &&
					std::is_nothrow_constructible<storage,
					decltype(std::declval<ThisType>().xy()),
					decltype(std::declval<ThisType>().yy()),
					decltype(std::declval<ThisType>().yz())>::value;

				using type = R3_elmt<ComponentType, storage>;

				JKL_GPU_EXECUTABLE static constexpr type make(ThisType&& m) noexcept(is_noexcept) {
					return{ direct_construction{},
						std::forward<ThisType>(m).xy(),
						std::forward<ThisType>(m).yy(),
						std::forward<ThisType>(m).yz() };
				}
			};

			template <class ThisType>
			struct row_type_helper<2, ThisType> {
				using storage = detail::tuple<
					typename row_type_base<ThisType>::xz_type,
					typename row_type_base<ThisType>::yz_type,
					typename row_type_base<ThisType>::zz_type>;

				static constexpr bool is_noexcept =
					noexcept(std::declval<ThisType>().xz()) &&
					noexcept(std::declval<ThisType>().yz()) &&
					noexcept(std::declval<ThisType>().zz()) &&
					std::is_nothrow_constructible<storage,
					decltype(std::declval<ThisType>().xz()),
					decltype(std::declval<ThisType>().yz()),
					decltype(std::declval<ThisType>().zz())>::value;

				using type = R3_elmt<ComponentType, storage>;

				JKL_GPU_EXECUTABLE static constexpr type make(ThisType&& m) noexcept(is_noexcept) {
					return{ direct_construction{},
						std::forward<ThisType>(m).xz(),
						std::forward<ThisType>(m).yz(),
						std::forward<ThisType>(m).zz() };
				}
			};

			template <std::size_t I, class ThisType>
			using row_type = typename row_type_helper<I, ThisType>::type;

		public:
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR row_type<I, sym3_elmt&> get() &
				noexcept(row_type<I, sym3_elmt&>::is_noexcept)
			{
				return row_type<I, sym3_elmt&>::make(*this);
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr row_type<I, sym3_elmt const&> get() const&
				noexcept(row_type<I, sym3_elmt const&>::is_noexcept)
			{
				return row_type<I, sym3_elmt const&>::make(*this);
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR row_type<I, sym3_elmt&&> get() &&
				noexcept(row_type<I, sym3_elmt&&>::is_noexcept)
			{
				return row_type<I, sym3_elmt&&>::make(std::move(*this));
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr row_type<I, sym3_elmt const&&> get() const&&
				noexcept(row_type<I, sym3_elmt const&&>::is_noexcept)
			{
				return row_type<I, sym3_elmt const&&>::make(std::move(*this));
			}


			// xx-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(xx()))
				-> std::enable_if_t<I == 0 && J == 0, decltype(xx())>
			{
				return xx();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(xx()))
				-> std::enable_if_t<I == 0 && J == 0, decltype(xx())>
			{
				return xx();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).xx()))
				-> std::enable_if_t<I == 0 && J == 0, decltype(std::move(*this).xx())>
			{
				return std::move(*this).xx();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).xx()))
				-> std::enable_if_t<I == 0 && J == 0, decltype(std::move(*this).xx())>
			{
				return std::move(*this).xx();
			}

			// xy-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(xy()))
				-> std::enable_if_t<(I == 0 && J == 1) || (I == 1 && J == 0), decltype(xy())>
			{
				return xy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(xy()))
				-> std::enable_if_t<(I == 0 && J == 1) || (I == 1 && J == 0), decltype(xy())>
			{
				return xy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).xy()))
				-> std::enable_if_t<(I == 0 && J == 1) || (I == 1 && J == 0), decltype(std::move(*this).xy())>
			{
				return std::move(*this).xy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).xy()))
				-> std::enable_if_t<(I == 0 && J == 1) || (I == 1 && J == 0), decltype(std::move(*this).xy())>
			{
				return std::move(*this).xy();
			}

			// xz-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(xz()))
				-> std::enable_if_t<(I == 0 && J == 2) || (I == 2 && J == 0), decltype(xz())>
			{
				return xz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(xz()))
				-> std::enable_if_t<(I == 0 && J == 2) || (I == 2 && J == 0), decltype(xz())>
			{
				return xz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).xz()))
				-> std::enable_if_t<(I == 0 && J == 2) || (I == 2 && J == 0), decltype(std::move(*this).xz())>
			{
				return std::move(*this).xz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).xz()))
				-> std::enable_if_t<(I == 0 && J == 2) || (I == 2 && J == 0), decltype(std::move(*this).xz())>
			{
				return std::move(*this).xz();
			}

			// yy-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(yy()))
				-> std::enable_if_t<I == 1 && J == 1, decltype(yy())>
			{
				return yy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(yy()))
				-> std::enable_if_t<I == 1 && J == 1, decltype(yy())>
			{
				return yy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).yy()))
				-> std::enable_if_t<I == 1 && J == 1, decltype(std::move(*this).yy())>
			{
				return std::move(*this).yy();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).yy()))
				-> std::enable_if_t<I == 1 && J == 1, decltype(std::move(*this).yy())>
			{
				return std::move(*this).yy();
			}

			// yz-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(yz()))
				-> std::enable_if_t<(I == 1 && J == 2) || (I == 2 && J == 1), decltype(yz())>
			{
				return yz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(yz()))
				-> std::enable_if_t<(I == 1 && J == 2) || (I == 2 && J == 1), decltype(yz())>
			{
				return yz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).yz()))
				->std::enable_if_t<(I == 1 && J == 2) || (I == 2 && J == 1), decltype(std::move(*this).yz())>
			{
				return std::move(*this).yz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).yz()))
				->std::enable_if_t<(I == 1 && J == 2) || (I == 2 && J == 1), decltype(std::move(*this).yz())>
			{
				return std::move(*this).yz();
			}

			// zz-component accessors
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() & noexcept(noexcept(zz()))
				-> std::enable_if_t<I == 2 && J == 2, decltype(zz())>
			{
				return zz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const& noexcept(noexcept(zz()))
				-> std::enable_if_t<I == 2 && J == 2, decltype(zz())>
			{
				return zz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto get() && noexcept(noexcept(std::move(*this).zz()))
				->std::enable_if_t<I == 2 && J == 2, decltype(std::move(*this).zz())>
			{
				return std::move(*this).zz();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr auto get() const&& noexcept(noexcept(std::move(*this).zz()))
				->std::enable_if_t<I == 2 && J == 2, decltype(std::move(*this).zz())>
			{
				return std::move(*this).zz();
			}


			// Default constructor; components might be filled with garbages
			sym3_elmt() = default;
									
			// Convert from sym3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<sym3_elmt,
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(constructor_provider(that.xx(), that.xy(), that.xz(),
					that.yy(), that.yz(), that.zz()))) :
				constructor_provider(that.xx(), that.xy(), that.xz(),
					that.yy(), that.yz(), that.zz()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<sym3_elmt,
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(constructor_provider(
					std::move(that).xx(), std::move(that).xy(), std::move(that).xz(),
					std::move(that).yy(), std::move(that).yz(), std::move(that).zz()))) :
				constructor_provider(std::move(that).xx(), std::move(that).xy(), std::move(that).xz(),
					std::move(that).yy(), std::move(that).yz(), std::move(that).zz()) {}

			// Convert from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(noexcept(constructor_provider(
					that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>(),
					that.template get<1, 1>(), that.template get<1, 2>(), that.template get<2, 2>()))) :
				constructor_provider(that.template get<0, 0>(), that.template get<0, 1>(), that.template get<0, 2>(),
					that.template get<1, 1>(), that.template get<1, 2>(), that.template get<2, 2>()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(noexcept(constructor_provider(std::move(that).template get<0, 0>(),
					std::move(that).template get<0, 1>(), std::move(that).template get<0, 2>(),
					std::move(that).template get<1, 1>(), std::move(that).template get<1, 2>(),
					std::move(that).template get<2, 2>()))) :
				constructor_provider(std::move(that).template get<0, 0>(),
					std::move(that).template get<0, 1>(), std::move(that).template get<0, 2>(),
					std::move(that).template get<1, 1>(), std::move(that).template get<1, 2>(),
					std::move(that).template get<2, 2>()) {}

		private:
			template <class Matrix>
			static constexpr constructor_provider check_and_forward(Matrix&& m) {
				return m.is_symmetric() ?
					constructor_provider{ std::forward<Matrix>(m).template get<0, 0>(),
					std::forward<Matrix>(m).template get<0, 1>(), std::forward<Matrix>(m).template get<0, 2>(),
					std::forward<Matrix>(m).template get<1, 1>(), std::forward<Matrix>(m).template get<1, 2>(),
					std::forward<Matrix>(m).template get<2, 2>() } :
					throw input_validity_error<sym3_elmt>{ "jkl::math: the matrix is not symmetric" };
			}

		public:
			// Convert from gl3_elmt of other component type (throwing)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr sym3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				constructor_provider(check_and_forward(that)) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr sym3_elmt(gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				constructor_provider(check_and_forward(std::move(that))) {}

			// Convert from gl3_elmt of other component type (symmetrization)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				auto_reform)
				noexcept(noexcept(constructor_provider(that.template get<0, 0>(),
				(that.template get<0, 1>() + that.template get<1, 0>()) / 2,
					(that.template get<0, 2>() + that.template get<2, 0>()) / 2,
					that.template get<1, 1>(),
					(that.template get<1, 2>() + that.template get<2, 1>()) / 2,
					that.template get<1, 1>()))) :
				constructor_provider(that.template get<0, 0>(),
				(that.template get<0, 1>() + that.template get<1, 0>()) / 2,
				(that.template get<0, 2>() + that.template get<2, 0>()) / 2,
					that.template get<1, 1>(),
					(that.template get<1, 2>() + that.template get<2, 1>()) / 2,
					that.template get<1, 1>()) {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr sym3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				auto_reform)
				noexcept(noexcept(constructor_provider(std::move(that).template get<0, 0>(),
				(std::move(that).template get<0, 1>() + std::move(that).template get<1, 0>()) / 2,
					(std::move(that).template get<0, 2>() + std::move(that).template get<2, 0>()) / 2,
					std::move(that).template get<1, 1>(),
					(std::move(that).template get<1, 2>() + std::move(that).template get<2, 1>()) / 2,
					std::move(that).template get<2, 2>()))) :
				constructor_provider(std::move(that).template get<0, 0>(),
					(std::move(that).template get<0, 1>() + std::move(that).template get<1, 0>()) / 2,
					(std::move(that).template get<0, 2>() + std::move(that).template get<2, 0>()) / 2,
					std::move(that).template get<1, 1>(),
					(std::move(that).template get<1, 2>() + std::move(that).template get<2, 1>()) / 2,
					std::move(that).template get<2, 2>()) {}
			

			// Copy and move
			sym3_elmt(sym3_elmt const&) = default;
			sym3_elmt(sym3_elmt&&) = default;
			sym3_elmt& operator=(sym3_elmt const&) & = default;
			sym3_elmt& operator=(sym3_elmt&&) & = default;

			// Assignment from sym3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<sym3_elmt,
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(xx() = that.xx()) &&
					noexcept(xy() = that.xy()) &&
					noexcept(xz() = that.xz()) &&
					noexcept(yy() = that.yy()) &&
					noexcept(yz() = that.yz()) &&
					noexcept(zz() = that.zz()))
			{
				xx() = that.xx();
				xy() = that.xy();
				xz() = that.xz();
				yy() = that.yy();
				yz() = that.yz();
				zz() = that.zz();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<sym3_elmt,
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(xx() = std::move(that).xx()) &&
					noexcept(xy() = std::move(that).xy()) &&
					noexcept(xz() = std::move(that).xz()) &&
					noexcept(yy() = std::move(that).yy()) &&
					noexcept(yz() = std::move(that).yz()) &&
					noexcept(zz() = std::move(that).zz()))
			{
				xx() = std::move(that).xx();
				xy() = std::move(that).xy();
				xz() = std::move(that).xz();
				yy() = std::move(that).yy();
				yz() = std::move(that).yz();
				zz() = std::move(that).zz();
				return *this;
			}

			// Assignment from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(xx() = that.template get<0, 0>()) &&
					noexcept(xy() = that.template get<0, 1>()) &&
					noexcept(xz() = that.template get<0, 2>()) &&
					noexcept(yy() = that.template get<1, 1>()) &&
					noexcept(yz() = that.template get<1, 2>()) &&
					noexcept(zz() = that.template get<2, 2>()))
			{
				xx() = that.template get<0, 0>();
				xy() = that.template get<0, 1>();
				xz() = that.template get<0, 2>();
				yy() = that.template get<1, 1>();
				yz() = that.template get<1, 2>();
				zz() = that.template get<2, 2>();
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(xx() = std::move(that).template get<0, 0>()) &&
					noexcept(xy() = std::move(that).template get<0, 1>()) &&
					noexcept(xz() = std::move(that).template get<0, 2>()) &&
					noexcept(yy() = std::move(that).template get<1, 1>()) &&
					noexcept(yz() = std::move(that).template get<1, 2>()) &&
					noexcept(zz() = std::move(that).template get<2, 2>()))
			{
				xx() = std::move(that).template get<0, 0>();
				xy() = std::move(that).template get<0, 1>();
				xz() = std::move(that).template get<0, 2>();
				yy() = std::move(that).template get<1, 1>();
				yz() = std::move(that).template get<1, 2>();
				zz() = std::move(that).template get<2, 2>();
				return *this;
			}

			// Assignment from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR sym3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_symmetric() )
					throw input_validity_error<sym3_elmt>{ "jkl::math: the matrix is not symmetric" };
				return assign_no_check(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR sym3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_symmetric() )
					throw input_validity_error<sym3_elmt>{ "jkl::math: the matrix is not symmetric" };
				return assign_no_check(std::move(that));
			}			


		private:
			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr decltype(auto) det_impl(Matrix&& m)
				noexcept(noexcept(
					std::forward<Matrix>(m).template get<0, 0>() * (
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 2>() -
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 1>()) +
					std::forward<Matrix>(m).template get<0, 1>() * (
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 0>() -
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 2>()) +
					std::forward<Matrix>(m).template get<0, 2>() * (
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 1>() -
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 0>())))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (
					std::forward<Matrix>(m).template get<0, 0>() * (
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 2>() -
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 1>()) +
					std::forward<Matrix>(m).template get<0, 1>() * (
						std::forward<Matrix>(m).template get<1, 2>() *
						std::forward<Matrix>(m).template get<2, 0>() -
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 2>()) +
					std::forward<Matrix>(m).template get<0, 2>() * (
						std::forward<Matrix>(m).template get<1, 0>() *
						std::forward<Matrix>(m).template get<2, 1>() -
						std::forward<Matrix>(m).template get<1, 1>() *
						std::forward<Matrix>(m).template get<2, 0>()));
			}

			template <class Matrix>
			JKL_GPU_EXECUTABLE static constexpr decltype(auto) trace_impl(Matrix&& m)
				noexcept(noexcept(
					std::forward<Matrix>(m).template get<0, 0>() +
					std::forward<Matrix>(m).template get<1, 1>() +
					std::forward<Matrix>(m).template get<2, 2>()))
			{
				// MSVC2015 has a bug giving warning C4552 when decltype(auto) is the return type.
				// To workaround this bug, the expressions are wrapped with parantheses
				return (
					std::forward<Matrix>(m).template get<0, 0>() +
					std::forward<Matrix>(m).template get<1, 1>() +
					std::forward<Matrix>(m).template get<2, 2>());
			}

		public:
			JKL_GPU_EXECUTABLE constexpr decltype(auto) det() const&
				noexcept(noexcept(det_impl(*this)))
			{
				return det_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) det() &&
				noexcept(noexcept(det_impl(std::move(*this))))
			{
				return det_impl(std::move(*this));
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) trace() const&
				noexcept(noexcept(trace_impl(*this)))
			{
				return trace_impl(*this);
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) trace() &&
				noexcept(noexcept(trace_impl(std::move(*this))))
			{
				return trace_impl(std::move(*this));
			}


			JKL_GPU_EXECUTABLE constexpr sym3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<sym3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR sym3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<sym3_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr sym3_elmt operator-() const&
				noexcept(noexcept(sym3_elmt{ -xx(), -xy(), -xz(), -yy(), -yz(), -zz() }))
			{
				return{ -xx(), -xy(), -xz(), -yy(), -yz(), -zz() };
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR sym3_elmt operator-() && noexcept(noexcept(sym3_elmt{
				-std::move(*this).xx(),
				-std::move(*this).xy(),
				-std::move(*this).xz(),
				-std::move(*this).yy(),
				-std::move(*this).yz(),
				-std::move(*this).zz() }))
			{
				return{ -std::move(*this).xx(), -std::move(*this).xy(), -std::move(*this).xz(),
					-std::move(*this).yy(), -std::move(*this).yz(), -std::move(*this).zz() };
			}

		private:
			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR auto& inplace_add_impl(OtherMatrix&& m) noexcept(
				noexcept(xx() += std::forward<OtherMatrix>(m).xx()) &&
				noexcept(xy() += std::forward<OtherMatrix>(m).xy()) &&
				noexcept(xz() += std::forward<OtherMatrix>(m).xz()) &&
				noexcept(yy() += std::forward<OtherMatrix>(m).yy()) &&
				noexcept(yz() += std::forward<OtherMatrix>(m).yz()) &&
				noexcept(zz() += std::forward<OtherMatrix>(m).zz()))
			{
				xx() += std::forward<OtherMatrix>(m).xx();
				xy() += std::forward<OtherMatrix>(m).xy();
				xz() += std::forward<OtherMatrix>(m).xz();
				yy() += std::forward<OtherMatrix>(m).yy();
				yz() += std::forward<OtherMatrix>(m).yz();
				zz() += std::forward<OtherMatrix>(m).zz();
				return *this;
			}

			template <class OtherMatrix>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR auto& inplace_sub_impl(OtherMatrix&& m) noexcept(
				noexcept(xx() -= std::forward<OtherMatrix>(m).xx()) &&
				noexcept(xy() -= std::forward<OtherMatrix>(m).xy()) &&
				noexcept(xz() -= std::forward<OtherMatrix>(m).xz()) &&
				noexcept(yy() -= std::forward<OtherMatrix>(m).yy()) &&
				noexcept(yz() -= std::forward<OtherMatrix>(m).yz()) &&
				noexcept(zz() -= std::forward<OtherMatrix>(m).zz()))
			{
				xx() -= std::forward<OtherMatrix>(m).xx();
				xy() -= std::forward<OtherMatrix>(m).xy();
				xz() -= std::forward<OtherMatrix>(m).xz();
				yy() -= std::forward<OtherMatrix>(m).yy();
				yz() -= std::forward<OtherMatrix>(m).yz();
				zz() -= std::forward<OtherMatrix>(m).zz();
				return *this;
			}

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator+=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_add_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator+=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_add_impl(std::move(that));
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator-=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(inplace_add_impl(that)))
			{
				return inplace_sub_impl(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator-=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(inplace_add_impl(std::move(that))))
			{
				return inplace_sub_impl(std::move(that));
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator*=(OtherComponentType const& k) noexcept(
				noexcept(xx() *= k) &&
				noexcept(xy() *= k) &&
				noexcept(xz() *= k) &&
				noexcept(yy() *= k) &&
				noexcept(yz() *= k) &&
				noexcept(zz() *= k))
			{
				xx() *= k;
				xy() *= k;
				xz() *= k;
				yy() *= k;
				yz() *= k;
				zz() *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt& operator/=(OtherComponentType const& k) noexcept(
				noexcept(xx() /= k) &&
				noexcept(xy() /= k) &&
				noexcept(xz() /= k) &&
				noexcept(yy() /= k) &&
				noexcept(yz() /= k) &&
				noexcept(zz() /= k))
			{
				xx() /= k;
				xy() /= k;
				xz() /= k;
				yy() /= k;
				yz() /= k;
				zz() /= k;
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr sym3_elmt t() const&
				noexcept(std::is_nothrow_copy_constructible<sym3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR sym3_elmt t() &&
				noexcept(std::is_nothrow_move_constructible<sym3_elmt>::value)
			{
				return std::move(*this);
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const noexcept(
					noexcept(xx() == that.xx()) &&
					noexcept(xy() == that.xy()) &&
					noexcept(xz() == that.xz()) &&
					noexcept(yy() == that.yy()) &&
					noexcept(yz() == that.yz()) &&
					noexcept(zz() == that.zz()))
			{
				return xx() == that.xx()
					&& xy() == that.xy()
					&& xz() == that.xz()
					&& yy() == that.yy()
					&& yz() == that.yz()
					&& zz() == that.zz();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const
				noexcept(noexcept((*this) == that))
			{
				return !((*this) == that);
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator==(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const noexcept(
					noexcept(xx() == that.template get<0, 0>()
						&& xy() == that.template get<0, 1>()
						&& xz() == that.template get<0, 2>()
						&& yx() == that.template get<1, 0>()
						&& yy() == that.template get<1, 1>()
						&& yz() == that.template get<1, 2>()
						&& zx() == that.template get<2, 0>()
						&& zy() == that.template get<2, 1>()
						&& zz() == that.template get<2, 2>()))
			{
				return xx() == that.template get<0, 0>()
					&& xy() == that.template get<0, 1>()
					&& xz() == that.template get<0, 2>()
					&& yx() == that.template get<1, 0>()
					&& yy() == that.template get<1, 1>()
					&& yz() == that.template get<1, 2>()
					&& zx() == that.template get<2, 0>()
					&& zy() == that.template get<2, 1>()
					&& zz() == that.template get<2, 2>();
			}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) const
				noexcept(noexcept((*this) == that))
			{
				return !((*this) == that);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const&
				noexcept(noexcept(jkl::math::is_invertible(det())))
			{
				return jkl::math::is_invertible(det());
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR bool is_invertible() &&
				noexcept(noexcept(jkl::math::is_invertible(std::move(*this).det())))
			{
				return jkl::math::is_invertible(std::move(*this).det());
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept(noexcept(
				close_to_one(get<0>().normsq()) &&
				close_to_one(get<1>().normsq()) &&
				close_to_one(get<2>().normsq()) &&
				close_to_zero(dot(get<0>(), get<1>())) &&
				close_to_zero(dot(get<1>(), get<2>())) &&
				close_to_zero(dot(get<2>(), get<0>()))))
			{
				return
					close_to_one(get<0>().normsq()) &&
					close_to_one(get<1>().normsq()) &&
					close_to_one(get<2>().normsq()) &&
					close_to_zero(dot(get<0>(), get<1>())) &&
					close_to_zero(dot(get<1>(), get<2>())) &&
					close_to_zero(dot(get<2>(), get<0>()));
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const
				noexcept(noexcept(det() > jkl::math::zero<ComponentType>() && is_orthogonal()))
			{
				return det() > jkl::math::zero<ComponentType>() && is_orthogonal();
			}

			JKL_GPU_EXECUTABLE constexpr bool is_symmetric() const noexcept
			{
				return true;
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const
				noexcept(noexcept(xx() > jkl::math::zero<ComponentType>() &&
					xx() * yy() > xy() * xy() &&
					det() > jkl::math::zero<ComponentType>()))
			{
				return xx() > jkl::math::zero<ComponentType>() &&
					xx() * yy() > xy() * xy() &&
					det() > jkl::math::zero<ComponentType>();
			}

			JKL_GPU_EXECUTABLE static constexpr sym3_elmt zero()
				noexcept(std::is_nothrow_constructible<sym3_elmt,
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>()),
					decltype(jkl::math::zero<ComponentType>())>::value)
			{
				return{ jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>(),
					jkl::math::zero<ComponentType>() };
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.det()))
		{
			return m.det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) det(sym3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).det()))
		{
			return std::move(m).det();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.trace()))
		{
			return m.trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) trace(sym3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).trace()))
		{
			return std::move(m).trace();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(sym3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_orthogonal(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_orthogonal()))
		{
			return m.is_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_special_orthogonal()))
		{
			return m.is_special_orthogonal();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_symmetric(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_symmetric()))
		{
			return m.is_symmetric();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr bool is_positive_definite(sym3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.is_positive_definite()))
		{
			return m.is_positive_definite();
		}

		// 3x3 positive definite matrix
		namespace detail {
			// To suppress generation of inherited constructors
			template <class ComponentType, class Storage, class StorageTraits>
			struct posdef3_elmt_base : sym3_elmt<ComponentType, Storage, StorageTraits> {
				posdef3_elmt_base() = default;

			private:
				using base_type = sym3_elmt<ComponentType, Storage, StorageTraits>;
				using target_type = posdef3_elmt<ComponentType, Storage, StorageTraits>;

				template <class Matrix>
				static constexpr base_type sym3_check_and_forward(Matrix&& m) {
					return m.is_positive_definite() ? std::forward<Matrix>(m) :
						throw input_validity_error<target_type>{ "jkl::math: the matrix is not positive-definite" };
				}

				template <class Matrix>
				static constexpr base_type gl3_check_and_forward(Matrix&& m) {
					return m.is_positive_definite() ? base_type{ std::forward<Matrix>(m), no_validity_check{} } :
						throw input_validity_error<target_type>{ "jkl::math: the matrix is not positive-definite" };
				}

			protected:
				// [NOTE]
				// I think it would be a bad idea to add assert() inside no-check constructors,
				// because the main cause of the break of invertibility is the accumulation of
				// floating-point operation errors. It would be frequently OK if the invertibility
				// condition is just "broken slightly" by such an error, and
				// those spurious failures indeed occur quite occasionally.
				// Or even "not-very-slight-violations" might sometimes be OK as well.
				// Adding assert() may make the class too strict.
				
				template <class... Args>
				struct is_nothrow_constructible : std::false_type {};
				
				template <class ArgXX, class ArgXY, class ArgXZ, class ArgYY, class ArgYZ, class ArgZZ>
				struct is_nothrow_constructible<ArgXX, ArgXY, ArgXZ, ArgYY, ArgYZ, ArgZZ, no_validity_check>
				{
					static constexpr bool value =
						std::is_nothrow_constructible<base_type,
						ArgXX, ArgXY, ArgXZ, ArgYY, ArgYZ, ArgZZ>::value;
				};

				template <class Matrix>
				struct is_nothrow_constructible<Matrix, no_validity_check>
				{
					static constexpr bool value =
						std::is_nothrow_constructible<base_type, Matrix>::value;
				};

				// No check component-wise constructor
				template <class ArgXX, class ArgXY, class ArgXZ, class ArgYY, class ArgYZ, class ArgZZ>
				JKL_GPU_EXECUTABLE constexpr posdef3_elmt_base(forward_to_storage_tag,
					ArgXX&& xx, ArgXY&& xy, ArgXZ&& xz, ArgYY&& yy, ArgYZ&& yz, ArgZZ&& zz,
					no_validity_check)
					noexcept(noexcept(base_type{
					std::forward<ArgXX>(xx), std::forward<ArgXY>(xy), std::forward<ArgXZ>(xz),
					std::forward<ArgYY>(yy), std::forward<ArgYZ>(yz), std::forward<ArgZZ>(zz) })) :
					base_type{ std::forward<ArgXX>(xx), std::forward<ArgXY>(xy), std::forward<ArgXZ>(xz),
					std::forward<ArgYY>(yy), std::forward<ArgYZ>(yz), std::forward<ArgZZ>(zz) } {}

				// Checking component-wise constructor
				template <class ArgXX, class ArgXY, class ArgXZ, class ArgYY, class ArgYZ, class ArgZZ>
				constexpr posdef3_elmt_base(forward_to_storage_tag,
					ArgXX&& xx, ArgXY&& xy, ArgXZ&& xz, ArgYY&& yy, ArgYZ&& yz, ArgZZ&& zz) :
					base_type{ sym3_check_and_forward(sym3_elmt<ComponentType,
						detail::tuple<ArgXX&&, ArgXY&&, ArgXZ&&, ArgYY&&, ArgYZ&&, ArgZZ&&>>{
					std::forward<ArgXX>(xx), std::forward<ArgXY>(xy), std::forward<ArgXZ>(xz),
						std::forward<ArgYY>(yy), std::forward<ArgYZ>(yz), std::forward<ArgZZ>(zz) }) } {}

				// No check gl3_elmt constructor
				template <class Matrix>
				JKL_GPU_EXECUTABLE constexpr posdef3_elmt_base(forward_to_storage_tag,
					Matrix&& m, no_validity_check)
					noexcept(noexcept(base_type{ std::forward<Matrix>(m), no_validity_check{} })) :
					base_type{ std::forward<Matrix>(m), no_validity_check{} } {}

				// Checking gl3_elmt constructor
				template <class Matrix>
				constexpr posdef3_elmt_base(forward_to_storage_tag, Matrix&& m) :
					base_type{ gl3_check_and_forward(std::forward<Matrix>(m)) } {}

				// Polar decomposition constructor
				template <class Matrix>
				CONSTEXPR_LAMBDA posdef3_elmt_base(forward_to_storage_tag,
					Matrix&& m, auto_reform) : base_type{
						polar_positive_part<ComponentType, Storage, StorageTraits>(std::forward<Matrix>(m)) } {}

			};
		}

		template <class ComponentType, class Storage, class StorageTraits>
		class posdef3_elmt : public tmp::generate_constructors<
			detail::posdef3_elmt_base<ComponentType, Storage, StorageTraits>,
			detail::forward_to_storage_tag,
			tmp::copy_or_move_n<ComponentType, 6>,
			tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 6>,
			std::tuple<std::tuple<no_validity_check>>>>
		{
			using base_type = tmp::generate_constructors<
				detail::posdef3_elmt_base<ComponentType, Storage, StorageTraits>,
				detail::forward_to_storage_tag,
				tmp::copy_or_move_n<ComponentType, 6>,
				tmp::concat_tuples<tmp::copy_or_move_n<ComponentType, 6>,
				std::tuple<std::tuple<no_validity_check>>>>;

			using sym3_elmt_type = sym3_elmt<ComponentType, Storage, StorageTraits>;

		public:
			using base_type::base_type;

			// Initialize to the unity
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt() noexcept(noexcept(base_type{
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(),
				no_validity_check{} })) : 
				base_type{ jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(),
				jkl::math::zero<ComponentType>(),
				jkl::math::unity<ComponentType>(), no_validity_check{} } {}
			
			// Convert from posdef3_elmt of other element type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<posdef3_elmt,
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ that.xx(), that.xy(), that.xz(),
				that.yy(), that.yz(), that.zz(), no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<posdef3_elmt,
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ std::move(that).xx(), std::move(that).xy(), std::move(that).xz(),
				std::move(that).yy(), std::move(that).yz(), std::move(that).zz(), no_validity_check{} } {}

			// Convert from sym3_elmt of other element type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&>::value) :
				base_type{ that.xx(), that.xy(), that.xz(),
				that.yy(), that.yz(), that.zz(), no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check)
				noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value) :
				base_type{ std::move(that).xx(), std::move(that).xy(), std::move(that).xz(),
				std::move(that).yy(), std::move(that).yz(), std::move(that).zz(), no_validity_check{} } {}

			// Convert from sym3_elmt of other element type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				base_type{ that.xx(), that.xy(), that.xz(), that.yy(), that.yz(), that.zz() } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				base_type{ std::move(that).xx(), std::move(that).xy(), std::move(that).xz(),
				std::move(that).yy(), std::move(that).yz(), std::move(that).zz() } {}

			// Convert from gl3_elmt of other element type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				no_validity_check) noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const&,
					no_validity_check>::value) :
				base_type{ detail::forward_to_storage_tag{}, that, no_validity_check{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			JKL_GPU_EXECUTABLE constexpr posdef3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				no_validity_check) noexcept(std::is_nothrow_constructible<sym3_elmt_type,
					gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>,
					no_validity_check>::value) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), no_validity_check{} } {}

			// Convert from gl3_elmt of other element type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr posdef3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) :
				base_type{ detail::forward_to_storage_tag{}, that } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			constexpr posdef3_elmt(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that) } {}
			
			// Convert from GL3_elmt of other element type (polar decomposition)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			CONSTEXPR_LAMBDA posdef3_elmt(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that,
				auto_reform) :
				base_type{ detail::forward_to_storage_tag{}, that, auto_reform{} } {}

			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_convertible<OtherComponentType, ComponentType>::value>>
			CONSTEXPR_LAMBDA posdef3_elmt(
				GL3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that,
				auto_reform) :
				base_type{ detail::forward_to_storage_tag{}, std::move(that), auto_reform{} } {}


			// Copy and move
			posdef3_elmt(posdef3_elmt const&) = default;
			posdef3_elmt(posdef3_elmt&&) = default;
			posdef3_elmt& operator=(posdef3_elmt const&) & = default;
			posdef3_elmt& operator=(posdef3_elmt&&) & = default;
			
			// Assignment from posdef3_elmt of other component type
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<posdef3_elmt,
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) = that))
			{
				static_cast<sym3_elmt_type&>(*this) = that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<!std::is_same<posdef3_elmt,
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>>::value &&
				std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) = std::move(that)))
			{
				static_cast<sym3_elmt_type&>(*this) = std::move(that);
				return *this;
			}
			
			// Assignment from sym3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& assign_no_check(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) = that))
			{
				static_cast<sym3_elmt_type&>(*this) = that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& assign_no_check(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) = std::move(that)))
			{
				static_cast<sym3_elmt_type&>(*this) = std::move(that);
				return *this;
			}

			// Assignment from sym3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_positive_definite() )
					throw input_validity_error<posdef3_elmt>{ "jkl::math: the matrix is not positive-definite" };
				static_cast<sym3_elmt_type&>(*this) = that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_positive_definite() )
					throw input_validity_error<posdef3_elmt>{ "jkl::math: the matrix is not positive-definite" };
				static_cast<sym3_elmt_type&>(*this) = std::move(that);
				return *this;
			}

			// Assignment from gl3_elmt of other component type (no check)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
				noexcept(noexcept(sym3_elmt_type::assign_no_check(that)))
			{
				sym3_elmt_type::assign_no_check(that);
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& assign_no_check(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
				noexcept(noexcept(sym3_elmt_type::assign_no_check(std::move(that))))
			{
				sym3_elmt_type::assign_no_check(std::move(that));
				return *this;
			}

			// Assignment from gl3_elmt of other component type (checking)
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType const&>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that) &
			{
				if( !that.is_positive_definite() )
					throw input_validity_error<posdef3_elmt>{ "jkl::math: the matrix is not positive-definite" };
				return assign_no_check(that);
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits,
				class = std::enable_if_t<std::is_assignable<ComponentType&, OtherComponentType>::value>>
			GENERALIZED_CONSTEXPR posdef3_elmt& operator=(
				gl3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that) &
			{
				if( !that.is_positive_definite() )
					throw input_validity_error<posdef3_elmt>{ "jkl::math: the matrix is not positive-definite" };
				return assign_no_check(std::move(that));
			}


			// Remove mutable lvalue element accessors
			JKL_GPU_EXECUTABLE constexpr decltype(auto) xx() const&
				noexcept(noexcept(std::declval<base_type const&>().xx()))
			{
				return static_cast<base_type const&>(*this).xx();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) xx() &&
				noexcept(noexcept(std::declval<base_type&&>().xx()))
			{
				return static_cast<base_type&&>(*this).xx();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) xx() const&&
				noexcept(noexcept(std::declval<base_type const&&>().xx()))
			{
				return static_cast<base_type const&&>(*this).xx();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) xy() const&
				noexcept(noexcept(std::declval<base_type const&>().xy()))
			{
				return static_cast<base_type const&>(*this).xy();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) xy() &&
				noexcept(noexcept(std::declval<base_type&&>().xy()))
			{
				return static_cast<base_type&&>(*this).xy();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) xy() const&&
				noexcept(noexcept(std::declval<base_type const&&>().xy()))
			{
				return static_cast<base_type const&&>(*this).xy();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) xz() const&
				noexcept(noexcept(std::declval<base_type const&>().xz()))
			{
				return static_cast<base_type const&>(*this).xz();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) xz() &&
				noexcept(noexcept(std::declval<base_type&&>().xz()))
			{
				return static_cast<base_type&&>(*this).xz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) xz() const&&
				noexcept(noexcept(std::declval<base_type const&&>().xz()))
			{
				return static_cast<base_type const&&>(*this).xz();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) yx() const&
				noexcept(noexcept(std::declval<base_type const&>().yx()))
			{
				return static_cast<base_type const&>(*this).yx();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) yx() &&
				noexcept(noexcept(std::declval<base_type&&>().yx()))
			{
				return static_cast<base_type&&>(*this).yx();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) yx() const&&
				noexcept(noexcept(std::declval<base_type const&&>().yx()))
			{
				return static_cast<base_type const&&>(*this).yx();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) yy() const&
				noexcept(noexcept(std::declval<base_type const&>().yy()))
			{
				return static_cast<base_type const&>(*this).yy();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) yy() &&
				noexcept(noexcept(std::declval<base_type&&>().yy()))
			{
				return static_cast<base_type&&>(*this).yy();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) yy() const&&
				noexcept(noexcept(std::declval<base_type const&&>().yy()))
			{
				return static_cast<base_type const&&>(*this).yy();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) yz() const&
				noexcept(noexcept(std::declval<base_type const&>().yz()))
			{
				return static_cast<base_type const&>(*this).yz();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) yz() &&
				noexcept(noexcept(std::declval<base_type&&>().yz()))
			{
				return static_cast<base_type&&>(*this).yz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) yz() const&&
				noexcept(noexcept(std::declval<base_type const&&>().yz()))
			{
				return static_cast<base_type const&&>(*this).yz();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) zx() const&
				noexcept(noexcept(std::declval<base_type const&>().zx()))
			{
				return static_cast<base_type const&>(*this).zx();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zx() &&
				noexcept(noexcept(std::declval<base_type&&>().zx()))
			{
				return static_cast<base_type&&>(*this).zx();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zx() const&&
				noexcept(noexcept(std::declval<base_type const&&>().zx()))
			{
				return static_cast<base_type const&&>(*this).zx();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) zy() const&
				noexcept(noexcept(std::declval<base_type const&>().zy()))
			{
				return static_cast<base_type const&>(*this).zy();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zy() &&
				noexcept(noexcept(std::declval<base_type&&>().zy()))
			{
				return static_cast<base_type&&>(*this).zy();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zy() const&&
				noexcept(noexcept(std::declval<base_type const&&>().zy()))
			{
				return static_cast<base_type const&&>(*this).zy();
			}

			JKL_GPU_EXECUTABLE constexpr decltype(auto) zz() const&
				noexcept(noexcept(std::declval<base_type const&>().zz()))
			{
				return static_cast<base_type const&>(*this).zz();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) zz() &&
				noexcept(noexcept(std::declval<base_type&&>().zz()))
			{
				return static_cast<base_type&&>(*this).zz();
			}
			JKL_GPU_EXECUTABLE constexpr decltype(auto) zz() const&&
				noexcept(noexcept(std::declval<base_type const&&>().zz()))
			{
				return static_cast<base_type const&&>(*this).zz();
			}


			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<base_type const&>().template get<I>()))
			{
				return static_cast<base_type const&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<base_type&&>().template get<I>()))
			{
				return static_cast<base_type&&>(*this).template get<I>();
			}
			template <std::size_t I>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<base_type const&&>().template get<I>()))
			{
				return static_cast<base_type const&&>(*this).template get<I>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&
				noexcept(noexcept(std::declval<base_type const&>().template get<I, J>()))
			{
				return static_cast<base_type const&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR decltype(auto) get() &&
				noexcept(noexcept(std::declval<base_type&&>().template get<I, J>()))
			{
				return static_cast<base_type&&>(*this).template get<I, J>();
			}
			template <std::size_t I, std::size_t J>
			JKL_GPU_EXECUTABLE constexpr decltype(auto) get() const&&
				noexcept(noexcept(std::declval<base_type const&&>().template get<I, J>()))
			{
				return static_cast<base_type const&&>(*this).template get<I, J>();
			}
			

			JKL_GPU_EXECUTABLE constexpr posdef3_elmt operator+() const&
				noexcept(std::is_nothrow_copy_constructible<posdef3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR posdef3_elmt operator+() &&
				noexcept(std::is_nothrow_move_constructible<posdef3_elmt>::value)
			{
				return std::move(*this);
			}

			// Remove unary minus operators
		private:
			JKL_GPU_EXECUTABLE constexpr sym3_elmt_type operator-() const&;
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR sym3_elmt_type operator-() &&;

		public:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator+=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that)
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) += that))
			{
				static_cast<sym3_elmt_type&>(*this) += that;
				return *this;
			}
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator+=(
				posdef3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that)
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) += std::move(that)))
			{
				static_cast<sym3_elmt_type&>(*this) += std::move(that);
				return *this;
			}

			// Remove -= operators
		private:
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt_type& operator-=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits> const& that);
			template <class OtherComponentType, class OtherStorage, class OtherStorageTraits>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR sym3_elmt_type& operator-=(
				sym3_elmt<OtherComponentType, OtherStorage, OtherStorageTraits>&& that);

		public:
			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() *= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator*=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) *= k))
			{
				assert(k > jkl::math::zero<ComponentType>());
				static_cast<sym3_elmt_type&>(*this) *= k;
				return *this;
			}

			template <class OtherComponentType,
				class = decltype(std::declval<ComponentType&>() /= std::declval<OtherComponentType const&>())>
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt& operator/=(OtherComponentType const& k)
				noexcept(noexcept(static_cast<sym3_elmt_type&>(*this) /= k))
			{
				assert(k > jkl::math::zero<ComponentType>());
				static_cast<sym3_elmt_type&>(*this) /= k;
				return *this;
			}


			JKL_GPU_EXECUTABLE constexpr posdef3_elmt t() const&
				noexcept(std::is_nothrow_copy_constructible<posdef3_elmt>::value)
			{
				return *this;
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR posdef3_elmt t() &&
				noexcept(std::is_nothrow_move_constructible<posdef3_elmt>::value)
			{
				return std::move(*this);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_invertible() const noexcept
			{
				return true;
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const noexcept
			{
				return true;
			}


			// Division
		private:
			template <class Posdef3>
			JKL_GPU_EXECUTABLE static GENERALIZED_CONSTEXPR posdef3_elmt inv_impl(Posdef3&& m) noexcept(
				noexcept(sym3_elmt_type{
					m.template get<1, 1>() * m.template get<2, 2>() -
					m.template get<1, 2>() * m.template get<2, 1>(),
					m.template get<0, 2>() * m.template get<2, 1>() -
					m.template get<0, 1>() * m.template get<2, 2>(),
					m.template get<0, 1>() * m.template get<1, 2>() -
					m.template get<0, 2>() * m.template get<1, 1>(),
					m.template get<0, 0>() * m.template get<2, 2>() -
					m.template get<0, 2>() * m.template get<2, 0>(),
					m.template get<0, 2>() * m.template get<1, 0>() -
					m.template get<0, 0>() * m.template get<1, 2>(),
					m.template get<0, 0>() * m.template get<1, 1>() -
					m.template get<0, 1>() * m.template get<1, 0>() }) &&
					noexcept(std::declval<posdef3_elmt&>() /= std::forward<Posdef3>(m).det()))
			{
				posdef3_elmt ret_value{
					m.template get<1, 1>() * m.template get<2, 2>() -
					m.template get<1, 2>() * m.template get<2, 1>(),
					m.template get<0, 2>() * m.template get<2, 1>() -
					m.template get<0, 1>() * m.template get<2, 2>(),
					m.template get<0, 1>() * m.template get<1, 2>() -
					m.template get<0, 2>() * m.template get<1, 1>(),
					m.template get<0, 0>() * m.template get<2, 2>() -
					m.template get<0, 2>() * m.template get<2, 0>(),
					m.template get<0, 2>() * m.template get<1, 0>() -
					m.template get<0, 0>() * m.template get<1, 2>(),
					m.template get<0, 0>() * m.template get<1, 1>() -
					m.template get<0, 1>() * m.template get<1, 0>(),
					no_validity_check{} };
				ret_value /= std::forward<Posdef3>(m).det();
				return ret_value;
			}

		public:
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt inv() const&
				noexcept(noexcept(inv_impl(*this)))
			{
				return inv_impl(*this);
			}
			JKL_GPU_EXECUTABLE GENERALIZED_CONSTEXPR posdef3_elmt inv() &&
				noexcept(noexcept(inv_impl(std::move(*this))))
			{
				return inv_impl(std::move(*this));
			}
		};

		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(posdef3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.t()))
		{
			return m.t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) transpose(posdef3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).t()))
		{
			return std::move(m).t();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(posdef3_elmt<ComponentType, Storage, StorageTraits> const& m)
			noexcept(noexcept(m.inv()))
		{
			return m.inv();
		}
		template <class ComponentType, class Storage, class StorageTraits>
		JKL_GPU_EXECUTABLE constexpr decltype(auto) inv(posdef3_elmt<ComponentType, Storage, StorageTraits>&& m)
			noexcept(noexcept(std::move(m).inv()))
		{
			return std::move(m).inv();
		}


		//// Binary operations for symmetric 3D matrices

		namespace detail {
			template <class T>
			using get_sym3_elmt = get_gl3_elmt_impl<T, sym3_elmt>;

			template <class T>
			using get_posdef3_elmt = get_gl3_elmt_impl<T, posdef3_elmt>;

			template <template <class, class> class Template, class LeftOperand, class RightOperand>
			using dispatch_to_sym3_or_gl3 =
				std::conditional_t<get_sym3_elmt<LeftOperand>::value,
					std::conditional_t<get_sym3_elmt<RightOperand>::value,
						Template<
							typename get_sym3_elmt<LeftOperand>::type,
							typename get_sym3_elmt<RightOperand>::type
						>,
						std::conditional_t<get_gl3_elmt<RightOperand>::value,
							Template<
								typename get_sym3_elmt<LeftOperand>::type,
								typename get_gl3_elmt<RightOperand>::type
							>,
							empty_type
						>
					>,
					std::conditional_t<get_gl3_elmt<LeftOperand>::value,
						std::conditional_t<get_sym3_elmt<RightOperand>::value,
							Template<
								typename get_gl3_elmt<LeftOperand>::type,
								typename get_sym3_elmt<RightOperand>::type
							>,
							std::conditional_t<get_gl3_elmt<RightOperand>::value,
								Template<
									typename get_gl3_elmt<LeftOperand>::type,
									typename get_sym3_elmt<RightOperand>::type
								>,
								empty_type
							>
						>,
						empty_type
					>
				>;


			template <class LeftOperand, class RightOperand>
			struct get_sym3_elmt_add_result_impl : dispatch_to_sym3_or_gl3<
				get_sym3_elmt_add_result_impl, LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_add_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_add_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				gl3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_add_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<false,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_add_result_impl<
				posdef3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				posdef3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_sym3_elmt_add_result = typename get_sym3_elmt_add_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;
			

			template <class LeftOperand, class RightOperand>
			struct get_sym3_elmt_sub_result_impl : dispatch_to_sym3_or_gl3<
				get_sym3_elmt_sub_result_impl, LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_sub_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_binary_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_sub_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				gl3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_sub_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<false,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_sym3_elmt_sub_result = typename get_sym3_elmt_sub_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;

			
			template <class LeftOperand, class RightOperand>
			struct get_sym3_elmt_mult_result_impl : dispatch_to_sym3_or_gl3<
				get_sym3_elmt_mult_result_impl, LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_mult_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				gl3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				gl3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				sym3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<false,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				posdef3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				posdef3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_mult_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				posdef3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_GL3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_mult_result_impl<
				GL3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				posdef3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_GL3_elmt_result<false,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_sym3_elmt_mult_result = typename get_sym3_elmt_mult_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;


			template <class LeftOperand, class RightOperand>
			struct get_sym3_elmt_div_result_impl : dispatch_to_sym3_or_gl3<
				get_sym3_elmt_div_result_impl, LeftOperand, RightOperand> {};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_div_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				posdef3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_mult_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_div_result_impl<
				sym3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = sym3_elmt_gl3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_div_result_impl<
				posdef3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				posdef3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_mult_result<
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftComponentType, class LeftStorage, class LeftStorageTraits,
				class RightComponentType, class RightStorage, class RightStorageTraits>
			struct get_sym3_elmt_div_result_impl<
				posdef3_elmt<LeftComponentType, LeftStorage, LeftStorageTraits>,
				GL3_elmt<RightComponentType, RightStorage, RightStorageTraits>>
			{
				using type = posdef3_elmt_GL3_elmt_result<true,
					LeftComponentType, LeftStorage, LeftStorageTraits,
					RightComponentType, RightStorage, RightStorageTraits>;
			};

			template <class LeftOperand, class RightOperand>
			using get_sym3_elmt_div_result = typename get_sym3_elmt_div_result_impl<
				tmp::remove_cvref_t<LeftOperand>,
				tmp::remove_cvref_t<RightOperand>>::type;
			

			template <class Scalar, class Matrix, bool from_left>
			struct get_sym3_elmt_scalar_mult_result_impl_impl {
				static constexpr bool value = false;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_sym3_elmt_scalar_mult_result_impl_impl<Scalar,
				sym3_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = sym3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, bool from_left, class ComponentType, class Storage, class StorageTraits>
			struct get_sym3_elmt_scalar_mult_result_impl_impl<Scalar,
				posdef3_elmt<ComponentType, Storage, StorageTraits>, from_left>
			{
				using type = posdef3_elmt_scalar_mult_result<Scalar, from_left,
					ComponentType, Storage, StorageTraits>;

				// Remove from the overload set if Scalar is not compatible with ComponentType
				static constexpr bool value = !std::is_same<type,
					no_operation_tag<no_operation_reason::component_type_not_compatible>>::value;
			};

			template <class Scalar, class Matrix, bool from_left>
			struct get_sym3_elmt_scalar_mult_result_impl : std::conditional_t<
				get_sym3_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>::value,
				get_sym3_elmt_scalar_mult_result_impl_impl<Scalar, Matrix, from_left>,
				get_sym3_elmt_scalar_mult_result_impl_impl<void, void, false>> {};

			template <class Scalar, class Matrix, bool from_left>
			using get_sym3_elmt_scalar_mult_result = typename get_sym3_elmt_scalar_mult_result_impl<
				tmp::remove_cvref_t<Scalar>,
				tmp::remove_cvref_t<Matrix>, from_left>::type;


			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<sym3_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = sym3_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args...>::value)
				{
					return{ std::forward<Args>(args)... };
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct call_unchecking<posdef3_elmt<ComponentType, Storage, StorageTraits>>
			{
				using result_type = posdef3_elmt<ComponentType, Storage, StorageTraits>;

				template <class... Args>
				JKL_GPU_EXECUTABLE static constexpr result_type make(Args&&... args)
					noexcept(std::is_nothrow_constructible<result_type, Args..., no_validity_check>::value)
				{
					return{ std::forward<Args>(args)..., no_validity_check{} };
				}
			};

			template <class ResultType>
			struct sym3_add_impl {};

			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_add_impl<sym3_elmt<ComponentType, Storage, StorageTraits>> {
				using result_type = sym3_elmt<ComponentType, Storage, StorageTraits>;

				template <class LeftOperand, class RightOperand>
				JKL_GPU_EXECUTABLE static constexpr result_type op(LeftOperand&& a, RightOperand&& b)
					noexcept(noexcept(result_type{
					std::forward<LeftOperand>(a).xx() + std::forward<RightOperand>(b).xx(),
					std::forward<LeftOperand>(a).xy() +	std::forward<RightOperand>(b).xy(),
					std::forward<LeftOperand>(a).xz() +	std::forward<RightOperand>(b).xz(),
					std::forward<LeftOperand>(a).yy() + std::forward<RightOperand>(b).yy(),
					std::forward<LeftOperand>(a).yz() + std::forward<RightOperand>(b).yz(),
					std::forward<LeftOperand>(a).zz() + std::forward<RightOperand>(b).zz() }))
				{
					return{
						std::forward<LeftOperand>(a).xx() + std::forward<RightOperand>(b).xx(),
						std::forward<LeftOperand>(a).xy() +	std::forward<RightOperand>(b).xy(),
						std::forward<LeftOperand>(a).xz() +	std::forward<RightOperand>(b).xz(),
						std::forward<LeftOperand>(a).yy() + std::forward<RightOperand>(b).yy(),
						std::forward<LeftOperand>(a).yz() + std::forward<RightOperand>(b).yz(),
						std::forward<LeftOperand>(a).zz() + std::forward<RightOperand>(b).zz()
					};
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_add_impl<posdef3_elmt<ComponentType, Storage, StorageTraits>> {
				using result_type = posdef3_elmt<ComponentType, Storage, StorageTraits>;

				template <class LeftOperand, class RightOperand>
				JKL_GPU_EXECUTABLE static constexpr result_type op(LeftOperand&& a, RightOperand&& b)
					noexcept(noexcept(result_type{
					std::forward<LeftOperand>(a).xx() + std::forward<RightOperand>(b).xx(),
					std::forward<LeftOperand>(a).xy() +	std::forward<RightOperand>(b).xy(),
					std::forward<LeftOperand>(a).xz() +	std::forward<RightOperand>(b).xz(),
					std::forward<LeftOperand>(a).yy() + std::forward<RightOperand>(b).yy(),
					std::forward<LeftOperand>(a).yz() + std::forward<RightOperand>(b).yz(),
					std::forward<LeftOperand>(a).zz() + std::forward<RightOperand>(b).zz(),
					no_validity_check{} }))
				{
					return{
						std::forward<LeftOperand>(a).xx() + std::forward<RightOperand>(b).xx(),
						std::forward<LeftOperand>(a).xy() +	std::forward<RightOperand>(b).xy(),
						std::forward<LeftOperand>(a).xz() +	std::forward<RightOperand>(b).xz(),
						std::forward<LeftOperand>(a).yy() + std::forward<RightOperand>(b).yy(),
						std::forward<LeftOperand>(a).yz() + std::forward<RightOperand>(b).yz(),
						std::forward<LeftOperand>(a).zz() + std::forward<RightOperand>(b).zz(),
						no_validity_check{}
					};
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_add_impl<gl3_elmt<ComponentType, Storage, StorageTraits>> {
				using result_type = gl3_elmt<ComponentType, Storage, StorageTraits>;

				template <class LeftOperand, class RightOperand>
				JKL_GPU_EXECUTABLE static constexpr result_type op(LeftOperand&& a, RightOperand&& b)
					noexcept(noexcept(result_type{
						std::forward<LeftOperand>(a).template get<0, 0>() +
						std::forward<RightOperand>(b).template get<0, 0>(),
						std::forward<LeftOperand>(a).template get<0, 1>() +
						std::forward<RightOperand>(b).template get<0, 1>(),
						std::forward<LeftOperand>(a).template get<0, 2>() +
						std::forward<RightOperand>(b).template get<0, 2>(),
						std::forward<LeftOperand>(a).template get<1, 0>() +
						std::forward<RightOperand>(b).template get<1, 0>(),
						std::forward<LeftOperand>(a).template get<1, 1>() +
						std::forward<RightOperand>(b).template get<1, 1>(),
						std::forward<LeftOperand>(a).template get<1, 2>() +
						std::forward<RightOperand>(b).template get<1, 2>(),
						std::forward<LeftOperand>(a).template get<2, 0>() +
						std::forward<RightOperand>(b).template get<2, 0>(),
						std::forward<LeftOperand>(a).template get<2, 1>() +
						std::forward<RightOperand>(b).template get<2, 1>(),
						std::forward<LeftOperand>(a).template get<2, 2>() +
						std::forward<RightOperand>(b).template get<2, 2>() }))
				{
					return{
						std::forward<LeftOperand>(a).template get<0, 0>() +
						std::forward<RightOperand>(b).template get<0, 0>(),
						std::forward<LeftOperand>(a).template get<0, 1>() +
						std::forward<RightOperand>(b).template get<0, 1>(),
						std::forward<LeftOperand>(a).template get<0, 2>() +
						std::forward<RightOperand>(b).template get<0, 2>(),
						std::forward<LeftOperand>(a).template get<1, 0>() +
						std::forward<RightOperand>(b).template get<1, 0>(),
						std::forward<LeftOperand>(a).template get<1, 1>() +
						std::forward<RightOperand>(b).template get<1, 1>(),
						std::forward<LeftOperand>(a).template get<1, 2>() +
						std::forward<RightOperand>(b).template get<1, 2>(),
						std::forward<LeftOperand>(a).template get<2, 0>() +
						std::forward<RightOperand>(b).template get<2, 0>(),
						std::forward<LeftOperand>(a).template get<2, 1>() +
						std::forward<RightOperand>(b).template get<2, 1>(),
						std::forward<LeftOperand>(a).template get<2, 2>() +
						std::forward<RightOperand>(b).template get<2, 2>()
					};
				}
			};


			template <class ResultType>
			struct sym3_sub_impl {};

			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_sub_impl<sym3_elmt<ComponentType, Storage, StorageTraits>> {
				using result_type = sym3_elmt<ComponentType, Storage, StorageTraits>;

				template <class LeftOperand, class RightOperand>
				JKL_GPU_EXECUTABLE static constexpr result_type op(LeftOperand&& a, RightOperand&& b)
					noexcept(noexcept(result_type{
					std::forward<LeftOperand>(a).xx() - std::forward<RightOperand>(b).xx(),
					std::forward<LeftOperand>(a).xy() -	std::forward<RightOperand>(b).xy(),
					std::forward<LeftOperand>(a).xz() -	std::forward<RightOperand>(b).xz(),
					std::forward<LeftOperand>(a).yy() - std::forward<RightOperand>(b).yy(),
					std::forward<LeftOperand>(a).yz() - std::forward<RightOperand>(b).yz(),
					std::forward<LeftOperand>(a).zz() - std::forward<RightOperand>(b).zz() }))
				{
					return{
						std::forward<LeftOperand>(a).xx() - std::forward<RightOperand>(b).xx(),
						std::forward<LeftOperand>(a).xy() -	std::forward<RightOperand>(b).xy(),
						std::forward<LeftOperand>(a).xz() -	std::forward<RightOperand>(b).xz(),
						std::forward<LeftOperand>(a).yy() - std::forward<RightOperand>(b).yy(),
						std::forward<LeftOperand>(a).yz() - std::forward<RightOperand>(b).yz(),
						std::forward<LeftOperand>(a).zz() - std::forward<RightOperand>(b).zz()
					};
				}
			};

			template <class ComponentType, class Storage, class StorageTraits>
			struct sym3_sub_impl<gl3_elmt<ComponentType, Storage, StorageTraits>> {
				using result_type = gl3_elmt<ComponentType, Storage, StorageTraits>;

				template <class LeftOperand, class RightOperand>
				JKL_GPU_EXECUTABLE static constexpr result_type op(LeftOperand&& a, RightOperand&& b)
					noexcept(noexcept(result_type{
						std::forward<LeftOperand>(a).template get<0, 0>() -
						std::forward<RightOperand>(b).template get<0, 0>(),
						std::forward<LeftOperand>(a).template get<0, 1>() -
						std::forward<RightOperand>(b).template get<0, 1>(),
						std::forward<LeftOperand>(a).template get<0, 2>() -
						std::forward<RightOperand>(b).template get<0, 2>(),
						std::forward<LeftOperand>(a).template get<1, 0>() -
						std::forward<RightOperand>(b).template get<1, 0>(),
						std::forward<LeftOperand>(a).template get<1, 1>() -
						std::forward<RightOperand>(b).template get<1, 1>(),
						std::forward<LeftOperand>(a).template get<1, 2>() -
						std::forward<RightOperand>(b).template get<1, 2>(),
						std::forward<LeftOperand>(a).template get<2, 0>() -
						std::forward<RightOperand>(b).template get<2, 0>(),
						std::forward<LeftOperand>(a).template get<2, 1>() -
						std::forward<RightOperand>(b).template get<2, 1>(),
						std::forward<LeftOperand>(a).template get<2, 2>() -
						std::forward<RightOperand>(b).template get<2, 2>() }))
				{
					return{
						std::forward<LeftOperand>(a).template get<0, 0>() -
						std::forward<RightOperand>(b).template get<0, 0>(),
						std::forward<LeftOperand>(a).template get<0, 1>() -
						std::forward<RightOperand>(b).template get<0, 1>(),
						std::forward<LeftOperand>(a).template get<0, 2>() -
						std::forward<RightOperand>(b).template get<0, 2>(),
						std::forward<LeftOperand>(a).template get<1, 0>() -
						std::forward<RightOperand>(b).template get<1, 0>(),
						std::forward<LeftOperand>(a).template get<1, 1>() -
						std::forward<RightOperand>(b).template get<1, 1>(),
						std::forward<LeftOperand>(a).template get<1, 2>() -
						std::forward<RightOperand>(b).template get<1, 2>(),
						std::forward<LeftOperand>(a).template get<2, 0>() -
						std::forward<RightOperand>(b).template get<2, 0>(),
						std::forward<LeftOperand>(a).template get<2, 1>() -
						std::forward<RightOperand>(b).template get<2, 1>(),
						std::forward<LeftOperand>(a).template get<2, 2>() -
						std::forward<RightOperand>(b).template get<2, 2>()
					};
				}
			};
		}

		// Binary addition of sym3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator+(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::sym3_add_impl<detail::get_sym3_elmt_add_result<LeftOperand, RightOperand>>::op(
				std::forward<LeftOperand>(a), std::forward<RightOperand>(b))))
			-> detail::get_sym3_elmt_add_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_sym3_elmt_add_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot add two sym3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot add two sym3_elmt's; failed to deduce the resulting storage type");

			return detail::sym3_add_impl<result_type>::op(
				std::forward<LeftOperand>(a), std::forward<RightOperand>(b));
		}

		// Binary subtraction of sym3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator-(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::sym3_sub_impl<detail::get_sym3_elmt_sub_result<LeftOperand, RightOperand>>::op(
				std::forward<LeftOperand>(a), std::forward<RightOperand>(b))))
			-> detail::get_sym3_elmt_sub_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_sym3_elmt_sub_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot subtract two sym3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot subtract two sym3_elmt's; failed to deduce the resulting storage type");

			return detail::sym3_sub_impl<detail::get_sym3_elmt_sub_result<LeftOperand, RightOperand>>::op(
				std::forward<LeftOperand>(a), std::forward<RightOperand>(b));
		}

		// Binary multiplication of sym3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator*(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_sym3_elmt_mult_result<LeftOperand, RightOperand>>::make(
					// First row
					a.template get<0, 0>() * b.template get<0, 0>() +
					a.template get<0, 1>() * b.template get<1, 0>() +
					a.template get<0, 2>() * b.template get<2, 0>(),
					a.template get<0, 0>() * b.template get<0, 1>() +
					a.template get<0, 1>() * b.template get<1, 1>() +
					a.template get<0, 2>() * b.template get<2, 1>(),
					a.template get<0, 0>() * b.template get<0, 2>() +
					a.template get<0, 1>() * b.template get<1, 2>() +
					a.template get<0, 2>() * b.template get<2, 2>(),
					// Second row
					a.template get<1, 0>() * b.template get<0, 0>() +
					a.template get<1, 1>() * b.template get<1, 0>() +
					a.template get<1, 2>() * b.template get<2, 0>(),
					a.template get<1, 0>() * b.template get<0, 1>() +
					a.template get<1, 1>() * b.template get<1, 1>() +
					a.template get<1, 2>() * b.template get<2, 1>(),
					a.template get<1, 0>() * b.template get<0, 2>() +
					a.template get<1, 1>() * b.template get<1, 2>() +
					a.template get<1, 2>() * b.template get<2, 2>(),
					// Third row
					a.template get<2, 0>() * b.template get<0, 0>() +
					a.template get<2, 1>() * b.template get<1, 0>() +
					a.template get<2, 2>() * b.template get<2, 0>(),
					a.template get<2, 0>() * b.template get<0, 1>() +
					a.template get<2, 1>() * b.template get<1, 1>() +
					a.template get<2, 2>() * b.template get<2, 1>(),
					a.template get<2, 0>() * b.template get<0, 2>() +
					a.template get<2, 1>() * b.template get<1, 2>() +
					a.template get<2, 2>() * b.template get<2, 2>())))
			-> detail::get_sym3_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_sym3_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot multiply two sym3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply two sym3_elmt's; failed to deduce the resulting storage type");
			
			return detail::call_unchecking<result_type>::make(
				// First row
				a.template get<0, 0>() * b.template get<0, 0>() +
				a.template get<0, 1>() * b.template get<1, 0>() +
				a.template get<0, 2>() * b.template get<2, 0>(),
				a.template get<0, 0>() * b.template get<0, 1>() +
				a.template get<0, 1>() * b.template get<1, 1>() +
				a.template get<0, 2>() * b.template get<2, 1>(),
				a.template get<0, 0>() * b.template get<0, 2>() +
				a.template get<0, 1>() * b.template get<1, 2>() +
				a.template get<0, 2>() * b.template get<2, 2>(),
				// Second row
				a.template get<1, 0>() * b.template get<0, 0>() +
				a.template get<1, 1>() * b.template get<1, 0>() +
				a.template get<1, 2>() * b.template get<2, 0>(),
				a.template get<1, 0>() * b.template get<0, 1>() +
				a.template get<1, 1>() * b.template get<1, 1>() +
				a.template get<1, 2>() * b.template get<2, 1>(),
				a.template get<1, 0>() * b.template get<0, 2>() +
				a.template get<1, 1>() * b.template get<1, 2>() +
				a.template get<1, 2>() * b.template get<2, 2>(),
				// Third row
				a.template get<2, 0>() * b.template get<0, 0>() +
				a.template get<2, 1>() * b.template get<1, 0>() +
				a.template get<2, 2>() * b.template get<2, 0>(),
				a.template get<2, 0>() * b.template get<0, 1>() +
				a.template get<2, 1>() * b.template get<1, 1>() +
				a.template get<2, 2>() * b.template get<2, 1>(),
				a.template get<2, 0>() * b.template get<0, 2>() +
				a.template get<2, 1>() * b.template get<1, 2>() +
				a.template get<2, 2>() * b.template get<2, 2>());
		}


		// Binary division of sym3_elmt's
		template <class LeftOperand, class RightOperand>
		JKL_GPU_EXECUTABLE constexpr auto operator/(LeftOperand&& a, RightOperand&& b)
			noexcept(noexcept(std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv()))
			-> detail::get_sym3_elmt_mult_result<LeftOperand, RightOperand>
		{
			using result_type = detail::get_sym3_elmt_mult_result<LeftOperand, RightOperand>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot divide two sym3_elmt's; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot divide two sym3_elmt's; failed to deduce the resulting storage type");

			return std::forward<LeftOperand>(a) * std::forward<RightOperand>(b).inv();
		}
		

		// Scalar multiplication of sym3_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
				std::forward<Matrix>(m).xx() * k,
				std::forward<Matrix>(m).xy() * k,
				std::forward<Matrix>(m).xz() * k,
				std::forward<Matrix>(m).yy() * k,
				std::forward<Matrix>(m).yz() * k,
				std::forward<Matrix>(m).zz() * k)))
			-> detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply sym3_elmt with a scalar; failed to deduce the resulting storage type");
			
			using component_type = typename result_type::component_type;
			assert(!detail::get_posdef3_elmt<result_type>::value || component_type(k) > zero<component_type>());

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).xx() * k,
				std::forward<Matrix>(m).xy() * k,
				std::forward<Matrix>(m).xz() * k,
				std::forward<Matrix>(m).yy() * k,
				std::forward<Matrix>(m).yz() * k,
				std::forward<Matrix>(m).zz() * k);
		}

		// Scalar multiplication of sym3_elmt's from left
		template <class Scalar, class Matrix>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Scalar const& k, Matrix&& m)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, true>>::make(
				k * std::forward<Matrix>(m).xx(),
				k * std::forward<Matrix>(m).xy(),
				k * std::forward<Matrix>(m).xz(),
				k * std::forward<Matrix>(m).yy(),
				k * std::forward<Matrix>(m).yz(),
				k * std::forward<Matrix>(m).zz())))
			-> detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, true>
		{
			using result_type = detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, true>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::component_type_not_compatible>>::value,
				"jkl::math: cannot multiply sym3_elmt with a scalar; failed to deduce the resulting component type");
			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot multiply sym3_elmt with a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_posdef3_elmt<result_type>::value || component_type(k) > zero<component_type>());

			return detail::call_unchecking<result_type>::make(
				k * std::forward<Matrix>(m).xx(),
				k * std::forward<Matrix>(m).xy(),
				k * std::forward<Matrix>(m).xz(),
				k * std::forward<Matrix>(m).yy(),
				k * std::forward<Matrix>(m).yz(),
				k * std::forward<Matrix>(m).zz());
		}

		// Scalar division of sym3_elmt's from right
		template <class Matrix, class Scalar>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Matrix&& m, Scalar const& k)
			noexcept(noexcept(detail::call_unchecking<
				detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>>::make(
				std::forward<Matrix>(m).xx() / k,
				std::forward<Matrix>(m).xy() / k,
				std::forward<Matrix>(m).xz() / k,
				std::forward<Matrix>(m).yy() / k,
				std::forward<Matrix>(m).yz() / k,
				std::forward<Matrix>(m).zz() / k)))
			-> detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>
		{
			using result_type = detail::get_sym3_elmt_scalar_mult_result<Scalar, Matrix, false>;

			static_assert(!std::is_same<result_type,
				no_operation_tag<no_operation_reason::storage_not_compatible>>::value,
				"jkl::math: cannot divide sym3_elmt by a scalar; failed to deduce the resulting storage type");

			using component_type = typename result_type::component_type;
			assert(!detail::get_posdef3_elmt<result_type>::value || component_type(k) > zero<component_type>());

			return detail::call_unchecking<result_type>::make(
				std::forward<Matrix>(m).xx() / k,
				std::forward<Matrix>(m).xy() / k,
				std::forward<Matrix>(m).xz() / k,
				std::forward<Matrix>(m).yy() / k,
				std::forward<Matrix>(m).yz() / k,
				std::forward<Matrix>(m).zz() / k);
		}
	}
}