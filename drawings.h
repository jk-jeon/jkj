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

/// This file is a collection of useful utilities related to OpenGL and generic 3D graphic rendering

#pragma once
#include <gl/glew.h>
#ifdef JKL_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuda/helper.h"
#endif
#include "numerical_lie_group.h"
#include "tmp/assert_helper.h"

namespace jkj {
	namespace drawings {
		// OpenGL error checking helpers
		namespace {
		#ifndef JKL_OPENGL_ENUM_TO_STRING_CASE
		#define JKL_OPENGL_ENUM_TO_STRING_CASE(e)		case e: return #e
			inline char const* error_string_from_enum(GLenum error) {
				switch( error ) {
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_NO_ERROR);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_INVALID_ENUM);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_INVALID_VALUE);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_INVALID_OPERATION);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_STACK_OVERFLOW);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_STACK_UNDERFLOW);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_OUT_OF_MEMORY);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_INVALID_FRAMEBUFFER_OPERATION);
					JKL_OPENGL_ENUM_TO_STRING_CASE(GL_CONTEXT_LOST);
				}
				return "<unknown>";
			}
		#undef JKL_OPENGL_ENUM_TO_STRING_CASE
		#endif
		}

		// Throw an exception on error

		class opengl_error : public std::runtime_error {
		public:
			opengl_error(GLenum code, std::string const& string_description ={})
				: std::runtime_error{ string_description + ": " + error_string_from_enum(code) }, code{ code } {}

			GLenum code;
		};

		inline void check_error(std::string const& description ={}) {
			auto result = glGetError();
			if( result != GL_NO_ERROR ) {
				throw opengl_error{ result, description };
			}
		}

		template <class Function, class... Args, class = decltype(std::declval<Function>()(std::declval<Args>()...))>
		inline void check_error(Function&& ftn, Args&&... args) {
			std::forward<Function>(ftn)(std::forward<Args>(args)...);
			check_error();
		}

		// RAII-style wrapper for mapped resources
		class mapped_host_ptr {
		public:
			mapped_host_ptr(GLuint buffer_name, GLenum access) : buffer_name{ buffer_name } {
				ptr = glMapNamedBuffer(buffer_name, access);
				if( ptr == nullptr )
					check_error("glMapBuffer failed");
			}
			~mapped_host_ptr() {
				glUnmapNamedBuffer(buffer_name);
			}

			// Move-only
			mapped_host_ptr(mapped_host_ptr const&) = delete;
			mapped_host_ptr& operator=(mapped_host_ptr const&) = delete;
			mapped_host_ptr(mapped_host_ptr&& that) : buffer_name{ that.buffer_name }, ptr{ that.ptr }
			{
				that.buffer_name = 0;
				that.ptr = nullptr;
			}
			mapped_host_ptr& operator=(mapped_host_ptr&& that) {
				std::swap(buffer_name, that.buffer_name);
				std::swap(ptr, that.ptr);
				return *this;
			}

			void* get() const noexcept {
				return ptr;
			}

		private:
			GLuint			buffer_name = 0;
			void*			ptr = nullptr;
		};
	#ifdef JKL_USE_CUDA
		class mapped_device_ptr {
		public:
			mapped_device_ptr(cudaGraphicsResource_t rsrc, cudaStream_t stream) : rsrc{ rsrc }, stream{ stream } {
				jkj::cuda::check_error(cudaGraphicsMapResources(1, &rsrc, stream));
				try {
					jkj::cuda::check_error(cudaGraphicsResourceGetMappedPointer(&ptr, &size, rsrc));
				}
				catch( ... ) {
					cudaGraphicsUnmapResources(1, &rsrc, stream);
					throw;
				}

			}
			~mapped_device_ptr() {
				cudaGraphicsUnmapResources(1, &rsrc, stream);
			}

			// Move-only
			mapped_device_ptr(mapped_host_ptr const&) = delete;
			mapped_device_ptr& operator=(mapped_device_ptr const&) = delete;
			mapped_device_ptr(mapped_device_ptr&& that) : rsrc{ that.rsrc }, stream{ that.stream },
				ptr{ that.ptr }, size{ that.size }
			{
				that.rsrc = nullptr;
				that.ptr = nullptr;
			}
			mapped_device_ptr& operator=(mapped_device_ptr&& that) {
				std::swap(rsrc, that.rsrc);
				std::swap(stream, that.stream);
				std::swap(ptr, that.ptr);
				std::swap(size, that.size);
				return *this;
			}

			void* get() const noexcept {
				return ptr;
			}

		private:
			cudaGraphicsResource_t	rsrc = nullptr;
			cudaStream_t			stream = nullptr;
			void*					ptr = nullptr;
			std::size_t				size = 0;
		};

		// Since cudaGraphicsMapResources()/cudaGraphicsUnmapResources() are quite heavy,
		// it is convenient to wrap multiple pointers into one object.
		template <std::size_t number_of_rsrcs>
		class mapped_device_ptrs {
		public:
			static constexpr std::size_t number_of_resources = number_of_rsrcs;

			// If some component of rsrcs is nullptr, the corresponding mapped pointer becomes nullptr, 
			// rather than to throw an exception. This is for convenience, but perhaps it is a bad design.
			template <class CudaGraphicsResourceTuple>
			mapped_device_ptrs(CudaGraphicsResourceTuple const& rsrcs, cudaStream_t stream) : m_stream{ stream }
			{
				initialize_rsrcs<0>(rsrcs);
				if( m_number_of_non_null_rsrcs == 0 ) {
					m_owns = false;
					return;
				}

				jkj::cuda::check_error(cudaGraphicsMapResources(m_number_of_non_null_rsrcs, m_rsrcs.data(), stream));
				try {
					get_mapped_pointer<0>(rsrcs);
				}
				catch( ... ) {
					cudaGraphicsUnmapResources(m_number_of_non_null_rsrcs, m_rsrcs.data(), stream);
					throw;
				}

				m_owns = true;
			}
			~mapped_device_ptrs() {
				if( m_owns )
					cudaGraphicsUnmapResources(m_number_of_non_null_rsrcs, m_rsrcs.data(), m_stream);
			}

			// Move-only
			mapped_device_ptrs(mapped_device_ptrs const&) = delete;
			mapped_device_ptrs& operator=(mapped_device_ptrs const&) = delete;
			mapped_device_ptrs(mapped_device_ptrs&& that) : m_stream{ that.stream }
			{
				if( that.m_owns ) {
					m_rsrcs = that.m_rsrcs;
					m_ptrs = that.m_ptrs;
					m_sizes = that.m_sizes;
					m_number_of_non_null_rsrcs = that.m_number_of_non_null_rsrcs;
					m_owns = true;
					that.m_owns = false;
				}
			}
			mapped_device_ptrs& operator=(mapped_device_ptrs&& that)
			{
				if( m_owns && that.m_owns ) {
					std::swap(m_rsrcs, that.m_rsrcs);
					std::swap(m_stream, that.m_stream);
					std::swap(m_ptrs, that.m_ptrs);
					std::swap(m_sizes, that.m_sizes);
					std::swap(m_number_of_non_null_rsrcs, that.m_number_of_non_null_rsrcs);
				}
				else if( m_owns && !that.m_owns ) {
					that.m_rsrcs = m_rsrcs;
					that.m_stream = m_stream;
					that.m_ptrs = m_ptrs;
					that.m_sizes = m_sizes;
					that.m_number_of_non_null_rsrcs = m_number_of_non_null_rsrcs;
					that.m_owns = true;
					m_owns = false;
				}
				else if( !m_owns && that.m_owns ) {
					m_rsrcs = that.m_rsrcs;
					m_stream = that.m_stream;
					m_ptrs = that.m_ptrs;
					m_sizes = that.m_sizes;
					m_number_of_non_null_rsrcs = that.m_number_of_non_null_rsrcs;
					m_owns = true;
					that.m_owns = false;
				}
				return *this;
			}

			template <std::size_t idx>
			void* get() const noexcept {
				return m_ptrs[idx];
			}
			void* get(std::size_t idx) const noexcept {
				return m_ptrs[idx];
			}

			template <std::size_t idx>
			std::size_t get_size() const noexcept {
				return m_sizes[idx];
			}
			std::size_t get_size(std::size_t idx) const noexcept {
				return m_sizes[idx];
			}

		private:
			std::array<cudaGraphicsResource_t, number_of_rsrcs>	m_rsrcs;
			cudaStream_t										m_stream = nullptr;
			std::array<void*, number_of_rsrcs>					m_ptrs{};	// Zero-initialize
			std::array<std::size_t, number_of_rsrcs>			m_sizes{};	// Zero-initialize
			int													m_number_of_non_null_rsrcs = 0;
			bool												m_owns = false;

			template <std::size_t idx, class CudaGraphicsResourceTuple,
				class = std::enable_if_t<idx != number_of_rsrcs>>
			void initialize_rsrcs(CudaGraphicsResourceTuple const& rsrcs) {
				if( std::get<idx>(rsrcs) != nullptr ) 
					m_rsrcs[m_number_of_non_null_rsrcs++] = std::get<idx>(rsrcs);

				initialize_rsrcs<idx + 1>(rsrcs);
			}
			template <std::size_t idx, class CudaGraphicsResourceTuple,
				class = std::enable_if_t<idx == number_of_rsrcs>, class = void>
			void initialize_rsrcs(CudaGraphicsResourceTuple const& rsrcs) {}

			template <std::size_t idx, class CudaGraphicsResourceTuple,
				class = std::enable_if_t<idx != number_of_rsrcs>>
			void get_mapped_pointer(CudaGraphicsResourceTuple const& rsrcs) {
				if( std::get<idx>(rsrcs) != nullptr ) {
					jkj::cuda::check_error(
						cudaGraphicsResourceGetMappedPointer(&m_ptrs[idx], &m_sizes[idx], std::get<idx>(rsrcs)));
				}
				else {
					m_ptrs[idx] = nullptr;
					m_sizes[idx] = 0;
				}

				get_mapped_pointer<idx + 1>(rsrcs);
			}

			template <std::size_t idx, class CudaGraphicsResourceTuple,
				class = std::enable_if_t<idx == number_of_rsrcs>, class = void>
			void get_mapped_pointer(CudaGraphicsResourceTuple const& rsrcs) {}
		};
	#endif

		// Initialize GLEW and other necessary things
		// This function can be called multiple times (thanks to the mechanism of Meyer's singleton),
		// but should be called after a proper OpenGL context has been attached to the calling thread.
		// The first call to this function will remove all errors enqueued in the current context.
		inline void global_initialize() {
			struct initializer {
				initializer() {
					if( glewInit() != GLEW_OK ) {
						throw std::runtime_error{ "jkj::drawings: GLEW Initialization failed!" };
					}

					// GL_ARB_direct_state_access is required
					if( !GLEW_ARB_direct_state_access )
						throw std::runtime_error{ "jkj::drawings: your platform does not support "
						"GL_ARB_direct_state_access feature (OpenGL version 4.5); "
						"please try to update your graphic driver"};

					// Empty the OpenGL error queue
					while( glGetError() != GL_NO_ERROR );
				}
			};

			static initializer init{};
		}

		// Camera class
		template <class Float>
		struct camera {
			// Translation part is the position of the camera
			// Column vectors of the rotation part are the camera coordinate frame vectors
			// That is, m_camera is the transform from the camera frame to the world frame
			jkj::math::SE3_elmt<Float>				pose;
			// The distance of the rotation center
			Float									look_at_distance;

			// Get position
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto& x() noexcept {
				return pose.translation().x();
			}
			JKL_GPU_EXECUTABLE constexpr auto const& x() const noexcept {
				return pose.translation().x();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto& y() noexcept {
				return pose.translation().y();
			}
			JKL_GPU_EXECUTABLE constexpr auto const& y() const noexcept {
				return pose.translation().y();
			}
			JKL_GPU_EXECUTABLE NONCONST_CONSTEXPR auto& z() noexcept {
				return pose.translation().z();
			}
			JKL_GPU_EXECUTABLE constexpr auto const& z() const noexcept {
				return pose.translation().z();
			}

			// x-axis of the camera coordinate frame
			JKL_GPU_EXECUTABLE constexpr auto right_vector() const noexcept {
				return jkj::math::R3_elmt<Float>{ pose.rotation()[0][0], pose.rotation()[1][0], pose.rotation()[2][0] };
			}
			// y-axis of the camera coordinate frame
			JKL_GPU_EXECUTABLE constexpr auto direction_vector() const noexcept {
				return jkj::math::R3_elmt<Float>{ pose.rotation()[0][1], pose.rotation()[1][1], pose.rotation()[2][1] };
			}
			// z-axis of the camera coordinate frame
			JKL_GPU_EXECUTABLE constexpr auto up_vector() const noexcept {
				return jkj::math::R3_elmt<Float>{ pose.rotation()[0][2], pose.rotation()[1][2], pose.rotation()[2][2] };
			}
			// The point where the camera is looking at
			JKL_GPU_EXECUTABLE constexpr auto look_at() const noexcept {
				return pose.translation() + direction_vector() * look_at_distance;
			}
		};


		struct none {};

		namespace detail {
			template <class... Types>
			struct attribute;

			template <class UnderlyingType, class... OtherTypes>
			struct attribute<UnderlyingType, OtherTypes...> : attribute<OtherTypes...> {
				using type = UnderlyingType;
				static constexpr std::size_t attribute_size = sizeof(UnderlyingType);
				UnderlyingType	obj;

				attribute() = default;

				template <class FirstArg, class... RemainingArgs>
				JKL_GPU_EXECUTABLE attribute(FirstArg&& first, RemainingArgs&&... remaining_args)
					: obj(std::forward<FirstArg>(first)),
					attribute<OtherTypes...>{ std::forward<RemainingArgs>(remaining_args)... } {}
			};

			template<class... OtherTypes>
			struct attribute<none, OtherTypes...> : attribute<OtherTypes...> {
				using type = none;
				static constexpr std::size_t attribute_size = 0;
				attribute() = default;

				template <class FirstArg, class... RemainingArgs>
				JKL_GPU_EXECUTABLE attribute(FirstArg&&, RemainingArgs&&... remaining_args)
					: attribute<OtherTypes...>{ std::forward<RemainingArgs>(remaining_args)... } {}
			};

			template <>
			struct attribute<> {};
		}

		template <
			class Position = jkj::math::R3_elmtf,
			class Normal = none,
			class Color = none,
			class TexCoord = none
		>
		struct vertex : private detail::attribute<TexCoord, Color, Normal, Position>
		{
		private:
			using attribute_position = detail::attribute<Position>;
			using attribute_normal = detail::attribute<Normal, Position>;
			using attribute_color = detail::attribute<Color, Normal, Position>;
			using attribute_tex_coord = detail::attribute<TexCoord, Color, Normal, Position>;

			template <class MemPtr>
			static auto offset_from_mem_ptr(MemPtr ptr) noexcept {
				return reinterpret_cast<std::size_t>(&((vertex*)(nullptr)->*ptr));
			}

		public:
			vertex() = default;
			JKL_GPU_EXECUTABLE constexpr vertex(
				Position const& pos,
				Normal const& normal={},
				Color const& color={},
				TexCoord const& tex_coord={}) :
				attribute_tex_coord{ tex_coord, color, normal, pos } {}

			JKL_GPU_EXECUTABLE auto& position() noexcept {
				return attribute_position::obj;
			}
			JKL_GPU_EXECUTABLE constexpr auto& position() const noexcept {
				return attribute_position::obj;
			}

			JKL_GPU_EXECUTABLE auto& normal() noexcept {
				return attribute_normal::obj;
			}
			JKL_GPU_EXECUTABLE constexpr auto& normal() const noexcept {
				return attribute_normal::obj;
			}

			JKL_GPU_EXECUTABLE auto& color() noexcept {
				return attribute_color::obj;
			}
			JKL_GPU_EXECUTABLE constexpr auto& color() const noexcept {
				return attribute_color::obj;
			}

			JKL_GPU_EXECUTABLE auto& tex_coord() noexcept {
				return attribute_tex_coord::obj;
			}
			JKL_GPU_EXECUTABLE constexpr auto& tex_coord() const noexcept {
				return attribute_tex_coord::obj;
			}

			JKL_GPU_EXECUTABLE constexpr explicit operator Position const&() const noexcept {
				return position();
			}

			using position_type = typename attribute_position::type;
			using normal_type = typename attribute_normal::type;
			using color_type = typename attribute_color::type;
			using tex_coord_type = typename attribute_tex_coord::type;

			static constexpr std::size_t position_size = attribute_position::attribute_size;
			static constexpr std::size_t normal_size = attribute_normal::attribute_size;
			static constexpr std::size_t color_size = attribute_color::attribute_size;
			static constexpr std::size_t tex_coord_size = attribute_tex_coord::attribute_size;

			/// Be careful!
			/// Use these may lead to undefined behaviour since vertex is not a standard layout type
			JKL_GPU_EXECUTABLE static auto position_offset() noexcept {
				return offset_from_mem_ptr(&attribute_position::obj);
			}
			JKL_GPU_EXECUTABLE static auto normal_offset() noexcept {
				return offset_from_mem_ptr(&attribute_normal::obj);
			}
			JKL_GPU_EXECUTABLE static auto color_offset() noexcept {
				return offset_from_mem_ptr(&attribute_color::obj);
			}
			JKL_GPU_EXECUTABLE static auto tex_coord_offset() noexcept {
				return offset_from_mem_ptr(&attribute_tex_coord::obj);
			}
		};

		template <class UInt>
		struct line : private jkj::math::R2_elmt<UInt> {
		private:
			using base_type = jkj::math::R2_elmt<UInt>;
		public:
			using base_type::base_type;
			using component_type = typename base_type::component_type;
			using base_type::components;
			static constexpr auto draw_mode = GL_LINES;

			UInt& first() noexcept { return base_type::x(); }
			constexpr UInt const& first() const noexcept { return base_type::x(); }
			UInt& second() noexcept { return base_type::y(); }
			constexpr UInt const& second() const noexcept { return base_type::y(); }
		};

		template <class UInt>
		struct triangle : private jkj::math::R3_elmt<UInt> {
		private:
			using base_type = jkj::math::R3_elmt<UInt>;
		public:
			using base_type::base_type;
			using component_type = typename base_type::component_type;
			using base_type::components;
			static constexpr auto draw_mode = GL_TRIANGLES;

			UInt& first() noexcept { return base_type::x(); }
			constexpr UInt const& first() const noexcept { return base_type::x(); }
			UInt& second() noexcept { return base_type::y(); }
			constexpr UInt const& second() const noexcept { return base_type::y(); }
			UInt& third() noexcept { return base_type::z(); }
			constexpr UInt const& third() const noexcept { return base_type::z(); }
		};

		namespace detail {
			template <std::size_t idx, class T>
			struct get_impl;

			template <class T>
			struct get_impl<0, T> {
				static auto& get(T& a) {
					return a.first();
				}
				static constexpr auto const& get(T const& a) {
					return a.first();
				}
			};

			template <class T>
			struct get_impl<1, T> {
				static auto& get(T& a) {
					return a.second();
				}
				static constexpr auto const& get(T const& a) {
					return a.second();
				}
			};

			template <class T>
			struct get_impl<2, T> {
				static auto& get(T& a) {
					return a.third();
				}
				static constexpr auto const& get(T const& a) {
					return a.third();
				}
			};
		}
	}

	template <std::size_t idx, class UInt>
	JKL_GPU_EXECUTABLE auto& get(drawings::line<UInt>& a) noexcept {
		return drawings::detail::get_impl<idx, drawings::line<UInt>>::get(a);
	}
	template <std::size_t idx, class UInt>
	JKL_GPU_EXECUTABLE constexpr auto const& get(drawings::line<UInt> const& a) noexcept {
		return drawings::detail::get_impl<idx, drawings::line<UInt>>::get(a);
	}
	template <std::size_t idx, class UInt>
	JKL_GPU_EXECUTABLE auto& get(drawings::triangle<UInt>& a) noexcept {
		return drawings::detail::get_impl<idx, drawings::triangle<UInt>>::get(a);
	}
	template <std::size_t idx, class UInt>
	JKL_GPU_EXECUTABLE constexpr auto const& get(drawings::triangle<UInt> const& a) noexcept {
		return drawings::detail::get_impl<idx, drawings::triangle<UInt>>::get(a);
	}
	
	namespace drawings {
		namespace detail {
			template <class T>
			using component_type_t = typename T::component_type;

			template <class T, class = void>
			struct has_component_type : std::false_type {};

			template <class T>
			struct has_component_type<T, std::void_t<component_type_t<T>>> : std::true_type {};

			template <class T, class = void>
			struct has_components : std::false_type {};

			template <class T>
			struct has_components<T, std::void_t<decltype(T::components)>> : std::true_type {};

			template <class T, class = void>
			struct has_draw_mode : std::false_type {};

			template <class T>
			struct has_draw_mode<T, std::void_t<decltype(T::draw_mode)>> : std::true_type {};

			template <class Type, int v_or_i>
			struct type_to_glenum {
				template <int = v_or_i, class = void>
				struct assertion {};

				template <class dummy>
				struct assertion<0, dummy> {
					static_assert(jkj::tmp::assert_helper<Type>::value,
						"The type is incompatible with vertex_buffer: the only allowed component_type is one of "
						"char, unsigned char, short, unsigned short, int, unsigned int, float, and double");
				};

				template <class dummy>
				struct assertion<1, dummy> {
					static_assert(jkj::tmp::assert_helper<Type>::value,
						"The type is incompatible with index_buffer: the only allowed component_type is one of "
						"char, unsigned char, short, unsigned short, int, unsigned int, float, and double");
				};
			};

			template <int v_or_i>
			struct type_to_glenum<char, v_or_i> : std::integral_constant<GLenum, GL_BYTE> {};
			template <int v_or_i>
			struct type_to_glenum<unsigned char, v_or_i> : std::integral_constant<GLenum, GL_UNSIGNED_BYTE> {};
			template <int v_or_i>
			struct type_to_glenum<short, v_or_i> : std::integral_constant<GLenum, GL_SHORT> {};
			template <int v_or_i>
			struct type_to_glenum<unsigned short, v_or_i> : std::integral_constant<GLenum, GL_UNSIGNED_SHORT> {};
			template <int v_or_i>
			struct type_to_glenum<int, v_or_i> : std::integral_constant<GLenum, GL_INT> {};
			template <int v_or_i>
			struct type_to_glenum<unsigned int, v_or_i> : std::integral_constant<GLenum, GL_UNSIGNED_INT> {};
			template <int v_or_i>
			struct type_to_glenum<float, v_or_i> : std::integral_constant<GLenum, GL_FLOAT> {};
			template <int v_or_i>
			struct type_to_glenum<double, v_or_i> : std::integral_constant<GLenum, GL_DOUBLE> {};

			template <
				class Type,
				class = std::enable_if_t<has_component_type<Type>::value && has_components<Type>::value>
			>
			constexpr bool check_type() {
				static_assert(sizeof(Type) == Type::components * sizeof(component_type_t<Type>),
					"The type is incompatible with vertex_buffer: the type is not a simple array of component_type");
				static_assert(Type::components == 1 || Type::components == 2 ||
					Type::components == 3 || Type::components == 4,
					"The type is incompatible with vertex_buffer: \"components\" must be one of 1, 2, 3, and 4");
				return true;
			}
			template <
				class Type,
				class = std::enable_if_t<!has_component_type<Type>::value>,
				class = void
			>
			constexpr bool check_type() {
				static_assert(jkj::tmp::assert_helper<Type>::value,
					"The type is incompatible with vertex_buffer: the type does not has the member type \"component_type\"");
				return true;
			}
			template <
				class Type,
				class = std::enable_if_t<has_component_type<Type>::value && !has_components<Type>::value>,
				class = void, class = void
			>
			constexpr bool check_type() {
				static_assert(jkj::tmp::assert_helper<Type>::value,
					"The type is incompatible with vertex_buffer: the type does not has the static member \"components\"");
				return true;
			}

			template <class Type>
			struct set_attribute_base {
				static constexpr bool compatible_type = check_type<Type>();
				static_assert(std::is_trivially_copyable<Type>::value,
					"The type is incompatible with vertex_buffer: the type is not trivially copyable");
				static_assert(std::is_trivially_destructible<Type>::value,
					"The type is incompatible with vertex_buffer: the type is not trivially destructible");
				
				static constexpr GLenum component_type_to_enum =
					type_to_glenum<component_type_t<Type>, 0>::value;

				template <class Vertex, GLint size, GLenum type, GLsizei stride>
				struct set_impl {
					template <std::size_t idx, class = void>
					struct impl {};

					// Position
					template <class dummy>
					struct impl<0, dummy> {
						void operator()(void const* offset) {
							static_assert(size == 2 || size == 3 || size == 4,
								"The type is incompatible with vertex_buffer: position_type must have 2, 3, or 4 components");
							static_assert(type == GL_SHORT || type == GL_INT || type == GL_FLOAT || type == GL_DOUBLE,
								"The type is incompatible with vertex_buffer: the component_type of position_type must be one of "
								"short, int, float, and double");
							glVertexPointer(size, type, stride, offset);
							glEnableClientState(GL_VERTEX_ARRAY);
						}
					};

					// Normal
					template <class dummy>
					struct impl<1, dummy> {
						void operator()(void const* offset) {
							static_assert(size == 3,
								"The type is incompatible with vertex_buffer: normal_type must have 3 components");
							static_assert(type == GL_BYTE || type == GL_SHORT || type == GL_INT || type == GL_FLOAT || type == GL_DOUBLE,
								"The type is incompatible with vertex_buffer: the component_type of normal_type must be one of "
								"char, short, int, float, and double");
							glNormalPointer(type, stride, offset);
							glEnableClientState(GL_NORMAL_ARRAY);
						}
					};

					// Color
					template <class dummy>
					struct impl<2, dummy> {
						void operator()(void const* offset) {
							static_assert(size == 3 || size == 4,
								"The type is incompatible with vertex_buffer: color_type must have 3 or 4 components");
							glColorPointer(size, type, stride, offset);
							glEnableClientState(GL_COLOR_ARRAY);
						}
					};

					// Texture coordinate
					template <class dummy>
					struct impl<3, dummy> {
						void operator()(void const* offset) {
							static_assert(type == GL_SHORT || type == GL_INT || type == GL_FLOAT || type == GL_DOUBLE,
								"The type is incompatible with vertex_buffer: the component_type of tex_coord_type must be one of "
								"short, int, float, and double");
							glTexCoordPointer(size, type, stride, offset);
							glEnableClientState(GL_TEXTURE_COORD_ARRAY);
						}
					};
				};

				template <class Vertex, std::size_t idx>
				void set(std::size_t offset) {
					using impl_type = typename set_impl<Vertex, static_cast<GLint>(Type::components),
						component_type_to_enum, static_cast<GLsizei>(sizeof(Vertex))>::template impl<idx>;
					impl_type{}(reinterpret_cast<void const*>(offset));
				}
			};
			template <>
			struct set_attribute_base<none> {
				template <class Vertex, std::size_t idx>
				void set(std::size_t) {}
			};

			template <class Position, class Vertex>
			struct set_position_attribute : set_attribute_base<Position> {
				void operator()() {
					set_attribute_base<Position>::template set<Vertex, 0>(Vertex::position_offset());
				}
			};
			template <class Normal, class Vertex>
			struct set_normal_attribute : set_attribute_base<Normal> {
				void operator()() {
					set_attribute_base<Normal>::template set<Vertex, 1>(Vertex::normal_offset());
				}
			};
			template <class Color, class Vertex>
			struct set_color_attribute : set_attribute_base<Color> {
				void operator()() {
					set_attribute_base<Color>::template set<Vertex, 2>(Vertex::color_offset());
				}
			};
			template <class TexCoord, class Vertex>
			struct set_tex_coord_attribute : set_attribute_base<TexCoord> {
				void operator()() {
					set_attribute_base<TexCoord>::template set<Vertex, 3>(Vertex::tex_coord_offset());
				}
			};
		}

		// A light-weight wrapper for (const ptr, size) pair
		// This class does not do any memory management; it is not an "owner" of the buffer
		template <class T, class SizeType = std::size_t>
		class buffer_span {
		public:
			using component_type = T;
			using size_type = SizeType;

			JKL_GPU_EXECUTABLE constexpr buffer_span(T const* ptr, size_type sz) noexcept
				: ptr{ ptr }, sz{ sz } {}

			JKL_GPU_EXECUTABLE T const* data() const noexcept { return ptr; }
			JKL_GPU_EXECUTABLE size_type size() const noexcept { return sz; }
			JKL_GPU_EXECUTABLE component_type const& operator[](size_type idx) const noexcept {
				assert(idx < sz);
				return ptr[idx];
			}
			JKL_GPU_EXECUTABLE auto byte_size() const noexcept {
				return sz * sizeof(T);
			}
			
		private:
			T const*		ptr;
			size_type		sz;
		};

		template <class T, class SizeType>
		buffer_span<T, SizeType> make_span(T const* ptr, SizeType sz) noexcept {
			return{ ptr, sz };
		}

		template <class Container,
			class = decltype(std::declval<Container const&>().data()),
			class = decltype(std::declval<Container const&>().size())>
		buffer_span<typename Container::value_type, typename Container::size_type>
			make_span(Container const& c)
			noexcept(noexcept(c.data()) && noexcept(c.size()))
		{
			return{ c.data(), c.size() };
		}

		// OpenGL vertex buffer object wrapper
		// VertexType can be vertex_type<Position, Normal, Color, TexCoord>
		// It can also be a custom data type, but restrictions on such a data type is complex
		template <class VertexType>
		class vertex_buffer {
		public:
			using vertex_type = VertexType;
			using position_type = typename vertex_type::position_type;
			using normal_type = typename vertex_type::normal_type;
			using color_type = typename vertex_type::color_type;
			using tex_coord_type = typename vertex_type::tex_coord_type;
			using value_type = vertex_type;

			template <class IndexUnitType>
			friend class index_buffer;
			template <class... BufferTypes>
			friend class buffer_ref_tuple;

			vertex_buffer() {
				check_error("Cannot create vertex_buffer: OpenGL context is broken");
				check_error(glGenVertexArrays, 1, &vao_id);
			}

			vertex_buffer(std::size_t capacity) : vertex_buffer{}  {
				allocate_buffer(capacity);
			}

			~vertex_buffer() {
				destroy_buffer();
				glDeleteVertexArrays(1, &vao_id);
			}

			// Perhaps it is possible to define a proper copy semantics, but I'm not sure how to do
			vertex_buffer(vertex_buffer const&) = delete;
			vertex_buffer& operator=(vertex_buffer const&) = delete;

			// Move operations: steal the resources (TBD)
			
			void copy_from_host(buffer_span<vertex_type> src) {
				if( src.size() > 0 ) {
					prepare_enough_buffer(src.size());

					mapped_host_ptr dst{ vbo_id, GL_WRITE_ONLY };
					memcpy(dst.get(), src.data(), src.byte_size());
				}
				size_ = src.size();
			}

		#ifdef JKL_USE_CUDA
			void copy_from_device(buffer_span<vertex_type> src, cudaStream_t stream = nullptr) {
				if( src.size() > 0 ) {
					prepare_enough_buffer(src.size());

					mapped_device_ptr dst{ cuda_rsrc, stream };
					jkj::cuda::check_error(cudaMemcpyAsync(dst.get(), src.data(),
						src.byte_size(), cudaMemcpyDeviceToDevice, stream));
				}
				size_ = src.size();
			}
		#endif

			void copy_to_host(vertex_type* dst) {
				if( size_ > 0 ) {
					mapped_host_ptr src{ vbo_id, GL_READ_ONLY };
					memcpy(dst, src.get(), sizeof(vertex_type) * size_);
				}
			}

		#ifdef JKL_USE_CUDA
			void copy_to_device(vertex_type* dst, cudaStream_t stream = nullptr) {
				if( size_ > 0 ) {
					mapped_device_ptr src{ cuda_rsrc, stream };
					jkj::cuda::check_error(cudaMemcpyAsync(dst, src.get(),
						sizeof(vertex_type) * size_, cudaMemcpyDeviceToDevice, stream));
				}
			}
		#endif

			auto size() const noexcept {
				return size_;
			}

			auto capacity() const noexcept {
				return capacity_;
			}

			auto draw_size() const noexcept {
				return size_;
			}

			void draw(GLenum mode = GL_POINTS) const {
				if( size_ > 0 ) {
					prepare_draw();
					check_error(glDrawArrays, mode, 0, GLsizei(size_));
				}
			}

		private:
			GLuint					vao_id;
			GLuint					vbo_id;
		#ifdef JKL_USE_CUDA
			cudaGraphicsResource_t	cuda_rsrc = nullptr;
		#endif

			std::size_t				size_ = 0;
			std::size_t				capacity_ = 0;

			void destroy_buffer() noexcept {
				if( capacity_ != 0 ) {
				#ifdef JKL_USE_CUDA
					cudaGraphicsUnregisterResource(cuda_rsrc);
					cuda_rsrc = nullptr;
				#endif
					glDeleteBuffers(1, &vbo_id);
					capacity_ = 0;
					size_ = 0;
				}
			}

			void allocate_buffer(std::size_t n) {
				check_error(glGenBuffers, 1, &vbo_id);
				check_error(glBindBuffer, GL_ARRAY_BUFFER, vbo_id);
				check_error(glBufferData, GL_ARRAY_BUFFER, sizeof(vertex_type) * n, nullptr, GL_DYNAMIC_DRAW);
			#ifdef JKL_USE_CUDA
				jkj::cuda::check_error(cudaGraphicsGLRegisterBuffer(&cuda_rsrc,
					vbo_id, cudaGraphicsRegisterFlagsWriteDiscard));
			#endif
				capacity_ = n;
			}

			void prepare_enough_buffer(std::size_t n) {
				if( capacity_ < n ) {
					destroy_buffer();
					allocate_buffer(n);
				}
				else {
					check_error(glBindBuffer, GL_ARRAY_BUFFER, vbo_id);
				}
			}

			void prepare_draw() const {
				check_error(glBindVertexArray, vao_id);
				check_error(glBindBuffer, GL_ARRAY_BUFFER, vbo_id);
				detail::set_position_attribute<position_type, vertex_type>{}();
				detail::set_normal_attribute<normal_type, vertex_type>{}();
				detail::set_color_attribute<color_type, vertex_type>{}();
				detail::set_tex_coord_attribute<tex_coord_type, vertex_type>{}();
			}
		};

		// OpenGL index buffer object wrapper
		// IndexUnitType can be either:
		// - An "array-like" type which has member type "component_type" and static constexpr member "components",
		//   where component_type is one of unsigned char, unsigned short, and unsigned int
		// - One of unsigned char, unsigned short, and unsigned int
		template <class IndexUnitType>
		class index_buffer {
			static constexpr bool unit_is_an_array = detail::has_component_type<IndexUnitType>::value
				&& detail::has_components<IndexUnitType>::value;

			template <bool is_array = unit_is_an_array, class = void>
			struct get_component_type {};

			template <class dummy>
			struct get_component_type<true, dummy> {
				using type = typename IndexUnitType::component_type;
			};

			template <class dummy>
			struct get_component_type<false, dummy> {
				using type = IndexUnitType;
			};

		public:
			using index_unit_type = IndexUnitType;
			using component_type = typename get_component_type<>::type;
			using value_type = index_unit_type;

			template <class... BufferTypes>
			friend class buffer_ref_tuple;

			index_buffer() {
				check_error("Cannot create index_buffer: OpenGL context is broken");
			}

			index_buffer(std::size_t capacity) : index_buffer{} {
				allocate_buffer(capacity);
			}

			~index_buffer() {
				destroy_buffer();
			}

			// Perhaps it is possible to define a proper copy semantics, but I'm not sure how to do
			index_buffer(index_buffer const&) = delete;
			index_buffer& operator=(index_buffer const&) = delete;

			// Move operations: steal the resources (TBD)

			void copy_from_host(buffer_span<index_unit_type> src) {
				if( src.size() > 0 ) {
					prepare_enough_buffer(src.size());

					mapped_host_ptr dst{ ibo_id, GL_WRITE_ONLY };
					memcpy(dst.get(), src.data(), src.byte_size());
				}
				size_ = src.size();
			}

		#ifdef JKL_USE_CUDA
			void copy_from_device(buffer_span<index_unit_type> src, cudaStream_t stream = nullptr) {
				if( src.size() > 0 ) {
					prepare_enough_buffer(src.size());

					mapped_device_ptr dst{ cuda_rsrc, stream };
					jkj::cuda::check_error(cudaMemcpyAsync(dst.get(), src.data(),
						src.byte_size(), cudaMemcpyDeviceToDevice, stream));
				}
				size_ = src.size();
			}
		#endif

			void copy_to_host(index_unit_type* dst) {
				if( size_ > 0 ) {
					mapped_host_ptr src{ ibo_id, GL_READ_ONLY };
					memcpy(dst, src, sizeof(index_unit_type) * size_);
				}
			}

		#ifdef JKL_USE_CUDA
			void copy_to_device(index_unit_type* dst, cudaStream_t stream = cudaStream_t(0)) {
				if( size_ > 0 ) {
					mapped_device_ptr src{ cuda_rsrc, stream };
					jkj::cuda::check_error(cudaMemcpyAsync(dst, src.get(),
						sizeof(index_unit_type) * size_, cudaMemcpyDeviceToDevice, stream));
				}
			}
		#endif

			auto size() const noexcept {
				return size_;
			}

			auto capacity() const noexcept {
				return capacity_;
			}

			auto draw_size() const noexcept {
				return size_;
			}

			template <class VertexType>
			void draw(vertex_buffer<VertexType> const& vbuffer, GLenum mode) const {
				if( size_ > 0 ) {
					vbuffer.prepare_draw();
					check_error(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, ibo_id);
					check_error(glDrawElements, mode,
						GLsizei(size_ * number_of_indices_per_unit<>::value),
						component_type_enum, nullptr);
				}
			}

			template <class VertexType, class = std::enable_if_t<detail::has_draw_mode<IndexUnitType>::value, VertexType>>
			void draw(vertex_buffer<VertexType> const& vbuffer) const {
				draw(vbuffer, IndexUnitType::draw_mode);
			}

		private:
			GLuint					ibo_id;
		#ifdef JKL_USE_CUDA
			cudaGraphicsResource_t	cuda_rsrc = nullptr;
		#endif

			std::size_t				size_ = 0;
			std::size_t				capacity_ = 0;

			template <bool is_array = unit_is_an_array, class = void>
			struct number_of_indices_per_unit {};

			template <class dummy>
			struct number_of_indices_per_unit<true, dummy>
				: std::integral_constant<std::size_t, index_unit_type::components> {};

			template <class dummy>
			struct number_of_indices_per_unit<false, dummy>
				: std::integral_constant<std::size_t, 1> {};

			static constexpr GLenum component_type_enum = detail::type_to_glenum<component_type, 1>::value;

			void destroy_buffer() noexcept {
				if( capacity_ != 0 ) {
				#ifdef JKL_USE_CUDA
					cudaGraphicsUnregisterResource(cuda_rsrc);
					cuda_rsrc = nullptr;
				#endif
					glDeleteBuffers(1, &ibo_id);
					capacity_ = 0;
					size_ = 0;
				}
			}

			void allocate_buffer(std::size_t n) {
				check_error(glGenBuffers, 1, &ibo_id);
				check_error(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, ibo_id);
				check_error(glBufferData, GL_ELEMENT_ARRAY_BUFFER, sizeof(index_unit_type) * n, nullptr, GL_DYNAMIC_DRAW);
			#ifdef JKL_USE_CUDA
				jkj::cuda::check_error(cudaGraphicsGLRegisterBuffer(&cuda_rsrc,
					ibo_id, cudaGraphicsRegisterFlagsWriteDiscard));
			#endif
				capacity_ = n;
			}

			void prepare_enough_buffer(std::size_t n) {
				if( capacity_ < n ) {
					destroy_buffer();
					allocate_buffer(n);
				}
				else {
					check_error(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, ibo_id);
				}
			}
		};

		// Light-weight reference container for simultaneous copy from device
		template <class... BufferTypes>
		class buffer_ref_tuple {
		public:
			template <std::size_t idx>
			using buffer_type = std::tuple_element_t<idx, std::tuple<BufferTypes...>>;
			template <std::size_t idx>
			using value_type = typename std::tuple_element_t<idx, std::tuple<BufferTypes...>>::value_type;
			static constexpr std::size_t number_of_buffers = sizeof...(BufferTypes);

			explicit buffer_ref_tuple(BufferTypes&... buffer_refs)
				: buffer_refs{ buffer_refs... }
			#ifdef JKL_USE_CUDA
				, copy_streams{ cudaEventDefault, number_of_buffers }
			#endif
			{}

			template <std::size_t idx>
			auto& get() noexcept {
				return std::get<idx>(buffer_refs);
			}
			template <std::size_t idx>
			auto& get() const noexcept {
				return std::get<idx>(buffer_refs);
			}

			// Just perform serialized multiple calls to copy_from_host
			template <class... BufferSpans>
			void copy_from_host(BufferSpans const&... srcs) {
				copy_from_host_impl(std::forward_as_tuple(srcs...));
			}

		#ifdef JKL_USE_CUDA
			// Only one pair of calls to cudaGraphicsMapResources()/cudaGraphicsUnmapResources()
			template <class BufferSpanTuple, class = std::enable_if_t<jkj::tmp::is_tuple<BufferSpanTuple>::value>>
			void copy_from_device(BufferSpanTuple const& srcs, cudaStream_t stream = nullptr) {
				// Prepare buffers
				prepare_enough_buffer_impl(srcs);

				// Map CUDA resources
				mapped_device_ptrs<number_of_buffers> dsts{
					make_array_of_cuda_rsrcs(std::make_index_sequence<number_of_buffers>{}), stream };

				// Perform copies simultaneously
				copy_streams.fork(stream, [&]() {
					copy_from_device_impl(dsts, srcs);
				});
			}

			template <class... BufferSpans>
			void copy_from_device(cudaStream_t stream, BufferSpans const&... srcs) {
				copy_from_device(std::forward_as_tuple(srcs...), stream);
			}
		#endif

		private:
			std::tuple<BufferTypes&...>	buffer_refs;

			template <std::size_t idx = 0, class BufferSpanTuple,
				class = std::enable_if_t<idx != number_of_buffers>>
			void copy_from_host_impl(BufferSpanTuple const& srcs) {
				using std::get;
				get<idx>(buffer_refs).copy_from_host(get<idx>(srcs));
				copy_from_host_impl<idx + 1>(srcs);
			}

			template <std::size_t idx, class BufferSpanTuple,
				class = std::enable_if_t<idx == number_of_buffers>, class = void>
			void copy_from_host_impl(BufferSpanTuple const&) {}

		#ifdef JKL_USE_CUDA
			jkj::cuda::stream_fork		copy_streams;

			template <std::size_t idx = 0, class BufferSpanTuple, class = std::enable_if_t<idx != number_of_buffers>>
			void prepare_enough_buffer_impl(BufferSpanTuple const& srcs) {
				using std::get;
				auto& buffer = get<idx>(buffer_refs);
				buffer.prepare_enough_buffer(get<idx>(srcs).size());
				prepare_enough_buffer_impl<idx + 1>(srcs);
			}

			template <std::size_t idx, class BufferSpanTuple, class = std::enable_if_t<idx == number_of_buffers>, class = void>
			void prepare_enough_buffer_impl(BufferSpanTuple const&) {}

			template <std::size_t... I>
			auto make_array_of_cuda_rsrcs(std::index_sequence<I...>)
			{
				return std::array<cudaGraphicsResource_t, sizeof...(I)>{ {
						std::get<I>(buffer_refs).cuda_rsrc...
					} };
			}

			template <std::size_t idx = 0, class BufferSpanTuple,
				class = std::enable_if_t<idx != number_of_buffers>>
			void copy_from_device_impl(mapped_device_ptrs<number_of_buffers>& dsts, BufferSpanTuple const& srcs)
			{
				using std::get;
				if( get<idx>(srcs).size() > 0 ) {
					auto dst_ptr = dsts.template get<idx>();
					auto& src = get<idx>(srcs);
					jkj::cuda::check_error(cudaMemcpyAsync(dst_ptr, src.data(),
						src.byte_size(), cudaMemcpyDeviceToDevice, copy_streams[idx]));
				}
				auto& buffer = get<idx>(buffer_refs);
				buffer.size_ = get<idx>(srcs).size();
				
				copy_from_device_impl<idx + 1>(dsts, srcs);
			}

			template <std::size_t idx, class BufferSpanTuple,
				class = std::enable_if_t<idx == number_of_buffers>, class = void>
			void copy_from_device_impl(mapped_device_ptrs<number_of_buffers>&, BufferSpanTuple const&) {}
		#endif
		};

		template <class... BufferTypes>
		auto make_buffer_ref_tuple(BufferTypes&... buffer_refs) {
			return buffer_ref_tuple<BufferTypes...>{ buffer_refs... };
		}
	}
}