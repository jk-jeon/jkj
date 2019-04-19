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
#include "detail/opengl_draw_form_base.h"
#include "../strmnet/node.h"
// I will replace this with std::optional since it is available
#include "../optional.h"

/// A simple OpenGL renderer attached to nana::nested_form
/// Vertex type and index type can be configured using template parameters.
/// Other common components are factored in the implementation class detail::opengl_draw_form_base,
/// which is a non-template class.

// Coordinate frame
//
//   z |
//     |  / y
//     | /
//     |/______ x
//
//

namespace jkj {
	namespace gui {
#ifdef JKL_USE_CUDA
		struct gpu_input_tag {
			gpu_input_tag(int gpu_id) : gpu_id{ gpu_id } {}
			int gpu_id;
		};
#endif

		template <class VertexType = drawings::vertex<
			math::R3_elmtf,
			math::R3_elmtf,
			math::R3_elmt<std::uint8_t>>,
			class IndexType = drawings::triangle<unsigned int>>
		class opengl_draw_form :
			public strmnet::node<opengl_draw_form<VertexType, IndexType>, strmnet::update_policy_fps>,
			public detail::opengl_draw_form_base
		{
		public:
			using crtp_base_type = typename strmnet::node<opengl_draw_form, strmnet::update_policy_fps>::crtp_base_type;
			friend crtp_base_type;

			using camera_t = typename detail::opengl_draw_form_base::camera_t;
			using vertex_t = VertexType;
			using index_t = IndexType;

			using vertex_buffer_type = drawings::vertex_buffer<vertex_t>;
			using index_buffer_type = drawings::index_buffer<index_t>;
			using buffer_refs = drawings::buffer_ref_tuple<vertex_buffer_type, index_buffer_type>;

			opengl_draw_form(nana::form const& parent,
				nana::rectangle const& region = {}, float target_fps = 60.0f)
				: detail::opengl_draw_form_base{ parent, region, process_mutex(), output_mutex() }
			{
				// Set target FPS
				set_fps(target_fps);
			}
			opengl_draw_form(nana::form const& parent, view_settings_t const& cs,
				nana::rectangle const& region = {}, float target_fps = 60.0f)
				: detail::opengl_draw_form_base{ parent, cs, region, process_mutex(), output_mutex() }
			{
				// Set target FPS
				set_fps(target_fps);
			}			

		protected:
			// Vertex & index buffers are lazy-intialized; the first call to get_buffer() will create them.
			template <class... BufferSpans>
			void input(BufferSpans const&... srcs) {
				get_buffers().copy_from_host(srcs...);
			}

#ifdef JKL_USE_CUDA
			// For the case of GPU buffers, the buffers are also created at the call to get_buffers_stream_pair().
			// If get_buffers_stream_pair() is called after the buffers are already created with
			// gpu_id that is different from that associated to the buffers,
			// the buffers are destroyed and recreated according to the new gpu_id.
			// Frankly speaking, I consider this design awful, but I don't have any better idea to make it possible
			// to pass buffers created on a non-default GPU to another thread, partially due to the (miserable) fact
			// that the selceted GPU ID is a thread local global variable in CUDA.

			template <class... BufferSpans>
			void input(gpu_input_tag tag, BufferSpans const&... srcs) {
				// It is not necessary to call cudaStreamScynchronize(), since cudaGraphicsUnmapResources() already
				// synchronizes the stream.
				auto buffers_stream_pair = get_buffers_stream_pair(tag.gpu_id);
				buffers_stream_pair.first.copy_from_device(buffers_stream_pair.second, srcs...);
			}
#endif
			using detail::opengl_draw_form_base::before_work;
			void after_work() {
				m_buffers.reset();
				detail::opengl_draw_form_base::after_work();
			}

			// Rendering routine
			using detail::opengl_draw_form_base::process;

			// Let the window show the drawn contents
			// This function should be called only after process() have been succesfully called
			using detail::opengl_draw_form_base::produce;

			// Vertex / Index buffers
			struct buffer_wrapper {
				buffer_wrapper() : refs{ vertices, indices }
				{
#ifdef JKL_USE_CUDA
					jkj::cuda::check_error(cudaGetDevice(&current_gpu_id));
#endif
				}

				vertex_buffer_type					vertices;
				index_buffer_type					indices;
				buffer_refs							refs;
#ifdef JKL_USE_CUDA
				int									current_gpu_id;
				cuda::stream						stream;
#endif
			};
			optional<buffer_wrapper> m_buffers;
			
			auto& get_buffers() noexcept
			{
				if( !m_buffers ) {
					m_buffers.emplace();
				}
				return m_buffers->refs;
			}

#ifdef JKL_USE_CUDA
			std::pair<buffer_refs&, cudaStream_t> get_buffers_stream_pair(int gpu_id) noexcept
			{
				if( !m_buffers || m_buffers->current_gpu_id != gpu_id ) {
					cuda::check_error(cudaSetDevice(gpu_id));
					m_buffers.emplace();
				}
				return{ m_buffers->refs, m_buffers->stream.get() };
			}
#endif

			void draw() const override {
				if( m_buffers ) {
					m_buffers->indices.draw(m_buffers->vertices);
				}
			}
		};
	}
}