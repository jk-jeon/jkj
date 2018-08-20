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
#include <functional>
#include <memory>
#include <nana/gui/widgets/form.hpp>
#include "../../drawings.h"
#include "../../utilities.h"
#include "../../tmp/identity.h"

namespace jkl {
	namespace gui {
		namespace detail {
			class opengl_draw_form_base : public nana::nested_form, private util::fps_counter {
			public:
				using camera_t = drawings::camera<float>;

			private:
				template <typename T>
				static T relaxed_load(T x) {
					return x;
				}
				template <typename T>
				static T relaxed_load(std::atomic<T> const& x) {
					return x.load(std::memory_order_relaxed);
				}

				template <template <class> class AtomicOrNonatomic>
				struct view_settings_impl {
					AtomicOrNonatomic<camera_t>	initial_camera;

					AtomicOrNonatomic<float>	move_speed;	// unit/pixel
					AtomicOrNonatomic<float>	zoom_speed;	// unit/pixel
					AtomicOrNonatomic<float>	rot_speed;	// rad/pixel

					AtomicOrNonatomic<float>	field_of_view;
					AtomicOrNonatomic<float>	near_plane;
					AtomicOrNonatomic<float>	far_plane;

					constexpr view_settings_impl(camera_t const& initial_camera,
						float move_speed, float zoom_speed, float rot_speed,
						float field_of_view, float near_plane, float far_plane) :
						initial_camera(initial_camera),
						move_speed{ move_speed }, zoom_speed{ zoom_speed }, rot_speed{ rot_speed },
						field_of_view{ field_of_view }, near_plane{ near_plane }, far_plane{ far_plane } {}

					template <template <class> class Other>
					constexpr view_settings_impl(view_settings_impl<Other> const& other) :
						initial_camera(relaxed_load(other.initial_camera)),
						move_speed{ relaxed_load(other.move_speed) },
						zoom_speed{ relaxed_load(other.zoom_speed) },
						rot_speed{ relaxed_load(other.rot_speed) },
						field_of_view{ relaxed_load(other.field_of_view) },
						near_plane{ relaxed_load(other.near_plane) },
						far_plane{ relaxed_load(other.far_plane) } {}
				};

			public:
				using view_settings_t = view_settings_impl<tmp::identity_t>;
				camera_t const& camera() const noexcept;
				void initial_camera(camera_t const& initial_camera);
				camera_t initial_camera() const;
				void camera_speed(float move_speed, float zoom_speed, float rot_speed);
				float move_speed() const;
				float zoom_speed() const;
				float rot_speed() const;
				void camera_projection(float field_of_view, float near_plane, float far_plane);
				float field_of_view() const;
				float near_plane() const;
				float far_plane() const;
				void view_settings(view_settings_t const& new_settings);
				view_settings_t view_settings() const;

				using fps_counter::fps;

				// Return camera to the initial position set by initial_camera()
				void reset_camera() noexcept;

				// Draw coordinate axis
				void draw_axis(bool draw) noexcept;
				bool draw_axis() const noexcept;
				bool toggle_draw_axis() noexcept;			// returns new value

				void axis_length(float length) noexcept;
				float axis_length() noexcept;

				// Draw camera center
				void draw_camera_center(bool draw) noexcept;
				bool draw_camera_center() const noexcept;
				bool toggle_draw_camera_center() noexcept;	// returns new value

				void camera_center_radius(float radius) noexcept;
				float camera_center_radius() noexcept;

				// Wireframe rendering mode
				void wireframe(bool enable) noexcept;
				bool wireframe() const noexcept;
				bool toggle_wireframe() noexcept;			// returns new value

				// Back-face culling mode
				void back_face_culling(bool enable) noexcept;
				bool back_face_culling() const noexcept;
				bool toggle_back_face_culling() noexcept;	// returns new value

			protected:
				opengl_draw_form_base(nana::form const& parent, view_settings_t const& cs,
					nana::rectangle const& region, std::mutex& process_mutex, std::mutex& output_mutex);
				opengl_draw_form_base(nana::form const& parent,
					nana::rectangle const& region, std::mutex& process_mutex, std::mutex& output_mutex);
				~opengl_draw_form_base();

				void before_work();
				void after_work();

				// Rendering routine
				void process();
				// Let the window show the drawn contents
				// This function should be called only after process() have been succesfully called
				// Don't call this function inside the main thread (the thread called nana::exec())!
				void produce();

			private:
				struct impl_base;
				struct impl;
				std::unique_ptr<impl> const	m_impl_ptr;
				std::mutex&					m_process_mutex;
				std::mutex&					m_output_mutex;

				virtual void draw() const = 0;
			};
		}
	}
}