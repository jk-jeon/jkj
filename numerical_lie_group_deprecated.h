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
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "portability.h"

#ifdef JKL_USE_CUDA
#include <vector_types.h>
#endif

namespace jkl {
	using std::get;

#ifdef JKL_USE_CUDA	
	namespace detail {
		template <class T, std::size_t N>
		struct to_cuda_builtin {
			using type = T[N];
		};

		template <>
		struct to_cuda_builtin<char, 1> {
			using type = char1;
		};
		template <>
		struct to_cuda_builtin<char, 2> {
			using type = char2;
		};
		template <>
		struct to_cuda_builtin<char, 3> {
			using type = char3;
		};
		template <>
		struct to_cuda_builtin<char, 4> {
			using type = char4;
		};
		template <>
		struct to_cuda_builtin<unsigned char, 1> {
			using type = uchar1;
		};
		template <>
		struct to_cuda_builtin<unsigned char, 2> {
			using type = uchar2;
		};
		template <>
		struct to_cuda_builtin<unsigned char, 3> {
			using type = uchar3;
		};
		template <>
		struct to_cuda_builtin<unsigned char, 4> {
			using type = uchar4;
		};

		template <>
		struct to_cuda_builtin<short, 1> {
			using type = short1;
		};
		template <>
		struct to_cuda_builtin<short, 2> {
			using type = short2;
		};
		template <>
		struct to_cuda_builtin<short, 3> {
			using type = short3;
		};
		template <>
		struct to_cuda_builtin<short, 4> {
			using type = short4;
		};
		template <>
		struct to_cuda_builtin<unsigned short, 1> {
			using type = ushort1;
		};
		template <>
		struct to_cuda_builtin<unsigned short, 2> {
			using type = ushort2;
		};
		template <>
		struct to_cuda_builtin<unsigned short, 3> {
			using type = ushort3;
		};
		template <>
		struct to_cuda_builtin<unsigned short, 4> {
			using type = ushort4;
		};

		template <>
		struct to_cuda_builtin<int, 1> {
			using type = int1;
		};
		template <>
		struct to_cuda_builtin<int, 2> {
			using type = int2;
		};
		template <>
		struct to_cuda_builtin<int, 3> {
			using type = int3;
		};
		template <>
		struct to_cuda_builtin<int, 4> {
			using type = int4;
		};
		template <>
		struct to_cuda_builtin<unsigned int, 1> {
			using type = uint1;
		};
		template <>
		struct to_cuda_builtin<unsigned int, 2> {
			using type = uint2;
		};
		template <>
		struct to_cuda_builtin<unsigned int, 3> {
			using type = uint3;
		};
		template <>
		struct to_cuda_builtin<unsigned int, 4> {
			using type = uint4;
		};

		template <>
		struct to_cuda_builtin<long, 1> {
			using type = long1;
		};
		template <>
		struct to_cuda_builtin<long, 2> {
			using type = long2;
		};
		template <>
		struct to_cuda_builtin<long, 3> {
			using type = long3;
		};
		template <>
		struct to_cuda_builtin<long, 4> {
			using type = long4;
		};
		template <>
		struct to_cuda_builtin<unsigned long, 1> {
			using type = ulong1;
		};
		template <>
		struct to_cuda_builtin<unsigned long, 2> {
			using type = ulong2;
		};
		template <>
		struct to_cuda_builtin<unsigned long, 3> {
			using type = ulong3;
		};
		template <>
		struct to_cuda_builtin<unsigned long, 4> {
			using type = ulong4;
		};
		
		template <>
		struct to_cuda_builtin<long long, 1> {
			using type = longlong1;
		};
		template <>
		struct to_cuda_builtin<long long, 2> {
			using type = longlong2;
		};
		template <>
		struct to_cuda_builtin<long long, 3> {
			using type = longlong3;
		};
		template <>
		struct to_cuda_builtin<long long, 4> {
			using type = longlong4;
		};
		template <>
		struct to_cuda_builtin<unsigned long long, 1> {
			using type = ulonglong1;
		};
		template <>
		struct to_cuda_builtin<unsigned long long, 2> {
			using type = ulonglong2;
		};
		template <>
		struct to_cuda_builtin<unsigned long long, 3> {
			using type = ulonglong3;
		};
		template <>
		struct to_cuda_builtin<unsigned long long, 4> {
			using type = ulonglong4;
		};

		template <>
		struct to_cuda_builtin<float, 1> {
			using type = float1;
		};
		template <>
		struct to_cuda_builtin<float, 2> {
			using type = float2;
		};
		template <>
		struct to_cuda_builtin<float, 3> {
			using type = float3;
		};
		template <>
		struct to_cuda_builtin<float, 4> {
			using type = float4;
		};

		template <>
		struct to_cuda_builtin<double, 1> {
			using type = double1;
		};
		template <>
		struct to_cuda_builtin<double, 2> {
			using type = double2;
		};
		template <>
		struct to_cuda_builtin<double, 3> {
			using type = double3;
		};
		template <>
		struct to_cuda_builtin<double, 4> {
			using type = double4;
		};

		template <class T, std::size_t N>
		struct has_cuda_builtin : std::conditional_t<
			std::is_same<typename to_cuda_builtin<T, N>::type, T[N]>::value,
			std::false_type,
			std::true_type> {};
	}

	template <class T, std::size_t N, bool use_cuda_builtin = detail::has_cuda_builtin<T, N>::value>
	class gpu_array;
	
	template <class T, std::size_t N>
	class gpu_array<T, N, false>
	{
		T	m_data[N];
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE T& operator[](std::size_t i) noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE constexpr T const& operator[](std::size_t i) const noexcept { return m_data[i]; }
	};

	template <class T>
	class gpu_array<T, 1, false> {
		T m_data;
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x) noexcept : m_data{ x } {}
		JKL_GPU_EXECUTABLE T& operator[](std::size_t i) noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE constexpr T const& operator[](std::size_t i) const noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data; }
	};

	template <class T>
	class gpu_array<T, 2, false> {
		T m_data[2];
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x, T const& y) noexcept : m_data{ x, y } {}
		JKL_GPU_EXECUTABLE T& operator[](std::size_t i) noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE constexpr T const& operator[](std::size_t i) const noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data[1]; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data[1]; }
	};

	template <class T>
	class gpu_array<T, 3, false> {
		T m_data[3];
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x, T const& y, T const& z) noexcept : m_data{ x, y, z } {}
		JKL_GPU_EXECUTABLE T& operator[](std::size_t i) noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE constexpr T const& operator[](std::size_t i) const noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data[1]; }
		JKL_GPU_EXECUTABLE T& z() noexcept { return m_data[2]; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data[1]; }
		JKL_GPU_EXECUTABLE constexpr T const& z() const noexcept { return m_data[2]; }
	};

	template <class T>
	class gpu_array<T, 4, false> {
		T m_data[4];
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& w, T const& x, T const& y, T const& z) noexcept
			: m_data{ w, x, y, z } {}
		JKL_GPU_EXECUTABLE T& operator[](std::size_t i) noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE constexpr T const& operator[](std::size_t i) const noexcept { return m_data[i]; }
		JKL_GPU_EXECUTABLE T& w() noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data[1]; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data[2]; }
		JKL_GPU_EXECUTABLE T& z() noexcept { return m_data[3]; }
		JKL_GPU_EXECUTABLE constexpr T const& w() const noexcept { return m_data[0]; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data[1]; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data[2]; }
		JKL_GPU_EXECUTABLE constexpr T const& z() const noexcept { return m_data[3]; }
	};

	template <class T>
	class gpu_array<T, 1, true> {
		using internal_type = typename detail::to_cuda_builtin<T, 1>::type;
		internal_type m_data;
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(internal_type const& data) noexcept : m_data(data) {}
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x) noexcept : m_data{ x } {}
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data.x; }
	};

	template <class T>
	class gpu_array<T, 2, true> {
		using internal_type = typename detail::to_cuda_builtin<T, 2>::type;
		internal_type m_data;
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(internal_type const& data) noexcept : m_data(data) {}
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x, T const& y) noexcept : m_data{ x, y } {}
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data.y; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data.y; }
	};

	template <class T>
	class gpu_array<T, 3, true> {
		using internal_type = typename detail::to_cuda_builtin<T, 3>::type;
		internal_type m_data;
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(internal_type const& data) noexcept : m_data(data) {}
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& x, T const& y, T const& z) noexcept : m_data{ x, y, z } {}
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data.y; }
		JKL_GPU_EXECUTABLE T& z() noexcept { return m_data.z; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data.y; }
		JKL_GPU_EXECUTABLE constexpr T const& z() const noexcept { return m_data.z; }
	};

	template <class T>
	class gpu_array<T, 4, true> {
		using internal_type = typename detail::to_cuda_builtin<T, 4>::type;
		internal_type m_data;
	public:
		gpu_array() = default;
		JKL_GPU_EXECUTABLE constexpr gpu_array(internal_type const& data) noexcept : m_data(data) {}
		JKL_GPU_EXECUTABLE constexpr gpu_array(T const& w, T const& x, T const& y, T const& z) noexcept
			: m_data{ w, x, y, z } {}
		JKL_GPU_EXECUTABLE T& w() noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE T& x() noexcept { return m_data.y; }
		JKL_GPU_EXECUTABLE T& y() noexcept { return m_data.z; }
		JKL_GPU_EXECUTABLE T& z() noexcept { return m_data.w; }
		JKL_GPU_EXECUTABLE constexpr T const& w() const noexcept { return m_data.x; }
		JKL_GPU_EXECUTABLE constexpr T const& x() const noexcept { return m_data.y; }
		JKL_GPU_EXECUTABLE constexpr T const& y() const noexcept { return m_data.z; }
		JKL_GPU_EXECUTABLE constexpr T const& z() const noexcept { return m_data.w; }
	};

	template <class T, std::size_t N>
	using array_t = gpu_array<T, N>;

	namespace detail {
		template <std::size_t idx, class T, std::size_t N, bool use_cuda_builtin>
		struct gpu_array_get_impl;

		template <std::size_t idx, class T, std::size_t N>
		struct gpu_array_get_impl<idx, T, N, false> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, N, false>& a) noexcept {
				return a[idx];
			}
			JKL_GPU_EXECUTABLE static constexpr auto& get(gpu_array<T, N, false> const& a) noexcept {
				return a[idx];
			}
		};
		
		template <class T>
		struct gpu_array_get_impl<0, T, 1, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 1, true>& a) noexcept {
				return a.x();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 1, true> const& a) noexcept {
				return a.x();
			}
		};

		template <class T>
		struct gpu_array_get_impl<0, T, 2, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 2, true>& a) noexcept {
				return a.x();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 2, true> const& a) noexcept {
				return a.x();
			}
		};
		template <class T>
		struct gpu_array_get_impl<1, T, 2, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 2, true>& a) noexcept {
				return a.y();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 2, true> const& a) noexcept {
				return a.y();
			}
		};

		template <class T>
		struct gpu_array_get_impl<0, T, 3, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 3, true>& a) noexcept {
				return a.x();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 3, true> const& a) noexcept {
				return a.x();
			}
		};
		template <class T>
		struct gpu_array_get_impl<1, T, 3, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 3, true>& a) noexcept {
				return a.y();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 3, true> const& a) noexcept {
				return a.y();
			}
		};
		template <class T>
		struct gpu_array_get_impl<2, T, 3, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 3, true>& a) noexcept {
				return a.z();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 3, true> const& a) noexcept {
				return a.z();
			}
		};

		template <class T>
		struct gpu_array_get_impl<0, T, 4, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 4, true>& a) noexcept {
				return a.w();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 4, true> const& a) noexcept {
				return a.w();
			}
		};
		template <class T>
		struct gpu_array_get_impl<1, T, 4, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 4, true>& a) noexcept {
				return a.x();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 4, true> const& a) noexcept {
				return a.x();
			}
		};
		template <class T>
		struct gpu_array_get_impl<2, T, 4, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 4, true>& a) noexcept {
				return a.y();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 4, true> const& a) noexcept {
				return a.y();
			}
		};
		template <class T>
		struct gpu_array_get_impl<3, T, 4, true> {
			JKL_GPU_EXECUTABLE static auto& get(gpu_array<T, 4, true>& a) noexcept {
				return a.z();
			}
			JKL_GPU_EXECUTABLE static constexpr auto const& get(gpu_array<T, 4, true> const& a) noexcept {
				return a.z();
			}
		};
	}

	template <std::size_t idx, class T, std::size_t N, bool use_cuda_builtin>
	JKL_GPU_EXECUTABLE auto& get(gpu_array<T, N, use_cuda_builtin>& a) noexcept {
		return detail::gpu_array_get_impl<idx, T, N, use_cuda_builtin>::get(a);
	}
	template <std::size_t idx, class T, std::size_t N, bool use_cuda_builtin>
	JKL_GPU_EXECUTABLE constexpr auto const& get(gpu_array<T, N, use_cuda_builtin> const& a) noexcept {
		return detail::gpu_array_get_impl<idx, T, N, use_cuda_builtin>::get(a);
	}

#else
	template <class T, std::size_t N>
	using array_t = std::array<T, N>;
#endif

	namespace math {
		// Get zero element of the given type
		namespace detail {
			template <class T, class = void>
			struct has_zero : std::false_type {};

			template <class T>
			struct has_zero<T, std::void_t<decltype(T::zero())>> : std::true_type {};
		}

		template <class T, class = std::enable_if_t<detail::has_zero<std::decay_t<T>>::value>>
		JKL_GPU_EXECUTABLE static constexpr auto zero() noexcept(noexcept(T::zero())) {
			return T::zero();
		}

		template <class T, class = std::enable_if_t<!detail::has_zero<std::decay_t<T>>::value>, class = void>
		JKL_GPU_EXECUTABLE static constexpr auto zero() noexcept(noexcept(T(0))) {
			return T(0);
		}

		// Get unity element of the given type
		namespace detail {
			template <class T, class = void>
			struct has_unity : std::false_type {};

			template <class T>
			struct has_unity<T, std::void_t<decltype(T::unity())>> : std::true_type {};
		}

		template <class T, class = std::enable_if_t<detail::has_unity<std::decay_t<T>>::value>>
		JKL_GPU_EXECUTABLE static constexpr auto unity() noexcept(noexcept(T::unity())) {
			return T::unity();
		}

		template <class T, class = std::enable_if_t<!detail::has_unity<std::decay_t<T>>::value>, class = void>
		JKL_GPU_EXECUTABLE static constexpr auto unity() noexcept(noexcept(T(1))) {
			return T(1);
		}

		// Get inverse
		// The name "general_inverse" is somewhat ugly, but I think the name "inv" can be potentially dangerous,
		// since there can be other functions with the same name in the namespace jkl::math.
		// Universal reference is too greedy to define the default fallback behavior.
		namespace detail {
			template <class T, class = void>
			struct has_mem_inv : std::false_type {};

			template <class T>
			struct has_mem_inv<T, std::void_t<decltype(&T::inv)>> : std::true_type {};
		}

		template <class T, class = std::enable_if_t<detail::has_mem_inv<std::decay_t<T>>::value>>
		JKL_GPU_EXECUTABLE static constexpr auto general_inverse(T&& x) noexcept(noexcept(std::declval<T>().inv())) {
			return std::forward<T>(x).inv();
		}
		
		template <class T, class = std::enable_if_t<!detail::has_mem_inv<std::decay_t<T>>::value>, class = void>
		JKL_GPU_EXECUTABLE static constexpr auto general_inverse(T&& x) noexcept(noexcept(unity<std::decay_t<T>>() / std::declval<T>())) {
			return unity<std::decay_t<T>>() / std::forward<T>(x);
		}

		template <class Float = float>
		struct numeric_consts {
			using number_type = Float;
			static constexpr Float pi = Float(3.141592653589793238462643383279502884197169399375105820974944592307816406286);
			static constexpr Float eps = std::numeric_limits<Float>().epsilon() * 10;
			static constexpr Float sqrt_pi = Float(1.77245385090551602729816748334114518279754945612238712821380778985291128459103);
			static constexpr Float exp1 = Float(2.71828182845904523536028747135266249775724709369995957496696762772407663035354);
		};

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE static constexpr bool close_to(Float1 const& x, Float2 const& y) noexcept {
			return x - y < numeric_consts<std::decay_t<decltype(x - y)>>::eps &&
				y - x < numeric_consts<std::decay_t<decltype(x - y)>>::eps;
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool close_to_zero(Float const& x) noexcept {
			return close_to(x, jkl::math::zero<Float>());
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool slightly_larger(Float1 const& x, Float2 const& y) noexcept {
			return x > y + numeric_consts<std::decay_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool slightly_smaller(Float1 const& x, Float2 const& y) noexcept {
			return x < y - numeric_consts<std::decay_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool almost_larger(Float1 const& x, Float2 const& y) noexcept {
			return x > y - numeric_consts<std::decay_t<decltype(x - y)>>::eps;
		}
		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr bool almost_smaller(Float1 const& x, Float2 const& y) noexcept {
			return x < y + numeric_consts<std::decay_t<decltype(x - y)>>::eps;
		}

		// A flag class to request to constructors not to check preconditions
		// A constructor may throw if this flag is not set and the input violates the preconditions
		struct do_not_check_validity {};

		// N-dimensional Eulclidean space
		template <class Float, std::size_t N>
		class Rn_elmt {
			array_t<Float, N> r_;
		public:
			using element_type = Float;
			static constexpr std::size_t components = N;

			// Default constructor; components are filled with garbages
			Rn_elmt() = default;

			// Construction from initilizer-list
			// - Lists with length longer than N are trimmed
			// - Undefined behaviour if the list has shorter length
			Rn_elmt(std::initializer_list<Float> list) noexcept {
				auto itr = list.begin();
				for( std::size_t i = 0; i < N; ++i, ++itr )
					r_[i] = *itr;
			}

			// Convert to vector of other element type
			// enable_if swith is mandatory; otherwise, the compiler may think any 
			// Rn_elmt<Float1> is convertible to Rn_elmt<Float2>, even if 
			// the construction from one to another may fail
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value && 
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE operator Rn_elmt<OtherFloat, N>() const noexcept {
				Rn_elmt<OtherFloat, N> ret_value;
				for( std::size_t i = 0; i < N; ++i )
					ret_value[i] = r_[i];
				return ret_value;
			}
			
			JKL_GPU_EXECUTABLE Float& operator[](std::size_t idx) noexcept { return r_[idx]; }
			JKL_GPU_EXECUTABLE constexpr Float const& operator[](std::size_t idx) const noexcept { return r_[idx]; }

			JKL_GPU_EXECUTABLE Float normsq() const noexcept {
				Float sum = 0;
				for( std::size_t i = 0; i < N; ++i )
					sum += r_[i] * r_[i];
				return sum;
			}

			JKL_GPU_EXECUTABLE auto norm() const noexcept {
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE Rn_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE Rn_elmt operator-() const noexcept {
				Rn_elmt ret_value;
				for( std::size_t i = 0; i < N; ++i )
					ret_value[i] = -r_[i];
				return ret_value;
			}
			
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(Rn_elmt<OtherFloat, N> const& that) noexcept {
				for( std::size_t i = 0; i < N; ++i )
					r_[i] += that[i];
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(Rn_elmt<OtherFloat, N> const& that) noexcept {
				for( std::size_t i = 0; i < N; ++i )
					r_[i] -= that[i];
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				for( std::size_t i = 0; i < N; ++i )
					r_[i] *= k;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(OtherFloat const& k) noexcept {
				for( std::size_t i = 0; i < N; ++i )
					r_[i] /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE auto& normalize() noexcept {
				if( normsq() != 0 )
					return operator/=(norm());
				else
					return *this;
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE bool operator==(Rn_elmt<OtherFloat, N> const& v) const noexcept {
				for( std::size_t i = 0; i < N; ++i )
					if( r_[i] != v[i] )
						return false;
				return true;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE bool operator!=(Rn_elmt<OtherFloat, N> const& v) const noexcept {
				for( std::size_t i = 0; i < N; ++i )
					if( r_[i] != v[i] )
						return true;
				return false;
			}

			JKL_GPU_EXECUTABLE static Rn_elmt zero() noexcept {
				Rn_elmt ret;
				for( std::size_t i = 0; i < N; ++i )
					ret[i] = 0;
				return ret;
			}
		};

		// 2-dimensional Eulclidean space
		template <class Float>
		using R2_elmt = Rn_elmt<Float, 2>;
		template <class Float>
		class Rn_elmt<Float, 2> {
			array_t<Float, 2> r_;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 2;

			// Default constructor; components are filled with garbages
			Rn_elmt() = default;

			JKL_GPU_EXECUTABLE constexpr Rn_elmt(Float const& x, Float const& y) noexcept : r_{ x, y } {}

			// Convert to vector of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator R2_elmt<OtherFloat>() const noexcept {
				return R2_elmt<OtherFloat>(x(), y());
			}
			
			/* Do not use these members */
			JKL_GPU_EXECUTABLE Float& operator[](std::size_t idx) noexcept;
			JKL_GPU_EXECUTABLE constexpr Float const& operator[](std::size_t idx) const noexcept;

			JKL_GPU_EXECUTABLE Float& x() noexcept { return get<0>(r_); }
			JKL_GPU_EXECUTABLE Float& y() noexcept { return get<1>(r_); }

			JKL_GPU_EXECUTABLE constexpr Float const& x() const noexcept { return get<0>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& y() const noexcept { return get<1>(r_); }

			JKL_GPU_EXECUTABLE constexpr auto normsq() const noexcept { return x() * x() + y() * y(); }

			JKL_GPU_EXECUTABLE auto norm() const noexcept {
				using std::sqrt;
				return sqrt(normsq());
			}
			
			JKL_GPU_EXECUTABLE constexpr Rn_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator-() const noexcept {
				return{ -x(), -y() };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(R2_elmt<OtherFloat> const& that) noexcept {
				x() += that.x();
				y() += that.y();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(R2_elmt<OtherFloat> const& that) noexcept {
				x() -= that.x();
				y() -= that.y();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				x() *= k;
				y() *= k;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(OtherFloat const& k) noexcept {
				x() /= k;
				y() /= k;
				return *this;
			}

			JKL_GPU_EXECUTABLE auto& normalize() noexcept {
				if( normsq() != 0 )
					return operator/=(norm());
				else
					return *this;
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(R2_elmt<OtherFloat> const& v) const noexcept {
				return x() == v.x() && y() == v.y();
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(R2_elmt<OtherFloat> const& v) const noexcept {
				return !(*this == v);
			}

			JKL_GPU_EXECUTABLE static constexpr Rn_elmt zero() noexcept {
				return{ jkl::math::zero<Float>(), jkl::math::zero<Float>() };
			}
		};

		// 3-dimensional Euclidean space
		template <class Float>
		using R3_elmt = Rn_elmt<Float, 3>;
		template <class Float>
		class Rn_elmt<Float, 3> {
			array_t<Float, 3> r_;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 3;

			// Default constructor; components are filled with garbages
			Rn_elmt() = default;

			JKL_GPU_EXECUTABLE constexpr Rn_elmt(Float const& x, Float const& y, Float const& z) noexcept
				: r_{ x, y, z } {}

			// Convert to vector of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator R3_elmt<OtherFloat>() const noexcept {
				return R3_elmt<OtherFloat>(x(), y(), z());
			}
			
			/* Do not use these members */
			JKL_GPU_EXECUTABLE Float& operator[](std::size_t idx) noexcept;
			JKL_GPU_EXECUTABLE constexpr Float const& operator[](std::size_t idx) const noexcept;
			
			JKL_GPU_EXECUTABLE Float& x() noexcept { return get<0>(r_); }
			JKL_GPU_EXECUTABLE Float& y() noexcept { return get<1>(r_); }
			JKL_GPU_EXECUTABLE Float& z() noexcept { return get<2>(r_); }

			JKL_GPU_EXECUTABLE constexpr Float const& x() const noexcept { return get<0>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& y() const noexcept { return get<1>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& z() const noexcept { return get<2>(r_); }

			JKL_GPU_EXECUTABLE constexpr auto normsq() const noexcept { return x() * x() + y() * y() + z() * z(); }

			JKL_GPU_EXECUTABLE auto norm() const noexcept {
				using std::sqrt;
				return sqrt(normsq());
			}

			JKL_GPU_EXECUTABLE constexpr Rn_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE constexpr Rn_elmt operator-() const noexcept {
				return{ -x(), -y(), -z() };
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(R3_elmt<OtherFloat> const& that) noexcept {
				x() += that.x();
				y() += that.y();
				z() += that.z();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(R3_elmt<OtherFloat> const& that) noexcept {
				x() -= that.x();
				y() -= that.y();
				z() -= that.z();
				return *this;
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				x() *= k;
				y() *= k;
				z() *= k;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(OtherFloat const& k) noexcept {
				x() /= k;
				y() /= k;
				z() /= k;
				return *this;
			}
			JKL_GPU_EXECUTABLE auto& normalize() noexcept {
				if( normsq() != 0 )
					return operator/=(norm());
				else
					return *this;
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(R3_elmt<OtherFloat> const& v) const noexcept {
				return x() == v.x() && y() == v.y() && z() == v.z();
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(R3_elmt<OtherFloat> const& v) const noexcept {
				return !(*this == v);
			}

			JKL_GPU_EXECUTABLE static constexpr Rn_elmt zero() noexcept {
				return{ jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>() };
			}
		};

		// General linear group of degree 2
		template <class Float>
		class gl2_elmt;
		template <class Float>
		class GL2_elmt : public gl2_elmt<Float> {
		public:
			FORCEINLINE JKL_GPU_EXECUTABLE constexpr GL2_elmt() noexcept
				: gl2_elmt<Float>{ jkl::math::unity<Float>(), jkl::math::zero<Float>(),
				jkl::math::zero<Float>(), jkl::math::unity<Float>() } {}

			// For some reason, copy constructor of gl2_elmt (even when explicitly defaulted)
			// is not treated as a constexpr function in some compilers.
			// Some operations require classes in this file to be trivially copyable, so 
			// providing custom constexpr copy constructor seems to be not a good solution.
			// Hence, we avoid use of copy constructor here.
			JKL_GPU_EXECUTABLE constexpr GL2_elmt(gl2_elmt<Float> const& m, do_not_check_validity) noexcept
				: gl2_elmt<Float>{ m[0][0], m[0][1], m[1][0], m[1][1] } {}

			JKL_GPU_EXECUTABLE constexpr GL2_elmt(Float const& r11, Float const& r12,
				Float const& r21, Float const& r22,
				do_not_check_validity) noexcept
				: gl2_elmt<Float>{ r11, r12, r21, r22 } {}

			constexpr GL2_elmt(gl2_elmt<Float> const& m)
				: GL2_elmt(m.det() != 0 ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not invertible!"),
					do_not_check_validity{}) {}

			constexpr GL2_elmt(Float const& r11, Float const& r12,
				Float const& r21, Float const& r22)
				: GL2_elmt(gl2_elmt<Float>{ r11, r12, r21, r22 }) {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
				JKL_GPU_EXECUTABLE constexpr operator GL2_elmt<OtherFloat>() const noexcept {
				return{ gl2_elmt<OtherFloat>(*this), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto const* operator[](std::size_t idx) const noexcept {
				return static_cast<gl2_elmt<Float> const&>(*this)[idx];
			}

			JKL_GPU_EXECUTABLE constexpr GL2_elmt t() const noexcept {
				return{ static_cast<gl2_elmt<Float> const&>(*this).t(), do_not_check_validity{} };
			}

			using gl2_elmt<Float>::det;
			JKL_GPU_EXECUTABLE constexpr GL2_elmt inv() const noexcept {
				return{ (*this)[1][1] / det(), -(*this)[0][1] / det(),
					-(*this)[1][0] / det(), (*this)[0][0] / det(),
					do_not_check_validity{}
				};
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(GL2_elmt<OtherFloat> const& that) noexcept {
				static_cast<gl2_elmt<Float>&>(*this) *= that;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(GL2_elmt<OtherFloat> const& that) noexcept {
				return (*this) *= that.inv();
			}

		private:
			// Addition/subtraction must be disabled
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE gl2_elmt<Float>& operator+=(gl2_elmt<OtherFloat> const& that) noexcept;
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE gl2_elmt<Float>& operator-=(gl2_elmt<OtherFloat> const& that) noexcept;
		};

		// I want to define these functions inside GL2_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(GL2_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(GL2_elmt<Float> const& m) noexcept {
			return m.inv();
		}

		// Matrix of size 2x2
		template <class Float>
		class gl2_elmt {
			Float r_[2][2];
		public:
			using element_type = Float;
			static constexpr std::size_t components = 4;

			// Default constructor; components are filled with garbages
			gl2_elmt() = default;

			JKL_GPU_EXECUTABLE constexpr gl2_elmt(Float const& r11, Float const& r12,
				Float const& r21, Float const& r22) noexcept
				: r_{ { r11, r12 }, { r21, r22 } } {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator gl2_elmt<OtherFloat>() const noexcept {
				return gl2_elmt<OtherFloat>(r_[0][0], r_[0][1], r_[1][0], r_[1][1]);
			}

			JKL_GPU_EXECUTABLE auto* operator[](std::size_t idx) noexcept { return r_[idx]; }
			JKL_GPU_EXECUTABLE constexpr auto const* operator[](std::size_t idx) const noexcept { return r_[idx]; }

			// I don't know why, but NVCC complains it can't deduce the return type of det().
			// It is totally a nonsense, but I should have explicitly written "Float" here...
			JKL_GPU_EXECUTABLE constexpr Float det() const noexcept {
				return r_[0][0]*r_[1][1] - r_[0][1]*r_[1][0];
			}

			JKL_GPU_EXECUTABLE constexpr auto trace() const noexcept {
				return r_[0][0] + r_[1][1];
			}

			JKL_GPU_EXECUTABLE constexpr gl2_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE constexpr gl2_elmt operator-() const noexcept {
				return{ -r_[0][0], -r_[0][1], -r_[1][0], -r_[1][1] };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(gl2_elmt<OtherFloat> const& that) noexcept {
				r_[0][0] += that.r_[0][0];
				r_[0][1] += that.r_[0][1];
				r_[1][0] += that.r_[1][0];
				r_[1][1] += that.r_[1][1];
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(gl2_elmt<OtherFloat> const& that) noexcept {
				r_[0][0] -= that.r_[0][0];
				r_[0][1] -= that.r_[0][1];
				r_[1][0] -= that.r_[1][0];
				r_[1][1] -= that.r_[1][1];
				return *this;
			}

			template <class OtherFloat,
				class = decltype(std::declval<Float&>() *= std::declval<OtherFloat>())>
			JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				r_[0][0] *= k;
				r_[0][1] *= k;
				r_[1][0] *= k;
				r_[1][1] *= k;
				return *this;
			}
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<OtherFloat, Float>::value>>
			JKL_GPU_EXECUTABLE auto& operator*=(gl2_elmt<OtherFloat> const& that) noexcept {
				auto new_value = *this * that;
				return *this = new_value;
			}

			JKL_GPU_EXECUTABLE constexpr gl2_elmt t() const noexcept {
				return{ r_[0][0], r_[1][0],
					r_[0][1], r_[1][1] };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(gl2_elmt<OtherFloat> const& m) const noexcept {
				return r_[0][0] == m[0][0] && r_[0][1] == m[0][1] &&
					r_[1][0] == m[1][0] && r_[1][1] == m[1][1];
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(gl2_elmt<OtherFloat> const& m) const noexcept {
				return !(*this == m);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept {
				return
					close_to(R2_elmt<Float>{ r_[0][0], r_[0][1] }.normsq(), 1) &&
					close_to(R2_elmt<Float>{ r_[1][0], r_[1][1] }.normsq(), 1) &&
					close_to_zero(dot(R2_elmt<Float>{ r_[0][0], r_[0][1] },
						R2_elmt<Float>{ r_[1][0], r_[1][1] }));
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const noexcept {
				return det() > 0 && is_orthogonal();
			}

			JKL_GPU_EXECUTABLE constexpr bool is_symmetric() const noexcept {
				return close_to(r_[0][1], r_[1][0]);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const noexcept {
				return is_symmetric() && r_[0][0] > 0 && det() > 0;
			}

			// Acting of R^2
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr auto transform(R2_elmt<OtherFloat> const& v) const noexcept {
				return (*this) * v;
			}

			JKL_GPU_EXECUTABLE static constexpr gl2_elmt zero() noexcept {
				return{ jkl::math::zero<Float>(), jkl::math::zero<Float>(),
					jkl::math::zero<Float>(), jkl::math::zero<Float>() };
			}

			JKL_GPU_EXECUTABLE constexpr static GL2_elmt<Float> unity() noexcept {
				return{};
			}
		};

		// I want to define these functions inside gl2_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto det(gl2_elmt<Float> const& m) noexcept {
			return m.det();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto trace(gl2_elmt<Float> const& m) noexcept {
			return m.trace();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(gl2_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_orthogonal(gl2_elmt<Float> const& m) noexcept {
			return m.is_orthogonal();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal(gl2_elmt<Float> const& m) noexcept {
			return m.is_special_orthogonal();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_symmetric(gl2_elmt<Float> const& m) noexcept {
			return m.is_symmetric();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_positive_definite(gl2_elmt<Float> const& m) noexcept {
			return m.is_positive_definite();
		}

		// General linear group of degree 3
		template <class Float>
		class gl3_elmt;
		template <class Float>
		class GL3_elmt : public gl3_elmt<Float> {
		public:
			FORCEINLINE JKL_GPU_EXECUTABLE constexpr GL3_elmt() noexcept
				: gl3_elmt<Float>{ jkl::math::unity<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>(),
				jkl::math::zero<Float>(), jkl::math::unity<Float>(), jkl::math::zero<Float>(),
				jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::unity<Float>() } {}

			// For some reason, copy constructor of gl3_elmt (even when explicitly defaulted)
			// is not treated as a constexpr function in some compilers.
			// Some operations require classes in this file to be trivially copyable, so 
			// providing custom constexpr copy constructor seems to be not a good solution.
			// Hence, we avoid use of copy constructor here.
			JKL_GPU_EXECUTABLE constexpr GL3_elmt(gl3_elmt<Float> const& m, do_not_check_validity) noexcept
				: gl3_elmt<Float>{ m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2] } {}

			JKL_GPU_EXECUTABLE constexpr GL3_elmt(Float const& r11, Float const& r12, Float const& r13,
				Float const& r21, Float const& r22, Float const& r23,
				Float const& r31, Float const& r32, Float const& r33,
				do_not_check_validity) noexcept
				: gl3_elmt<Float>{ r11, r12, r13, r21, r22, r23, r31, r32, r33 } {}
			
			constexpr GL3_elmt(gl3_elmt<Float> const& m)
				: GL3_elmt(m.det() != 0 ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not invertible!"),
					do_not_check_validity{}) {}

			constexpr GL3_elmt(Float const& r11, Float const& r12, Float const& r13,
				Float const& r21, Float const& r22, Float const& r23,
				Float const& r31, Float const& r32, Float const& r33)
				: GL3_elmt(gl3_elmt<Float>{ r11, r12, r13, r21, r22, r23, r31, r32, r33 }) {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator GL3_elmt<OtherFloat>() const noexcept {
				return{ gl3_elmt<OtherFloat>(*this), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto const* operator[](std::size_t idx) const noexcept {
				return static_cast<gl3_elmt<Float> const&>(*this)[idx];
			}

			JKL_GPU_EXECUTABLE constexpr GL3_elmt t() const noexcept {
				return{ static_cast<gl3_elmt<Float> const&>(*this).t(), do_not_check_validity{} };
			}

			using gl3_elmt<Float>::det;
			JKL_GPU_EXECUTABLE constexpr GL3_elmt inv() const noexcept {
				return{ gl3_elmt<Float>{
					(*this)[1][1]*(*this)[2][2] - (*this)[1][2]*(*this)[2][1],
					(*this)[0][2]*(*this)[2][1] - (*this)[0][1]*(*this)[2][2],
					(*this)[0][1]*(*this)[1][2] - (*this)[0][2]*(*this)[1][1],
					(*this)[1][2]*(*this)[2][0] - (*this)[1][0]*(*this)[2][2],
					(*this)[0][0]*(*this)[2][2] - (*this)[0][2]*(*this)[2][0],
					(*this)[0][2]*(*this)[1][0] - (*this)[0][0]*(*this)[1][2],
					(*this)[1][0]*(*this)[2][1] - (*this)[1][1]*(*this)[2][0],
					(*this)[0][1]*(*this)[2][0] - (*this)[0][0]*(*this)[2][1],
					(*this)[0][0]*(*this)[1][1] - (*this)[0][1]*(*this)[1][0]
				} / det(), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto operator/(GL3_elmt const& that) const noexcept {
				return (*this) * that.inv();
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(GL3_elmt<OtherFloat> const& that) noexcept {
				static_cast<gl3_elmt<Float>&>(*this) *= that;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(GL3_elmt<OtherFloat> const& that) noexcept {
				return (*this) *= that.inv();
			}

		private:
			// Addition/subtraction must be disabled
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE gl3_elmt<Float>& operator+=(gl3_elmt<OtherFloat> const& that) noexcept;
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE gl3_elmt<Float>& operator-=(gl3_elmt<OtherFloat> const& that) noexcept;
		};

		// I want to define these functions inside GL3_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(GL3_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(GL3_elmt<Float> const& m) noexcept {
			return m.inv();
		}

		// Matrix of size 3x3
		template <class Float>
		class gl3_elmt {
			Float r_[3][3];
		public:
			using element_type = Float;
			static constexpr std::size_t components = 9;

			// Default constructor; components are filled with garbages
			gl3_elmt() = default;

			JKL_GPU_EXECUTABLE constexpr gl3_elmt(Float const& r11, Float const& r12, Float const& r13,
				Float const& r21, Float const& r22, Float const& r23,
				Float const& r31, Float const& r32, Float const& r33) noexcept
				: r_{ { r11, r12, r13 }, { r21, r22, r23 }, { r31, r32, r33 } } {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
				JKL_GPU_EXECUTABLE constexpr operator gl3_elmt<OtherFloat>() const noexcept {
				return gl3_elmt<OtherFloat>(r_[0][0], r_[0][1], r_[0][2],
					r_[1][0], r_[1][1], r_[1][2],
					r_[2][0], r_[2][1], r_[2][2]);
			}

			JKL_GPU_EXECUTABLE auto* operator[](std::size_t idx) noexcept { return r_[idx]; }
			JKL_GPU_EXECUTABLE constexpr auto const* operator[](std::size_t idx) const noexcept { return r_[idx]; }

			JKL_GPU_EXECUTABLE constexpr Float det() const noexcept {
				return r_[0][0] * (r_[1][1] * r_[2][2] - r_[1][2] * r_[2][1])
					+ r_[0][1] * (r_[1][2] * r_[2][0] - r_[1][0] * r_[2][2])
					+ r_[0][2] * (r_[1][0] * r_[2][1] - r_[1][1] * r_[2][0]);
			}

			JKL_GPU_EXECUTABLE constexpr auto trace() const noexcept {
				return r_[0][0] + r_[1][1] + r_[2][2];
			}

			JKL_GPU_EXECUTABLE constexpr gl3_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE constexpr gl3_elmt operator-() const noexcept {
				return{ -r_[0][0], -r_[0][1], -r_[0][2],
					-r_[1][0], -r_[1][1], -r_[1][2],
					-r_[2][0], -r_[2][1], -r_[2][2] };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(gl3_elmt<OtherFloat> const& that) noexcept {
				r_[0][0] += that.r_[0][0];
				r_[0][1] += that.r_[0][1];
				r_[0][2] += that.r_[0][2];
				r_[1][0] += that.r_[1][0];
				r_[1][1] += that.r_[1][1];
				r_[1][2] += that.r_[1][2];
				r_[2][0] += that.r_[2][0];
				r_[2][1] += that.r_[2][1];
				r_[2][2] += that.r_[2][2];
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(gl3_elmt<OtherFloat> const& that) noexcept {
				r_[0][0] -= that.r_[0][0];
				r_[0][1] -= that.r_[0][1];
				r_[0][2] -= that.r_[0][2];
				r_[1][0] -= that.r_[1][0];
				r_[1][1] -= that.r_[1][1];
				r_[1][2] -= that.r_[1][2];
				r_[2][0] -= that.r_[2][0];
				r_[2][1] -= that.r_[2][1];
				r_[2][2] -= that.r_[2][2];
				return *this;
			}
			template <class OtherFloat,
				class = decltype(std::declval<Float&>() *= std::declval<OtherFloat>())>
				JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				r_[0][0] *= k;
				r_[0][1] *= k;
				r_[0][2] *= k;
				r_[1][0] *= k;
				r_[1][1] *= k;
				r_[1][2] *= k;
				r_[2][0] *= k;
				r_[2][1] *= k;
				r_[2][2] *= k;
				return *this;
			}
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<OtherFloat, Float>::value>>
				JKL_GPU_EXECUTABLE auto& operator*=(gl3_elmt<OtherFloat> const& that) noexcept {
				auto new_value = *this * that;
				return *this = new_value;
			}

			JKL_GPU_EXECUTABLE constexpr gl3_elmt t() const noexcept {
				return{ r_[0][0], r_[1][0], r_[2][0],
					r_[0][1], r_[1][1], r_[2][1],
					r_[0][2], r_[1][2], r_[2][2] };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(gl3_elmt<OtherFloat> const& m) const noexcept {
				return r_[0][0] == m[0][0] && r_[0][1] == m[0][1] && r_[0][2] == m[0][2] &&
					r_[1][0] == m[1][0] && r_[1][1] == m[1][1] && r_[1][2] == m[1][2] &&
					r_[2][0] == m[2][0] && r_[2][1] == m[2][1] && r_[2][2] == m[2][2];
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(gl3_elmt<OtherFloat> const& m) const noexcept {
				return !(*this == m);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_orthogonal() const noexcept {
				return
					close_to(R3_elmt<Float>{ r_[0][0], r_[0][1], r_[0][2] }.normsq(), 1) &&
					close_to(R3_elmt<Float>{ r_[1][0], r_[1][1], r_[1][2] }.normsq(), 1) &&
					close_to(R3_elmt<Float>{ r_[2][0], r_[2][1], r_[2][2] }.normsq(), 1) &&
					close_to_zero(dot(R3_elmt<Float>{ r_[0][0], r_[0][1], r_[0][2] },
						R3_elmt<Float>{ r_[1][0], r_[1][1], r_[1][2] })) &&
					close_to_zero(dot(R3_elmt<Float>{ r_[1][0], r_[1][1], r_[1][2] },
						R3_elmt<Float>{ r_[2][0], r_[2][1], r_[2][2] })) &&
					close_to_zero(dot(R3_elmt<Float>{ r_[2][0], r_[2][1], r_[2][2] },
						R3_elmt<Float>{ r_[0][0], r_[0][1], r_[0][2] }));
			}

			JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal() const noexcept {
				return det() > 0 && is_orthogonal();
			}

			JKL_GPU_EXECUTABLE constexpr bool is_symmetric() const noexcept {
				return close_to(r_[0][1], r_[1][0]) && close_to(r_[1][2], r_[2][1]) && close_to(r_[2][0], r_[0][2]);
			}

			JKL_GPU_EXECUTABLE constexpr bool is_positive_definite() const noexcept {
				return is_symmetric() && r_[0][0] > 0 &&
					gl2_elmt<Float>{ r_[0][0], r_[0][1], r_[1][0], r_[1][1] }.det() > 0 && det() > 0;
			}

			// Acting on R^3
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr auto transform(R3_elmt<OtherFloat> const& v) const noexcept {
				return (*this) * v;
			}

			JKL_GPU_EXECUTABLE constexpr static gl3_elmt zero() noexcept {
				return{ jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>(),
					jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>(),
					jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>() };
			}

			JKL_GPU_EXECUTABLE constexpr static GL3_elmt<Float> unity() noexcept {
				return{};
			}
		};

		// I want to define these functions inside gl3_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto det(gl3_elmt<Float> const& m) noexcept {
			return m.det();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto trace(gl3_elmt<Float> const& m) noexcept {
			return m.trace();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(gl3_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_orthogonal(gl3_elmt<Float> const& m) noexcept {
			return m.is_orthogonal();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_special_orthogonal(gl3_elmt<Float> const& m) noexcept {
			return m.is_special_orthogonal();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_symmetric(gl3_elmt<Float> const& m) noexcept {
			return m.is_symmetric();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr bool is_positive_definite(gl3_elmt<Float> const& m) noexcept {
			return m.is_positive_definite();
		}

		// Unit quaternions
		template <class Float>
		using su2_elmt = R3_elmt<Float>;
		template <class Float>
		class SU2_elmt {
			array_t<Float, 4> r_;	// w, x, y, z
		public:
			using element_type = Float;
			static constexpr std::size_t components = 4;

			FORCEINLINE JKL_GPU_EXECUTABLE constexpr SU2_elmt() noexcept
				: r_{ jkl::math::unity<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>() } {}

			FORCEINLINE JKL_GPU_EXECUTABLE constexpr SU2_elmt(Float const& w, Float const& x, Float const& y, Float const& z,
				do_not_check_validity) noexcept : r_{ w, x, y, z } {}

			FORCEINLINE JKL_GPU_EXECUTABLE constexpr SU2_elmt(Float const& scalar, R3_elmt<Float> const& vector, 
				do_not_check_validity) noexcept : SU2_elmt(scalar, vector.x(), vector.y(), vector.z(), do_not_check_validity{}) {}

			constexpr SU2_elmt(Float const& w, Float const& x, Float const& y, Float const& z)
				: r_{ !close_to(w*w + x*x + y*y + z*z, 1) ?
				throw std::invalid_argument("jkl::math: Assertion failed! Normalization is required!") : w,
				x, y, z } {}

			constexpr SU2_elmt(Float const& scalar, R3_elmt<Float> const& vector)
				: SU2_elmt(scalar, vector.x(), vector.y(), vector.z()) {}
			
			// Convert to quaternion of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator SU2_elmt<OtherFloat>() const noexcept {
				return{ w(), x(), y(), z(), do_not_check_validity() };
			}

			JKL_GPU_EXECUTABLE constexpr Float const& w() const noexcept { return get<0>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& x() const noexcept { return get<1>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& y() const noexcept { return get<2>(r_); }
			JKL_GPU_EXECUTABLE constexpr Float const& z() const noexcept { return get<3>(r_); }

			JKL_GPU_EXECUTABLE constexpr auto const& scalar_part() const noexcept { return w(); }
			JKL_GPU_EXECUTABLE constexpr R3_elmt<Float> vector_part() const noexcept { return{ x(), y(), z() }; }

			JKL_GPU_EXECUTABLE static SU2_elmt exp(su2_elmt<Float> const& exponent) noexcept {
				Float w, x, y, z;
				w = cos(exponent.norm() / 2);
				if( almost_smaller(exponent.norm(), 0) ) {
					x = exponent.x() / 2;
					y = exponent.y() / 2;
					z = exponent.z() / 2;
				}
				else {
					x = exponent.x() / exponent.norm() * sin(exponent.norm() / 2);
					y = exponent.y() / exponent.norm() * sin(exponent.norm() / 2);
					z = exponent.z() / exponent.norm() * sin(exponent.norm() / 2);
				}
				return{ w, x, y, z, do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE su2_elmt<Float> log() const noexcept {
				auto angle_ = angle();
				auto v = su2_elmt<Float>{ x(), y(), z() };

				if( angle_ > numeric_consts<Float>::pi ) {
					angle_ = 2 * numeric_consts<Float>::pi - angle_;
					v = -v;
				}
				auto s = sin(angle_/2);
				if( close_to_zero(s) )
					return 2 * v;
				else
					return (angle_/s) * v;
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt inv() const noexcept {
				return{ w(), -x(), -y(), -z(), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt const& operator+() const noexcept {
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr SU2_elmt operator-() const noexcept {
				return{ -w(), -x(), -y(), -z(), do_not_check_validity{} };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(SU2_elmt<OtherFloat> const& q) noexcept {
				auto new_value = *this * q;
				return *this = new_value;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(SU2_elmt<OtherFloat> const& q) noexcept {
				return (*this) *= q.inv();
			}

			JKL_GPU_EXECUTABLE auto angle() const noexcept {	// from 0 to 2pi
				auto ww = w();
				if( ww > 1 )
					ww = 1;
				if( ww < -1 )
					ww = -1;
				return 2 * acos(ww);
			}
			JKL_GPU_EXECUTABLE auto abs_angle() const noexcept {	// convert [0,2pi) to (-pi,pi] and take abs()
				auto angle_ = angle();
				if( angle_ > numeric_consts<Float>::pi )
					return 2 * numeric_consts<Float>::pi - angle_;
				else
					return angle_;
			}

			// Acting on R^3
			// MSVC2015 fails to evaluate this function as a constant expression if the return type is 'auto'
			// Hence, we have to explicitly write the return type
			// This issue is fixed in MSVC2017
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr auto rotate(R3_elmt<OtherFloat> const& v) const noexcept
				-> R3_elmt<std::decay_t<decltype(
				(1 - 2*y()*y() - 2*z()*z()) * v.x() + 2*(x()*y() - z()*w()) * v.y() + 2*(x()*z() + y()*w()) * v.z())>>
			{
				return{ (1 - 2*y()*y() - 2*z()*z()) * v.x() + 2*(x()*y() - z()*w()) * v.y() + 2*(x()*z() + y()*w()) * v.z(),
					2*(y()*x() + z()*w()) * v.x() + (1 - 2*z()*z() - 2*x()*x()) * v.y() + 2*(y()*z() - x()*w()) * v.z(),
					2*(z()*x() - y()*w()) * v.x() + 2*(z()*y() + x()*w()) * v.y() + (1 - 2*x()*x() - 2*y()*y()) * v.z() };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(SU2_elmt<OtherFloat> const& q) const noexcept {
				return w() == q.w() && x() == q.x() && y() == q.y() && z() == q.z();
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(SU2_elmt<OtherFloat> const& q) const noexcept {
				return !(*this == q);
			}

			JKL_GPU_EXECUTABLE static constexpr SU2_elmt unity() noexcept { return{}; }
		};

		// I want to define these functions inside SU2_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE auto log(SU2_elmt<Float> const& q) noexcept {
			return q.log();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(SU2_elmt<Float> const& q) noexcept {
			return q.inv();
		}

		// Rotation matrices
		template <class Float>
		using so3_elmt = R3_elmt<Float>;
		template <class Float>
		class SO3_elmt : public GL3_elmt<Float> {
		public:
			FORCEINLINE JKL_GPU_EXECUTABLE constexpr SO3_elmt() noexcept : GL3_elmt<Float>{} {}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt(gl3_elmt<Float> const& m, do_not_check_validity) noexcept
				: GL3_elmt<Float>(m, do_not_check_validity{}) {}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt(Float const& r11, Float const& r12, Float const& r13,
				Float const& r21, Float const& r22, Float const& r23,
				Float const& r31, Float const& r32, Float const& r33,
				do_not_check_validity) noexcept
				: GL3_elmt<Float>{ r11, r12, r13, r21, r22, r23, r31, r32, r33, do_not_check_validity{} } {}

			constexpr SO3_elmt(gl3_elmt<Float> const& m)
				: GL3_elmt<Float>(m.is_special_orthogonal() ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not special orthogonal!"),
					do_not_check_validity{}) {}

			constexpr SO3_elmt(Float const& r11, Float const& r12, Float const& r13,
				Float const& r21, Float const& r22, Float const& r23,
				Float const& r31, Float const& r32, Float const& r33)
				: SO3_elmt(gl3_elmt<Float>{ r11, r12, r13, r21, r22, r23, r31, r32, r33 }) {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator SO3_elmt<OtherFloat>() const noexcept {
				return{ gl3_elmt<OtherFloat>(*this), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto const* operator[](std::size_t idx) const noexcept {
				return static_cast<gl3_elmt<Float> const&>(*this)[idx];
			}

			// Exponential map
			JKL_GPU_EXECUTABLE static auto exp(so3_elmt<Float> const& exponent) noexcept {
				return SO3_elmt(SU2_elmt<Float>::exp(exponent));
			};

			// Universal covering map
			JKL_GPU_EXECUTABLE explicit constexpr SO3_elmt(SU2_elmt<Float> const& q) noexcept
				: SO3_elmt(
					1 - 2 * (q.y()*q.y() + q.z()*q.z()), 2 * (q.x()*q.y() - q.z()*q.w()), 2 * (q.z()*q.x() + q.y()*q.w()),
					2 * (q.x()*q.y() + q.z()*q.w()), 1 - 2 * (q.z()*q.z() + q.x()*q.x()), 2 * (q.y()*q.z() - q.x()*q.w()),
					2 * (q.z()*q.x() - q.y()*q.w()), 2 * (q.y()*q.z() + q.x()*q.w()), 1 - 2 * (q.x()*q.x() + q.y()*q.y()),
					do_not_check_validity{}) {}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt t() const noexcept {
				return{ static_cast<gl3_elmt<Float> const&>(*this).t(), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr SO3_elmt inv() const noexcept {
				return t();
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(SO3_elmt<OtherFloat> const& that) noexcept {
				static_cast<gl3_elmt<Float>&>(*this) *= that;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(SO3_elmt<OtherFloat> const& that) noexcept {
				return (*this) *= that.t();
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr auto rotate(R3_elmt<OtherFloat> const& v) const noexcept {
				return (*this) * v;
			}

			// Logarithmic map
			JKL_GPU_EXECUTABLE so3_elmt<Float> log() const {
				// Take antisymmetric part
				auto x = ((*this)[2][1] - (*this)[1][2]) / 2;
				auto y = ((*this)[0][2] - (*this)[2][0]) / 2;
				auto z = ((*this)[1][0] - (*this)[0][1]) / 2;
				// Compute angle
				auto angle_cos = ((*this)[0][0] + (*this)[1][1] + (*this)[2][2] - 1) / 2;
				auto angle = acos(angle_cos);
				// arcsin(norm) / norm
				if( almost_smaller(angle, 0) )
					return{ x, y, z };
				else if( almost_larger(angle, numeric_consts<Float>::pi) ) {
					x = sqrt(((*this)[0][0] + 1) / 2);
					y = sqrt(((*this)[1][1] + 1) / 2);
					z = sqrt(((*this)[2][2] + 1) / 2);
					if( (*this)[0][1] < 0 && (*this)[0][2] < 0 )
						return{ -x, y, z };
					else if( (*this)[0][1] < 0 && (*this)[0][2] >= 0 )
						return{ x, -y, z };
					else if( (*this)[0][1] >= 0 && (*this)[0][2] < 0 )
						return{ x, y, -z };
					else
						return{ x, y, z };
				}
				auto d = angle / sin(angle);
				return{ x*d, y*d, z*d };
			}

			// Inverse covering map
			JKL_GPU_EXECUTABLE auto find_quaternion() const {
				return SU2_elmt<Float>::exp(log());
			}

			// Get Euler angles; find (theta_x, theta_y, theta_z) with R = R_z(theta_z)R_y(theta_y)R_x(theta_x)
			JKL_GPU_EXECUTABLE R3_elmt<Float> euler_angles() const {
				Float theta_x, theta_y, theta_z;
				if( close_to((*this)[2][0], Float(1)) ) {
					theta_z = 0;
					theta_y = -numeric_consts<Float>::pi / 2;
					theta_x = atan2(-(*this)[0][1], -(*this)[0][2]);
				}
				else if( close_to((*this)[2][0], Float(-1)) ) {
					theta_z = 0;
					theta_y = numeric_consts<Float>::pi / 2;
					theta_x = atan2((*this)[0][1], (*this)[0][2]);
				}
				else {
					theta_y = -asin((*this)[2][0]);
					theta_x = atan2((*this)[2][1] / cos(theta_y), (*this)[2][2] / cos(theta_y));
					theta_z = atan2((*this)[1][0] / cos(theta_y), (*this)[0][0] / cos(theta_y));
				}
				return{ theta_x, theta_y, theta_z };
			}

			JKL_GPU_EXECUTABLE constexpr static SO3_elmt unity() noexcept {
				return{};
			}
			JKL_GPU_EXECUTABLE static SO3_elmt rotx(Float const& theta) noexcept {
				return SO3_elmt::exp({ theta, jkl::math::zero<Float>(), jkl::math::zero<Float>() });
			}
			JKL_GPU_EXECUTABLE static SO3_elmt roty(Float const& theta) noexcept {
				return SO3_elmt::exp({ jkl::math::zero<Float>(), theta, jkl::math::zero<Float>() });
			}
			JKL_GPU_EXECUTABLE static SO3_elmt rotz(Float const& theta) noexcept {
				return SO3_elmt::exp({ jkl::math::zero<Float>(), jkl::math::zero<Float>(), theta });
			}
			JKL_GPU_EXECUTABLE static SO3_elmt euler_to_SO3(R3_elmt<Float> const& euler_angles) noexcept {
				return rotz(euler_angles.z()) * roty(euler_angles.y()) * rotx(euler_angles.x());
			}
		};

		// I want to define these functions inside SO3_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(SO3_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(SO3_elmt<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE auto log(SO3_elmt<Float> const& r) {
			return r.log();
		}

		// 6-dimensional Lie algebra se(3)
		template <class Float>
		class se3_elmt {
			su2_elmt<Float>		m_rot;
			R3_elmt<Float>		m_trans;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 6;

			// Default constructor; components are filled with garbages
			se3_elmt() = default;

			JKL_GPU_EXECUTABLE constexpr se3_elmt(su2_elmt<Float> const& rotation_part,
				R3_elmt<Float> const& translation_part) noexcept
				: m_rot{ rotation_part }, m_trans{ translation_part } {}

			// Convert to vector of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator se3_elmt<OtherFloat>() const noexcept {
				return{ m_rot, m_trans };
			}

			JKL_GPU_EXECUTABLE auto& rotation_part() noexcept {
				return m_rot;
			}
			JKL_GPU_EXECUTABLE constexpr auto const& rotation_part() const noexcept {
				return m_rot;
			}
			JKL_GPU_EXECUTABLE auto& translation_part() noexcept {
				return m_trans;
			}
			JKL_GPU_EXECUTABLE constexpr auto const& translation_part() const noexcept {
				return m_trans;
			}

			JKL_GPU_EXECUTABLE constexpr se3_elmt const& operator+() const noexcept {
				return *this;
			}
			JKL_GPU_EXECUTABLE constexpr se3_elmt operator-() const noexcept {
				return{ -rotation_part(), -translation_part() };
			}
			
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator+=(se3_elmt<OtherFloat> const& that) noexcept {
				rotation_part() += that.rotation_part();
				translation_part() += that.translation_part();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator-=(se3_elmt<OtherFloat> const& that) noexcept {
				rotation_part() -= that.rotation_part();
				translation_part() -= that.translation_part();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(OtherFloat const& k) noexcept {
				rotation_part() *= k;
				translation_part() *= k;
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(OtherFloat const& k) noexcept {
				rotation_part() /= k;
				translation_part() /= k;
				return *this;
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(se3_elmt<OtherFloat> const& that) const noexcept {
				return rotation_part() == that.rotation_part() && translation_part() == that.translation_part();
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(se3_elmt<OtherFloat> const& that) const noexcept {
				return !(*this == that);
			}

			JKL_GPU_EXECUTABLE static constexpr se3_elmt zero() noexcept {
				return{ su2_elmt<Float>::zero(), R3_elmt<Float>::zero() };
			}
		};

		// Rigid transforms
		template <class Float>
		class SE3_elmt {
			SU2_elmt<Float> m_rot;
			R3_elmt<Float>	m_trans;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 7;

			JKL_GPU_EXECUTABLE constexpr SE3_elmt() noexcept
				: m_rot{ SU2_elmt<Float>::unity() }, m_trans{ R3_elmt<Float>::zero() } {}

			JKL_GPU_EXECUTABLE constexpr SE3_elmt(SU2_elmt<Float> const& rotation_q, 
				R3_elmt<Float> const& translation) noexcept
				: m_rot{ rotation_q }, m_trans{ translation } {}
			JKL_GPU_EXECUTABLE SE3_elmt(SO3_elmt<Float> const& rotation, 
				R3_elmt<Float> const& translation) noexcept
				: m_rot{ rotation.find_quaternion() }, m_trans{ translation } {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator SE3_elmt<OtherFloat>() const noexcept {
				return SE3_elmt<OtherFloat>(m_rot, m_trans);
			}

			JKL_GPU_EXECUTABLE auto& rotation_q() noexcept {
				return m_rot;
			}
			JKL_GPU_EXECUTABLE constexpr auto const& rotation_q() const noexcept {
				return m_rot;
			}
			JKL_GPU_EXECUTABLE constexpr auto const rotation() const noexcept {
				return SO3_elmt<Float>{ m_rot };
			}
			JKL_GPU_EXECUTABLE auto& translation() noexcept {
				return m_trans;
			}
			JKL_GPU_EXECUTABLE constexpr auto const& translation() const noexcept {
				return m_trans;
			}

			// Exponential map
			JKL_GPU_EXECUTABLE static SE3_elmt exp(se3_elmt<Float> const& exponent) noexcept {
				auto rot = SU2_elmt<Float>::exp(exponent.rotation_part());
				auto const angle = exponent.rotation_part().norm();
				Float alpha, beta;
				if( almost_smaller(angle, 0) ) {
					alpha = Float(1) / 2 - angle*angle / 24;
					beta = Float(1) / 6 - angle*angle / 120;
				}
				else {
					alpha = (1 - cos(angle)) / (angle * angle);
					beta = (angle - sin(angle)) / (angle * angle * angle);
				}
				auto cross_prod = cross(exponent.rotation_part(), exponent.translation_part());
				auto trans = exponent.translation_part() + alpha * cross_prod +
					beta * cross(exponent.rotation_part(), cross_prod);

				return{ rot, trans };
			}

			JKL_GPU_EXECUTABLE constexpr SE3_elmt inv() const noexcept {
				return{ rotation_q().inv(), -(rotation_q().inv().rotate(translation())) };
			}

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator*=(SE3_elmt<OtherFloat> const& that) noexcept {
				translation() += rotation_q().rotate(that.translation());
				rotation_q() *= that.rotation_q();
				return *this;
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE auto& operator/=(SE3_elmt<OtherFloat> const& that) noexcept {
				return (*this) *= that.inv();
			}

			// Logarithmic map
			JKL_GPU_EXECUTABLE se3_elmt<Float> log() const {
				auto rotation_part = rotation_q().log();
				auto const angle = rotation_part.norm();
				Float beta;
				if( almost_smaller(angle, 0) )
					beta = Float(1) / 12 + angle / 360;
				else
					beta = (1 - (angle / 2) / tan(angle / 2)) / (angle * angle);

				auto cross_prod = cross(rotation_part, translation());
				auto translation_part = translation() - cross_prod/2 +
					beta * cross(rotation_part, cross_prod);
				return{ rotation_part, translation_part };
			}

			// Acting on R^3
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr auto transform(R3_elmt<OtherFloat> const& point) const noexcept {
				return rotation_q().rotate(point) + translation();
			}
			
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator==(SE3_elmt<OtherFloat> const& that) const noexcept {
				return (translation() == that.translation()) &&
					(rotation_q() == that.rotation_q() || rotation_q() == -that.rotation_q());
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr bool operator!=(SE3_elmt<OtherFloat> const& that) const noexcept {
				return (translation() != that.translation()) ||
					(rotation_q() != that.rotation_q() && rotation_q() != -that.rotation_q());
			}

			JKL_GPU_EXECUTABLE static constexpr SE3_elmt unity() noexcept {
				return{};
			}
		};

		// I want to define these functions inside SE3_elmt as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(SE3_elmt<Float> const& t) noexcept {
			return t.inv();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE auto log(SE3_elmt<Float> const& t) {
			return t.log();
		}

		// 2x2 Symmetric matrices
		template <class Float>
		class sym_2x2 {
			Float xx_;
			Float yy_;
			Float xy_;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 3;

			// Default constructor; components are filled with garbages
			sym_2x2() = default;

			JKL_GPU_EXECUTABLE constexpr sym_2x2(gl2_elmt<Float> const& m, do_not_check_validity) noexcept
				: xx_(m[0][0]), yy_(m[1][1]), xy_(m[0][1]) {}
			JKL_GPU_EXECUTABLE constexpr sym_2x2(Float const& xx, Float const& yy, Float const& xy) noexcept
				: xx_(xx), yy_(yy), xy_(xy) {}

			constexpr sym_2x2(gl2_elmt<Float> const& m)
				: sym_2x2(is_symmetric(m) ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not symmetric!"),
					do_not_check_validity{}) {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator sym_2x2<OtherFloat>() const noexcept {
				return sym_2x2<OtherFloat>(xx_, yy_, xy_);
			}

			JKL_GPU_EXECUTABLE Float& xx() noexcept { return xx_; }
			JKL_GPU_EXECUTABLE Float& xy() noexcept { return xy_; }
			JKL_GPU_EXECUTABLE Float& yx() noexcept { return xy_; }
			JKL_GPU_EXECUTABLE Float& yy() noexcept { return yy_; }

			JKL_GPU_EXECUTABLE constexpr Float const& xx() const noexcept { return xx_; }
			JKL_GPU_EXECUTABLE constexpr Float const& xy() const noexcept { return xy_; }
			JKL_GPU_EXECUTABLE constexpr Float const& yx() const noexcept { return xy_; }
			JKL_GPU_EXECUTABLE constexpr Float const& yy() const noexcept { return yy_; }

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator gl2_elmt<OtherFloat>() const noexcept {
				return gl2_elmt<OtherFloat>(xx_, xy_, xy_, yy_);
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator GL2_elmt<OtherFloat>() const noexcept {
				return{ static_cast<gl2_elmt<OtherFloat>>(*this) };
			}

			JKL_GPU_EXECUTABLE constexpr array_t<Float, 2> operator[](std::size_t idx) const noexcept {
				return static_cast<gl2_elmt<Float>>(*this)[idx];
			}

			// I don't know why, but NVCC complains it can't deduce the return type of det().
			// It is totally a nonsense, but I should have explicitly written "Float" here...
			JKL_GPU_EXECUTABLE constexpr Float det() const noexcept {
				return xx() * yy() - xy() * yx();
			}

			JKL_GPU_EXECUTABLE constexpr auto trace() const noexcept {
				return xx() + yy();
			}

			JKL_GPU_EXECUTABLE constexpr auto t() const noexcept {
				return *this;
			}
		};

		// I want to define these functions inside sym_2x2 as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto det(sym_2x2<Float> const& m) noexcept {
			return m.det();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto trace(sym_2x2<Float> const& m) noexcept {
			return m.trace();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(sym_2x2<Float> const& m) noexcept {
			return m.t();
		}

		// 3x3 Symmetric matrices
		template <class Float>
		class sym_3x3 {
			Float xx_ = 1;
			Float yy_ = 1;
			Float zz_ = 1;
			Float xy_ = 0;
			Float yz_ = 0;
			Float zx_ = 0;
		public:
			using element_type = Float;
			static constexpr std::size_t components = 6;

			// Default constructor; components are filled with garbages
			sym_3x3() = default;

			JKL_GPU_EXECUTABLE constexpr sym_3x3(gl3_elmt<Float> const& m, do_not_check_validity) noexcept
				: xx_(m[0][0]), yy_(m[1][1]), zz_(m[2][2]), xy_(m[0][1]), yz_(m[1][2]), zx_(m[2][0]) {}
			JKL_GPU_EXECUTABLE constexpr sym_3x3(Float const& xx, Float const& yy, Float const& zz,
				Float const& xy, Float const& yz, Float const& zx) noexcept
				: xx_(xx), yy_(yy), zz_(zz), xy_(xy), yz_(yz), zx_(zx) {}

			constexpr sym_3x3(gl3_elmt<Float> const& m)
				: sym_3x3(is_symmetric(m) ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not symmetric!"),
					do_not_check_validity{}) {}

			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator sym_3x3<OtherFloat>() const noexcept {
				return sym_3x3<OtherFloat>(xx_, yy_, zz_, xy_, yz_, zx_);
			}

			JKL_GPU_EXECUTABLE Float& xx() noexcept { return xx_; }
			JKL_GPU_EXECUTABLE Float& xy() noexcept { return xy_; }
			JKL_GPU_EXECUTABLE Float& xz() noexcept { return zx_; }
			JKL_GPU_EXECUTABLE Float& yx() noexcept { return xy_; }
			JKL_GPU_EXECUTABLE Float& yy() noexcept { return yy_; }
			JKL_GPU_EXECUTABLE Float& yz() noexcept { return yz_; }
			JKL_GPU_EXECUTABLE Float& zx() noexcept { return zx_; }
			JKL_GPU_EXECUTABLE Float& zy() noexcept { return yz_; }
			JKL_GPU_EXECUTABLE Float& zz() noexcept { return zz_; }

			JKL_GPU_EXECUTABLE constexpr Float const& xx() const noexcept { return xx_; }
			JKL_GPU_EXECUTABLE constexpr Float const& xy() const noexcept { return xy_; }
			JKL_GPU_EXECUTABLE constexpr Float const& xz() const noexcept { return zx_; }
			JKL_GPU_EXECUTABLE constexpr Float const& yx() const noexcept { return xy_; }
			JKL_GPU_EXECUTABLE constexpr Float const& yy() const noexcept { return yy_; }
			JKL_GPU_EXECUTABLE constexpr Float const& yz() const noexcept { return yz_; }
			JKL_GPU_EXECUTABLE constexpr Float const& zx() const noexcept { return zx_; }
			JKL_GPU_EXECUTABLE constexpr Float const& zy() const noexcept { return yz_; }
			JKL_GPU_EXECUTABLE constexpr Float const& zz() const noexcept { return zz_; }

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator gl3_elmt<OtherFloat>() const noexcept {
				return gl3_elmt<OtherFloat>(xx_, xy_, zx_, xy_, yy_, yz_, zx_, yz_, zz_);
			}
			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator GL3_elmt<OtherFloat>() const noexcept {
				return{ static_cast<gl3_elmt<OtherFloat>>(*this) };
			}

			JKL_GPU_EXECUTABLE constexpr array_t<Float, 3> operator[](std::size_t idx) const noexcept {
				return static_cast<gl3_elmt<Float>>(*this)[idx];
			}

			JKL_GPU_EXECUTABLE constexpr auto det() const noexcept {
				return static_cast<gl3_elmt<Float>>(*this).det();
			}

			JKL_GPU_EXECUTABLE constexpr auto trace() const noexcept {
				return xx() + yy() + zz();
			}

			JKL_GPU_EXECUTABLE constexpr auto t() const noexcept {
				return *this;
			}
		};

		// I want to define these functions inside sym_3x3 as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto det(sym_3x3<Float> const& m) noexcept {
			return m.det();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto trace(sym_3x3<Float> const& m) noexcept {
			return m.trace();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(sym_3x3<Float> const& m) noexcept {
			return m.t();
		}

		// 2x2 Symmetric positive-definite matrices
		template <class Float>
		class pos_def_2x2 : public sym_2x2<Float>{
		public:
			using element_type = Float;
			static constexpr std::size_t components = 3;

			JKL_GPU_EXECUTABLE constexpr pos_def_2x2() noexcept
				: sym_2x2<Float>{ jkl::math::unity<Float>(), jkl::math::unity<Float>(), jkl::math::zero<Float>() } {}
			JKL_GPU_EXECUTABLE constexpr pos_def_2x2(gl2_elmt<Float> const& m, do_not_check_validity) noexcept
				: sym_2x2<Float>(m[0][0], m[1][1], m[0][1]) {}
			JKL_GPU_EXECUTABLE constexpr pos_def_2x2(Float const& xx, Float const& yy, Float const& xy, 
				do_not_check_validity) noexcept
				: sym_2x2<Float>(xx, yy, xy) {}

			constexpr pos_def_2x2(gl2_elmt<Float> const& m)
				: pos_def_2x2(is_positive_definite(m) ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not positive-definite!"),
					do_not_check_validity{}) {}
			constexpr pos_def_2x2(Float const& xx, Float const& yy, Float const& xy)
				: pos_def_2x2(xx > 0 && xx * yy > xy * xy ? xx
				: throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not positive-definite!"),
				yy, xy, do_not_check_validity{}) {}
			
			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator pos_def_2x2<OtherFloat>() const noexcept {
				return pos_def_2x2<OtherFloat>(xx(), yy(), xy(), do_not_check_validity{});
			}

			JKL_GPU_EXECUTABLE constexpr Float const& xx() const noexcept { return sym_2x2<Float>::xx(); }
			JKL_GPU_EXECUTABLE constexpr Float const& xy() const noexcept { return sym_2x2<Float>::xy(); }
			JKL_GPU_EXECUTABLE constexpr Float const& yx() const noexcept { return sym_2x2<Float>::yx(); }
			JKL_GPU_EXECUTABLE constexpr Float const& yy() const noexcept { return sym_2x2<Float>::yy(); }

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator GL2_elmt<OtherFloat>() const noexcept {
				return{ static_cast<gl2_elmt<OtherFloat>>(*this), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto t() const noexcept {
				return *this;
			}

			using sym_2x2<Float>::det;
			JKL_GPU_EXECUTABLE constexpr pos_def_2x2 inv() const noexcept {
				return{ yy() / det(), xx() / det(), -xy() / det(), do_not_check_validity{} };
			}
		};

		// I want to define these functions inside pos_def_2x2 as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(pos_def_2x2<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(pos_def_2x2<Float> const& m) noexcept {
			return m.inv();
		}

		// 3x3 Symmetric positive-definite matrices
		template <class Float>
		class pos_def_3x3 : sym_3x3<Float> {
		public:
			using element_type = Float;
			static constexpr std::size_t components = 6;

			JKL_GPU_EXECUTABLE constexpr pos_def_3x3() noexcept
				: sym_3x3<Float>{ jkl::math::unity<Float>(), jkl::math::unity<Float>(), jkl::math::unity<Float>(),
				jkl::math::zero<Float>(), jkl::math::zero<Float>(), jkl::math::zero<Float>() } {};
			JKL_GPU_EXECUTABLE constexpr pos_def_3x3(gl3_elmt<Float> const& m, do_not_check_validity) noexcept
				: sym_3x3<Float>(m[0][0], m[1][1], m[2][2], m[0][1], m[1][2], m[2][0]) {}
			JKL_GPU_EXECUTABLE constexpr pos_def_3x3(Float const& xx, Float const& yy, Float const& zz,
				Float const& xy, Float const& yz, Float const& zx, do_not_check_validity) noexcept
				: sym_3x3<Float>(xx, yy, zz, xy, yz, zx) {}

			constexpr pos_def_3x3(gl3_elmt<Float> const& m)
				: pos_def_3x3(is_positive_definite(m) ? m :
					throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not positive-definite!"),
					do_not_check_validity{}) {}
			constexpr pos_def_3x3(Float const& xx, Float const& yy, Float const& zz,
				Float const& xy, Float const& yz, Float const& zx)
				: pos_def_3x3(xx > 0 && xx * yy > xy * xy && gl3_elmt<Float>{ xx, xy, zx, xy, yy, yz, zx, yz, zz }.det() > 0 ? xx
					: throw std::invalid_argument("jkl::math: Assertion failed! The matrix is not positive-definite!"),
					yy, zz, xy, yz, zx, do_not_check_validity{}) {}
			
			// Convert to matrix of other element type
			template <class OtherFloat,
				class = std::enable_if_t<std::is_convertible<Float, OtherFloat>::value &&
				!std::is_same<Float, OtherFloat>::value>>
			JKL_GPU_EXECUTABLE constexpr operator pos_def_3x3<OtherFloat>() const noexcept {
				return pos_def_3x3<OtherFloat>(xx(), yy(), zz(), xy(), yz(), zx(), do_not_check_validity{});
			}
			
			JKL_GPU_EXECUTABLE constexpr Float const& xx() const noexcept { return sym_3x3<Float>::xx(); }
			JKL_GPU_EXECUTABLE constexpr Float const& xy() const noexcept { return sym_3x3<Float>::xy(); }
			JKL_GPU_EXECUTABLE constexpr Float const& xz() const noexcept { return sym_3x3<Float>::xz(); }
			JKL_GPU_EXECUTABLE constexpr Float const& yx() const noexcept { return sym_3x3<Float>::yx(); }
			JKL_GPU_EXECUTABLE constexpr Float const& yy() const noexcept { return sym_3x3<Float>::yy(); }
			JKL_GPU_EXECUTABLE constexpr Float const& yz() const noexcept { return sym_3x3<Float>::yz(); }
			JKL_GPU_EXECUTABLE constexpr Float const& zx() const noexcept { return sym_3x3<Float>::zx(); }
			JKL_GPU_EXECUTABLE constexpr Float const& zy() const noexcept { return sym_3x3<Float>::zy(); }
			JKL_GPU_EXECUTABLE constexpr Float const& zz() const noexcept { return sym_3x3<Float>::zz(); }

			template <class OtherFloat>
			JKL_GPU_EXECUTABLE constexpr operator GL3_elmt<OtherFloat>() const noexcept {
				return{ static_cast<gl3_elmt<OtherFloat>>(*this), do_not_check_validity{} };
			}

			JKL_GPU_EXECUTABLE constexpr auto t() const noexcept {
				return *this;
			}

			JKL_GPU_EXECUTABLE constexpr pos_def_3x3 inv() const noexcept {
				return{ static_cast<GL3_elmt<Float>>(*this).inv(), do_not_check_validity{} };
			}
		};

		// I want to define these functions inside pos_def_3x3 as friends,
		// but then nvcc fails to compile this header file saying "assertion failed"
		// It seems that the problem is somewhat related to 'constexpr', but 
		// I really don't have any idea what's wrong...
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto transpose(pos_def_3x3<Float> const& m) noexcept {
			return m.t();
		}
		template <class Float>
		JKL_GPU_EXECUTABLE constexpr auto inv(pos_def_3x3<Float> const& m) noexcept {
			return m.inv();
		}


		/// Operations on Rn_elmt

		namespace detail {
			template <class Float, std::size_t N>
			struct binary_op_impl {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE auto operator()(Kernel&& k, First const& v, Second const& w) const noexcept {
					Rn_elmt<Float, N> result;
					for( std::size_t i = 0; i < N; ++i )
						result[i] = k(v[i], w[i]);
					return result;
				}
			};

			template <class Float>
			struct binary_op_impl<Float, 2> {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(Kernel&& k, First const& v, Second const& w) const noexcept {
					return R2_elmt<Float>{ k(v.x(), w.x()), k(v.y(), w.y()) };
				}
			};

			template <class Float>
			struct binary_op_impl<Float, 3> {
				template <class Kernel, class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(Kernel&& k, First const& v, Second const& w) const noexcept {
					return R3_elmt<Float>{ k(v.x(), w.x()), k(v.y(), w.y()), k(v.z(), w.z()) };
				}
			};

			template <class Float>
			struct scalar_wrapper {
				Float const& k;
				JKL_GPU_EXECUTABLE constexpr auto operator[](std::size_t idx) const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto x() const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto y() const noexcept { return k; }
				JKL_GPU_EXECUTABLE constexpr auto z() const noexcept { return k; }
			};
			template <class Float>
			JKL_GPU_EXECUTABLE constexpr auto wrap_scalar(Float const& k) noexcept {
				return scalar_wrapper<Float>{ k };
			}

			struct sum_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(First const& a, Second const& b) const noexcept {
					return a + b;
				}
			};

			struct diff_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(First const& a, Second const& b) const noexcept {
					return a - b;
				}
			};

			struct mult_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(First const& a, Second const& b) const noexcept {
					return a * b;
				}
			};

			struct div_kernel {
				template <class First, class Second>
				JKL_GPU_EXECUTABLE constexpr auto operator()(First const& a, Second const& b) const noexcept {
					return a / b;
				}
			};
		}

		template <class Float1, class Float2, std::size_t N>
		JKL_GPU_EXECUTABLE constexpr auto operator+(Rn_elmt<Float1, N> const& v, Rn_elmt<Float2, N> const& w) noexcept
			-> Rn_elmt<std::decay_t<decltype(v[0] + w[0])>, N>
		{
			return detail::binary_op_impl<std::decay_t<decltype(v[0] + w[0])>, N>{}(
				detail::sum_kernel{}, v, w);
		}

		template <class Float1, class Float2, std::size_t N>
		JKL_GPU_EXECUTABLE constexpr auto operator-(Rn_elmt<Float1, N> const& v, Rn_elmt<Float2, N> const& w) noexcept
			-> Rn_elmt<std::decay_t<decltype(v[0] - w[0])>, N>
		{
			return detail::binary_op_impl<std::decay_t<decltype(v[0] - w[0])>, N>{}(
				detail::diff_kernel{}, v, w);
		}

		template <class Float1, class Float2, std::size_t N,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Rn_elmt<Float1, N> const& v, Float2 const& k) noexcept
			-> Rn_elmt<std::decay_t<decltype(v[0] * k)>, N>
		{
			return detail::binary_op_impl<std::decay_t<decltype(v[0] * k)>, N>{}(
				detail::mult_kernel{}, v, detail::wrap_scalar(k));
		}

		template <class Float1, class Float2, std::size_t N,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, Rn_elmt<Float2, N> const& v) noexcept
			-> Rn_elmt<std::decay_t<decltype(k * v[0])>, N>
		{
			return detail::binary_op_impl<std::decay_t<decltype(k * v[0])>, N>{}(
				detail::mult_kernel{}, detail::wrap_scalar(k), v);
		}

		template <class Float1, class Float2, std::size_t N,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator/(Rn_elmt<Float1, N> const& v, Float2 const& k) noexcept
			-> Rn_elmt<std::decay_t<decltype(v[0] / k)>, N>
		{
			return detail::binary_op_impl<std::decay_t<decltype(v[0] / k)>, N>{}(
				detail::div_kernel{}, v, detail::wrap_scalar(k));
		}

		template <class Float, std::size_t N>
		JKL_GPU_EXECUTABLE constexpr auto normsq(Rn_elmt<Float, N> const& v) noexcept {
			return v.normsq();
		}

		template <class Float, std::size_t N>
		JKL_GPU_EXECUTABLE auto norm(Rn_elmt<Float, N> const& v) noexcept {
			return v.norm();
		}

		template <class Float, std::size_t N>
		JKL_GPU_EXECUTABLE Rn_elmt<Float, N> normalize(Rn_elmt<Float, N> const& v) noexcept {
			if( v.normsq() == 0 )
				return Rn_elmt<Float, N>::zero();
			else
				return v / v.norm();
		}

		template <class Float1, class Float2, std::size_t N>
		JKL_GPU_EXECUTABLE auto dot(Rn_elmt<Float1, N> const& v, Rn_elmt<Float2, N> const& w) noexcept {
			auto sum = Float1(0) * Float2(0);
			for( std::size_t i = 0; i < N; ++i )
				sum += v[i] * w[i];
			return sum;
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto dot(R2_elmt<Float1> const& v, R2_elmt<Float2> const& w) noexcept {
			return v.x() * w.x() + v.y() * w.y();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto dot(R3_elmt<Float1> const& v, R3_elmt<Float2> const& w) noexcept {
			return v.x() * w.x() + v.y() * w.y() + v.z() * w.z();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto outer(R2_elmt<Float1> const& v, R2_elmt<Float2> const& w) noexcept
			-> gl2_elmt<std::decay_t<decltype(v.x() * w.x())>>
		{
			return{
				v.x()*w.x(), v.x()*w.y(),
				v.y()*w.x(), v.y()*w.y() };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto outer(R3_elmt<Float1> const& v, R3_elmt<Float2> const& w) noexcept
			-> gl3_elmt<std::decay_t<decltype(v.x() * w.x())>>
		{
			return{
				v.x()*w.x(), v.x()*w.y(), v.x()*w.z(),
				v.y()*w.x(), v.y()*w.y(), v.y()*w.z(),
				v.z()*w.x(), v.z()*w.y(), v.z()*w.z()
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto signed_area(R2_elmt<Float1> const& v, R2_elmt<Float2> const& w) noexcept {
			return v.x() * w.y() - v.y() * w.x();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto cross(R3_elmt<Float1> const& v, R3_elmt<Float2> const& w) noexcept
			-> R3_elmt<std::decay_t<decltype(v.y()*w.z() - v.z()*w.y())>>
		{
			return{
				v.y()*w.z() - v.z()*w.y(), 
				v.z()*w.x() - v.x()*w.z(), 
				v.x()*w.y() - v.y()*w.x()
			};
		}


		/// Operations on gl2_elmt

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator+(gl2_elmt<Float1> const& a, gl2_elmt<Float2> const& b) noexcept
			-> gl2_elmt<std::decay_t<decltype(a[0][0] + b[0][0])>>
		{
			return{
				a[0][0] + b[0][0], a[0][1] + b[0][1],
				a[1][0] + b[1][0], a[1][1] + b[1][1]
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator-(gl2_elmt<Float1> const& a, gl2_elmt<Float2> const& b) noexcept
			-> gl2_elmt<std::decay_t<decltype(a[0][0] - b[0][0])>>
		{
			return{
				a[0][0] - b[0][0], a[0][1] - b[0][1],
				a[1][0] - b[1][0], a[1][1] - b[1][1]
			};
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl2_elmt<Float1> const& a, Float2 const& k) noexcept
			-> gl2_elmt<std::decay_t<decltype(a[0][0] * k)>>
		{
			return{
				a[0][0] * k, a[0][1] * k, 
				a[1][0] * k, a[1][1] * k
			};
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, gl2_elmt<Float2> const& a) noexcept
			-> gl2_elmt<std::decay_t<decltype(k * a[0][0])>>
		{
			return{
				k * a[0][0], k * a[0][1],
				k * a[1][0], k * a[1][1]
			};
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator/(gl2_elmt<Float1> const& a, Float2 const& k) noexcept
			-> gl2_elmt<std::decay_t<decltype(a[0][0] / k)>>
		{
			return{
				a[0][0] / k, a[0][1] / k,
				a[1][0] / k, a[1][1] / k
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl2_elmt<Float1> const& a, gl2_elmt<Float2> const& b) noexcept
			-> gl2_elmt<std::decay_t<decltype(a[0][0]*b[0][0] + a[0][1]*b[1][0])>>
		{
			return{
				a[0][0]*b[0][0] + a[0][1]*b[1][0],
				a[0][0]*b[0][1] + a[0][1]*b[1][1],
				a[1][0]*b[0][0] + a[1][1]*b[1][0],
				a[1][0]*b[0][1] + a[1][1]*b[1][1]
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(GL2_elmt<Float1> const& a, GL2_elmt<Float2> const& b) noexcept
			-> GL2_elmt<std::decay_t<decltype(a[0][0]*b[0][0] + a[0][1]*b[1][0])>>
		{
			return{
				static_cast<gl2_elmt<Float1> const&>(a) * static_cast<gl2_elmt<Float2> const&>(b),
				do_not_check_validity{}
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(gl2_elmt<Float1> const& a, GL2_elmt<Float2> const& b) noexcept
			-> gl2_elmt<std::decay_t<decltype((a[0][0]*b[1][1] - a[0][1]*b[1][0]) / b.det())>>
		{
			return{
				(a[0][0]*b[1][1] - a[0][1]*b[1][0]) / b.det(),
				(a[0][1]*b[0][0] - a[0][0]*b[0][1]) / b.det(),
				(a[1][0]*b[1][1] - a[1][1]*b[1][0]) / b.det(),
				(a[1][1]*b[0][0] - a[1][0]*b[0][1]) / b.det()
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(GL2_elmt<Float1> const& a, GL2_elmt<Float2> const& b) noexcept
			-> GL2_elmt<std::decay_t<decltype((a[0][0]*b[1][1] - a[0][1]*b[1][0]) / b.det())>>
		{
			return{ static_cast<gl2_elmt<Float1> const&>(a) / b, do_not_check_validity{} };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl2_elmt<Float1> const& a, R2_elmt<Float2> const& v) noexcept
			-> R2_elmt<std::decay_t<decltype(a[0][0] * v.x() + a[0][1] * v.y())>>
		{
			return{
				a[0][0] * v.x() + a[0][1] * v.y(),
				a[1][0] * v.x() + a[1][1] * v.y()
			};
		}


		/// Operations on gl3_elmt

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator+(gl3_elmt<Float1> const& a, gl3_elmt<Float2> const& b) noexcept
			-> gl3_elmt<std::decay_t<decltype(a[0][0] + b[0][0])>>
		{
			return{
				a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2],
				a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2],
				a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator-(gl3_elmt<Float1> const& a, gl3_elmt<Float2> const& b) noexcept
			-> gl3_elmt<std::decay_t<decltype(a[0][0] - b[0][0])>>
		{
			return{
				a[0][0] - b[0][0], a[0][1] - b[0][1], a[0][2] - b[0][2],
				a[1][0] - b[1][0], a[1][1] - b[1][1], a[1][2] - b[1][2],
				a[2][0] - b[2][0], a[2][1] - b[2][1], a[2][2] - b[2][2]
			};
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl3_elmt<Float1> const& a, Float2 const& k) noexcept
			-> gl3_elmt<std::decay_t<decltype(a[0][0] * k)>>
		{
			return{
				a[0][0] * k, a[0][1] * k, a[0][2] * k,
				a[1][0] * k, a[1][1] * k, a[1][2] * k,
				a[2][0] * k, a[2][1] * k, a[2][2] * k
			};
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, gl3_elmt<Float2> const& a) noexcept
			-> gl3_elmt<std::decay_t<decltype(k * a[0][0])>>
		{
			return{
				k * a[0][0], k * a[0][1], k * a[0][2],
				k * a[1][0], k * a[1][1], k * a[1][2],
				k * a[2][0], k * a[2][1], k * a[2][2] };
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator/(gl3_elmt<Float1> const& a, Float2 const& k) noexcept
			-> gl3_elmt<std::decay_t<decltype(a[0][0] / k)>>
		{
			return{
				a[0][0] / k, a[0][1] / k, a[0][2] / k,
				a[1][0] / k, a[1][1] / k, a[1][2] / k,
				a[2][0] / k, a[2][1] / k, a[2][2] / k
			};
		}


		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl3_elmt<Float1> const& a, gl3_elmt<Float2> const& b) noexcept
			-> gl3_elmt<std::decay_t<decltype(a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0])>>
		{
			return{
				a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
				a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
				a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2],
				a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
				a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
				a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2],
				a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
				a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
				a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(GL3_elmt<Float1> const& a, GL3_elmt<Float2> const& b) noexcept
			-> GL3_elmt<std::decay_t<decltype(a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0])>>
		{
			return{
				static_cast<gl3_elmt<Float1> const&>(a) * static_cast<gl3_elmt<Float2> const&>(b),
				do_not_check_validity{}
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(SO3_elmt<Float1> const& a, SO3_elmt<Float2> const& b) noexcept
			-> SO3_elmt<std::decay_t<decltype(a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0])>>
		{
			return{
				static_cast<gl3_elmt<Float1> const&>(a) * static_cast<gl3_elmt<Float2> const&>(b),
				do_not_check_validity{}
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(gl3_elmt<Float1> const& a, GL3_elmt<Float2> const& b) noexcept
			-> decltype(a * b.inv())
		{
			return a * b.inv();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(GL3_elmt<Float1> const& a, GL3_elmt<Float2> const& b) noexcept
			-> decltype(a * b.inv())
		{
			return a * b.inv();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(SO3_elmt<Float1> const& a, SO3_elmt<Float2> const& b) noexcept
			-> decltype(a * b.inv())
		{
			return a * b.inv();
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(gl3_elmt<Float1> const& a, R3_elmt<Float2> const& v) noexcept
			-> R3_elmt<std::decay_t<decltype(a[0][0] * v.x() + a[0][1] * v.y() + a[0][2] * v.z())>>
		{
			return{
				a[0][0] * v.x() + a[0][1] * v.y() + a[0][2] * v.z(),
				a[1][0] * v.x() + a[1][1] * v.y() + a[1][2] * v.z(),
				a[2][0] * v.x() + a[2][1] * v.y() + a[2][2] * v.z()
			};
		}


		/// Operations on SU2_elmt

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(SU2_elmt<Float1> const& p, SU2_elmt<Float2> const& q) noexcept
			-> SU2_elmt<std::decay_t<decltype(p.w() * q.w() - dot(p.vector_part(), q.vector_part()))>>
		{
			return{ 
				p.w() * q.w() - dot(p.vector_part(), q.vector_part()),
				p.w() * q.vector_part() + q.w() * p.vector_part() + cross(p.vector_part(), q.vector_part()),
				do_not_check_validity{}
			};
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(SU2_elmt<Float1> const& p, SU2_elmt<Float2> const& q) noexcept
			-> decltype(p * q.inv())
		{
			return p * q.inv();
		}
		
		template <class Float1, class Float2, class Float3>
		JKL_GPU_EXECUTABLE auto SU2_interpolation(SU2_elmt<Float1> const& p, SU2_elmt<Float2> const& q, 
			Float3 const& t) noexcept
		{
			auto vec = (p.inv() * q).log() * t;
			return p * SU2_elmt<typename decltype(vec)::element_type>::exp(vec);
		}


		/// Operations on se3_elmt

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator+(se3_elmt<Float1> const& x, se3_elmt<Float2> const& y) noexcept
			-> se3_elmt<typename std::decay_t<decltype(x.rotation_part() + y.rotation_part())>::element_type>
		{
			return{ x.rotation_part() + y.rotation_part(), 
				x.translation_part() + y.translation_part() };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator-(se3_elmt<Float1> const& x, se3_elmt<Float2> const& y) noexcept
			-> se3_elmt<typename std::decay_t<decltype(x.rotation_part() - y.rotation_part())>::element_type>
		{
			return{ x.rotation_part() - y.rotation_part(),
				x.translation_part() - y.translation_part() };
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(se3_elmt<Float1> const& v, Float2 const& k) noexcept
			-> se3_elmt<typename std::decay_t<decltype(v.rotation_part() * k)>::element_type>
		{
			return{ v.rotation_part() * k, v.translation_part() * k };
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, se3_elmt<Float2> const& v) noexcept
			-> se3_elmt<typename std::decay_t<decltype(k * v.rotation_part())>::element_type>
		{
			return{ k * v.rotation_part(), k * v.translation_part() };
		}

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator/(se3_elmt<Float1> const& v, Float2 const& k) noexcept
			-> se3_elmt<typename std::decay_t<decltype(v.rotation_part() / k)>::element_type>
		{
			return{ v.rotation_part() / k, v.translation_part() / k };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto commutator(se3_elmt<Float1> const& x, se3_elmt<Float2> const& y) noexcept
			-> se3_elmt<typename std::decay_t<decltype(cross(x.rotation_part(), y.rotation_part()))>::element_type>
		{
			return{ cross(x.rotation_part(), y.rotation_part()),
				cross(x.rotation_part(), y.translation_part()) - cross(y.rotation_part(), x.translation_part()) };
		}


		/// Operations on SE3_elmt

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(SE3_elmt<Float1> const& s, SE3_elmt<Float2> const& t) noexcept
			-> SE3_elmt<typename std::decay_t<decltype(s.rotation_q() * t.rotation_q())>::element_type>
		{
			return{ s.rotation_q() * t.rotation_q(), 
				s.rotation_q().rotate(t.translation()) + s.translation() };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator/(SE3_elmt<Float1> const& s, SE3_elmt<Float2> const& t) noexcept
			-> decltype(s * t.inv())
		{
			return s * t.inv();
		}

		template <class Float1, class Float2, class Float3>
		JKL_GPU_EXECUTABLE auto SE3_interpolation(SE3_elmt<Float1> const& p, SE3_elmt<Float2> const& q,
			Float3 const& t) noexcept
		{
			auto vec = (p.inv() * q).log() * t;
			return p * SE3_elmt<typename decltype(vec)::element_type>::exp(vec);
		}


		/// Operations on sym_2x2

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, sym_2x2<Float2> const& a) noexcept
			-> sym_2x2<std::decay_t<decltype(k * a.xx())>>
		{
			return{ k * a.xx(), k * a.yy(), k * a.xy() };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(sym_2x2<Float1> const& a, R2_elmt<Float2> const& v) noexcept
			-> R2_elmt<std::decay_t<decltype(a.xx() * v.x() + a.xy() * v.y())>>
		{
			return{
				a.xx() * v.x() + a.xy() * v.y(),
				a.yx() * v.x() + a.yy() * v.y()
			};
		}


		/// Operations on pos_def_2x2

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator+(pos_def_2x2<Float1> const& a, pos_def_2x2<Float2> const& b) noexcept
			-> pos_def_2x2<std::decay_t<decltype(a.xx() + b.xx())>>
		{
			return{
				a.xx() + b.xx(), a.yy() + b.yy(), a.xy() + b.xy(),
				do_not_check_validity{}
			};
		}


		/// Operations on sym_3x3

		template <class Float1, class Float2,
			class = std::enable_if_t<std::is_convertible<Float1, Float2>::value
			|| std::is_convertible<Float2, Float1>::value>>
		JKL_GPU_EXECUTABLE constexpr auto operator*(Float1 const& k, sym_3x3<Float2> const& a) noexcept
			-> sym_3x3<std::decay_t<decltype(k * a.xx())>>
		{
			return{ k * a.xx(), k * a.yy(), k * a.zz(), k * a.xy(), k * a.yz(), k * a.zx() };
		}

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator*(sym_3x3<Float1> const& a, R3_elmt<Float2> const& v) noexcept
			-> R3_elmt<std::decay_t<decltype(a.xx() * v.x() + a.xy() * v.y() + a.xz() * v.z())>>
		{
			return{
				a.xx() * v.x() + a.xy() * v.y() + a.xz() * v.z(),
				a.yx() * v.x() + a.yy() * v.y() + a.yz() * v.z(),
				a.zx() * v.x() + a.zy() * v.y() + a.zz() * v.z()
			};
		}


		/// Operations on pos_def_3x3

		template <class Float1, class Float2>
		JKL_GPU_EXECUTABLE constexpr auto operator+(pos_def_3x3<Float1> const& a, pos_def_3x3<Float2> const& b) noexcept
			-> pos_def_3x3<std::decay_t<decltype(a.xx() + b.xx())>>
		{
			return{
				a.xx() + b.xx(), a.yy() + b.yy(), a.zz() + b.zz(),
				a.xy() + b.xy(), a.yz() + b.yz(), a.zx() + b.zx(),
				do_not_check_validity{}
			};
		}


		template <class Float>
		struct numerical : public numeric_consts<Float> {
			template <std::size_t N>
			using Rn_elmt = math::Rn_elmt<Float, N>;
			using R2_elmt = math::R2_elmt<Float>;
			using R3_elmt = math::R3_elmt<Float>;
			using gl2_elmt = math::gl2_elmt<Float>;
			using gl3_elmt = math::gl3_elmt<Float>;
			using GL2_elmt = math::GL2_elmt<Float>;
			using GL3_elmt = math::GL3_elmt<Float>;
			using su2_elmt = math::su2_elmt<Float>;
			using SU2_elmt = math::SU2_elmt<Float>;
			using so3_elmt = math::so3_elmt<Float>;
			using SO3_elmt = math::SO3_elmt<Float>;
			using se3_elmt = math::se3_elmt<Float>;
			using SE3_elmt = math::SE3_elmt<Float>;
			using sym_2x2 = math::sym_2x2<Float>;
			using pos_def_2x2 = math::pos_def_2x2<Float>;
			using sym_3x3 = math::sym_3x3<Float>;
			using pos_def_3x3 = math::pos_def_3x3<Float>;
		};
		using numf = numerical<float>;
		using numd = numerical<double>;
		using numld = numerical<long double>;

		template <class Object>
		JKL_GPU_EXECUTABLE constexpr auto square(Object const& x) noexcept(noexcept(x * x))
		{
			return x * x;
		}

		inline namespace math_literals {
			// Degree to radian conversion
			JKL_GPU_EXECUTABLE constexpr float operator""_degf(long double deg) noexcept {
				return float(deg) * numf::pi / 180.0f;
			}
			JKL_GPU_EXECUTABLE constexpr double operator""_deg(long double deg) noexcept {
				return double(deg) * numd::pi / 180.0;
			}
			JKL_GPU_EXECUTABLE constexpr long double operator""_degl(long double deg) noexcept {
				return deg * numld::pi / 180.0l;
			}
		}
	}
};