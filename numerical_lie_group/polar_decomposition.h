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
#include "gl2_elmt.h"
#include "gl3_elmt.h"

namespace jkj {
	namespace math {
		// Compute sqrt(A^T A) for 2x2 matrix A
		template <class ComponentType,
			class Storage = ComponentType[3],
			class StorageTraits = default_storage_traits,
			class OtherStorage, class OtherStorageTraits>
		JKL_GPU_EXECUTABLE posdef2_elmt<ComponentType, Storage, StorageTraits>
			polar_positive_part(GL2_elmt<ComponentType, OtherStorage, OtherStorageTraits> const& m)
		{
			using ret_type = posdef2_elmt<ComponentType, Storage, StorageTraits>;

			auto determinant = m.det();
			bool const swap_rows = determinant < zero<ComponentType>();

			using swapped_matrix_type = gl2_elmt<ComponentType,
				detail::tuple<decltype(m.template get<1>()), decltype(m.template get<0>())>,
				default_storage_traits>;

			auto do_decomposition = [](auto&& m, auto&& determinant) {
				auto a_plus_d = m.get<0, 0>() + m.get<1, 1>();
				auto b_minus_c = m.get<0, 1>() - m.get<1, 0>();

				using std::sqrt;
				auto normalizer = sqrt(a_plus_d * a_plus_d + b_minus_c * b_minus_c);

				return ret_type{
					m.get<0, 0>() * m.get<0, 0>() + m.get<1, 0>() * m.get<1, 0>() + determinant,
					m.get<0, 0>() * m.get<0, 1>() + m.get<1, 0>() * m.get<1, 1>(),
					m.get<0, 1>() * m.get<0, 1>() + m.get<1, 1>() * m.get<1, 1>() + determinant,
					no_validity_check{}, } /
					std::move(normalizer);
			};

			if( swap_rows )
				return do_decomposition(swapped_matrix_type{ direct_construction{},
					m.template get<1>(), m.template get<0>() }, -std::move(determinant));
			else
				return do_decomposition(m, std::move(determinant));
		}
	}
}