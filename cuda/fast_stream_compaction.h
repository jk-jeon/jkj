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

/// An implementation of the Hybrid algorithm described in 
/// "A Fast Hybrid Approach for Stream Compaction on GPUs" by Vernon Rego, et al, 2016.
/// Note that this algorithm can run only on devices having the warpSize = 32;
/// otherwise, the behavior is undefined.
/// Following modifications are applied to the original algorithm explained in the paper:
///  - Input/output may not be simple linear arrays. Any RandomAccessIterators pointing to valid 
///    GPU memory addresses can be used. Thrust iterators are perhaps the best match.
///  - An input item satisfying the predicate may not be just merely copied to the destination.
///    It may be transformed to some other thing before written. This transformation may or may not
///    accept the offset in the output destination as the second argument.
///  - Grid-stride-loop is applied to maximize the occupancy.

#pragma once
#include <cassert>
#include <type_traits>
#include <cuda_runtime.h>
#include "../portability.h"
#include "helper.h"

namespace jkj {
	namespace cuda {
		namespace detail {
			/// Call trans(input, output_idx, pred_result) if the expression is well-defined; otherwise,
			/// Call trans(input, output_idx) if the expression is well-defined; otherwise, call trans(input).

			template <class Transform, class Input, class Counter, class PredResult, class = void>
			struct ternary_well_defined {
				static constexpr bool value = false;
			};

			template <class Transform, class Input, class Counter, class PredResult>
			struct ternary_well_defined<Transform, Input, Counter, PredResult,
				std::void_t<decltype(std::declval<Transform&>()(
					std::declval<Input>(), std::declval<Counter>(), std::declval<PredResult>()))>>
			{
				static constexpr bool value = true;
			};

			template <class Transform, class Input, class Counter, class = void>
			struct binary_well_defined {
				static constexpr bool value = false;
			};

			template <class Transform, class Input, class Counter>
			struct binary_well_defined<Transform, Input, Counter,
				std::void_t<decltype(std::declval<Transform&>()(
					std::declval<Input>(), std::declval<Counter>()))>>
			{
				static constexpr bool value = true;
			};
			
			template <class Transform, class Input, class Counter, class PredResult,
				bool ternary_well_defined_, bool binary_well_defined_>
			struct transform_impl {
				FORCEINLINE __device__ static auto apply(Transform& trans, Input&& input, Counter&&, PredResult&&) {
					return trans(std::forward<Input>(input));
				}
			};

			template <class Transform, class Input, class Counter, class PredResult, bool binary_well_defined_>
			struct transform_impl<Transform, Input, Counter, PredResult, true, binary_well_defined_> {
				FORCEINLINE __device__ static auto apply(Transform& trans, Input&& input, Counter&& output_idx, PredResult&& pred_result) {
					return trans(std::forward<Input>(input),
						std::forward<Counter>(output_idx), std::forward<PredResult>(pred_result));
				}
			};

			template <class Transform, class Input, class Counter, class PredResult>
			struct transform_impl<Transform, Input, Counter, PredResult, false, true> {
				FORCEINLINE __device__ static auto apply(Transform& trans, Input&& input, Counter&& output_idx, PredResult&&) {
					return trans(std::forward<Input>(input),
						std::forward<Counter>(output_idx));
				}
			};
			
			template <class Transform, class Input, class Counter, class PredResult>
			FORCEINLINE __device__ auto transform(Transform& trans, Input&& input, Counter&& output_idx, PredResult&& pred_result)
			{
				return transform_impl<Transform, Input, Counter, PredResult,
					ternary_well_defined<Transform, Input, Counter, PredResult>::value,
					binary_well_defined<Transform, Input, Counter>::value
				>::apply(trans, std::forward<Input>(input),
					std::forward<Counter>(output_idx), std::forward<PredResult>(pred_result));
			}

			// Get predicate qualification-removed return type of predicates
			// The input to the predicate can be either mutable or immutable reference.
			// For the former case, the predicate may mutate the passed input value.
			// If both mutable/immutable references can be passed, mutable reference is chosen.
			template <class Predicate, class ValueType, class = void>
			struct get_pred_result_type {
				// Generate an error message
				template <class T>
				struct assert_helper : std::false_type {};

				template <class dummy = void, class = void>
				struct pass_const_reference {
					static_assert(assert_helper<ValueType>::value,
						"jkj::cuda::fast_stream_compaction: predicate can't be evaluated for the given input");
				};

				template <class dummy>
				struct pass_const_reference<dummy,
					std::void_t<decltype(std::declval<Predicate&>()(std::declval<ValueType const&>()))>>
				{
					using type = std::remove_cv_t<std::remove_reference_t<
						decltype(std::declval<Predicate&>()(std::declval<ValueType const&>()))>>;
				};

				using type = typename pass_const_reference<void>::type;
			};

			template <class Predicate, class ValueType>
			struct get_pred_result_type<Predicate, ValueType,
				std::void_t<decltype(std::declval<Predicate&>()(std::declval<ValueType&>()))>>
			{
				using type = std::remove_cv_t<std::remove_reference_t<
					decltype(std::declval<Predicate&>()(std::declval<ValueType&>()))>>;
			};

			/// The main kernel routine
#if __CUDACC_VER_MAJOR__ > 8
#define JKL_CUDA_WARP_FUNC(func_name,...)		func_name##_sync(0xffffffff, __VA_ARGS__)
#else
#define JKL_CUDA_WARP_FUNC(func_name,...)		func_name(__VA_ARGS__)
#endif
			template <class InputIterator, class SizeType, class OutputIterator, class Counter, class Predicate, class Transform>
			__global__ void fast_stream_compaction_kernel(InputIterator first, SizeType const number_of_threads,
				OutputIterator result, Counter* counter_ptr, Predicate pred, Transform trans)
			{
				auto const group_size = warpSize * warpSize;
				auto const lane_id = threadIdx.x % warpSize;
				unsigned int votes, cnt;

				for( auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
					thread_idx < number_of_threads;
					thread_idx += blockDim.x * gridDim.x )
				{
					// One group per one warp
					auto group_idx = thread_idx / warpSize;

					// Stage 1 - store votes & cnt of each subgroup to the corresponding lane
					using value_type = typename thrust::iterator_traits<std::remove_cv_t<InputIterator>>::value_type;
					using pred_result_type = typename get_pred_result_type<Predicate, value_type>::type;
					value_type			input_result[32];
					pred_result_type	pred_result[32];
					for( unsigned int i = 0; i < warpSize; ++i ) {
						input_result[i] = first[group_idx * group_size + i * warpSize + lane_id];
						pred_result[i] = pred(input_result[i]);
						auto mask = JKL_CUDA_WARP_FUNC(__ballot, static_cast<bool>(pred_result[i]));

						if( lane_id == i ) {
							votes = mask;
							cnt = __popc(mask);
						}
					}

					// Stage 2 - perform prefix sum on each warp
					// Now, the cnt stored in the last lane is the number of elements in the group passing the predicate.
					for( unsigned int i = 1; i < warpSize; i <<= 1 ) {
						auto n = JKL_CUDA_WARP_FUNC(__shfl_up, cnt, i);
						if( lane_id >= i ) cnt += n;
					}

					// Stage 3 - add cnt stored in the last lane to the global counter
					// atomicAdd() will return the global offset of the warp.
					// This result is broadcasted to other lanes using __shfl()
					unsigned int group_offset, temp;
					if( lane_id == warpSize - 1 ) {
						temp = atomicAdd(counter_ptr, Counter(cnt));
						cnt = 0;
					}
					group_offset = JKL_CUDA_WARP_FUNC(__shfl, temp, warpSize - 1);

					// Stage 4 - calculate the total offset and perform gathering
					// Each lane is responsible for the lane_id'th element in each subgroup.
					for( unsigned int i = 0; i < warpSize; ++i ) {
						// The result of __ballot in the ith subgroup
						auto mask = JKL_CUDA_WARP_FUNC(__shfl, votes, i);

						// Offset inside group
						auto subgroup_offset = JKL_CUDA_WARP_FUNC(__shfl, cnt, i - 1);

						if( pred_result[i] ) {
							// Offset inside subgroup
							auto intra_subgroup_offset = __popc(mask & ((1 << lane_id) - 1));
							auto offset = group_offset + subgroup_offset + intra_subgroup_offset;
							result[offset] = transform(trans, input_result[i], Counter(offset), pred_result[i]);
						}
					}
				}
			}
#undef JKL_CUDA_WARP_FUNC

			/// The kernel routine for less than 1024 elements
			template <class InputIterator, class SizeType, class OutputIterator, class Counter, class Predicate, class Transform>
			__global__ void fast_stream_compaction_remaining_kernel(InputIterator first, SizeType number_of_threads,
				OutputIterator result, Counter* counter_ptr, Predicate pred, Transform trans)
			{
				for( auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
					thread_idx < number_of_threads;
					thread_idx += blockDim.x * gridDim.x )
				{
					using value_type = typename thrust::iterator_traits<std::remove_cv_t<InputIterator>>::value_type;
					value_type input_result = first[thread_idx];
					auto pred_result = pred(input_result);
					if( pred_result ) {
						auto offset = atomicAdd(counter_ptr, 1);
						result[offset] = transform(trans, input_result, Counter(offset), pred_result);
					}
				}
			}
		}
		
		/// Useful functors
		struct identity_functor {
			template <class T>
			__host__ __device__ T operator()(T const& x) const {
				return x;
			}
		};
		template <class T>
		struct array_functor {
			T* const ptr;
			template <class IndexType>
			__device__ T& operator()(IndexType idx) const {
				return ptr[idx];
			}
		};
		template <class T>
		auto make_array_functor(T* ptr) noexcept {
			return array_functor<T>{ ptr };
		}
		template <class T>
		auto make_const_array_functor(T const* ptr) noexcept {
			return array_functor<T const>{ ptr };
		}

		template <class Iterator>
		struct iterator_predicate {
			Iterator itr;
			template <class IndexType>
			__device__ bool operator()(IndexType idx) const {
				return itr[idx];
			}
		};
		template <class Iterator>
		auto make_iterator_predicate(Iterator const& itr) noexcept {
			return iterator_predicate<Iterator>{ itr };
		}

		/// Host-side function
		/// If grid_size == 0, grid dimension is automatically configured.
		template <class InputIterator, class SizeType, class OutputIterator, class Counter, class Predicate,
			class Transform = identity_functor, class = std::enable_if_t<!std::is_pointer<std::decay_t<Predicate>>::value>>
		void fast_stream_compaction_n(InputIterator&& first, SizeType n,
			OutputIterator&& result, Counter* counter_ptr, Predicate&& pred, Transform&& trans ={},
			cudaStream_t stream = nullptr, std::size_t block_size = 1024, std::size_t grid_size = 0)
		{
			// CUDA atomicAdd is supported only for int, unsigned int, and unsigned long long
			static_assert(std::is_same<Counter, int>::value || std::is_same<Counter, unsigned int>::value
				|| std::is_same<Counter, unsigned long long>::value,
				"jkj::cuda: counter_ptr must be of one of int*, unsigned int*, or unsigned long long*");
			assert(block_size % 32 == 0);
			jkj::cuda::check_error(cudaMemsetAsync(counter_ptr, 0, sizeof(Counter), stream));

			constexpr unsigned int subgroup_size = 32;
			constexpr unsigned int group_size = 1024;

			auto const number_of_groups = n / group_size;
			auto const number_of_threads = number_of_groups * subgroup_size;
			auto const residue_size = n - number_of_groups * group_size;

			// Automatically find the optimal grid dimension
			if( grid_size == 0 ) {
				grid_size = jkj::cuda::get_optimal_grid_size(
					detail::fast_stream_compaction_kernel<std::remove_reference_t<InputIterator>,
					SizeType, std::remove_reference_t<OutputIterator>,
					Counter, std::remove_reference_t<Predicate>,
					std::remove_reference_t<Transform>>, number_of_threads, block_size);
			}

			if( number_of_threads > 0 ) {
				detail::fast_stream_compaction_kernel<<<grid_size, block_size, 0, stream>>>(first,
					number_of_threads, result, counter_ptr, pred, trans);
			}

			if( residue_size > 0 ) {
				grid_size = (residue_size + block_size - 1) / block_size;
				detail::fast_stream_compaction_remaining_kernel<<<grid_size, block_size, 0, stream>>>(
					first + number_of_groups * group_size, residue_size, result, counter_ptr, pred, trans);
			}
		}

		/// Convenient overload for the case when Predicate is an array_functor
		template <class InputIterator, class SizeType, class OutputIterator, class Counter, class ConditionType, class Transform = identity_functor>
		void fast_stream_compaction_n(InputIterator&& first, SizeType size,
			OutputIterator&& result, Counter* counter_ptr, ConditionType const* conditions_ptr, Transform&& trans ={},
			cudaStream_t stream = nullptr, unsigned int block_dim = 1024, unsigned int grid_dim = 0)
		{
			fast_stream_compaction_n(std::forward<InputIterator>(first), size,
				result, counter_ptr, make_array_functor(conditions_ptr),
				std::forward<Transform>(trans), stream, block_dim, grid_dim);
		}
		
		/// Convenient overload for the case of begin/end pair
		/// pred can be either a bool-valued functor or a const pointer to a type that is contextually convertible to bool
		template <class InputIterator, class OutputIterator, class Counter, class Predicate, class Transform = identity_functor>
		void fast_stream_compaction(InputIterator&& first, InputIterator&& last,
			OutputIterator&& result, Counter* counter_ptr, Predicate&& pred, Transform&& trans ={},
			cudaStream_t stream = nullptr, unsigned int block_dim = 1024, unsigned int grid_dim = 0)
		{
			fast_stream_compaction(std::forward<InputIterator>(first), last - first,
				std::forward<OutputIterator>(result), counter_ptr,
				std::forward<Predicate>(pred), std::forward<Transform>(trans),
				stream, block_dim, grid_dim);
		}
	}
}