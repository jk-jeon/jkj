#pragma once
#include <array>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include "../bit_twiddling.h"
#include "../numerical_lie_group.h"
#include "device_algorithm.h"
#include "helper.h"
#include "thrust_allocators.h"

/// Implementation of "kANN on the GPU with Shifted Sorting", 2012, Li, et al., for 3-dimensional case.
/// In our usecase, there are much more query points than data points, so we don't sort them together.
/// Rather, each query point finds its position inside the sorted data points using binary search.
/// This has a further advantage that multiple queries can be requested without re-sorting the data points.

namespace jkj {
	/// 3D space filling curve encodings
	/// Input: three 21-bit integers
	/// Output: 63-bit Morton/Hilbert code

	/// Morton code
	struct morton_encode_3d {
		JKL_GPU_EXECUTABLE std::uint64_t operator()(std::uint32_t x, std::uint32_t y, std::uint32_t z) const
		{
			constexpr std::size_t bits_per_axis = 21;
			std::uint64_t interleaved = 0;
			for( std::uint32_t q = (1 << (bits_per_axis - 1)); q > 0; q >>= 1 ) {
				interleaved |= (x & q);
				interleaved <<= 1;
				interleaved |= (y & q);
				interleaved <<= 1;
				interleaved |= (z & q);
			}
			return interleaved;
		}
	};
	
	/// Hilbert code (reimplementation of John Skilling's algorithm)
	struct hilbert_encode_3d {
		JKL_GPU_EXECUTABLE std::uint64_t operator()(std::uint32_t x, std::uint32_t y, std::uint32_t z) const
		{
			constexpr std::size_t bits_per_axis = 21;
			for( std::uint32_t q = (1 << (bits_per_axis - 1)); q > 1; q >>= 1 ) {
				std::uint32_t mask = q - 1;

				// If the bit of x is set, invert lower bits of x
				if( x & q ) {
					// Invert
					x ^= mask;
				}
				// If the bit of y is set, invert lower bits of x
				if( y & q )
					x ^= mask;
				// Otherwise, exchange lower bits of y with x
				else {
					std::uint32_t t = (x ^ y) & mask;
					x ^= t;
					y ^= t;
				}
				// If the bit of z is set, invert lower bits of x
				if( z & q )
					x ^= mask;
				// Otherwise, exchange lower bits of z with x
				else {
					std::uint32_t t = (x ^ z) & mask;
					x ^= t;
					z ^= t;
				}
			}

			// Make interleaved combination of resulting x, y, z
			auto interleaved = morton_encode_3d{}(x, y, z);

			// Gray decode the above
			interleaved ^= (interleaved >> 32);
			interleaved ^= (interleaved >> 16);
			interleaved ^= (interleaved >> 8);
			interleaved ^= (interleaved >> 4);
			interleaved ^= (interleaved >> 2);
			interleaved ^= (interleaved >> 1);

			return interleaved;
		}
	};
	
	namespace cuda {
		// Unfortunately, NVCC requires kernel functions to be public, so we need a separate implementation class
		// with all the details open, and then wrap it with another class having the right interface.
		namespace shifted_sorting_kann_detail {
			template <class Neighbor>
			__device__ void bitonic_sort_step(Neighbor* load_buffer, bool* is_loaded,
				unsigned int load_idx, unsigned int major_step, unsigned int minor_step)
			{
				auto target_idx = load_idx ^ minor_step;

				// The threads with the lowest ids sort the array.
				if( target_idx > load_idx ) {
					if( (load_idx & major_step) == 0 ) {
						// Sort ascending
						if( !is_loaded[load_idx] || (is_loaded[target_idx] &&
							load_buffer[load_idx].value > load_buffer[target_idx].value) )
						{
							thrust::swap(is_loaded[load_idx], is_loaded[target_idx]);
							thrust::swap(load_buffer[load_idx], load_buffer[target_idx]);
						}
					}
					else {
						// Sort descending
						if( !is_loaded[target_idx] || (is_loaded[load_idx] &&
							load_buffer[load_idx].value < load_buffer[target_idx].value) )
						{
							thrust::swap(is_loaded[load_idx], is_loaded[target_idx]);
							thrust::swap(load_buffer[load_idx], load_buffer[target_idx]);
						}
					}
				}
			}

			// In the original paper, k threads in this kernel is gathered as a block, 
			// but here we use 2 * rounded_k threads per a (conceptual) block
			template <class PointType, class DistanceType, class Neighbor,
				class DataPointIterator, class QueryPointIterator, class OutputIterator>
			__global__ void load_kernel(unsigned int const k, unsigned int const rounded_k,
				DataPointIterator data_points, std::size_t const number_of_data_points,
				QueryPointIterator query_points, std::size_t const number_of_query_points,
				OutputIterator kann_results,
				unsigned int const* const data_indices_ptr, unsigned int const* const query_indices_ptr)
			{				
				extern __shared__ unsigned char dynamic_shared_memory[];

				auto load_idx = threadIdx.x % (2 * rounded_k);
				auto base_idx = threadIdx.x - load_idx;

				auto load_buffer = reinterpret_cast<Neighbor*>(dynamic_shared_memory) + base_idx;
				auto is_loaded = reinterpret_cast<bool*>(
					reinterpret_cast<Neighbor*>(dynamic_shared_memory) + blockDim.x) + base_idx;

				// Grid-stride loop
				auto tid = threadIdx.x + blockDim.x * blockIdx.x;
				auto query_idx = tid / (2 * rounded_k);
				while( query_idx < number_of_query_points )
				{
					auto iq = query_indices_ptr[query_idx];
					auto query_pt = (PointType const&)query_points[query_idx];

					// Load points
					unsigned int from = iq >= rounded_k ? iq - rounded_k : 0;
					unsigned int to = iq + rounded_k <= number_of_data_points ? iq + rounded_k : number_of_data_points;
					unsigned int loaded = to - from;

					if( load_idx < loaded ) {
						load_buffer[load_idx].index = data_indices_ptr[from + load_idx];
						auto data_pt = (PointType const&)data_points[data_indices_ptr[from + load_idx]];
						load_buffer[load_idx].value = normsq(query_pt - data_pt);
						is_loaded[load_idx] = true;
					}
					else
						is_loaded[load_idx] = false;

					__syncthreads();

					// Parallel bitonic sort
					for( unsigned int major_step = 2; major_step <= 2 * rounded_k; major_step <<= 1 ) {
						for( unsigned int minor_step = major_step >> 1; minor_step > 0; minor_step >>= 1 ) {
							bitonic_sort_step(load_buffer, is_loaded, load_idx, major_step, minor_step);
							__syncthreads();
						}
					}

					// Copy k sorted elements
					if( load_idx < k )
						kann_results[query_idx][load_idx] = load_buffer[load_idx];

					__syncthreads();

					tid += blockDim.x * gridDim.x;
					query_idx = tid / (2 * rounded_k);
				}
			}

			template <class Neighbor, class ValueType>
			struct neighbor_to_value : thrust::unary_function<Neighbor const&, ValueType> {
				__device__ ValueType operator()(Neighbor const& n) const {
					return n.value;
				}
			};

			// This is the most complicated part of the algorithm.
			// I'm not pretty sure about the correctness of this code, but the results anyway seems fine.
			template <class PointType, class DistanceType, class Neighbor,
				class DataPointIterator, class QueryPointIterator, class OutputIterator>
			__global__ void merge_kernel(unsigned int const k, unsigned int const rounded_k,
				DataPointIterator data_points, std::size_t const number_of_data_points,
				QueryPointIterator query_points, std::size_t const number_of_query_points,
				OutputIterator kann_results,
				unsigned int const* const data_indices_ptr, unsigned int const* const query_indices_ptr)
			{
				extern __shared__ unsigned char dynamic_shared_memory[];

				auto load_idx = threadIdx.x % (2 * rounded_k);
				auto result_idx = load_idx / 2;
				auto base_idx = threadIdx.x - load_idx;

				auto counters = reinterpret_cast<unsigned int*>(dynamic_shared_memory) + base_idx;
				auto current_results = reinterpret_cast<Neighbor*>(
					reinterpret_cast<unsigned int*>(dynamic_shared_memory) + blockDim.x) + (base_idx / 2);

				// Grid-stride loop
				auto tid = threadIdx.x + blockDim.x * blockIdx.x;
				auto query_idx = tid / (2 * rounded_k);
				while( query_idx < number_of_query_points )
				{
					// Initialize counter and load current best candidates
					if( load_idx % 2 != 0 && result_idx < k ) {
						counters[load_idx] = 1;
						current_results[result_idx] = kann_results[query_idx][result_idx];
					}
					else {
						counters[load_idx] = 0;
					}

					// Load points
					auto iq = query_indices_ptr[query_idx];
					auto query_pt = (PointType const&)query_points[query_idx];

					unsigned int from = iq >= rounded_k ? iq - rounded_k : 0;
					unsigned int to = iq + rounded_k <= number_of_data_points ? iq + rounded_k : number_of_data_points;
					unsigned int loaded = to - from;

					bool should_proceed = false;
					Neighbor new_candidate;
					if( load_idx < loaded ) {						
						new_candidate.index = data_indices_ptr[from + load_idx];
						auto data_pt = (PointType const&)data_points[data_indices_ptr[from + load_idx]];
						new_candidate.value = normsq(query_pt - data_pt);
						should_proceed = true;
					}

					__syncthreads();

					// Propose the position where the candidate might be inserted
					unsigned int loc;
					if( should_proceed ) {
						auto current_result_value = thrust::make_transform_iterator(current_results,
							neighbor_to_value<Neighbor, DistanceType>{});
						loc = jkj::cuda::binary_search_n(current_result_value, k, new_candidate.value)
							- current_result_value;
					}

					__syncthreads();

					// Update counter
					unsigned int offset;
					if( should_proceed ) {
						if( loc == k || (loc != 0 && new_candidate.index == current_results[loc - 1].index) )
							should_proceed = false;
						else
							offset = atomicAdd(&counters[loc * 2], 1);
					}
					
					__syncthreads();

					// In-place exclusive scan on counter
					for( unsigned int i = 1; i < 2 * k; i <<= 1 ) {
						unsigned int temp = 0;
						if( load_idx >= i )
							temp = counters[load_idx - i];
						__syncthreads();

						counters[load_idx] += temp;
						__syncthreads();
					}
					unsigned int prev_counter = 0;
					if( load_idx > 0 )
						prev_counter = counters[load_idx - 1];
					__syncthreads();
					counters[load_idx] = prev_counter;

					__syncthreads();

					// Collect current best candidates
					if( load_idx < k ) {
						unsigned int index = counters[load_idx * 2 + 1];
						if( index < k )
							kann_results[query_idx][index] = current_results[load_idx];
					}

					// Collect new candidates
					if( should_proceed ) {
						unsigned int index = counters[loc * 2] + offset;
						if( index < k )
							kann_results[query_idx][index] = new_candidate;
					}

					__syncthreads();

					tid += blockDim.x * gridDim.x;
					query_idx = tid / (2 * rounded_k);
				}
			}

			template <class Neighbor, class OutputIterator>
			__global__ void blockwise_sorting_kernel(unsigned int const k, unsigned int const rounded_k,
				std::size_t const number_of_query_points, OutputIterator kann_results)
			{
				extern __shared__ unsigned char dynamic_shared_memory[];

				auto load_idx = threadIdx.x % rounded_k;
				auto base_idx = threadIdx.x - load_idx;

				auto load_buffer = reinterpret_cast<Neighbor*>(dynamic_shared_memory) + base_idx;
				auto is_loaded = reinterpret_cast<bool*>(
					reinterpret_cast<Neighbor*>(dynamic_shared_memory) + blockDim.x) + base_idx;

				// Grid-stride loop
				auto tid = threadIdx.x + blockDim.x * blockIdx.x;
				auto query_idx = tid / rounded_k;
				while( query_idx < number_of_query_points )
				{
					// Load results
					if( load_idx < k ) {
						load_buffer[load_idx] = kann_results[query_idx][load_idx];
						is_loaded[load_idx] = true;
					}
					else
						is_loaded[load_idx] = false;

					__syncthreads();

					// Parallel bitonic sort
					// I want to wrap this loop as a __device__ function, but I can't sure what will gonna happen
					// if I call __syncthreads() inside a __device__ function...
					for( unsigned int major_step = 2; major_step <= rounded_k; major_step <<= 1 ) {
						for( unsigned int minor_step = major_step >> 1; minor_step > 0; minor_step >>= 1 ) {
							bitonic_sort_step(load_buffer, is_loaded, load_idx, major_step, minor_step);
							__syncthreads();
						}
					}

					// Copy sorted result back
					if( load_idx < k )
						kann_results[query_idx][load_idx] = load_buffer[load_idx];

					__syncthreads();

					tid += blockDim.x * gridDim.x;
					query_idx = tid / rounded_k;
				}
			}

			template <class PointType, class DistanceType, class SpaceFilling3D, class DataPointIterator>
			struct impl {
				using point_type = PointType;
				using distance_type = DistanceType;

				struct neighbor {
					unsigned int	index;	// index in the data array
					distance_type	value;	// squared distance
				};

				impl() : child_streams(cudaEventDefault, 5)
				{
					// Precalculate optimal block sizes
					int device_id;
					jkj::cuda::check_error(cudaGetDevice(&device_id));
					cudaDeviceProp device_prop;
					jkj::cuda::check_error(cudaGetDeviceProperties(&device_prop, device_id));
					max_threads_per_block = unsigned int(device_prop.maxThreadsPerBlock);
					// max_threads_per_block must be power of 2
					auto log2 = jkj::math::ilog2(max_threads_per_block);
					max_threads_per_block = 1u << log2;
				}

				void presort_data(DataPointIterator new_data_points, std::size_t number_of_data_points,
					point_type const& lower_bound, point_type const& upper_bound, cudaStream_t stream = nullptr)
				{
					data_points = std::move(new_data_points);
					scale = upper_bound - lower_bound;
					scale.x() = distance_type(0.8 / scale.x());
					scale.y() = distance_type(0.8 / scale.y());
					scale.z() = distance_type(0.8 / scale.z());

					base_point = lower_bound;

					child_streams.record_parent(stream);
					for( unsigned int j = 0; j < 5; ++j ) {
						child_streams.fork_individual(stream, j, [&](cudaStream_t child_stream) {
							codes[j].resize(number_of_data_points);
							indices[j].resize(number_of_data_points);

							// Compute codes for data points
							thrust::for_each_n(thrust::cuda::par.on(child_stream),
								thrust::make_counting_iterator(0u), number_of_data_points,
								compute_code_kernel_data{ encoder, data_points, base_point, scale,
								distance_type(j * 0.05), codes[j].data().get(), indices[j].data().get() });

							// Sort all points with their indices
							thrust::sort_by_key(thrust::cuda::par(allocs[j]).on(child_stream),
								codes[j].begin(), codes[j].end(), indices[j].begin());
						});
					}
				}
				
				// results[query_idx][nbhd_idx] for query_idx in [0,number_of_query_points),
				// nbhd_idx in [0,k) must reference to the nbhd_idx'th neighborhood of query_idx'th query point
				template <class QueryPointIterator, class OutputIterator>
				void compute_kann(unsigned int k, QueryPointIterator&& query_points, std::size_t number_of_query_points,
					OutputIterator&& results, cudaStream_t stream = nullptr)
				{
					assert(k != 0 && k <= codes[0].size());
					kann_load_buffer.resize(2 * k * number_of_query_points);

					// Compute codes for query points and find their position in the sorted data points
					child_streams.record_parent(stream);
					for( unsigned int j = 0; j < 5; ++j ) {
						child_streams.fork_individual(stream, j, [&](cudaStream_t child_stream) {
							query_indices[j].resize(number_of_query_points);
							thrust::for_each_n(thrust::cuda::par.on(child_stream),
								thrust::make_counting_iterator(0u), number_of_query_points,
								compute_code_kernel_query<QueryPointIterator>{
								encoder, query_points, base_point, scale, distance_type(j * 0.05),
								codes[j].data().get(), unsigned int(codes[j].size()),
								query_indices[j].data().get() });
						});
					}

					// rounded_k is the smallest power of 2 that is larger than or equal to k
					auto rounded_k = 1u << jkj::math::ilog2(2 * k - 1);
					assert(rounded_k <= max_threads_per_block);

					// Load phase
					auto shared_memory_size = (sizeof(neighbor) + sizeof(bool)) * max_threads_per_block;
					auto grid_size = jkj::cuda::get_optimal_grid_size(
						load_kernel<point_type, distance_type, neighbor,
						DataPointIterator, std::decay_t<QueryPointIterator>, std::decay_t<OutputIterator>>,
						2 * rounded_k * number_of_query_points, max_threads_per_block, shared_memory_size);
					
					load_kernel<point_type, distance_type, neighbor>
						<<<grid_size, max_threads_per_block, shared_memory_size, stream>>>
						(k, rounded_k, data_points, codes[0].size(), query_points, number_of_query_points,
							results, indices[0].data().get(), query_indices[0].data().get());
					
					// Merge phase
					shared_memory_size = sizeof(unsigned int) * max_threads_per_block
						+ sizeof(neighbor) * (max_threads_per_block / 2);
					grid_size = jkj::cuda::get_optimal_grid_size(
						merge_kernel<point_type, distance_type, neighbor,
						DataPointIterator, std::decay_t<QueryPointIterator>, std::decay_t<OutputIterator>>,
						2 * rounded_k * number_of_query_points, max_threads_per_block, shared_memory_size);

					auto sorting_shared_memory_size = (sizeof(neighbor) + sizeof(bool)) * max_threads_per_block;
					auto sorting_grid_size = jkj::cuda::get_optimal_grid_size(
						blockwise_sorting_kernel<neighbor, std::decay_t<OutputIterator>>,
						rounded_k * number_of_query_points, max_threads_per_block, sorting_shared_memory_size);

					for( unsigned int j = 1; j < 5; ++j ) {
						merge_kernel<point_type, distance_type, neighbor>
							<<<grid_size, max_threads_per_block, shared_memory_size, stream>>>
							(k, rounded_k, data_points, codes[j].size(), query_points, number_of_query_points,
								results, indices[j].data().get(), query_indices[j].data().get());

						blockwise_sorting_kernel<neighbor>
							<<<sorting_grid_size, max_threads_per_block, sorting_shared_memory_size, stream>>>
							(k, rounded_k, number_of_query_points, results);
					}
				}

				unsigned int										max_threads_per_block;

				template <class T>
				using device_vector = thrust::device_vector<T, jkj::cuda::uninitialized_allocator<T>>;

				DataPointIterator									data_points;
				SpaceFilling3D										encoder;
				point_type											base_point;
				point_type											scale;

				std::array<device_vector<std::uint64_t>, 5>			codes;
				std::array<device_vector<unsigned int>, 5>			indices;
				std::array<device_vector<unsigned int>, 5>			query_indices;
				device_vector<neighbor>								kann_load_buffer;

				jkj::cuda::stream_fork								child_streams;
				std::array<jkj::cuda::cached_allocator<8192>, 5>	allocs;

				static __device__ std::uint64_t encode(SpaceFilling3D const& encoder, point_type pt,
					point_type const& base_point, point_type const& scale, distance_type shift)
				{
					pt -= base_point;
					pt.x() = pt.x() * scale.x() + shift;
					pt.y() = pt.y() * scale.y() + shift;
					pt.z() = pt.z() * scale.z() + shift;

					// Each dimension is divided by 2^21 grids
					static constexpr std::uint32_t scale_multiplier = std::uint32_t(1) << std::uint32_t(21);

					auto x = std::uint32_t(pt.x() * scale_multiplier);
					auto y = std::uint32_t(pt.y() * scale_multiplier);
					auto z = std::uint32_t(pt.z() * scale_multiplier);

					return encoder(x, y, z);
				}

				struct compute_code_kernel_data {
					SpaceFilling3D									encoder;
					DataPointIterator								points;
					point_type										base_point;
					point_type										scale;
					distance_type									shift;
					std::uint64_t* const							codes_ptr;
					unsigned int* const								indices_ptr;

					__device__ void operator()(unsigned int idx) const {
						// I wanted use static_cast rather than C-style cast here, but NVCC doesn't accept that.
						// It seems that the problem is related to "explicit" specifier,
						// but I don't know the exact reason. Perhaps another bug of NVCC?
						codes_ptr[idx] = encode(encoder, (point_type const&)points[idx], base_point, scale, shift);
						indices_ptr[idx] = idx;
					}
				};

				template <class QueryPointIterator>
				struct compute_code_kernel_query {
					SpaceFilling3D									encoder;
					std::decay_t<QueryPointIterator>				query_points;
					point_type										base_point;
					point_type										scale;
					distance_type									shift;
					std::uint64_t const* const						codes_ptr;
					unsigned int									number_of_data_points;
					unsigned int* const								query_indices_ptr;

					__device__ void operator()(unsigned int idx) const {
						// I wanted use static_cast rather than C-style cast here, but NVCC doesn't accept that.
						// It seems that the problem is related to "explicit" specifier,
						// but I don't know the exact reason. Perhaps another bug of NVCC?
						auto code = encode(encoder, (point_type const&)query_points[idx], base_point, scale, shift);
						query_indices_ptr[idx] = jkj::cuda::binary_search_n(codes_ptr, number_of_data_points, code) - codes_ptr;
					}
				};
			};
		}

		template <class PointType, class DistanceType, class SpaceFilling3D = hilbert_encode_3d,
			class DataPointIterator = PointType const*>
		class shifted_sorting_kann : private shifted_sorting_kann_detail::impl<PointType, DistanceType,
			SpaceFilling3D, DataPointIterator> {
			using impl_type = shifted_sorting_kann_detail::impl<PointType, DistanceType,
				SpaceFilling3D, DataPointIterator>;
		public:
			using impl_type::impl_type;
			using point_type = typename impl_type::point_type;
			using neighbor = typename impl_type::neighbor;
			using impl_type::presort_data;
			using impl_type::compute_kann;
		};
	}
}