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
#include <map>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/contiguous_storage.h>

namespace jkl {
	namespace cuda {
		// cached_allocator: a simple allocator for caching allocation requests
		// This is a modification of the code brought from:
		// https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
		template <std::size_t min_block_size>
		class cached_allocator {
		public:
			// Just allocate bytes
			using value_type = unsigned char;

			~cached_allocator() {
				// Release all allocations when cached_allocator goes out of scope
				release_all();
			}

			value_type* allocate(std::ptrdiff_t num_bytes) {
				// Regularize num_bytes to make it an integer multiple of min_block_size
				// This will lower the chance that large number of blocks of similar size are never used
				num_bytes = ((num_bytes + min_block_size - 1) / min_block_size) * min_block_size;

				value_type* result = nullptr;

				// Search the cache for a free block having enough size
				free_blocks_type::iterator free_block = free_blocks.lower_bound(num_bytes);

				// Not found
				if( free_block == free_blocks.end() ) {
					// No allocation of the enough size exists
					// Create a new one with cuda::malloc
					try {
						// Allocate memory and convert cuda::pointer to raw pointer
						result = thrust::cuda::malloc<value_type>(num_bytes).get();
					}
					catch( ... ) {
						// If allocation fails, then retry after releasing all free blocks
						// If this still fails, an exception will be thrown
						release_free_blocks();
						result = thrust::cuda::malloc<value_type>(num_bytes).get();
					}
					// Insert the allocated pointer into the allocated_blocks map
					allocated_blocks.insert(std::make_pair(result, num_bytes));
				}
				// Found
				else {
					result = free_block->second;
					auto allocated_size = free_block->first;
					free_blocks.erase(free_block);

					// Insert the allocated pointer into the allocated_blocks map
					allocated_blocks.insert(std::make_pair(result, allocated_size));
				}

				return result;
			}

			void deallocate(value_type* ptr, size_t n) {
				// Erase the allocated block from the allocated blocks map
				allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
				std::ptrdiff_t num_bytes = iter->second;
				allocated_blocks.erase(iter);

				/*************************** TODO ***************************/
				// Insert the block into the free blocks map
				// I think this implementation is wrong, since this line makes deallocate() impossible to be noexcept.
				// A better approach is to preallocate a free block node when allocate() is called,
				// and then replace its contents with ptr. But I'm too lazy to fix this...
				free_blocks.insert(std::make_pair(num_bytes, ptr));
			}

		private:
			typedef std::multimap<std::ptrdiff_t, value_type*>	free_blocks_type;
			typedef std::map<value_type*, std::ptrdiff_t>		allocated_blocks_type;

			free_blocks_type      free_blocks;
			allocated_blocks_type allocated_blocks;

			// thrust::cuda::free will throw if cudaFree() fails.
			// According to NVIDIA's documentation, possible return values of cudaFree() is one of
			// cudaSuccess, cudaErrorInvalidDevicePointer, and cudaErrorInitializationError.
			// cudaErrorInvalidDevicePointer is returned when the pointer passed to cudaFree() was
			// not valid. cudaErrorInitializationError is returned when the CUDA runtime has failed
			// to be initialized. Both of these error cases are out of our concern, fortunately.
			// However, cudaFree() may return an error code caused by some previous asynchronous call.
			// I'm not sure how to gracefully deal with such an error, but I think ignoring them
			// here causes no problem. Perhaps, those errors will continue to be returned from
			// other CUDA runtime calls as long as the root cause of the error is not resolved,
			// so ignoring those will be not harmful. But I'm not sure, since I couldn't find any
			// documentation explaning this behaviour...

			void release_free_blocks() noexcept {
				for( auto i = free_blocks.begin(); i != free_blocks.end(); ++i ) {
					try {
						// transform the pointer to cuda::pointer before calling cuda::free
						thrust::cuda::free(thrust::cuda::pointer<value_type>(i->second));
					}
					catch( ... ) {
						// ignore
					}
				}				
			}
			void release_allocated_blocks() noexcept {
				for( auto i = allocated_blocks.begin(); i != allocated_blocks.end(); ++i ) {
					try {
						// transform the pointer to cuda::pointer before calling cuda::free
						thrust::cuda::free(thrust::cuda::pointer<value_type>(i->first));
					}
					catch( ... ) {
						// ignore
					}
				}
			}
			void release_all() noexcept {
				release_free_blocks();
				release_allocated_blocks();
			}
		};

		// Allocator with trivial initialization
		template<typename T>
		class uninitialized_allocator : public thrust::device_malloc_allocator<T> {
		public:
			__host__ __device__ void construct(T* p) {}

			// forward everything else with at least one argument to the default
			template<typename Arg1, typename... Args>
			__host__ __device__ void construct(T* p, Arg1 &&arg1, Args&&... args) {
				thrust::detail::allocator_traits<thrust::device_malloc_allocator<T>>::construct(
					*this, p, std::forward<Arg1>(arg1), std::forward<Args>(args)...);
			}

			// thrust::device_malloc_allocator<T> is broken since deallocate() is not noexcept.
			// This causes the program to silently shutdown when detructing a thrust::device_vector<T>
			// after an asynchronous call has caused an error.
			using pointer = typename thrust::device_malloc_allocator<T>::pointer;
			__host__ void deallocate(pointer p, size_type cnt) noexcept {
				// Just ignore exceptions. Is there any better solution?
				try {
					thrust::device_malloc_allocator<T>::deallocate(p, cnt);
				}
				catch( ... ) {}
			}
		};

		// Tuple decay iterator
		namespace detail {
			template <class TupleIterator, std::size_t I>
			struct tuple_iterator_get_types {
			private:
				using tuple_reference = typename thrust::iterator_traits<TupleIterator>::reference;
				using const_correct_value = std::remove_pointer_t<
					decltype(thrust::raw_pointer_cast(&std::declval<tuple_reference>()))>;

			public:
				using reference = typename thrust::tuple_element<I, const_correct_value>::type;
				using value_type = typename thrust::tuple_element<I,
					typename thrust::iterator_traits<TupleIterator>::value_type>::type;
			};
		}
		template <class TupleIterator, std::size_t I>
		struct tuple_access_iterator : thrust::iterator_adaptor<
			tuple_access_iterator<TupleIterator, I>,
			TupleIterator,
			typename detail::tuple_iterator_get_types<TupleIterator, I>::value_type,
			thrust::use_default, thrust::use_default,
			typename detail::tuple_iterator_get_types<TupleIterator, I>::reference>
		{
			using super_t = thrust::iterator_adaptor<
				tuple_access_iterator,
				TupleIterator,
				typename detail::tuple_iterator_get_types<TupleIterator, I>::value_type,
				thrust::use_default, thrust::use_default,
				typename detail::tuple_iterator_get_types<TupleIterator, I>::reference>;

			using super_t::super_t;

			friend class thrust::iterator_core_access;

		private:
			__host__ __device__ typename super_t::reference dereference() const
			{
				return thrust::get<I>(*thrust::raw_pointer_cast(&*base()));
			}
		};
	}
}

// This is perhaps a terrible hacking, but it seems that there is no other way
// to make thrust::device_vector::resize and thrust::device_vector::clear not to launch a useless kernel

namespace thrust {
	namespace detail {
		template<typename T, typename Pointer, typename Size>
		__host__ __device__ void destroy_range(
			jkl::cuda::uninitialized_allocator<T> &a, Pointer p, Size n) {}

		template<typename T, typename Pointer, typename Size>
		__host__ __device__ void default_construct_range(
			jkl::cuda::uninitialized_allocator<T> &a, Pointer p, Size n) {}
	}
}