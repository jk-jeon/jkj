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

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include "tmp.h"

namespace jkl {
	namespace memory {
		namespace detail {
			// The minimal cell of data
			template <std::size_t size>
			struct cell {
				union {
					std::uint8_t	data[size];
					cell*			next;
				};
				cell() noexcept : next{ this + 1 } {}
			};

			// Chunk of cells
			template <std::size_t cell_size, std::size_t number_of_cells>
			struct chunk {
				cell<cell_size>		cells[number_of_cells];
				chunk() noexcept {
					cells[number_of_cells - 1].next = nullptr;
				}
			};

			// Singleton linked-list of cells that can be allocated
			template <std::size_t cell_size, std::size_t cells_per_chunk>
			class free_cell_list {
				template <class T>
				struct node {
					T			data;
					node*		next;
					node() noexcept : next{ nullptr } {}
				};

				using cell_type = cell<cell_size>;
				using chunk_type = chunk<cell_size, cells_per_chunk>;
				struct chunk_with_cell_node {
					chunk_type			data_chunk;
					node<cell_type*>	cell_ptr_node;
				};

				// Linked-list of free cell pointers
				std::atomic<node<cell_type*>*>				m_free_cell_ptr_list;
				// Linked-list of node<cell_type*> that can be reclaimed
				std::atomic<node<cell_type*>*>				m_empty_node_list;
				// Linked-list of data chunks
				std::atomic<node<chunk_with_cell_node>*>	m_chunk_list;

				free_cell_list() noexcept
					: m_free_cell_ptr_list{ nullptr }, m_empty_node_list{ nullptr }, m_chunk_list{ nullptr } {}

				// The destructor is called only once by the main thread
				~free_cell_list() {
					// Delete all chunk nodes
					auto current_chunk_node = m_chunk_list.load(std::memory_order_relaxed);
					while( current_chunk_node ) {
						auto temp = current_chunk_node->next;
						delete current_chunk_node;
						current_chunk_node = temp;
					}
				}

			public:
				static free_cell_list& instance() noexcept {
					static free_cell_list inst;
					return inst;
				}

				// Get a free cell
				// May throw std::bad_alloc
				cell_type* allocate() {
					node<cell_type*>* current_cell_ptr_node = m_free_cell_ptr_list.load(std::memory_order_relaxed);
					node<cell_type*>* next_cell_ptr_node = nullptr;
					cell_type* ret_value;

					// Update m_free_cell_ptr_list to its next and cache the previous head
					do {
						// If there is no free cell, allocate a new chunk and add it at the front of the chunk list
						if( !current_cell_ptr_node ) {
							auto new_chunk_node = new node<chunk_with_cell_node>;

							// Out of memory
							if( !new_chunk_node )
								throw std::bad_alloc{};

							new_chunk_node->next = m_chunk_list.load(std::memory_order_relaxed);
							while( !m_chunk_list.compare_exchange_weak(new_chunk_node->next, new_chunk_node,
								std::memory_order_release, std::memory_order_relaxed) );

							// We will return the first cell of the new chunk
							ret_value = &new_chunk_node->data.data_chunk.cells[0];

							// The empty node that will be used when deallocate() is called
							current_cell_ptr_node = &new_chunk_node->data.cell_ptr_node;

							goto add_empty_node_and_return;
						}

						next_cell_ptr_node = current_cell_ptr_node->next;
					} while( !m_free_cell_ptr_list.compare_exchange_weak(current_cell_ptr_node, next_cell_ptr_node,
						std::memory_order_release, std::memory_order_relaxed) );

					// Cache the pointer stored in the cached node
					ret_value = current_cell_ptr_node->data;

				add_empty_node_and_return:
					// The node will be no longer used, so push it at the front of the empty node list
					current_cell_ptr_node->next = m_empty_node_list.load(std::memory_order_relaxed);
					while( !m_empty_node_list.compare_exchange_weak(current_cell_ptr_node->next, current_cell_ptr_node,
						std::memory_order_release, std::memory_order_relaxed) );

					// Return the cached pointer
					return ret_value;
				}

				void deallocate(cell_type* c) noexcept {
					if( !c )
						return;

					// Get a cell pointer node in the empty node list
					node<cell_type*>* current_empty_node_node = m_empty_node_list.load(std::memory_order_relaxed);
					node<cell_type*>* next_empty_node_node = nullptr;

					// Update m_empty_node_list to its next and cache the previous head
					do {
						// deallocate() must form a pair with allocate(), and there must be at least 
						// one node in the empty node list before returning from allocate()
						assert(current_empty_node_node != nullptr);

						next_empty_node_node = current_empty_node_node->next;
					} while( !m_empty_node_list.compare_exchange_weak(current_empty_node_node, next_empty_node_node,
						std::memory_order_release, std::memory_order_relaxed) );

					// Push the node obtained above into the free cell pointer list
					current_empty_node_node->data = c;
					current_empty_node_node->next = m_free_cell_ptr_list.load(std::memory_order_relaxed);
					while( !m_free_cell_ptr_list.compare_exchange_weak(current_empty_node_node->next, current_empty_node_node,
						std::memory_order_release, std::memory_order_relaxed) );
				}
			};
		}

		// Fixed-size allocator
		// An instance of this class should not be shared between multiple threads
		// However, pointers obtained from allocate() can be shared between multiple instances of this class
		// That is, the following code is valid:
		//
		// pool<64> a, b;
		// auto ptr = a.allocate();
		// b.deallocate(ptr);
		//
		// Here, a and b can be located on different threads
		template <std::size_t size, std::size_t cells_per_chunk = 8192>
		class pool {
			using free_cell_list_type = detail::free_cell_list<size, cells_per_chunk>;
			using cell_type = detail::cell<size>;

			cell_type*		m_head = nullptr;
		public:
			~pool() {
				free_cell_list_type::instance().deallocate(m_head);
			}

			// May throw std::bad_alloc
			void* allocate() {
				cell_type* current_head = m_head;
				if( !current_head )
					current_head = free_cell_list_type::instance().allocate();
				m_head = current_head->next;
				return &current_head->data[0];
			}

			void deallocate(void* ptr) noexcept {
				auto new_head = reinterpret_cast<cell_type*>(ptr);
				cell_type* current_head = m_head;
				new_head->next = current_head;
				m_head = new_head;
			}
		};

		// Stateless STL style pool-based allocator
		// It can allocate/deallocate an arbitrary size of memory chunk
		// It holds an array of pools with different allocation sizes, and forwards the allocation/deallocation requests 
		// to an appropriate pool depending on the requested allocation size
		// maximum_alloc_count is the expected maximum value of the parameter passed into allocate() and deallocate() member functions

		struct default_large_alloc {
			static void* malloc(std::size_t n) {
				return ::malloc(n);
			}

			static void free(void* ptr) noexcept {
				::free(ptr);
			}
		};
		template <typename T, std::size_t maximum_alloc_count_ = 64, std::size_t cells_per_chunk = 8192, class LargeAlloc = default_large_alloc>
		class pool_allocator {
			using aligned_storage_t = std::aligned_storage_t<sizeof(T), alignof(T)>;

			// Number of pools needed
			static constexpr std::size_t number_of_allocs =
				(sizeof(T) * maximum_alloc_count_ + sizeof(aligned_storage_t) - 1) / sizeof(aligned_storage_t);
			// Maximum size of cell
			// If requested size of allocation is larger than this, pool_allocator just refers to the default large memory chunk allocator
			static constexpr std::size_t maximum_alloc_count = number_of_allocs * sizeof(aligned_storage_t) / sizeof(T);

			// Convert requested allocation size to pool index
			static std::size_t count_to_index(std::size_t count) noexcept {
				return (sizeof(T) * count + sizeof(aligned_storage_t) - 1) / sizeof(aligned_storage_t) - 1;
			}

			// The underlying type of the ith pool
			template <std::size_t index>
			using pool_type = pool<sizeof(aligned_storage_t) * (index + 1), cells_per_chunk>;

			// Call allocate() from the ith pool
			struct call_allocate {
				// May throw std::bad_alloc
				template <std::size_t index>
				void* operator()(void* pools_ptr) const {
					return reinterpret_cast<pool_type<index>*>(pools_ptr)->allocate();
				}
			};

			// Call deallocate() from the ith pool
			struct call_deallocate {
				template <std::size_t index>
				void operator()(void* pools_ptr, void* ptr) const noexcept {
					reinterpret_cast<pool_type<index>*>(pools_ptr)->deallocate(ptr);
				}
			};

			// Call the destructor of the ith pool
			struct call_destructor {
				template <std::size_t index>
				void operator()(void* pools_ptr) const noexcept {
					reinterpret_cast<pool_type<index>*>(pools_ptr)->~pool_type<index>();
				}
			};

			// Array of pools with different allocation size
			// Since each pool is of different type, a type-erasure mechanism must be applied
			class pools_type {
			public:
				void*& operator[](std::size_t index) noexcept {
					assert(index < number_of_allocs);
					return m_pools[index];
				}

				pools_type() noexcept {
					std::memset(m_pools, 0, sizeof(void*) * number_of_allocs);
				}

				// Calls the destructor of the each pool at destruction
				~pools_type() {
					for( std::size_t i = 0; i < number_of_allocs; ++i ) {
						tmp::call_by_index<number_of_allocs>(call_destructor{}, i, &m_pools[i]);
					}
				}

			private:
				void* m_pools[number_of_allocs];
			};

			// Each thread owns a separate instance of pool array
			thread_local static pools_type m_pools;

		public:
			// May throw std::bad_alloc
			T* allocate(std::size_t count) {
				// If the requested size is too big, refer to the defalut allocator
				if( count > maximum_alloc_count ) {
					auto* ptr = reinterpret_cast<T*>(LargeAlloc::malloc(sizeof(T) * count));
					if( ptr == nullptr )
						throw std::bad_alloc{};
					return ptr;
				}

				// Forward to the minimum-sized pool that can allocate a memory chunk of the given size
				auto index = count_to_index(count);
				return reinterpret_cast<T*>(tmp::call_by_index<number_of_allocs>(call_allocate{}, index, &m_pools[index]));
			}

			void deallocate(T* ptr, std::size_t count) noexcept {
				// If the requested size is too big, refer to the defalut allocator
				if( count > maximum_alloc_count )
					LargeAlloc::free(ptr);

				// Forward to the minimum-sized pool that can deallocate a memory chunk of the given size
				else {
					auto index = count_to_index(count);
					tmp::call_by_index<number_of_allocs>(call_deallocate{}, index, &m_pools[index], ptr);
				}
			}
		};

		template <typename T, std::size_t maximum_alloc_size, std::size_t cells_per_chunk, class LargeAlloc>
		thread_local typename pool_allocator<T, maximum_alloc_size, cells_per_chunk, LargeAlloc>::pools_type
			pool_allocator<T, maximum_alloc_size, cells_per_chunk, LargeAlloc>::m_pools;
	}
}