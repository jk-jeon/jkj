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

/// This header is a collection of reimplementations of utility functions 
/// provided in "helper_cuda.h", which is contained in some CUDA sample files,
/// plus some C++ wrappers for C-style CUDA functions

#pragma once
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>

namespace jkj {
	namespace cuda {
	#ifndef JKL_CUDA_ENUM_TO_STRING_CASE
	#define JKL_CUDA_ENUM_TO_STRING_CASE(e)		case e: return #e
		namespace {
			// CUDA Runtime API errors
		#ifdef __DRIVER_TYPES_H__
			inline char const* error_string_from_enum(cudaError_t error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaSuccess);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMissingConfiguration);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMemoryAllocation);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInitializationError);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchFailure);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorPriorLaunchFailure);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchTimeout);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchOutOfResources);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidDeviceFunction);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidConfiguration);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidDevice);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidValue);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidPitchValue);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidSymbol);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMapBufferObjectFailed);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorUnmapBufferObjectFailed);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidHostPointer);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidDevicePointer);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidTexture);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidTextureBinding);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidChannelDescriptor);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidMemcpyDirection);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorAddressOfConstant);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorTextureFetchFailed);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorTextureNotBound);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorSynchronizationError);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidFilterSetting);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidNormSetting);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMixedDeviceExecution);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorCudartUnloading);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorUnknown);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNotYetImplemented);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMemoryValueTooLarge);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidResourceHandle);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNotReady);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInsufficientDriver);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorSetOnActiveProcess);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidSurface);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNoDevice);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorECCUncorrectable);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorSharedObjectSymbolNotFound);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorSharedObjectInitFailed);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorUnsupportedLimit);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorDuplicateVariableName);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorDuplicateTextureName);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorDuplicateSurfaceName);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorDevicesUnavailable);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidKernelImage);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNoKernelImageForDevice);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorIncompatibleDriverContext);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorPeerAccessAlreadyEnabled);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorPeerAccessNotEnabled);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorDeviceAlreadyInUse);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorProfilerDisabled);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorProfilerNotInitialized);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorProfilerAlreadyStarted);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorProfilerAlreadyStopped);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorAssert);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorTooManyPeers);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorHostMemoryAlreadyRegistered);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorHostMemoryNotRegistered);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorOperatingSystem);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorPeerAccessUnsupported);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchMaxDepthExceeded);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchFileScopedTex);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchFileScopedSurf);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorSyncDepthExceeded);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorLaunchPendingCountExceeded);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNotPermitted);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNotSupported);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorHardwareStackError);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorIllegalInstruction);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorMisalignedAddress);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidAddressSpace);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidPc);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorIllegalAddress);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidPtx);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorInvalidGraphicsContext);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorNvlinkUncorrectable);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorStartupFailure);
					JKL_CUDA_ENUM_TO_STRING_CASE(cudaErrorApiFailureBase);
				}
				return "<unknown>";
			}
		#endif

			// CUDA Device API errors
		#ifdef __cuda_cuda_h__
			inline char const* error_string_from_enum(CUresult error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_VALUE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_OUT_OF_MEMORY);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_INITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_DEINITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PROFILER_DISABLED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NO_DEVICE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_DEVICE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_IMAGE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_CONTEXT);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_MAP_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_UNMAP_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_ARRAY_IS_MAPPED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_ALREADY_MAPPED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NO_BINARY_FOR_GPU);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_ALREADY_ACQUIRED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_MAPPED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_ECC_UNCORRECTABLE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_UNSUPPORTED_LIMIT);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_SOURCE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_FILE_NOT_FOUND);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_OPERATING_SYSTEM);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_INVALID_HANDLE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_FOUND);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_NOT_READY);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_LAUNCH_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_LAUNCH_TIMEOUT);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_ASSERT);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_TOO_MANY_PEERS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUDA_ERROR_UNKNOWN);
				}
				return "<unknown>";
			}
		#endif

			// cuBLAS API errors
		#ifdef CUBLAS_API_H_
			char const* error_string_from_enum(cublasStatus_t error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_NOT_INITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ALLOC_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INVALID_VALUE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_ARCH_MISMATCH);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_MAPPING_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_EXECUTION_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUBLAS_STATUS_INTERNAL_ERROR);
				}
				return "<unknown>";
			}
		#endif

			// cuFFT API errors
		#ifdef _CUFFT_H_
			char const* error_string_from_enum(cufftResult error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_INVALID_PLAN);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_ALLOC_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_INVALID_TYPE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_INVALID_VALUE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_INTERNAL_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_EXEC_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_SETUP_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_INVALID_SIZE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUFFT_UNALIGNED_DATA);
				}
				return "<unknown>";
			}
		#endif

			// cuSPARSE API errors
		#ifdef CUSPARSEAPI
			char const* error_string_from_enum(cusparseStatus_t error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_NOT_INITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_ALLOC_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_INVALID_VALUE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_ARCH_MISMATCH);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_MAPPING_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_EXECUTION_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_INTERNAL_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
				}
				return "<unknown>";
			}
		#endif

			// cuRAND API errors
		#ifdef CURAND_H_
			char const* error_string_from_enum(curandStatus_t error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_VERSION_MISMATCH);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_NOT_INITIALIZED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_ALLOCATION_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_TYPE_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_OUT_OF_RANGE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_LAUNCH_FAILURE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_PREEXISTING_FAILURE);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_INITIALIZATION_FAILED);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_ARCH_MISMATCH);
					JKL_CUDA_ENUM_TO_STRING_CASE(CURAND_STATUS_INTERNAL_ERROR);
				}
				return "<unknown>";
			}
		#endif

			// NPP API errors
		#ifdef NV_NPPIDEFS_H
			char const* error_string_from_enum(NppStatus error) {
				switch( error ) {
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_NOT_SUPPORTED_MODE_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_RESIZE_NO_OPERATION_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_BAD_ARG_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_LUT_NUMBER_OF_LEVELS_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_TEXTURE_BIND_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_COEFF_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_RECT_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_QUAD_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_WRONG_INTERSECTION_ROI_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_NOT_EVEN_STEP_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_INTERPOLATION_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_RESIZE_FACTOR_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MEMFREE_ERR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MEMSET_ERR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MEMCPY_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MEM_ALLOC_ERR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_HISTO_NUMBER_OF_LEVELS_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MIRROR_FLIP_ERR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_INVALID_INPUT);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_ALIGNMENT_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_STEP_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_SIZE_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_POINTER_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_NULL_POINTER_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_CUDA_KERNEL_EXECUTION_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_NOT_IMPLEMENTED_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_ERROR);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_SUCCESS);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_WRONG_INTERSECTION_QUAD_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_MISALIGNED_DST_ROI_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_AFFINE_QUAD_INCORRECT_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_DOUBLE_SIZE_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_ODD_ROI_WARNING);
					JKL_CUDA_ENUM_TO_STRING_CASE(NPP_WRONG_INTERSECTION_ROI_WARNING);
				}
				return "<unknown>";
			}
		#endif
		}
	#undef JKL_CUDA_ENUM_TO_STRING_CASE
	#endif

		// Throw an exception on error
		
		template <class ErrorCode>
		class cuda_error : public std::runtime_error {
		public:
			cuda_error(ErrorCode code, std::string const& string_description)
				: std::runtime_error{ string_description }, code{ code } {}

			ErrorCode code;
		};
		template <class ErrorCode>
		auto make_cuda_error(ErrorCode code) {
			return cuda_error<ErrorCode>{ code, error_string_from_enum(code) };
		}

		template <class T>
		inline void check_error(T result) {
			if( result )
				throw make_cuda_error(result);
		}

		namespace detail {
			inline unsigned int sm_ver_to_cores(int major, int minor) {
				std::pair<int, unsigned int> sm_core_pairs[] = {
					{ 0x10, 8 },	// Tesla Generation (SM 1.0) G80 class
					{ 0x11, 8 },	// Tesla Generation (SM 1.1) G8x class
					{ 0x12, 8 },	// Tesla Generation (SM 1.2) G9x class
					{ 0x13, 8 },	// Tesla Generation (SM 1.3) GT200 class
					{ 0x20, 32 },	// Fermi Generation (SM 2.0) GF100 class
					{ 0x21, 48 },	// Fermi Generation (SM 2.1) GF10x class
					{ 0x30, 192 },	// Kepler Generation (SM 3.0) GK10x class
					{ 0x35, 192 }	// Kepler Generation (SM 3.5) GK11x class
				};

				auto sm_ver = (major << 4) + minor;
				for( std::size_t i = 0; i < std::extent<decltype(sm_core_pairs)>::value; ++i ) {
					if( sm_core_pairs[i].first == sm_ver )
						return sm_core_pairs[i].second;
				}
				return sm_core_pairs[7].second;
			}
		}

	#ifdef __CUDA_RUNTIME_H__
		// General GPU Device CUDA Initialization
		inline void gpu_initialize(int dev_id) {
			int device_count;
			check_error(cudaGetDeviceCount(&device_count));

			if( device_count == 0 )
				throw std::runtime_error{ "CUDA error: there is no GPU device supporting CUDA" };

			if( dev_id < 0 ) {
				dev_id = 0;
			}

			if( dev_id >= device_count ) {
				std::stringstream ss;
				ss << "CUDA error: " << device_count << " GPU devices are detected; "
					<< dev_id << " is not a valid device ID";
				throw std::runtime_error{ ss.str() };
			}

			cudaDeviceProp dev_prop;
			check_error(cudaGetDeviceProperties(&dev_prop, dev_id));

			if( dev_prop.computeMode == cudaComputeModeProhibited ) {
				std::stringstream ss;
				ss << "CUDA error: the specified GPU device (" << dev_id << ") is running in <Compute Mode Prohibited>";
				throw std::runtime_error{ ss.str() };
			}

			if( dev_prop.major < 1 ) {
				std::stringstream ss;
				ss << "CUDA error: the specified GPU device (" << dev_id << ") does not support CUDA";
				throw std::runtime_error{ ss.str() };
			}

			check_error(cudaSetDevice(dev_id));
		}
	#endif

		// This function returns the best GPU (with maximum GFLOPS)
		inline int get_max_gflops_device()
		{
			int device_count;
			check_error(cudaGetDeviceCount(&device_count));

			// Find the best major SM Architecture GPU device
			int best_sm_arch = 0;
			for( int current_device = 0; current_device < device_count; ++current_device ) {
				cudaDeviceProp dev_prop;
				check_error(cudaGetDeviceProperties(&dev_prop, current_device));
				if( dev_prop.computeMode != cudaComputeModeProhibited ) {
					if( dev_prop.major > 0 && dev_prop.major < 9999 ) {
						if( best_sm_arch < dev_prop.major )
							best_sm_arch = dev_prop.major;
					}
				}
			}

			// Find the best CUDA capable GPU device			
			int max_compute_perf = 0, max_perf_device = 0;
			for( int current_device = 0; current_device < device_count; ++current_device ) {
				cudaDeviceProp dev_prop;
				check_error(cudaGetDeviceProperties(&dev_prop, current_device));
				if( dev_prop.computeMode != cudaComputeModeProhibited ) {
					int sm_per_multiproc = 1;
					if( !(dev_prop.major == 9999 && dev_prop.minor == 9999) ) {
						sm_per_multiproc = detail::sm_ver_to_cores(dev_prop.major, dev_prop.minor);
					}

					int compute_perf  = dev_prop.multiProcessorCount * sm_per_multiproc * dev_prop.clockRate;

					if( compute_perf  > max_compute_perf ) {
						// If we find GPU with SM major > 2, search only these
						if( best_sm_arch > 2 ) {
							// If our device == best_sm_arch, choose this, or else pass
							if( dev_prop.major == best_sm_arch ) {
								max_compute_perf = compute_perf;
								max_perf_device = current_device;
							}
						}
						else {
							max_compute_perf = compute_perf;
							max_perf_device = current_device;
						}
					}
				}
			}

			return max_perf_device;
		}
		
		// General check for CUDA GPU SM Capabilities
		inline bool check_cuda_capabilities(int major_version, int minor_version)
		{			
			int dev;
			check_error(cudaGetDevice(&dev));

			cudaDeviceProp dev_prop;
			check_error(cudaGetDeviceProperties(&dev_prop, dev));

			if( (dev_prop.major > major_version) ||
				(dev_prop.major == major_version && dev_prop.minor >= minor_version) )
			{
				return true;
			}
			
			return false;
		}

		// Dereference a device pointer; T must be default constructible && trivially copyable
		// This function synchronizes the given stream
		template <class T>
		T dereference_device_ptr(T const* ptr, cudaStream_t s = nullptr)
		{
			T ret_value;
			check_error(cudaMemcpyAsync(&ret_value, ptr, sizeof(T), cudaMemcpyDeviceToHost, s));
			check_error(cudaStreamSynchronize(s));
			return ret_value;
		}

		// A "greedy" approach to find the (pseudo-)optimal grid size
		template <class Function>
		std::size_t get_optimal_grid_size(Function&& f, std::size_t total_size,
			std::size_t block_size, std::size_t dynamic_shared_mem = 0)
		{
			int max_active_blocks;
			check_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
				std::forward<Function>(f), int(block_size), dynamic_shared_mem));

			int dev;
			cudaGetDevice(&dev);

			// It seems that cudaGetDeviceProperties() is somewhat heavy.
			// To reduce the cost of calling it, we cache the result.
			struct multi_processor_count_cache_t {
				int dev;
				int value;

				multi_processor_count_cache_t(int dev) {
					recalculate(dev);
				}

				void recalculate(int new_dev) {
					cudaDeviceProp prop;
					check_error(cudaGetDeviceProperties(&prop, new_dev));
					value = prop.multiProcessorCount;
					dev = new_dev;
				}
			};
			thread_local multi_processor_count_cache_t multi_processor_count_cache{ dev };
			if( dev != multi_processor_count_cache.dev ) {
				multi_processor_count_cache.recalculate(dev);
			}

			auto grid_size = std::size_t(max_active_blocks * multi_processor_count_cache.value);
			auto max_grid_size = (total_size + block_size - 1) / block_size;
			if( grid_size > max_grid_size )
				grid_size = max_grid_size;

			return grid_size;
		}

		/// CUDA pinned memory allocator
		/// This functionality is already implemented in Thrust,
		/// but Thurst pinned_allocator use old cudaMallocHost() rather than new cudaHostAlloc().
		/// Note that in order to use the flag cudaHostAllocMapped, cudaSetDeviceFlags(cudaDeviceMapHost) must be called.

		template<typename T, int flags_ = cudaHostAllocDefault>
		class pinned_allocator
		{
		public:
			static constexpr int flags = flags_;

			using value_type = T;
			using pointer = T*;
			using const_pointer = T const*;
			using reference = T&;
			using const_reference = T const&;
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;

			template<typename U>
			struct rebind {
				using other = pinned_allocator<U>;
			};

			pinned_allocator() = default;
			pinned_allocator(pinned_allocator const&) = default;
			template<typename U>
			__host__ __device__ pinned_allocator(pinned_allocator<U> const &) noexcept {}

			__host__ __device__ pointer address(reference r) const noexcept { return &r; }
			__host__ __device__ const_pointer address(const_reference r) const noexcept { return &r; }

			__host__ pointer allocate(size_type n, const_pointer = nullptr) {
				if( n > max_size() ) {
					throw std::bad_alloc{};
				}

				pointer result = nullptr;
				if( cudaHostAlloc(reinterpret_cast<void**>(&result), n * sizeof(value_type), flags) ) {
					throw std::bad_alloc{};
				}

				return result;
			}

			__host__ void deallocate(pointer p, size_type n) noexcept {
				cudaFreeHost(p);
			}

			size_type max_size() const noexcept {
				return std::numeric_limits<size_type>::max() / sizeof(T);
			}

			__host__ __device__ bool operator==(pinned_allocator const& x) const noexcept { return true; }
			__host__ __device__ bool operator!=(pinned_allocator const& x) const noexcept { return !operator==(x); }
		};

		/// RAII wrappers
		
		// CUDA stream wrapper
		// Assumes unique ownership, unless the default stream is assigned
		class stream {
			cudaStream_t	m_stream;

		public:
			// Creates a stream; the default is non-blocking
			stream(unsigned int flags = cudaStreamNonBlocking) {
				check_error(cudaStreamCreateWithFlags(&m_stream, flags));
			}
			// Accept a previously constructed stream
			explicit stream(cudaStream_t s) noexcept
				: m_stream{ s } {}
			
			// Destroy the stream on destruction
			~stream() {
				if( m_stream != nullptr )
					cudaStreamDestroy(m_stream);
			}

			// Move-only
			stream(stream const&) = delete;
			stream& operator=(stream const&) = delete;
			stream(stream&& that) : m_stream{ that.release() } {}
			stream& operator=(stream&& that) {
				m_stream = that.release();
				return *this;
			}

			explicit operator cudaStream_t() const noexcept {
				return m_stream;
			}

			// std::unique_ptr-like functionalities
			cudaStream_t get() const noexcept {
				return m_stream;
			}
			cudaStream_t release() noexcept {
				auto ret_value = m_stream;
				m_stream = nullptr;
				return ret_value;
			}
			void reset(cudaStream_t s = nullptr) noexcept {
				if( m_stream != nullptr )
					cudaStreamDestroy(m_stream);
				m_stream = s;
			}

			// Construct a new stream and replace the previous
			// Destroy the previously assigned stream
			// Exception safety: if an exception is thrown during the creation of a new stream,
			//                   previous state is untouched.
			void emplace(unsigned int flags = cudaStreamNonBlocking) {
				cudaStream_t new_stream;
				check_error(cudaStreamCreateWithFlags(&new_stream, flags));
				if( m_stream != nullptr )
					cudaStreamDestroy(m_stream);
				m_stream = new_stream;
			}

			// Some basic stream functionalities
			void synchronize() const {
				check_error(cudaStreamSynchronize(m_stream));
			}
			void wait(cudaEvent_t e) const {
				check_error(cudaStreamWaitEvent(m_stream, e, 0));
			}
		};

		// CUDA event wrapper
		// Assumes unique ownership
		class event {
			cudaEvent_t	m_event;

		public:
			// Creates a event
			event(unsigned int flags = cudaEventDefault) {
				check_error(cudaEventCreateWithFlags(&m_event, flags));
			}
			// Accept a previously constructed event
			explicit event(cudaEvent_t e) noexcept
				: m_event{ e } {}

			// Destroy the event on destruction
			~event() {
				if( m_event != nullptr )
					cudaEventDestroy(m_event);
			}

			// Move-only
			event(event const&) = delete;
			event& operator=(event const&) = delete;
			event(event&& that) : m_event{ that.release() } {}
			event& operator=(event&& that) {
				m_event = that.release();
				return *this;
			}

			explicit operator cudaEvent_t() const noexcept {
				return m_event;
			}

			// std::unique_ptr-like functionalities
			cudaEvent_t get() const noexcept {
				return m_event;
			}
			cudaEvent_t release() noexcept {
				auto ret_value = m_event;
				m_event = nullptr;
				return ret_value;
			}
			void reset(cudaEvent_t e = nullptr) noexcept {
				if( m_event != nullptr )
					cudaEventDestroy(m_event);
				m_event = e;
			}

			// Construct a new event and replace the previous
			// Destroy the previously assigned event
			// Exception safety: if an exception is thrown during the creation of a new event,
			//                   previous state is untouched.
			void emplace(unsigned int flags = cudaEventDefault) {
				cudaEvent_t new_event;
				check_error(cudaEventCreateWithFlags(&new_event, flags));
				if( m_event != nullptr )
					cudaEventDestroy(m_event);
				m_event = new_event;
			}

			// Some basic event functionalities
			void record(cudaStream_t s = nullptr) const {
				check_error(cudaEventRecord(m_event, s));
			}
			void synchronize() const {
				check_error(cudaEventSynchronize(m_event));
			}
		};

		// Fork-Join stream synchronizer
		// Inspired from http://cedric-augonnet.com/declaring-dependencies-with-cudastreamwaitevent/
		class stream_fork {
			event							m_parent_event;
			using stream_event_pair = std::pair<stream, event>;
			std::vector<stream_event_pair>	m_childs;

		public:
			stream_fork(unsigned int parent_event_flag = cudaEventDefault) : m_parent_event{ parent_event_flag } {}
			stream_fork(unsigned int parent_event_flag, std::size_t size,
				unsigned int child_stream_flag = cudaStreamNonBlocking,
				unsigned int child_event_flag = cudaEventDefault) : m_parent_event{ parent_event_flag }
			{
				m_childs.reserve(size);
				for( std::size_t i = 0; i < size; ++i ) {
					m_childs.emplace_back(stream{ child_stream_flag }, event{ child_event_flag });
				}
			}

			void clear() {
				m_childs.clear();
			}

			void resize(std::size_t n,
				unsigned int child_stream_flag = cudaStreamNonBlocking,
				unsigned int child_event_flag = cudaEventDefault)
			{
				auto prev_size = m_childs.size();

				if( prev_size >= n ) {
					m_childs.erase(m_childs.begin() + n, m_childs.end());
				}
				else {
					m_childs.reserve(n);
					for( std::size_t i = 0; i < n - prev_size; ++i )
						m_childs.emplace_back(stream{ child_stream_flag }, event{ child_event_flag });
				}
			}

			// Access to chiled streams
			cudaStream_t operator[](std::size_t idx) const noexcept {
				return m_childs[idx].first.get();
			}

			class iterator {
				stream_event_pair const*	ptr = nullptr;
				std::size_t					idx = 0;

			public:
				friend stream_fork;
				constexpr iterator(stream_event_pair const* ptr, std::size_t idx) noexcept
					: ptr{ ptr }, idx{ idx } {}
				constexpr iterator() noexcept {}

				std::size_t index() const noexcept {
					return idx;
				}

				using value_type = cudaStream_t;
				using difference_type = std::ptrdiff_t;
				using reference = cudaStream_t;
				using pointer = cudaStream_t*;
				using iterator_category = std::random_access_iterator_tag;

				// Iterator concept
				reference operator*() const noexcept {
					return ptr[idx].first.get();
				}
				iterator& operator++() noexcept {
					++idx;
					return *this;
				}

				// InputIterator concept (without operator->)
				constexpr bool operator==(iterator const& that) const noexcept {
					return idx == that.idx;
				}
				constexpr bool operator!=(iterator const& that) const noexcept {
					return !(*this == that);
				}
				iterator operator++(int) noexcept {
					auto tmp = *this;
					++*this;
					return tmp;
				}

				// BidirectionalIterator concept
				iterator& operator--() noexcept {
					--idx;
					return *this;
				}
				iterator operator--(int) noexcept {
					auto tmp = *this;
					--*this;
					return tmp;
				}

				// RandomAccessIterator concept
				iterator& operator+=(difference_type n) noexcept {
					idx += n;
					return *this;
				}
				constexpr iterator operator+(difference_type n) const noexcept {
					return{ ptr, idx + n };
				}
				friend constexpr iterator operator+(difference_type n, iterator const& itr) noexcept {
					return itr + n;
				}
				iterator& operator-=(difference_type n) noexcept {
					idx -= n;
					return *this;
				}
				constexpr iterator operator-(difference_type n) const noexcept {
					return{ ptr, idx - n };
				}
				constexpr difference_type operator-(iterator const& that) const noexcept {
					return idx - that.idx;
				}
				reference operator[](std::size_t idx) const noexcept {
					return ptr[this->idx + idx].first.get();
				}
				constexpr bool operator<(iterator const& that) const noexcept {
					return idx < that.idx;
				}
				constexpr bool operator>(iterator const& that) const noexcept {
					return idx > that.idx;
				}
				constexpr bool operator<=(iterator const& that) const noexcept {
					return idx <= that.idx;
				}
				constexpr bool operator>=(iterator const& that) const noexcept {
					return idx >= that.idx;
				}
			};

			iterator begin() const noexcept {
				return{ m_childs.data(), 0 };
			}
			iterator end() const noexcept {
				return{ m_childs.data(), m_childs.size() };
			}
			iterator cbegin() const noexcept {
				return begin();
			}
			iterator cend() const noexcept {
				return end();
			}
			using const_iterator = iterator;

			// Fork streams from parent_stream at the begining, call proc, and then join the streams at the end
			// proc may accept the iterator pointing to the child stream array, or it may not:
			//  - Do std::forward<Procedure>(proc)(begin()) if it is a well-defined expression; otherwise,
			//  - Do std::forward<Procedure>(proc)(); otherwise, the program is ill-formed.
			template <class Procedure>
			void fork(cudaStream_t parent_stream, Procedure&& proc) const
			{
				fork(m_childs.size(), parent_stream, std::forward<Procedure>(proc));
			}

			// Same as the above, but number of child streams can be specified
			// The behaviour is undefined if the specified number is larger than the actual number of child streams.
			template <class Procedure>
			void fork(std::size_t n, cudaStream_t parent_stream, Procedure&& proc) const
			{
				struct fork_raii {
					std::size_t			n;
					cudaStream_t		parent_stream;
					stream_fork const&	sf;

					fork_raii(std::size_t n, cudaStream_t parent_stream, stream_fork const& sf)
						: n{ n }, parent_stream{ parent_stream }, sf{ sf }
					{
						sf.m_parent_event.record(parent_stream);
						for( std::size_t idx = 0; idx < n; ++idx ) {
							sf.m_childs[idx].first.wait(sf.m_parent_event.get());
						}
					}

					// I have no idea how to deal with errors returned inside the destructor
					~fork_raii() {
						for( std::size_t idx = 0; idx < n; ++idx ) {
							auto& c = sf.m_childs[idx];
							try {
								c.second.record(c.first.get());
							}
							catch( ... ) {
								// Just ignore...
							}
							// Just ignore the error...
							cudaStreamWaitEvent(parent_stream, c.second.get(), 0);
						}
					}
				} raii_obj{ n, parent_stream, *this };

				fork_impl<Procedure>::call(std::forward<Procedure>(proc), begin());
			}

		private:
			template <class Procedure, class = std::void_t<>>
			struct fork_impl {
				static void call(Procedure&& proc, iterator const& itr) {
					return std::forward<Procedure>(proc)();
				}
			};

			template <class Procedure>
			struct fork_impl<Procedure, std::void_t<decltype(std::declval<Procedure>()(iterator{}))>> {
				static void call(Procedure&& proc, iterator const& itr) {
					return std::forward<Procedure>(proc)(itr);
				}
			};
		};
	}
}