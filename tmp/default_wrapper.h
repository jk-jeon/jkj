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

namespace jkj {
	namespace tmp {
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// The "default" wrapper
		// When (partially or explicitly) specializing class templates, it is often desirable to 
		// use a huge part of the default definition of the templates, while just adding some more 
		// special features for the specialization. One way to do this is to make a separate base class 
		// implementing all the "common" features, and then do not specialize the base class.
		// A problem occurs when there are only "reusable parts" but no "common parts"; that is,  
		// the parts of the default definition that can be reused are different for each specialization.
		//
		// Here we provide an alternative method. Just define a template class as usual. If something 
		// inside the class should be done or annotated with the template parameter T, then 
		// do not use T directly and use the alias default_wrapper_t<T> defined below instead.
		// When you are specializing the template, derive from your_template_name<default_wrapper<T>>.
		// You only need to add or modify the things that are specially necessary for T, and everything else 
		// will be automatically included in the specialization. Here is a use-case:
		//
		// template <typename T> struct Sample {
		//   default_wrapper_t<T> f() { ... }
		//   default_wrapper_t<T> g() { ... }
		// };
		// // Modify g() when T is a pointer
		// template <typename T> struct Sample<T*> : Sample<default_wrapper<T*>> {
		//   T* g() { ... }   // The return type of f() is also T*
		// };
		// // Modify f() when T is a reference
		// template <typename T> struct Sample<T&> : Sample<default_wrapper<T&>> {
		//   T& f() { ... } // The return type of g() is also T&
		// };
		//
		// But be careful, if you override some method in the specialization, it will not be "really overriden", 
		// because other methods in the default definition never call the new version provided for the 
		// specialization. Also, default_wrapper only works with non-template type template parameters. 
		// To work with non-type template parameters or template template parameters, 
		// you need to write a similar template by yourself.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		struct default_wrapper {
			using type = T;
		};

		template <typename T>
		struct default_wrapper<default_wrapper<T>> {
			using type = typename default_wrapper<T>::type;
		};

		template <typename T>
		using default_wrapper_t = typename default_wrapper<T>::type;
	}
}