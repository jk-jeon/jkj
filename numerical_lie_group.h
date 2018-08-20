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

// By default, all types internally store an array of Element's.
// However, a user can change this if necessary, by changing Storage template parameter and
// providing an appropriate traits class. The role of the storage traits class is to provide
// the way to access elements from the given storage type.
// The default stroage traits' strategy of finding how to access elements is the following:
// 1. If the storage class is std::complex, treat the real part as the x-component
//    and the imaginary part as the y-component; otherwise,
// 2. If the storage class provides a member called x, y, z, or w,
//   - if that member is a member function, call it; or,
//   - if that member is a member object, use its value
//    (even if that member object is of callable type.).
// 3. Otherwise, if there is no member ftn/obj called x, y, z, or w,
//   - if the storage type is tuple-like, call an appropriate
//     tuple access function (jkl::tmp::get);
//   - otherwise, try to call operator[] instead; if there is no operator[] neither,
//     generates a compile error.
// Hence, followings are examples of classes that can be used as the storage with the default traits:
//  - built-in arrays,
//  - std::array, std::pair (for 2-element case), std::tuple
//  - Something like struct Point2D { double x, y; };
//  - Rn_elmt and other numerical Lie group classes themselves
// Construction of the storage is also dealt by the traits class.
// Parameters for construction are solely determined by the target class, not traits class;
// the role of the traits class is to translate those into a form that can be understood by
// the internal storage. To do that, the traits class should provide a wrapper type that will
// translate and forward the construction arguments into the storage.
// The default traits class does not do any non-trivial translation; it just forward
// all parameters to the storage class.
//
// In detail, a traits class should satisfy the following condition:
// 1. When used with a vector-like classes (e.g., Rn_elmt), one of followings must be defined,
//    where s is of the storage type, possibly cv & ref-qualified:
//    - static member functions x(s), y(s), z(s), and w(s) up to the required dimension, or
//    - a static member function get<I>(s), where I varies from 0 to the required dimension, or
//    - a static member function array_operator(s, idx), where idx is of an integral type.
//    x(s), y(s), z(s), and w(s) are preferred to get<I>(s) or array_opreatro(s, idx) when both of them exist,
//    while get<I>(s) is preferred to array_operator(s, idx), when both of them exist.
// 2. The result of the above functions is the corresponding component, correctly cv & ref-qualified.
// 3. When used with a matrix-like classes (e.g., gl2_elmt), one of the followings must be defined,
//    where s is of the storage type, possibly cv & ref-qualified:
//    - a static member function get<I>(s), where I varies from 0 to the required number of rows, or
//    - a static member function array_operator(s, idx), where idx is of an integral type.
//    get<I>(s) is preferred to array_operator(s, idx) when both of them exist.
// 4. The result r of the above function is a representation of the corresponding row;
//    then either get<J>(r) or array_operator(r, idx) should be evaluable in the same manner.
// 5. The traits class should provide a member type (either a template class or a template alias)
//    storage_type<Storage, TargetType>, which is the actual type that will be stored in TargetType
//    when Storage is given as the storage template parameter.
// 6. By defining a member template
//
//    template <std::size_t I, class Storage>
//    struct tuple_element;
//
//    classes in numerical_lie_group.h will become tuple-like. Hence, e.g. structured binding
//    will work with those classes. However, tuple_element is an optional feature.
//    Tuple-like get() access (either ADL or member function) will still work even without that.
// 7. A template typename storage_wrapper<Storage, TargetType> should be provided.
//    This is the type that is actually stored inside the TargetType. It should translate and
//    forward the constructor parameters passed by constructors of TargetType.
// 8. A function get_storage() should be provided:
//    - Its the only parameter is a possibly cv & ref-qualified storage_wrapper<Storage, TargetType>.
//    - It returns appropriately cv & ref-qualified actual storage.

#include "numerical_lie_group/general.h"
#include "numerical_lie_group/Rn_elmt.h"
#include "numerical_lie_group/gl2_elmt.h"
#include "numerical_lie_group/sym2_elmt.h"
#include "numerical_lie_group/gl3_elmt.h"
#include "numerical_lie_group/sym3_elmt.h"
#include "numerical_lie_group/rigid3d.h"
#include "numerical_lie_group/literals.h"