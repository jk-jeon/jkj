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

// This header contains three class templates that implement generic visitor pattern.
//
// The first class template,
// 
//   template <class ReturnType, class... AcceptorImpls>
//   struct visitor; // or, const_visitor
// 
// is the base class of visiting classes. ReturnType is the return type of the accept() method
// of visited classes as well as the return type of the visit() method of visiting classes.
// AcceptorImpls is the list of visited classes. For the "const_" version, visit() method
// takes reference-to-const; otherwise the method takes reference-to-non-const.
// On the other hand, the visit() method itself is always defined as const.
// For example, if the class hierarchy of visited classes is given as follows:
//
//                 Shape
//                   ¡ã
//      ¦£------------¦«-----------¦¤
//   Triangle    Rectangle    Ellipse
//
// the corresponding visitor definition should look like
//
//  class Triangle;
//  class Rectangle;
//  class Ellipse;
//  using ShapeVisitor = jkl::dp::const_visitor<void, Triangle, Rectangle, Ellipse>;
//
// The acceptor classes (i.e., Triangle, Rectangle, Ellipse in the above example)
// must be forward declared before the declaration of the visitor class
// (i.e., ShapeVisitor in the above example). However, the base acceptor class
// (i.e., Shape in the above example) need not be declared before them.
//
// The second class template,
//
//   template <class Visitor>
//   struct acceptor_base; // or, const_acceptor_base
//
// is the base class that declares the pure virtual method accept(). This class should be a
// base class of the base acceptor class (i.e., Shape in the above example).
// The "const_" version declares the method accept() as const, and should be used together
// with the const_visitor. The template parameter Visitor is the visitor class defined above
// (i.e., ShapeVisitor in the above example). 
// Hence, the definition of Shape class should be look like:
//
//   struct Shape : jkl::dp::const_acceptor_base<ShapeVisitor> {
//     // Some methods that are not addressed using visitor
//   };
//
// In fact, the base acceptor Shape is not strictly necessary;
// the class Shape can be omitted completely if not necessary. In this case,
// the class jkl::dp::const_acceptor_base<ShapeVisitor> will replace the role of Shape.
// One may let for example
//
//   using Shape = jkl::dp::const_acceptor_base<ShapeVisitor>;
//
// for convenience in this case.
//
// The third class template,
//
//   template <class Impl, class Visitor, class BaseClass = acceptor_base<Visitor>>
//   class acceptor; // or, const_acceptor
//
// is the CRTP base of acceptor classes (i.e., Triangle, Rectangle, Ellipse in the above example).
// Just as acceptor_base, the "const_" version should be used when const_visitor is used.
// The first template parameter Impl, is as usual in CRTP, the implementation class deriving
// from acceptor<Impl, Visitor, BaseClass>. The second template parameter Visitor is the
// visitor class defined before (i.e., ShapeVisitor in the above example).
// The third template parameter BaseClass is the base acceptor (i.e., Shape in the above example).
// As explained before, this parameter can be omitted and in that case
// acceptor_base (or const_acceptor_base) replaces that role.
// The definitions of the acceptor classes Triangle, Rectangle, Ellipse should look like:
//
//   class Triangle : public jkl::dp::const_acceptor<Triangle, ShapeVisitor, Shape> {
//     ...
//   };
//
//   class Rectangle : public jkl::dp::const_acceptor<Rectangle, ShapeVisitor, Shape> {
//     ...
//   };
//
//   class Ellipse : public jkl::dp::const_acceptor<Ellipse, ShapeVisitor, Shape> {
//     ...
//   };
//
// Then these classes automatically become subtypes of Shape, which should be
// a subtype of jkl::dp::const_acceptor_base<ShapeVisitor>,
// where the pure virtual method accept() is defined.
//
// Some examples of visitor implementations are given:
//
//   // Calculate the area of a given shape
//   class AreaVisitor : public ShapeVisitor {
//   public:
//     void visit(Triangle const& t) const {
//       ...
//     }
//     void visit(Rectangle const& r) const {
//       ...
//     }
//     void visit(Ellipse const& e) const {
//       ...
//     }
//   };
//
//   // Calculate the perimeter of a given shape
//   class PerimeterVisitor : public ShapeVisitor {
//   public:
//     void visit(Triangle const& t) const {
//       ...
//     }
//     void visit(Rectangle const& r) const {
//       ...
//     }
//     void visit(Ellipse const& e) const {
//       ...
//     }
//   };
//
// Note that the overloaded methods visit() of ShapeVisitor are pure virtual,
// so any visitor implementation deriving from ShapeVisitor must implement all three
// visit(Triangle&), visit(Rectangle&), visit(Ellipse&). Even if several of these
// do share the same implementation, all functions should be independently defined.
// This is because the acceptor calling the virtual method accept(ShapeVisitor&) does not
// know the actual runtime data type of Visitor. The accept(ShapeVisitor&) method calls
// the visit(...) method of the passed visitor, and because of this lack of type-knowledge,
// this method call should be virtual. Accordingly, all derivatives of ShapeVisitor must
// share the same list of virtual functions that need to be implemented.
// This possibly requires lots of, lots of boilerplates when many overloads of visit()
// indeed share the same implementation. In this reason, it is generally preferable
// to use std::variant & std::visit, rather than the classical OOP visitor pattern.
//
// The file structure may look like the following:
//
//   ShapeVisitor.h:
//     includes jkl/dp/visitor.h, forward declares Triangle, Rectangle, and Ellipse,
//     and defines ShapeVisitor alias. Optionally defines the interface Shape.
//   Triangle.h / Triangle.cpp:
//     includes ShapeVisitor.h, defines the class Triangle.
//   Rectangle.h / Rectangle.cpp:
//     includes ShapeVisitor.h, defines the class Rectangle.
//   Ellipse.h / Ellipse.cpp:
//     includes ShapeVisitor.h, defines the class Ellipse.
//   AreaVisitor.h / AreaVisitor.cpp:
//     includes ShapeVisitor.h; .cpp file may have to include all of
//     Triangle.h, Rectangle.h, and Ellipse.h. Defines the visitor calculating
//     the area of a given shape.
//   PerimeterVisitor.h / PerimeterVisitor.cpp:
//     includes ShapeVisitor.h; .cpp file may have to include all of
//     Triangle.h, Rectangle.h, and Ellipse.h. Defines the visitor calculating
//     the perimeter of a given shape.
//
// The dependency of visitor implementations on all the acceptor implementations can be
// in fact teared apart. For example, AreaVisitor.cpp can be separated into three:
// AreaVisitor_Triangle.cpp, AreaVisitor_Rectangle.cpp, and AreaVisitor_Ellipse.cpp,
// each including Triangle.h, Rectangle.h, and Ellipse.h, respectively.
// Note, however, that the list of visited acceptor implementation classes,
// should nonetheless be known a priori.

#pragma once
#include <tuple>
#include <type_traits>

namespace jkl {
	// Design Pattern namespace
	namespace dp {
		namespace detail {
			template <class ReturnType, class AcceptorImpl>
			struct visitor_fragment {
				virtual ReturnType visit(AcceptorImpl&) const = 0;
			};

			template <class... Fragments>
			struct visitor_impl : Fragments... {
				using Fragments::visit...;
			};
		}

		template <class Visitor>
		struct acceptor_base;

		template <class ReturnType, class... AcceptorImpls>
		struct visitor : detail::visitor_impl<detail::visitor_fragment<ReturnType, AcceptorImpls>...> {
			using return_type = ReturnType;
			using acceptor_base_type = acceptor_base<visitor>;
			using acceptor_list = std::tuple<AcceptorImpls...>;
		};

		template <class ReturnType, class... AcceptorImpls>
		using const_visitor = visitor<ReturnType, std::add_const_t<AcceptorImpls>...>;

		// Make sure that the common base class for acceptor implementations is a subtype of
		// one of these two classes:
		template <class Visitor>
		struct acceptor_base {
			using return_type = typename Visitor::return_type;

			virtual return_type accept(Visitor&) = 0;
			virtual ~acceptor_base() {}
		};
		template <class Visitor>
		struct const_acceptor_base {
			using return_type = typename Visitor::return_type;

			virtual return_type accept(Visitor&) const = 0;
			virtual ~const_acceptor_base() {}
		};

		// CRTP bases for acceptor implementations
		template <class Impl, class Visitor, class BaseClass = acceptor_base<Visitor>>
		class acceptor;
		template <class Impl, class Visitor, class BaseClass = const_acceptor_base<Visitor>>
		class const_acceptor;

		namespace detail {
			template <class Impl, class Visitor, bool is_const, class BaseClass>
			class acceptor_impl : public BaseClass {
			protected:
				// acceptor is a CRTP class
				acceptor_impl() = default;
				~acceptor_impl() = default;

			public:
				using return_type = typename Visitor::return_type;
				using acceptor_list = typename Visitor::acceptor_list;

			private:
				using acceptor = std::conditional_t<is_const,
					const_acceptor<Impl, Visitor, BaseClass>,
					acceptor<Impl, Visitor, BaseClass>
				>;

				friend acceptor;

				// Inspect convertibility of Impl into some type in the acceptor_list;
				// generate an error message if fails.
				template <class AcceptorList>
				struct inspect_convertibility;

				template <class... AcceptorImpls>
				struct inspect_convertibility<std::tuple<AcceptorImpls...>> {
					template <class... Bools>
					static constexpr bool calculate(Bools... v) noexcept {
						return (... | v);
					}

					static constexpr bool value = calculate(std::is_convertible_v<
						std::conditional_t<is_const, std::add_const_t<Impl>, Impl>&, AcceptorImpls&>...);
				};

				// Check if BaseClass actually derives from acceptor_base or const_acceptor_base;
				// generate an error message if fails.
				static_assert(is_const || std::is_base_of_v<acceptor_base<Visitor>, BaseClass>,
					"The specified base class does not derive from the proper acceptor_base<Visitor> class");
				static_assert(!is_const || std::is_base_of_v<const_acceptor_base<Visitor>, BaseClass>,
					"The specified base class does not derive from the proper const_acceptor_base<Visitor> class");

				template <class Acceptor>
				static return_type accept_impl(Acceptor& a, Visitor& v) {
					static_assert(inspect_convertibility<acceptor_list>::value,
						"The acceptor implementation cannot be converted to any of listed acceptor types");
					return v.visit(a);
				}
			};
		}

		template <class Impl, class Visitor, class BaseClass>
		class acceptor : public detail::acceptor_impl<Impl, Visitor, false, BaseClass> {
		public:
			using return_type = typename Visitor::return_type;

			return_type accept(Visitor& v) override {
				detail::acceptor_impl<Impl, Visitor, false, BaseClass>::accept_impl(
					static_cast<Impl&>(*this), v);
			}
		};

		template <class Impl, class Visitor, class BaseClass>
		class const_acceptor : public detail::acceptor_impl<Impl, Visitor, true, BaseClass> {
		public:
			using return_type = typename Visitor::return_type;

			return_type accept(Visitor& v) const override {
				detail::acceptor_impl<Impl, Visitor, true, BaseClass>::accept_impl(
					static_cast<Impl const&>(*this), v);
			}
		};
	}
}