//===-- gen/mlir/toir.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Generates MLIR for D expressions: integer and floating-point arithmetic,
// comparisons, casts and direct calls.
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include <mlir/IR/Block.h>
#include <mlir/IR/OpDefinition.h>
#if LDC_MLIR_ENABLED

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/init.h"
#include "dmd/mtype.h"
#include "dmd/visitor.h"
#include "gen/logger.h"
#include "gen/mlir/irgen.h"
#include "gen/mlir/types.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <type_traits>

using namespace dmd;

namespace ldc {

namespace {

bool isInteger(Type *t) {
  Type *bt = t->toBasetype();
  return bt->isTypeBasic() && isIntegral(bt);
}

bool isFloat(Type *t) {
  Type *bt = t->toBasetype();

  return bt->isTypeBasic() && isReal(bt);
}

bool isUnsignedInteger(Type *t) { return isUnsigned(t->toBasetype()); }
static Expression *skipOverCasts(Expression *e) {
  while (auto ce = e->isCastExp())
    e = ce->e1;
  return e;
}

class ToMLIRLValueVisitor : public Visitor {
  MLIRGen &gen;
  mlir::Value result;

public:
  explicit ToMLIRLValueVisitor(MLIRGen &gen) : gen(gen) {}
  mlir::Value getResult() { return result; }
  using Visitor::visit;

  void visit(Expression *e) override {
    error(e->loc, "`%s` is not an lvalue the MLIR backend can handle yet",
          e->toChars());
  }

  // lvalue
  void visit(VarExp *e) override {
    VarDeclaration *vd = e->var->isVarDeclaration();
    auto it = vd ? gen.locals.find(vd) : gen.locals.end();

    if (it == gen.locals.end()) {
      error(e->loc,
            "only parameters and local variables can be referenced by the "
            "MLIR backend yet, not `%s`",
            e->toChars());
      return;
    }

    result = it->second;
  }

  void visit(AssignExp *e) override {
    // generate expression first (e.g: (x = 1) += 1)
    if (!gen.genExpression(e)) {
      return;
    }

    result = gen.genLValue(e->e1);
  }
};

class ToMLIRVisitor : public Visitor {
  MLIRGen &gen;
  mlir::Value result;

public:
  mlir::Value castValue(mlir::Value v, Type *from, Type *to, const Loc &dloc) {
    mlir::Type toType = toMLIRType(gen.context, to, dloc);
    if (!toType)
      return {};

    mlir::Type fromType = v.getType();

    if (fromType == toType)
      return v;

    mlir::Location loc = gen.loc(dloc);

    const unsigned fromWidth = fromType.getIntOrFloatBitWidth();
    const unsigned toWidth = toType.getIntOrFloatBitWidth();

    // anything -> bool lowering it as `x != 0`
    if (to->toBasetype()->ty == TY::Tbool) {
      if (isInteger(from)) {
        mlir::Value zero = gen.builder.create<mlir::arith::ConstantOp>(
            loc, fromType, gen.builder.getIntegerAttr(fromType, 0));
        return gen.builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ne, v, zero);
      }

      if (isFloat(from)) {
        mlir::Value zero = gen.builder.create<mlir::arith::ConstantOp>(
            loc, fromType, gen.builder.getFloatAttr(fromType, 0.0));
        return gen.builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, v, zero);
      }
    }

    // int -> int
    if (isInteger(from) && isInteger(to)) {
      if (toWidth < fromWidth) {
        return gen.builder.create<mlir::arith::TruncIOp>(loc, toType, v);
      }

      if (isUnsignedInteger(from)) {
        return gen.builder.create<mlir::arith::ExtUIOp>(loc, toType, v);
      }

      return gen.builder.create<mlir::arith::ExtSIOp>(loc, toType, v);
    }

    // float -> int
    if (isFloat(from) && isInteger(to)) {
      if (isUnsignedInteger(to)) {
        return gen.builder.create<mlir::arith::FPToUIOp>(loc, toType, v);
      }

      return gen.builder.create<mlir::arith::FPToSIOp>(loc, toType, v);
    }

    // float -> float
    if (isFloat(from) && isFloat(to)) {
      if (toWidth < fromWidth) {
        return gen.builder.create<mlir::arith::TruncFOp>(loc, toType, v);
      }
      return gen.builder.create<mlir::arith::ExtFOp>(loc, toType, v);
    }

    // int -> float
    if (isInteger(from) && isFloat(to)) {
      if (isUnsignedInteger(from)) {
        return gen.builder.create<mlir::arith::UIToFPOp>(loc, toType, v);
      }

      return gen.builder.create<mlir::arith::SIToFPOp>(loc, toType, v);
    }

    error(dloc,
          "cast from `%s` to `%s` is not supported by the MLIR backend yet",
          from->toChars(), to->toChars());

    return {};
  }

  explicit ToMLIRVisitor(MLIRGen &gen) : gen(gen) {}
  mlir::Value getResult() { return result; }

  using Visitor::visit;

  template <typename SignedOp, typename UnsignedOp = SignedOp,
            typename FloatOp = void>
  mlir::Value emitBinOp(Type *opType, mlir::Value lhs, mlir::Value rhs,
                        mlir::Location loc) {
    if constexpr (!std::is_void_v<FloatOp>) {
      if (isFloat(opType)) {
        return gen.builder.create<FloatOp>(loc, lhs, rhs);
      }
    }

    // not float
    if (isUnsignedInteger(opType)) {
      return gen.builder.create<UnsignedOp>(loc, lhs, rhs);
    }

    return gen.builder.create<SignedOp>(loc, lhs, rhs);
  }

  template <typename SignedOp, typename UnsignedOp = SignedOp,
            typename FloatOp = void>
  void binAssign(BinAssignExp *e) {
    // byte c; c += 1; is transformed into cast(int)c += 1,
    // cast(int)c is rvalue and cannot be used as lvalue hence
    // casts should be skipped
    Expression *lvalExp = skipOverCasts(e->e1);
    Type *varType = lvalExp->type;
    Type *opType = e->e1->type;

    const bool isInt = isInteger(opType);
    const bool isFp = !std::is_void_v<FloatOp> && isFloat(opType);
    if (!isInt && !isFp) {
      visit(static_cast<Expression *>(e));
      return;
    }

    mlir::Value storage = gen.genLValue(lvalExp);
    if (!storage)
      return;

    mlir::Value rhs = gen.genExpression(e->e2);
    if (!rhs)
      return;

    rhs = castValue(rhs, e->e2->type, opType, e->e2->loc);
    if (!rhs)
      return;

    mlir::Location loc = gen.loc(e->loc);
    mlir::Value lhs = gen.builder.create<mlir::memref::LoadOp>(loc, storage);
    lhs = castValue(lhs, varType, opType, e->loc);
    if (!lhs)
      return;

    mlir::Value res =
        emitBinOp<SignedOp, UnsignedOp, FloatOp>(opType, lhs, rhs, loc);
    res = castValue(res, opType, varType, e->loc);
    if (!res)
      return;

    gen.builder.create<mlir::memref::StoreOp>(loc, res, storage);
    result = castValue(res, varType, e->type, e->loc);
  }

  template <typename SignedOp, typename UnsignedOp = SignedOp,
            typename FloatOp = void>
  void binOp(BinExp *e) {
    const bool isInt = isInteger(e->type);
    const bool isFp = !std::is_void_v<FloatOp> && isFloat(e->type);

    if (!isInt && !isFp) {
      visit(static_cast<Expression *>(e));
      return;
    }

    mlir::Value lhs = gen.genExpression(e->e1);
    if (!lhs)
      return;

    mlir::Value rhs = gen.genExpression(e->e2);
    if (!rhs)
      return;

    mlir::Location loc = gen.loc(e->loc);

    result = emitBinOp<SignedOp, UnsignedOp, FloatOp>(e->type, lhs, rhs, loc);
  }

  template <typename SignedOp, typename UnsignedOp = SignedOp>
  void shiftOp(BinExp *e) {
    if (!isInteger(e->type)) {
      visit(static_cast<Expression *>(e));
      return;
    }

    mlir::Value lhs = gen.genExpression(e->e1);
    if (!lhs)
      return;

    mlir::Value rhs = gen.genExpression(e->e2);
    if (!rhs)
      return;

    rhs = castValue(rhs, e->e2->type, e->e1->type, e->e2->loc);
    if (!rhs)
      return;

    mlir::Location loc = gen.loc(e->loc);

    if (isUnsignedInteger(e->e1->type)) {
      result = gen.builder.create<UnsignedOp>(loc, lhs, rhs);
    } else {
      result = gen.builder.create<SignedOp>(loc, lhs, rhs);
    }
  }

  void visit(Expression *expr) override {
    error(
        expr->loc,
        "expression `%s` of type `%s` is not supported by the MLIR backend yet",
        expr->toChars(), expr->type->toChars());
  }

  void visit(DeclarationExp *e) override {
    VarDeclaration *vd = e->declaration->isVarDeclaration();

    if (!vd) {
      if (e->declaration->isFuncDeclaration()) {
        error(e->loc, "nested functions and methods are not supported by the "
                      "MLIR backend yet");
        return;
      }

      error(e->loc, "declaration `%s` is not supported by the MLIR backend yet",
            e->declaration->toChars());
      return;
    }

    // manifest constants have no storage, see
    // https://dlang.org/spec/enum.html#manifest_constants
    if (!vd->canTakeAddressOf()) {
      return;
    }

    if (vd->isDataseg()) { // global variable
      error(e->loc, "static and global variables are not supported by the "
                    "MLIR backend yet");
      return;
    }

    mlir::Type type = toMLIRType(gen.context, vd->type, vd->loc);
    if (!type) {
      return;
    }

    gen.locals[vd] = gen.createAlloca(type, gen.loc(vd->loc));

    if (vd->_init) {
      if (ExpInitializer *ei = vd->_init->isExpInitializer()) {
        IF_LOG Logger::println("expression initializer");
        gen.genExpression(ei->exp);
      }
    }
  }

  void visit(AssignExp *e) override {
    mlir::Value storage = gen.genLValue(e->e1);
    if (!storage) {
      return;
    }

    mlir::Value rhs = gen.genExpression(e->e2);

    if (!rhs) {
      return;
    }

    gen.builder.create<mlir::memref::StoreOp>(gen.loc(e->loc), rhs, storage);

    result = rhs;
  }

  void visit(IntegerExp *e) override {
    mlir::Type type = toMLIRType(gen.context, e->type, e->loc);

    if (!type)
      return;

    llvm::APInt value(64, e->getInteger());
    value = value.zextOrTrunc(type.getIntOrFloatBitWidth());
    result = gen.builder.create<mlir::arith::ConstantOp>(
        gen.loc(e->loc), type, gen.builder.getIntegerAttr(type, value));
  }

  void visit(RealExp *e) override {
    mlir::Type type = toMLIRType(gen.context, e->type, e->loc);
    if (!type)
      return;
    result = gen.builder.create<mlir::arith::ConstantOp>(
        gen.loc(e->loc), type,
        gen.builder.getFloatAttr(type, static_cast<double>(e->value)));
  }

  void visit(CastExp *e) override {
    if (!isInteger(e->type) && !isFloat(e->type)) {
      visit(static_cast<Expression *>(e));
      return;
    }
    mlir::Value v = gen.genExpression(e->e1);
    if (!v)
      return;
    result = castValue(v, e->e1->type, e->type, e->loc);
  }

  void visit(NegExp *e) override {
    const bool isInt = isInteger(e->type);
    if (!isInt && !isFloat(e->type)) {
      visit(static_cast<Expression *>(e));
      return;
    }
    mlir::Value v = gen.genExpression(e->e1);
    if (!v)
      return;
    mlir::Location loc = gen.loc(e->loc);
    if (!isInt) {
      result = gen.builder.create<mlir::arith::NegFOp>(loc, v);
      return;
    }
    // arith has no integer negation; use `0 - x`.
    mlir::Value zero = gen.builder.create<mlir::arith::ConstantOp>(
        loc, v.getType(), gen.builder.getIntegerAttr(v.getType(), 0));
    result = gen.builder.create<mlir::arith::SubIOp>(loc, zero, v);
  }

  void visit(ComExp *e) override {
    if (!isInteger(e->type)) {
      visit(static_cast<Expression *>(e));
      return;
    }
    mlir::Value v = gen.genExpression(e->e1);
    if (!v)
      return;
    // `~x` is `x ^ -1`. `-1` is all ones in two's complement.
    mlir::Location loc = gen.loc(e->loc);
    mlir::Type type = v.getType();
    mlir::Value allOnes = gen.builder.create<mlir::arith::ConstantOp>(
        loc, type,
        gen.builder.getIntegerAttr(
            type, llvm::APInt::getAllOnes(type.getIntOrFloatBitWidth())));
    result = gen.builder.create<mlir::arith::XOrIOp>(loc, v, allOnes);
  }

  // rvalue
  void visit(VarExp *vexpr) override {
    mlir::Value storage = gen.genLValue(vexpr);

    if (!storage) {
      return;
    }

    result =
        gen.builder.create<mlir::memref::LoadOp>(gen.loc(vexpr->loc), storage);
  }

  void visit(AddExp *e) override {
    binOp<mlir::arith::AddIOp, mlir::arith::AddIOp, mlir::arith::AddFOp>(e);
  }
  void visit(MinExp *e) override {
    binOp<mlir::arith::SubIOp, mlir::arith::SubIOp, mlir::arith::SubFOp>(e);
  }
  void visit(MulExp *e) override {
    binOp<mlir::arith::MulIOp, mlir::arith::MulIOp, mlir::arith::MulFOp>(e);
  }
  void visit(DivExp *e) override {
    binOp<mlir::arith::DivSIOp, mlir::arith::DivUIOp, mlir::arith::DivFOp>(e);
  }
  void visit(ModExp *e) override {
    // `arith.remf` has fmod semantics (sign of the dividend), like D's `%`.
    binOp<mlir::arith::RemSIOp, mlir::arith::RemUIOp, mlir::arith::RemFOp>(e);
  }

  void visit(BinAssignExp *e) override {
    using namespace mlir::arith;
    switch (e->op) {
    case EXP::addAssign:
      return binAssign<AddIOp, AddIOp, AddFOp>(e);
    case EXP::minAssign:
      return binAssign<SubIOp, SubIOp, SubFOp>(e);
    case EXP::mulAssign:
      return binAssign<MulIOp, MulIOp, MulFOp>(e);
    case EXP::divAssign:
      return binAssign<DivSIOp, DivUIOp, DivFOp>(e);
    case EXP::modAssign:
      return binAssign<RemSIOp, RemUIOp, RemFOp>(e);
    case EXP::andAssign:
      return binAssign<AndIOp>(e);
    case EXP::orAssign:
      return binAssign<OrIOp>(e);
    case EXP::xorAssign:
      return binAssign<XOrIOp>(e);
    case EXP::leftShiftAssign:
      return binAssign<ShLIOp>(e);
    case EXP::rightShiftAssign:
      return binAssign<ShRSIOp, ShRUIOp>(e);
    case EXP::unsignedRightShiftAssign:
      return binAssign<ShRUIOp>(e);
    default:
      visit(static_cast<Expression *>(e));
    }
  }

  void visit(PostExp *e) override {
    mlir::Value storage = gen.genLValue(e->e1);
    if (!storage)
      return;

    mlir::Location loc = gen.loc(e->loc);
    mlir::Value old = gen.builder.create<mlir::memref::LoadOp>(loc, storage);
    mlir::Type type = old.getType();
    const bool inc = e->op == EXP::plusPlus;
    mlir::Value updated;

    if (isInteger(e->e1->type)) {
      mlir::Value one = gen.builder.create<mlir::arith::ConstantOp>(
          loc, type, gen.builder.getIntegerAttr(type, 1));
      if (inc) {
        updated = gen.builder.create<mlir::arith::AddIOp>(loc, old, one);
      } else {
        updated = gen.builder.create<mlir::arith::SubIOp>(loc, old, one);
      }
    } else if (isFloat(e->e1->type)) {
      mlir::Value one = gen.builder.create<mlir::arith::ConstantOp>(
          loc, type, gen.builder.getFloatAttr(type, 1.0));

      if (inc) {
        updated = gen.builder.create<mlir::arith::AddFOp>(loc, old, one);
      } else {
        updated = gen.builder.create<mlir::arith::SubFOp>(loc, old, one);
      }
    } else {
      visit(static_cast<Expression *>(e));
      return;
    }

    gen.builder.create<mlir::memref::StoreOp>(loc, updated, storage);
    result = old;
  }

  void visit(AndExp *e) override { binOp<mlir::arith::AndIOp>(e); }
  void visit(OrExp *e) override { binOp<mlir::arith::OrIOp>(e); }
  void visit(XorExp *e) override { binOp<mlir::arith::XOrIOp>(e); }
  void visit(ShlExp *e) override { shiftOp<mlir::arith::ShLIOp>(e); }
  void visit(ShrExp *e) override {
    shiftOp<mlir::arith::ShRSIOp, mlir::arith::ShRUIOp>(e);
  }
  void visit(UshrExp *e) override { shiftOp<mlir::arith::ShRUIOp>(e); }

  void visit(CmpExp *e) override {
    mlir::Value lhs = gen.genExpression(e->e1);
    mlir::Value rhs = gen.genExpression(e->e2);

    if (!lhs || !rhs)
      return;

    mlir::Location loc = gen.loc(e->loc);

    Type *t = e->e1->type->toBasetype();

    if (isInteger(t)) {
      const bool unsignedInt = isUnsignedInteger(t);

      mlir::arith::CmpIPredicate predicate;

      switch (e->op) {
      case EXP::lessThan:
        predicate = unsignedInt ? mlir::arith::CmpIPredicate::ult
                                : mlir::arith::CmpIPredicate::slt;
        break;
      case EXP::lessOrEqual:
        predicate = unsignedInt ? mlir::arith::CmpIPredicate::ule
                                : mlir::arith::CmpIPredicate::sle;
        break;
      case EXP::greaterThan:
        predicate = unsignedInt ? mlir::arith::CmpIPredicate::ugt
                                : mlir::arith::CmpIPredicate::sgt;
        break;
      case EXP::greaterOrEqual:
        predicate = unsignedInt ? mlir::arith::CmpIPredicate::uge
                                : mlir::arith::CmpIPredicate::sge;
        break;
      default:
        llvm_unreachable("unexpected CmpExp op");
      }

      result =
          gen.builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
      return;
    }

    if (isFloat(t)) {
      mlir::arith::CmpFPredicate predicate;

      switch (e->op) {
      case EXP::lessThan:
        predicate = mlir::arith::CmpFPredicate::OLT;
        break;
      case EXP::lessOrEqual:
        predicate = mlir::arith::CmpFPredicate::OLE;
        break;
      case EXP::greaterThan:
        predicate = mlir::arith::CmpFPredicate::OGT;
        break;
      case EXP::greaterOrEqual:
        predicate = mlir::arith::CmpFPredicate::OGE;
        break;
      default:
        llvm_unreachable("unexpected CmpExp op");
      }

      result =
          gen.builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs, rhs);
      return;
    }

    error(e->loc, "comparison of `%s` is not supported by the MLIR backend yet",
          e->e1->type->toChars());
  }

  void visit(EqualExp *e) override {
    mlir::Value lhs = gen.genExpression(e->e1);
    mlir::Value rhs = gen.genExpression(e->e2);

    if (!lhs || !rhs)
      return;

    mlir::Location loc = gen.loc(e->loc);
    Type *t = e->e1->type->toBasetype();

    const bool equal = e->op == EXP::equal;

    if (isInteger(t)) {
      mlir::arith::CmpIPredicate predicate =
          equal ? mlir::arith::CmpIPredicate::eq
                : mlir::arith::CmpIPredicate::ne;

      result =
          gen.builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
      return;
    }

    if (isFloat(t)) {
      // `==` is ordered (false if either side is NaN [NaN == Nan]), `!=` is
      // unordered (true if either side is NaN [NaN != Nan]), so `a != b` is
      // exactly `!(a == b)`
      mlir::arith::CmpFPredicate predicate =
          equal ? mlir::arith::CmpFPredicate::OEQ
                : mlir::arith::CmpFPredicate::UNE;

      result =
          gen.builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs, rhs);
      return;
    }

    error(e->loc, "equality on `%s` is not supported by the MLIR backend yet",
          e->e1->type->toChars());
  }

  void visit(NotExp *e) override {
    mlir::Value v = gen.genExpression(e->e1);

    if (!v)
      return;

    // `!x` is legal on any scalar (`!5`), so normalise to i1 first
    v = castValue(v, e->e1->type, Type::tbool, e->loc);

    if (!v)
      return;

    mlir::Location loc = gen.loc(e->loc);
    mlir::Value one = gen.builder.create<mlir::arith::ConstantOp>(
        loc, v.getType(), gen.builder.getIntegerAttr(v.getType(), 1));
    result = gen.builder.create<mlir::arith::XOrIOp>(loc, v, one);
  }

  void visit(CallExp *e) override {
    // `e->f` is the statically resolved callee; it is null for calls through
    // function pointers and delegates
    FuncDeclaration *fd = e->f;
    if (!fd) {
      error(e->loc, "indirect calls are not supported by the MLIR backend yet");
      return;
    }

    mlir::func::FuncOp callee = gen.getOrDeclareFunction(fd, e->loc);
    if (!callee)
      return;

    llvm::SmallVector<mlir::Value, 4> args;
    if (e->arguments) {
      for (Expression *arg : *e->arguments) {
        mlir::Value v = gen.genExpression(arg);
        if (!v)
          return;
        args.push_back(v);
      }
    }

    auto call =
        gen.builder.create<mlir::func::CallOp>(gen.loc(e->loc), callee, args);

    if (call.getNumResults() == 1)
      result = call.getResult(0);
  }

  void visit(IdentityExp *e) override {
    if (!isInteger(e->e1->type)) {
      error(e->loc, "`is` on `%s` is not supported by the MLIR backend yet",
            e->e1->type->toChars());
      return;
    }

    mlir::Value lhs = gen.genExpression(e->e1);
    if (!lhs)
      return;
    mlir::Value rhs = gen.genExpression(e->e2);
    if (!rhs)
      return;

    result = gen.builder.create<mlir::arith::CmpIOp>(
        gen.loc(e->loc),
        e->op == EXP::identity ? mlir::arith::CmpIPredicate::eq
                               : mlir::arith::CmpIPredicate::ne,
        lhs, rhs);
  }

  void visit(LogicalExp *e) override {
    // `&&` / `||` must not evaluate the rhs when the lhs decides the result,
    // so they need an scf.if yielding an i1
    error(e->loc,
          "`%s` is not supported by the MLIR backend yet (needs `scf.if`)",
          e->op == EXP::andAnd ? "&&" : "||");
  }
};
} // anonymous namespace

mlir::Value MLIRGen::genLValue(Expression *expr) {
  ToMLIRLValueVisitor visitor(*this);
  expr->accept(&visitor);
  return visitor.getResult();
}

mlir::Value MLIRGen::genExpression(Expression *expr) {
  ToMLIRVisitor visitor(*this);
  expr->accept(&visitor);
  return visitor.getResult();
}

mlir::Value MLIRGen::genCondition(Expression *expr) {
  ToMLIRVisitor visitor(*this);

  expr->accept(&visitor);
  mlir::Value value = visitor.getResult();

  if (!value) {
    return {};
  }

  return visitor.castValue(value, expr->type, Type::tbool, expr->loc);
}

} // namespace ldc

#endif // LDC_MLIR_ENABLED
