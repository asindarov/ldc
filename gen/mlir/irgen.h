//===-- gen/mlir/irgen.h - D AST to MLIR generation -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Entry point of the MLIR backend: walks a D module and emits `func` and
// `arith` dialect operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_MLIR_ENABLED

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

class Expression;
class FuncDeclaration;
class Module;
class Statement;
class VarDeclaration;
struct Loc;

namespace ldc {

class MLIRGen {
public:
  explicit MLIRGen(mlir::MLIRContext &context);

  mlir::MLIRContext &context;
  mlir::OpBuilder builder;

  /// Stack slots of parameters and local variables, as rank-0 memrefs
  llvm::DenseMap<VarDeclaration *, mlir::Value> locals;

  /// Returns the storage an lvalue denotes
  mlir::Value genLValue(Expression *expr);

  mlir::Value createAlloca(mlir::Type elemType, mlir::Location loc);

  mlir::Value genCondition(Expression *expr);

  bool isTerminated();

  /// Generates MLIR for the whole module. Returns null (after emitting errors)
  /// if anything in the module is not supported yet
  mlir::OwningOpRef<mlir::ModuleOp> genModule(Module *m);

  /// Generates the `func.func` definition for a function with a body.
  void genFunction(FuncDeclaration *fd);

  /// Returns the `func.func` for `fd`, creating a bodiless declaration if it
  /// has not been seen yet (e.g. a forward reference or an external function).
  /// Returns null (after emitting an error) if `fd` is not supported.
  mlir::func::FuncOp getOrDeclareFunction(FuncDeclaration *fd, const Loc &loc);

  /// Returns the value of an expression, or null if an error was emitted or
  /// the expression has type `void`.
  mlir::Value genExpression(Expression *expr);

  void genStatement(Statement *stmt);

  mlir::Location loc(const Loc &l);

private:
  mlir::ModuleOp module;
  mlir::Block *entryBlock = nullptr;
  mlir::Operation *lastAlloca = nullptr;
};

} // namespace ldc

#endif // LDC_MLIR_ENABLED
