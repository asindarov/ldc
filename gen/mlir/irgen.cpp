//===-- gen/mlir/irgen.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#if LDC_MLIR_ENABLED

#include "gen/mlir/irgen.h"

#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/globals.h"
#include "dmd/import.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/statement.h"
#include "dmd/visitor.h"
#include "gen/logger.h"
#include "gen/mlir/types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Verifier.h"
#include <iterator>

using namespace dmd;

namespace ldc {

namespace {

class ToMLIRDeclarationVisitor : public Visitor {
  MLIRGen &gen;

public:
  explicit ToMLIRDeclarationVisitor(MLIRGen &gen) : gen(gen) {}
  using Visitor::visit;

  void visit(Dsymbol *s) override {
    error(s->loc, "declaration `%s` is not supported by the MLIR backend yet",
          s->toChars());
  }

  void visit(Import *) override {}

  void visit(AttribDeclaration *ad) override {
    if (Dsymbols *members = include(ad, nullptr)) {
      for (Dsymbol *s : *members) {
        s->accept(this);
      }
    }
  }

  void visit(FuncDeclaration *fd) override {
    // Function aliases don't contribute new symbols, and bodiless declarations
    // are emitted on demand when they are called
    if (!fd->skipCodegen() && !fd->isFuncAliasDeclaration() && fd->fbody) {
      gen.genFunction(fd);
    }
  }
};

class ToMLIRStatementVisitor : public Visitor {
  MLIRGen &gen;

public:
  explicit ToMLIRStatementVisitor(MLIRGen &gen) : gen(gen) {}
  using Visitor::visit;

  void visit(Statement *s) override {
    error(s->loc, "this statement is not supported by the MLIR backend yet");
  }

  void visit(CompoundStatement *s) override {
    if (!s->statements)
      return;

    for (Statement *stmt : *s->statements) {
      if (!stmt)
        continue;


      llvm::errs() <<  "stmt @ " << stmt->loc.toChars() << " mightHaveTerminator = "
                  << gen.builder.getInsertionBlock()->mightHaveTerminator() << "\n";

      // Statements after a `return` are unreachable; `func.return` must be
      // the last operation of its block
      if (gen.builder.getInsertionBlock()->mightHaveTerminator())
        break;

      stmt->accept(this);
    }
  }

  void visit(IfStatement *s) override {
    mlir::Value condition = gen.genCondition(s->condition);

    mlir::Location loc = gen.loc(s->condition->loc);

    auto ifOp = gen.builder.create<mlir::scf::IfOp>(loc, condition, /* withElseRegion= */s->elsebody != nullptr);

    {
      mlir::OpBuilder::InsertionGuard guard(gen.builder);
      gen.builder.setInsertionPointToStart(ifOp.thenBlock());
      gen.genStatement(s->ifbody);
    }

    if (s->elsebody != nullptr) {
      mlir::OpBuilder::InsertionGuard guard(gen.builder);
      gen.builder.setInsertionPointToStart(ifOp.elseBlock());
      gen.genStatement(s->elsebody);
    }
  }

  void visit(ScopeStatement *s) override {
    if (s->statement)
      s->statement->accept(this);
  }

  void visit(ExpStatement *s) override {
    if (s->exp)
      gen.genExpression(s->exp);
  }

  void visit(ReturnStatement *s) override {
    llvm::SmallVector<mlir::Value, 1> operands;

    if (s->exp) {
      assert(s->exp->type->toBasetype()->ty != TY::Tvoid);

      mlir::Value value = gen.genExpression(s->exp);

      if (!value)
        return;
      operands.push_back(value);
    }

    gen.builder.create<mlir::func::ReturnOp>(gen.loc(s->loc), operands);
  }
};

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

MLIRGen::MLIRGen(mlir::MLIRContext &context)
    : context(context), builder(&context) {
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
}

mlir::Value MLIRGen::createAlloca(mlir::Type elemType, mlir::Location loc) {
  // Keep all allocas together att he top of the entry block, in declaration
  // order
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (lastAlloca) {
    builder.setInsertionPointAfter(lastAlloca);
  } else {
    builder.setInsertionPointToStart(entryBlock);
  }

  auto alloca = builder.create<mlir::memref::AllocaOp>(
      loc, mlir::MemRefType::get({}, elemType));

  lastAlloca = alloca;

  return alloca;
}

mlir::Location MLIRGen::loc(const Loc &l) {
  const char *filename = l.filename();
  return mlir::FileLineColLoc::get(&context, filename ? filename : "",
                                   l.linnum(), l.charnum());
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::genModule(Module *m) {
  IF_LOG Logger::println("MLIRGen::genModule(%s)", m->toPrettyChars());
  LOG_SCOPE

  mlir::OwningOpRef<mlir::ModuleOp> owningModule =
      mlir::ModuleOp::create(loc(m->loc));
  module = *owningModule;

  ToMLIRDeclarationVisitor visitor(*this);
  for (Dsymbol *s : *m->members) {
    s->accept(&visitor);
  }

  if (global.errors)
    return nullptr;

  if (mlir::failed(mlir::verify(module))) {
    error(m->loc, "generated MLIR for module `%s` failed verification",
          m->toPrettyChars());
    return nullptr;
  }

  return owningModule;
}

mlir::func::FuncOp MLIRGen::getOrDeclareFunction(FuncDeclaration *fd,
                                                 const Loc &dloc) {
  TypeFunction *tf = fd->type->isTypeFunction();

  if (tf->parameterList.varargs != VARARGnone) {
    error(dloc, "variadic functions are not supported by the MLIR backend yet");
    return {};
  }

  const char *name = mangleExact(fd);
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(name))
    return func;

  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (size_t i = 0, n = tf->parameterList.length(); i < n; i++) {
    Parameter *p = tf->parameterList[i];

    if (p->storageClass & (STCref | STCout | STClazy)) {
      error(dloc, "`ref`, `out` and `lazy` parameters are not supported by "
                  "the MLIR backend yet");
      return {};
    }

    mlir::Type type = toMLIRType(context, p->type, dloc);
    if (!type)
      return {};
    argTypes.push_back(type);
  }

  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (tf->next->toBasetype()->ty != TY::Tvoid) {
    mlir::Type type = toMLIRType(context, tf->next, dloc);
    if (!type)
      return {};
    resultTypes.push_back(type);
  }

  // This may be called in the middle of generating another function's body,
  // the builder is positioned inside that body, but the declaration must live
  // at the module level. So move the insertion point to the end of the module
  // and let the guard put it back when we return
  // so the caller can carry on exactly where it left off
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<mlir::func::FuncOp>(
      loc(fd->loc), name, builder.getFunctionType(argTypes, resultTypes));

  // The MLIR verifier requires bodiless declarations to be private.
  // `genFunction` makes the symbol public again when it adds the body.
  func.setPrivate();
  return func;
}

void MLIRGen::genFunction(FuncDeclaration *fd) {
  IF_LOG Logger::println("MLIRGen::genFunction(%s)", fd->toPrettyChars());
  LOG_SCOPE

  mlir::func::FuncOp func = getOrDeclareFunction(fd, fd->loc);
  if (!func)
    return;

  assert(func.isDeclaration() && "function defined twice");

  func.setPublic();

  entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  lastAlloca = nullptr;

  locals.clear();

  if (fd->parameters) {
    for (size_t i = 0; i < fd->parameters->length; i++) {
      VarDeclaration *vd = (*fd->parameters)[i];
      mlir::Value arg = entryBlock->getArgument(i);
      mlir::Location location = loc(vd->loc);
      mlir::Value storage = createAlloca(arg.getType(), location);
      builder.create<mlir::memref::StoreOp>(location, arg, storage);
      locals[vd] = storage;
    }
  }

  genStatement(fd->fbody);

  if (!builder.getInsertionBlock()->mightHaveTerminator() &&
      func.getNumResults() == 0) {
    builder.create<mlir::func::ReturnOp>(loc(fd->endloc));
  }
}

void MLIRGen::genStatement(Statement *stmt) {
  ToMLIRStatementVisitor visitor(*this);
  stmt->accept(&visitor);
}

bool MLIRGen::isTerminated() {
  mlir::Block *block = builder.getInsertionBlock();
  mlir::Block::iterator point = builder.getInsertionPoint();

  if (point == block->begin())
    return false;

  return std::prev(point)->hasTrait<mlir::OpTrait::IsTerminator>();
}

} // namespace ldc

#endif // LDC_MLIR_ENABLED
