//===-- gen/mlir/types.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "gen/mlir/types.h"

#include "dmd/errors.h"
#include "dmd/mtype.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace dmd;

namespace ldc {

mlir::Type toMLIRType(mlir::MLIRContext &context, Type *t, const Loc &loc) {
  Type *bt = t->toBasetype();

  // only basic types for now.
  if (bt->isTypeBasic()) {
    if (bt->ty == TY::Tbool) {
      return mlir::IntegerType::get(&context, 1);
    }

    if (isIntegral(bt)) {
      return mlir::IntegerType::get(&context, size(bt) * 8);
    }

    if (isReal(bt)) {
      switch (size(bt)) {
      case 4:
        return mlir::Float32Type::get(&context);
      case 8:
        return mlir::Float64Type::get(&context);
      default:
        break;
      }
    }
  }

  error(loc, "type `%s` is not supported by the MLIR backend yet",
        t->toChars());
  return {};
}

} // namespace ldc

#endif // LDC_MLIR_ENABLED
