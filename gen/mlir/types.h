//===-- gen/mlir/types.h - D to MLIR type mapping ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Maps D frontend types to MLIR builtin types for the MLIR backend.
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_MLIR_ENABLED

#include "mlir/IR/Types.h"

struct Loc;
class Type;

namespace mlir {
class MLIRContext;
}

namespace ldc {

/// Returns the MLIR type for the given D type, or a null type (after emitting
/// an error at `loc`) if the type is not supported by the MLIR backend yet
mlir::Type toMLIRType(mlir::MLIRContext &context, Type *t, const Loc &loc);

} // namespace ldc

#endif // LDC_MLIR_ENABLED
