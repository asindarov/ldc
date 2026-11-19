//===-- gen/mlir/lowering.h - MLIR to LLVM IR lowering ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Lowers the MLIR module produced by the MLIR backend to the `llvm` dialect
// and translates it to an `llvm::Module`.
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_MLIR_ENABLED

#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

struct Loc;

namespace ldc {

/// Runs the conversion passes on `module` and returns the translated
/// LLVM module, or null on failure
std::unique_ptr<llvm::Module> lowerToLLVMIR(mlir::ModuleOp module,
                                            llvm::LLVMContext &llvmContext,
                                            const Loc &loc);

} // namespace ldc

#endif // LDC_MLIR_ENABLED
