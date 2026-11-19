//===-- gen/mlir/lowering.cpp ---------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "gen/mlir/lowering.h"

#include "dmd/errors.h"
#include "dmd/globals.h"
#include "gen/logger.h"
#include "llvm/IR/Module.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace ldc {

std::unique_ptr<llvm::Module> lowerToLLVMIR(mlir::ModuleOp module,
                                            llvm::LLVMContext &llvmContext,
                                            const Loc &loc) {
  IF_LOG Logger::println("lowerToLLVMIR()");
  LOG_SCOPE

  mlir::PassManager pm(module.getContext());

#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  // Removes the casts the partial conversions left between dialects
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(module))) {
    error(loc, "lowering the MLIR module to the `llvm` dialect failed");
    return nullptr;
  }

  IF_LOG {
    Logger::println("Lowered to the llvm dialect:");
    module.dump();
  }

  std::unique_ptr<llvm::Module> result =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!result) {
    error(loc, "translating the MLIR module to LLVM IR failed");
    return nullptr;
  }

  return result;
}

} // namespace ldc

#endif // LDC_MLIR_ENABLED
