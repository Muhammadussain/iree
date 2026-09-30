//===- PluginRegistration.cpp - RISCVIME IREE plugin registration -------===//

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Target/LLVMIR/Dialect/RISCVIME/RISCVIMEToLLVMIRTranslation.h"  // ← ADD

namespace mlir::iree_compiler {

namespace {

struct RISCVIMESession
    : PluginSession<RISCVIMESession, EmptyPluginOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void registerDialects(DialectRegistry &registry) {
  registry.insert<riscv_ime::RISCVIMEDialect>();
  // Also register the LLVM IR translation interface so that
  // riscv_ime.intr.vmadotus can be lowered to @llvm.riscv.vmadotus
  // during serialization.
  riscv_ime::registerRISCVIMEDialectTranslation(registry);  // ← ADD
}
};

} // namespace

} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_hal_target_riscv_ime(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::RISCVIMESession>(
      "hal_target_riscv_ime");
  return true;
}