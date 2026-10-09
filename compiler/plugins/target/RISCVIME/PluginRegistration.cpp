//===- PluginRegistration.cpp - RISCVIME IREE plugin registration -------===//

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Target/LLVMIR/Dialect/RISCVIME/RISCVIMEToLLVMIRTranslation.h" 

namespace mlir::iree_compiler {

namespace {

struct RISCVIMESession
    : PluginSession<RISCVIMESession, EmptyPluginOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void registerDialects(DialectRegistry &registry) {
  registry.insert<riscv_ime::RISCVIMEDialect>();
  riscv_ime::registerRISCVIMEDialectTranslation(registry);  
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