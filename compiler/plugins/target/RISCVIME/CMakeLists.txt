iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_riscv_ime
  TARGET
    ::RISCVIME
)

iree_cc_library(
  NAME
    RISCVIME
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRRISCVIMEDialect
    MLIRRISCVIMETransforms
    MLIRRISCVIMEToLLVMIRTranslation     # ← ADD (naya)
    MLIRTargetLLVMIRExport               # ← ADD (naya)
    iree::compiler::PluginAPI          
  PUBLIC
)