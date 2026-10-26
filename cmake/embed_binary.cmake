# embed_binary.cmake — embed a binary file (the ocloc SPIR-V module) as a C byte array.
# Run as a script with E2_BIN_IN / E2_BIN_OUT / E2_BIN_FUNC.
file(READ "${E2_BIN_IN}" _hex HEX)
string(LENGTH "${_hex}" _hexlen)
math(EXPR _nbytes "${_hexlen} / 2")
string(REGEX REPLACE "(..)" "0x\\1," _bytes "${_hex}")
file(WRITE "${E2_BIN_OUT}"
     "/* generated from ${E2_BIN_IN} by cmake/embed_binary.cmake — do not edit */\n"
     "#include <stddef.h>\n"
     "#include \"gpu/l0.h\"\n"
     "static const unsigned char e2_spirv_bytes[] = {\n${_bytes}\n};\n"
     "const unsigned char *e2_l0_spirv(size_t *bytes) {\n"
     "    *bytes = ${_nbytes};\n"
     "    return e2_spirv_bytes;\n}\n")
