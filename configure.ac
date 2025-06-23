# 指定 autoconf 的最低版本
AC_PREREQ([2.65])

# 初始化项目基本信息
AC_INIT([isit2025], [1.0], [zhihang161013@outlook.com], [isit2025], [https://github.com/zhihang161013/isit2025])

# 指定项目的主源文件，用于验证当前目录
AC_CONFIG_SRCDIR([src/main.c])

# 指定生成的配置头文件
AC_CONFIG_HEADERS([include/config.h])

# 指定辅助脚本和宏文件目录
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

# 设置默认的 C 编译器标志
: ${CFLAGS="-g -O2 -Wall"}

# 初始化 libtool（如果需要支持共享库）
LT_INIT

# 检查是否有可用的 C 编译器
AC_PROG_CC

# 检查依赖库（jerasure）
AC_CHECK_LIB([jerasure], [jerasure_matrix_encode], [],
             [AC_MSG_ERROR([jerasure library is required but not found.])])

# 检查头文件
AC_CHECK_HEADERS([stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# 检查自定义宏 AX_EXT
AX_EXT

# 定义可选配置选项（如启用/禁用 SIMD 优化）
AC_ARG_ENABLE([simd],
              AS_HELP_STRING([--disable-simd], [Disable SIMD optimizations]),
              [if test "x$enableval" = "xno"; then
                 SIMD_FLAGS=""
                 echo "SIMD optimizations disabled."
               fi])

# 检查常用函数
AC_CHECK_FUNCS([malloc free memset memcpy])

# 指定需要生成的文件
AC_CONFIG_FILES([
    Makefile
    src/Makefile
    benchmarks/Makefile
])

# 输出配置
AC_OUTPUT