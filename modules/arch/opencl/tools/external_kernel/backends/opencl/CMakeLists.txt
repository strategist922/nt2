include(NT2Module)

set(NT2_TOOL_resource_compiler_ROOT ${NT2_SOURCE_ROOT}/modules/arch/opencl/tools/resource_compiler)
include_directories(${NT2_TOOL_resource_compiler_ROOT}/include)

macro(resource_compile src_file name file)
  nt2_module_tool_setup(resource_compiler)
  string(REGEX REPLACE "[/\\.-]" "_" ${src_file} ${name})
  set(${src_file} "nt2_resource_${${src_file}}.cpp")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${${src_file}}
                     COMMAND ${NT2_BINARY_DIR}/tools/resource_compiler/resource_compiler ${name} ${file}
                     DEPENDS ${file} ${ARGN}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    )
endmacro()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-fvisibility=hidden)
endif()

set(sources opencl.cpp)

#file(GLOB_RECURSE resources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/resources resources/*.map)
#foreach(resource ${resources})
#  resource_compile(src_file ${resource} ${CMAKE_CURRENT_SOURCE_DIR}/resources/${resource})
#  list(APPEND sources ${src_file})
#endforeach()

set(CMAKE_SHARED_MODULE_PREFIX)
add_library(opencl MODULE ${sources})
