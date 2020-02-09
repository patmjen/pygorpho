cmake_minimum_required (VERSION 3.10)

# Download and configure gorpho as part of CMake config
message ("Downloading gorpho")
configure_file(CMakeLists.txt.in gorpho-download/CMakeLists.txt)
execute_process (COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	RESULT_VARIABLE result
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gorpho-download)
if (result)
	message (FATAL_ERROR "CMake step for gorpho failed: ${result}")
endif ()
execute_process (COMMAND ${CMAKE_COMMAND} --build .
	RESULT_VARIABLE result
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gorpho-download)
if (result)
	message (FATAL_ERROR "Configure and build step for gorpho failed: ${result}")
endif ()

add_subdirectory (${CMAKE_CURRENT_BINARY_DIR}/gorpho-src
                  ${CMAKE_CURRENT_BINARY_DIR}/gorpho-build
				  EXCLUDE_FROM_ALL)

set (CMAKE_CXX_STANDARD 11)

set (SOURCE_DIR src)

# Setup pygorpho
project ("pygorpho" VERSION 0.5.0
                    DESCRIPTION "Python bindings for gorpho"
                    LANGUAGES "CUDA")

add_library (${PROJECT_NAME} SHARED ${SOURCE_DIR}/pygorpho.cu)
target_compile_options (${PROJECT_NAME} PUBLIC "--expt-relaxed-constexpr" "-lineinfo" "--std=c++14")
target_link_libraries (${PROJECT_NAME} gorpho)

if (WIN32)
	install (TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${SOURCE_DIR}/${PROJECT_NAME})
else ()
	install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${SOURCE_DIR}/${PROJECT_NAME})
endif ()