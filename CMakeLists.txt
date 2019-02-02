
cmake_minimum_required(VERSION 3.4.3)
include(ExternalProject)
find_package(Git REQUIRED)


project(CovarGibb
	LANGUAGES C CXX)

set(default_build_type "Release")

# if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC -Wall -pedantic -Wextra")
#   set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto -finline-limit=1024 -fno-semantic-interposition")
# elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -pedantic -Wextra")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC")
#   set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto")
# elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
#   set (CMAKE_CXX_FLAGS_RELEASE "-fast -qopt-zmm-usage=high -ansi-alias -qopenmp")
# endif ()

ExternalProject_Add(xsimd
  # GIT_REPOSITORY https://github.com/QuantStack/xsimd
	GIT_REPOSITORY https://github.com/chriselrod/xsimd
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include include)

option(BUILD_SHARED_LIBS "Build shared libs" ON)
# set(CMAKE_C_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -mprefer-vector-width=512 -flto")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC -Wall -pedantic -Wextra -fno-strict-aliasing")
  set (CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -fassociative-math -fno-signed-zeros -fno-trapping-math -mprefer-vector-width=512 -fno-semantic-interposition -DXSIMD_NO_DENORMALS -DXSIMD_NO_INIFINITIES -DXSIMD_NO_INFINITIES")#  -finline-limit=1024
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -pedantic -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -DXSIMD_NO_DENORMALS -DXSIMD_NO_INIFINITIES -DXSIMD_NO_INFINITIES")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set (CMAKE_CXX_FLAGS_RELEASE "-fast -qopt-zmm-usage=high -ansi-alias -qopenmp")
endif ()

add_library(gibbcovarrealism SHARED gibbs.cpp)
# add_dependencies(gibbcovarrealism libsleef)
# target_link_libraries(gibbcovarrealism sleef)


# cmake ../../CovarianceRealismGibbsModel/ -DCMAKE_BUILD_TYPE=Release
