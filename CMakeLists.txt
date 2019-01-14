
cmake_minimum_required(VERSION 3.4.3)
include(ExternalProject)
find_package(Git REQUIRED)


project(CovarGibb
	LANGUAGES C CXX)

set(default_build_type "Release")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC -Wall -pedantic -Wextra")
  set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto -finline-limit=1024 -fno-semantic-interposition")
  set (CMAKE_C_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -pedantic -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto")
  set (CMAKE_C_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set (CMAKE_CXX_FLAGS_RELEASE "-fast -qopt-zmm-usage=high -ansi-alias -qopenmp")
endif ()

ExternalProject_Add(libsleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib -DCMAKE_BUILD_TYPE=Release -DBUILD_DFT=FALSE -DBUILD_SHARED_LIBS=FALSE -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

option(BUILD_SHARED_LIBS "Build shared libs" ON)
# set(CMAKE_C_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -mprefer-vector-width=512 -flto")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC -Wall -pedantic -Wextra")
  set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto -finline-limit=1024 -fno-semantic-interposition")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -pedantic -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "-Ofast -mprefer-vector-width=512 -flto")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set (CMAKE_CXX_FLAGS_RELEASE "-fast -qopt-zmm-usage=high -ansi-alias -qopenmp")
endif ()

# if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
#   set (CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -mprefer-vector-width=512 -flto -finline-limit=1024 -fno-semantic-interposition")
# elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
#   set (CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -mprefer-vector-width=512 -flto")
# elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
#   set (CMAKE_Fortran_FLAGS_RELEASE "-fast -qopt-zmm-usage=high -ansi-alias -qopenmp")
# endif ()

# add_library(gibbcovarrealism SHARED process_inputs.cpp conditional_groups.cpp initialize_rngs.cpp gibbs.cpp)
add_library(gibbcovarrealism SHARED gibbs.cpp)
add_dependencies(gibbcovarrealism libsleef)
target_link_libraries(gibbcovarrealism sleef)
