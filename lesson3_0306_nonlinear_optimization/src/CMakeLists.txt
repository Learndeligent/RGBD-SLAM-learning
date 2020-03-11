# 添加opencv的依赖
find_package(OpenCV 3 REQUIRED)

# Eigen
include_directories( "/usr/include/eigen3" )

# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED )
# # list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

find_package(Sophus REQUIRED)
INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIRS})

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# 编译库文件
ADD_LIBRARY(base base.cpp)
TARGET_LINK_LIBRARIES(base
   ${OpenCV_LIBS}
	${PCL_LIBRARIES}  )

# 添加可执行文件
# add_executable(showRGBD showRGBD.cpp)
# target_link_libraries(showRGBD
#     base
# 	${OpenCV_LIBS}
# 	${PCL_LIBRARIES} )
	
add_executable(nonlinear_optimization nonlinear_optimization.cpp)
target_link_libraries(nonlinear_optimization
    base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Sophus_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba)


