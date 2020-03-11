# 添加opencv的依赖
find_package(OpenCV REQUIRED)

# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED )
# # list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

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
	
add_executable(pc_registration pc_registration.cpp)
target_link_libraries(pc_registration
    base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})
