add_executable(probe_main probe_main.cpp)
target_link_libraries(probe_main PRIVATE common)

add_executable(microbench_main microbench_main.cpp)
target_link_libraries(microbench_main PRIVATE common)

add_executable(dump_shader_stats dump_shader_stats.cpp)
target_link_libraries(dump_shader_stats PRIVATE common)
