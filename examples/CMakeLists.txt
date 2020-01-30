add_executable( dpa dpa.cxx )
target_link_libraries( dpa PUBLIC dvcxx )

add_executable( smr smr.cxx )
target_link_libraries( smr PUBLIC dvcxx )

add_executable( alloc_sfinae alloc_sfinae.cxx )
target_link_libraries( alloc_sfinae PUBLIC dvcxx )

add_executable( vector vector.cxx )
target_link_libraries( vector PUBLIC dvcxx )
