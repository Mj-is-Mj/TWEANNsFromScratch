# Dependencies

SYCL compilers sometimes don't play well with certain dependencies the project may have (i.e. doesn't play well with magnum for graphics), so dependencies should installed separately (i.e. should not be submodules of the project). 

Try to find ways to install dependencies as a separate CMake "environment", sort of like python venvs