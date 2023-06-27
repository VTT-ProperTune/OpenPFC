FROM ahojukka5/openpfc:base

WORKDIR /usr/src/openpfc
COPY . .

# Set environment variables (optional)
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

RUN ldconfig
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DUSE_CLANG_TIDY=OFF
RUN cmake --build build
RUN (cd build/tests && ctest)

# Generate coverage report
RUN lcov --capture --directory build/tests --output-file coverage.info
RUN lcov --extract coverage.info '*/openpfc/include/*' -o filtered_coverage.info
RUN genhtml filtered_coverage.info --output-directory coverage_report

# Generate documentation
RUN cmake --build build --target docs
