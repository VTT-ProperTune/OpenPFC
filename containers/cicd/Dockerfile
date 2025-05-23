# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

FROM ahojukka5/openpfc:base

WORKDIR /usr/src/openpfc
COPY . .

# Set environment variables (optional)
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

RUN ldconfig
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DUSE_CLANG_TIDY=OFF
RUN cmake --build build
RUN (cd build/tests && ctest)

# Generate coverage report
RUN lcov --gcov-tool gcov-12 --capture --directory build/tests --output-file coverage.info
RUN lcov --extract coverage.info '*/openpfc/include/*' -o filtered_coverage.info
RUN genhtml filtered_coverage.info --output-directory coverage_report

# Generate documentation
RUN git submodule update --init --recursive
RUN (cd docs/doxygen-awesome-css && git checkout v2.2.1)
RUN cmake --build build --target docs
