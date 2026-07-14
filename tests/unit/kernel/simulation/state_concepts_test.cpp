// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <array>

#include <openpfc/kernel/simulation/state_concepts.hpp>

using namespace pfc::field;

// Mock field types for concept testing

// Mock field satisfying FieldReadable and FieldWritable (similar to LocalField<T>, PaddedBrick<T>)
template <typename T>
struct MockField {
    using value_type = T;
    std::vector<T> data_;
    std::size_t nx_, ny_, nz_;

    MockField(std::size_t nx, std::size_t ny, std::size_t nz)
        : data_(nx * ny * nz), nx_(nx), ny_(ny), nz_(nz) {}

    std::size_t size() const noexcept { return data_.size(); }
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    
    T& operator()(int i, int j, int k) noexcept 
    { 
        std::size_t idx = i + j * nx_ + k * nx_ * ny_;
        return data_[idx];
    }
    
    const T& operator()(int i, int j, int k) const noexcept 
    { 
        std::size_t idx = i + j * nx_ + k * nx_ * ny_;
        return data_[idx];
    }
};

// Mock field satisfying only FieldReadable (const access only)
template <typename T>
struct MockConstField {
    using value_type = T;
    std::vector<T> data_;
    std::size_t nx_, ny_, nz_;

    MockConstField(std::size_t nx, std::size_t ny, std::size_t nz)
        : data_(nx * ny * nz), nx_(nx), ny_(ny), nz_(nz) {}

    std::size_t size() const noexcept { return data_.size(); }
    const T* data() const noexcept { return data_.data(); }
    
    const T& operator()(int i, int j, int k) const noexcept 
    { 
        std::size_t idx = i + j * nx_ + k * nx_ * ny_;
        return data_[idx];
    }
};

// Mock field with wrong operator() signature (does not satisfy FieldReadable/FieldWritable)
template <typename T>
struct MockIncompleteField {
    using value_type = T;
    std::vector<T> data_;

    MockIncompleteField(std::size_t size) : data_(size) {}

    std::size_t size() const noexcept { return data_.size(); }
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    // Missing operator()(int,int,int)
};

// Mock field const-only with wrong data() return (does not satisfy FieldReadable)
template <typename T>
struct MockBadConstField {
    using value_type = T;
    std::vector<T> data_;

    MockBadConstField(std::size_t size) : data_(size) {}

    std::size_t size() const noexcept { return data_.size(); }
    // Non-const data() - cannot be called on const reference
    T* data() { return data_.data(); }
    
    const T& operator()(int i, int j, int k) const noexcept 
    { 
        return data_[0]; // Simplified
    }
};

// Mock field similar to Array<T,D> - uses get_size() instead of size()
template <typename T, int D>
struct MockArrayField {
    using value_type = T;
    std::vector<T> data_;
    std::array<int, D> dims_;

    MockArrayField(const std::array<int, D>& dims) 
        : data_(dims[0] * dims[1] * dims[2]), dims_(dims) {}

    std::array<int, D> get_size() const noexcept { return dims_; }
    std::vector<T>& get_data() { return data_; }
    
    T& operator()(const std::array<int, D>& idx) noexcept { return data_[0]; }
    // Missing size(), operator()(int,int,int)
};

TEST_CASE("FieldReadable - interface requirements", "[state_concepts][unit]") {
    SECTION("MockField satisfies FieldReadable") {
        REQUIRE(FieldReadable<MockField<double>>);
        REQUIRE(FieldReadable<const MockField<double>>);
    }

    SECTION("MockConstField satisfies FieldReadable") {
        REQUIRE(FieldReadable<MockConstField<double>>);
        REQUIRE(FieldReadable<const MockConstField<double>>);
    }

    SECTION("MockIncompleteField does NOT satisfy FieldReadable") {
        REQUIRE_FALSE(FieldReadable<MockIncompleteField<double>>);
    }

    SECTION("MockBadConstField does NOT satisfy FieldReadable") {
        REQUIRE_FALSE(FieldReadable<MockBadConstField<double>>);
    }

    SECTION("std::vector does NOT satisfy FieldReadable (missing 3D operator())") {
        REQUIRE_FALSE(FieldReadable<std::vector<double>>);
    }

    SECTION("MockArrayField does NOT satisfy FieldReadable (different API)") {
        REQUIRE_FALSE(FieldReadable<MockArrayField<double, 3>>);
    }
}

TEST_CASE("FieldWritable - interface requirements", "[state_concepts][unit]") {
    SECTION("MockField satisfies FieldWritable") {
        REQUIRE(FieldWritable<MockField<double>>);
        // Const reference should NOT satisfy FieldWritable
        REQUIRE_FALSE(FieldWritable<const MockField<double>>);
    }

    SECTION("MockField allows non-const access via operator()") {
        MockField<double> f(10, 10, 10);
        f(0, 0, 0) = 3.14;
        REQUIRE(f(0, 0, 0) == 3.14);
    }

    SECTION("MockConstField does NOT satisfy FieldWritable") {
        REQUIRE_FALSE(FieldWritable<MockConstField<double>>);
        REQUIRE_FALSE(FieldWritable<const MockConstField<double>>);
    }

    SECTION("MockIncompleteField does NOT satisfy FieldWritable") {
        REQUIRE_FALSE(FieldWritable<MockIncompleteField<double>>);
    }
}

TEST_CASE("FieldWritable requires void* from data()", "[state_concepts][unit]") {
    SECTION("MockField provides void* convertible data()") {
        MockField<double> f(10, 10, 10);
        void* ptr = f.data();
        REQUIRE(ptr != nullptr);
    }

    SECTION("MockField non-const data() is writable") {
        MockField<double> f(10, 10, 10);
        double* ptr = f.data();
        ptr[0] = 2.71;
        REQUIRE(f(0, 0, 0) == 2.71);
    }
}

TEST_CASE("FieldWritable requires non-const element type via operator()", "[state_concepts][unit]") {
    SECTION("MockField returns non-const references from operator()") {
        MockField<double> f(5, 5, 5);
        f(1, 2, 3) = 1.23;
        const double& via_op = f(1, 2, 3);
        REQUIRE(via_op == 1.23);
    }
}

TEST_CASE("Field - composition of readable and writable", "[state_concepts][unit]") {
    SECTION("MockField satisfies Field (both readable and writable)") {
        REQUIRE(Field<MockField<double>>);
        // Const reference should NOT satisfy Field
        REQUIRE_FALSE(Field<const MockField<double>>);
    }

    SECTION("MockConstField does NOT satisfy Field (not writable)") {
        REQUIRE_FALSE(Field<MockConstField<double>>);
    }

    SECTION("std::vector does NOT satisfy Field (missing 3D operator())") {
        REQUIRE_FALSE(Field<std::vector<double>>);
    }
}

TEST_CASE("ConstField - alias for FieldReadable", "[state_concepts][unit]") {
    SECTION("ConstField<FieldReadable types> holds") {
        REQUIRE(ConstField<MockField<double>>);
        REQUIRE(ConstField<const MockField<double>>);
        REQUIRE(ConstField<MockConstField<double>>);
        REQUIRE(ConstField<const MockConstField<double>>);
    }

    SECTION("ConstField equivalent to FieldReadable") {
        REQUIRE(ConstField<MockField<double>> == FieldReadable<MockField<double>>);
        REQUIRE(ConstField<MockConstField<double>> == FieldReadable<MockConstField<double>>);
    }
}

TEST_CASE("ShapeCompatible - field compatibility checking", "[state_concepts][unit]") {
    SECTION("Same-size MockFields with same value_type are compatible") {
        REQUIRE(ShapeCompatible<MockField<double>, MockField<double>>);
        REQUIRE(ShapeCompatible<const MockField<double>, const MockField<double>>);
    }

    SECTION("Different value_type fields are NOT compatible") {
        REQUIRE_FALSE(ShapeCompatible<MockField<double>, MockField<float>>);
        REQUIRE_FALSE(ShapeCompatible<MockField<int>, MockField<double>>);
    }

    SECTION("Const and non-const fields are compatible if types match") {
        REQUIRE(ShapeCompatible<MockField<double>, const MockField<double>>);
        REQUIRE(ShapeCompatible<const MockField<double>, MockField<double>>);
    }

    SECTION("Fields with different value_type typedefs are NOT compatible") {
        struct OtherField {
            using value_type = float; // Different type
            std::vector<float> data_;
            
            OtherField(std::size_t size) : data_(size) {}
            std::size_t size() const { return data_.size(); }
            float* data() { return data_.data(); }
            const float* data() const { return data_.data(); }
            float& operator()(int, int, int) { return data_[0]; }
            const float& operator()(int, int, int) const { return data_[0]; }
        };
        
        REQUIRE_FALSE(ShapeCompatible<MockField<double>, OtherField>);
    }

    SECTION("MockConstField and MockField are compatible for same value_type") {
        REQUIRE(ShapeCompatible<MockConstField<double>, MockField<double>>);
        REQUIRE(ShapeCompatible<MockField<double>, MockConstField<double>>);
    }
}

TEST_CASE("AliasingSafe - prevents same-type aliasing", "[state_concepts][unit]") {
    SECTION("Different types are safe from aliasing") {
        REQUIRE(AliasingSafe<MockField<double>, std::vector<double>>);
        REQUIRE(AliasingSafe<MockField<double>, MockField<float>>);
    }

    SECTION("Same-type fields are NOT safe from aliasing") {
        REQUIRE_FALSE(AliasingSafe<MockField<double>, MockField<double>>);
        REQUIRE_FALSE(AliasingSafe<MockConstField<double>, MockConstField<double>>);
    }

    SECTION("const-correctness does NOT affect aliasing detection") {
        REQUIRE_FALSE(AliasingSafe<MockField<double>, const MockField<double>>);
        REQUIRE_FALSE(AliasingSafe<const MockField<double>, MockField<double>>);
    }

    SECTION("Reference and pointer wrappers are detected as same type") {
        REQUIRE_FALSE(AliasingSafe<MockField<double>&, MockField<double>&>);
        REQUIRE_FALSE(AliasingSafe<MockField<double>*, MockField<double>*>);
    }

    SECTION("Value types and references to same type are NOT safe") {
        REQUIRE_FALSE(AliasingSafe<MockField<double>, MockField<double>&>);
        REQUIRE_FALSE(AliasingSafe<MockField<double>, const MockField<double>&>);
    }
}
