// https://github.khronos.org/SYCL_Reference/iface/common-reference.html#common-r

// /opt/intel/oneapi/2025.3/bin/icpx

#pragma once

#include <sycl/sycl.hpp>

namespace sycl {

#define MAT_TEMPLATE_PARAMS typename dataT, int numRows, int numCols
#define MAT_TEMPLATE        template<MAT_TEMPLATE_PARAMS>
#define MAT_GENERIC         mat<dataT, numRows, numCols>

MAT_TEMPLATE
class mat {
    public:
        using Row_t = sycl::vec<dataT, numCols>;

        auto test = sycl::vec<float, 3>(1.0f, 2.0f, 3.0f);

        // Constructors and assignment
        mat(const mat&);
        mat(mat&&);
        mat& operator=(const mat&);
        mat& operator=(mat&&);

        // construct a matrix by v_00, v_01, ... v_10, v_11, ... given v_[row][col]
        template<typename... Args>
        mat(const Args&... args) {
            static_assert(sizeof...(args) == numRows*numCols, "Wrong number of elements");
            dataT tmp[] = { args... };
            int i = 0;
            ((data[i / numCols][i++ % numCols] = args), ...);
        }

        // Destructor (default)
        // ~mat();

        // Comparisons
        friend bool operator==(const mat& l, const mat& r) {
            for (int row = 0; row < numRows; ++row)
                if (l.data[row] != r.data[row]) return false;
            return true;
        }
        friend bool operator!=(const mat& l, const mat& r) { return !(r == l); }

        // Transform vector
        sycl::vec<dataT, numRows> operator*(const sycl::vec<dataT, numCols>&);


    private:
        Row_t data[numRows];
};

MAT_TEMPLATE
MAT_GENERIC::mat(const mat& o) {
    for (int r = 0; r < numRows; ++r)
    for (int c = 0; c < numCols; ++c) {
        data[r][c] = o.data[r][c];
    }
}

MAT_TEMPLATE
MAT_GENERIC::mat(mat&& o) {
    for (int r = 0; r < numRows; ++r) {
        data[r] = o.data[r]; // Is this correct for a rvalue copy?
    }
}

MAT_TEMPLATE
MAT_GENERIC& MAT_GENERIC::operator=(const mat& o) {
    for (int r = 0; r < numRows; ++r)
    for (int c = 0; c < numCols; ++c) {
        data[r][c] = o.data[r][c];
    }
    return *this;
}

MAT_TEMPLATE
MAT_GENERIC& MAT_GENERIC::operator=(mat&& o) {
    for (int r = 0; r < numRows; ++r) {
        data[r] = o.data[r]; // Is this correct for a rvalue copy?
    }
    return *this;
}

MAT_TEMPLATE
sycl::vec<dataT, numRows> MAT_GENERIC::operator*(const sycl::vec<dataT, numCols>& v) {
    sycl::vec<dataT, numRows> out;
    for (int r = 0; r < numRows; ++r) {
        out[r] = dataT{0};
        for (int c = 0; c < numCols; ++c) {
            out[r] += data[r][c] * v[c];
        }
    }
    return out;
}


} // namespace sycl;