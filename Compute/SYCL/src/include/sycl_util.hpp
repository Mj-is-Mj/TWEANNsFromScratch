#pragma once

#include <sycl/sycl.hpp>

auto constexpr Q_LAMBDA = [](sycl::exception_list el) {
    for (auto e : el) {
        try {
            std::rethrow_exception(e);
        } catch (std::exception const& e) {
            std::cout << "Caught SYCL exception:\n" << e.what() << std::endl;
        }
    }
};