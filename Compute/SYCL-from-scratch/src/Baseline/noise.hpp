#pragma once

#include "../include/doublebuf.hpp"

#include <sycl/sycl.hpp>

static auto constexpr Q_LAMBDA = [](sycl::exception_list el) {
    for (auto e : el) {
        try {
            std::rethrow_exception(e);
        } catch (std::exception const& e) {
            std::cout << "Caught SYCL exception:\n" << e.what() << std::endl;
        }
    }
};

class NoiseEnv {

    public:
        static constexpr float PIPI = 2.f*3.1415926535f;
        static constexpr float INV_PIPI = 1.f / PIPI;

        static constexpr size_t STEPS_PER_RANDO = 12;
        static constexpr float MAX_THETA = 0.01f*PIPI / float(STEPS_PER_RANDO);
        static constexpr float VLS = MAX_THETA*8.f; // Velocity limiter strength

        const size_t vg_width, vg_height;
        const size_t im_width, im_height;
        const size_t grid_stride;
        size_t step_num;

        NoiseEnv(size_t width, size_t height, size_t vgc=12);

        // Updates the vector grid, updating perlin noise
        void step();

        // Get random accelerations
        void randomize(); 

        // Renders image using perlin noise
        const char* render();

        // Debug rendering
        const char* debugRender();

        // Returns a pointer to the image data
        const char* getImg(size_t* s=nullptr);

    private:
        sycl::buffer<float, 2> racc_grid; // Determines the acceleration of rotation
        sycl::buffer<float, 2> rvel_grid; // Determines the velocity of rotation

        sycl::buffer<sycl::float2, 2> vec_grid; // Perlin noise vectors to be rotated
        DoubleBuf<sycl::buffer<sycl::uchar4, 2>> img; // Final output image

        sycl::queue q;
};