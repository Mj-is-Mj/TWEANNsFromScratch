#pragma once

#include "doublebuf.hpp"

#include <sycl/sycl.hpp>


#define PIPI        2.f*3.1415926535f
#define INV_PIPI    1.f / PIPI

class NoiseEnv {

    public:
        const size_t spr; // Steps per randomize
        const float max_theta;
        const float vls; // Velocity limiter strength

        const size_t vg_width, vg_height;
        const size_t im_width, im_height;
        const size_t grid_stride;
        size_t step_num;

        NoiseEnv(size_t width, size_t height, size_t vgc=12, size_t steps_per_rando=12, float max_angle=0.01f*PIPI, float vel_limiter=8.f);

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