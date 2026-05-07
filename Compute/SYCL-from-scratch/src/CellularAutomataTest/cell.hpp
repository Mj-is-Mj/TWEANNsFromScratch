#pragma once

// Standard includes
#include <cstdint>

// SYCL Includes
#include <sycl/sycl.hpp>

// Local includes
#include "doublebuf.hpp"

struct AdvancedParams_t {
    float 
        SAFE_INC    = 0.1f,    // 
        AGE_RATE    = 0.0002f, // How quickly life dies out
        REV_RATE    = 0.99f,   // How quickly life repopulates

        SPREAD      = 0.001f,  // How much should life spread

        SAFE_LOWER  = 0.3f,
        SAFE_HIGHER = 0.55f,

        LIFE_MIN    = 0.0f,    // The minimum amount of life
        LIFE_MAX    = 1.0f,    // The maximum amount of life
        LIFE_THRESH = 0.0f,    // The threshold above which is considered "alive"

        REVIVE_LOW  = 0.5f,
        REVIVE_HIGH = 1.0f,
        REVIVE_MID  = 0.6f;
    
    constexpr AdvancedParams_t() = default;
};

class CellEnv {
    public:

        static constexpr AdvancedParams_t DEFAULT_PARAMS;

        CellEnv(size_t width, size_t height, size_t search_radius, AdvancedParams_t adv_params=DEFAULT_PARAMS);

        // Clears buffers
        void cleanup();

        // Draws a square of a particular color
        void drawTo(size_t x, size_t y, size_t w, size_t h, float color[3]);

        // Progress simulation
        void step();

        // Return pointer to image data
        const uint8_t* const render();

    private:
        const size_t width,height; // Width and height of buffers
        const size_t radius; // Square radius to search within (area = (2r+1)^2)
        const AdvancedParams_t adv_params;
        size_t steps;

        sycl::queue q;
        DoubleBuf<sycl::buffer<sycl::float3, 2>> life_img;
        sycl::buffer<sycl::uchar4,2> out_img;
};