#include "noise.hpp"

#include <random>
using sycl::access::mode;


NoiseEnv::NoiseEnv(size_t width, size_t height, size_t vgc)
    : step_num(0),
    grid_stride(vgc),
    vg_width(1 + vgc),
    vg_height(1 + ((height+1)*vgc-1)/width),
    im_width(width),
    im_height(height),
    racc_grid(sycl::range<2>(vg_width, vg_height)),
    rvel_grid(sycl::range<2>(vg_width, vg_height)),
    vec_grid(sycl::range<2>(vg_width, vg_height)),
    img(sycl::range<2>(width, height)),
    q(sycl::default_selector_v, Q_LAMBDA)
{
    q.submit([&](sycl::handler& cgh) {

        auto w_racc = racc_grid.get_access<mode::write>(cgh);
        auto w_rvel = rvel_grid.get_access<mode::write>(cgh);
        auto w_vec  = vec_grid.get_access<mode::write>(cgh);

        // Avoid capturing `this`
        auto grid_width   = vg_width;
        auto grid_height  = vg_height;

        // Clear vector and matrix grids
        cgh.parallel_for(
            sycl::range<2>(grid_width, grid_height),
            [=](sycl::item<2> item)
        {
            w_racc[item] = sycl::float2(1.f, 0.f); // Equiv. to identity rotation
            w_rvel[item] = sycl::float2(1.f, 0.f); // Equiv. to identity rotation
            w_vec[item]  = sycl::float2(1.f, 0.f); // X axis
        });
    });


    q.submit([&](sycl::handler& cgh) {
        auto w_img  = img.write().get_access<mode::write>(cgh);

        // Avoid capturing `this`
        auto image_width  = im_width;
        auto image_height = im_height;

        // Clear output image
        cgh.parallel_for(
            sycl::range<2>(image_width, image_height),
            [=](sycl::item<2> item)
        {
            w_img[item] = sycl::uchar4{0};
        });
    });

    img.swap();

    randomize();
}

void NoiseEnv::step() {
    
    q.submit([&](sycl::handler& cgh) {
        auto r_racc  = racc_grid.get_access<mode::read>(cgh);
        auto rw_rvel = rvel_grid.get_access<mode::read_write>(cgh);
        auto rw_vec  = vec_grid.get_access<mode::read_write>(cgh);

        // Avoid capturing `this`
        auto grid_width   = vg_width;
        auto grid_height  = vg_height;

        // Vector-grid operations
        cgh.parallel_for(
            sycl::range<2>(grid_width, grid_height),
            [=](sycl::item<2> item) 
        {
            // Get rotated velocity vector
            sycl::float2 rvel  = rw_rvel[item];
            rvel = sycl::float2(
                rvel.x() * r_racc[item].x()
                 - rvel.y() * r_racc[item].y(),
                rvel.x() * r_racc[item].y()
                 + rvel.y() * r_racc[item].x()
            );

            // Weight velocity towards 0deg
            rvel += sycl::float2(VLS, 0.f);

            // Normalize and assign
            rvel /= rvel.x()*rvel.x() + rvel.y()*rvel.y();
            rw_rvel[item] = rvel;


            // Rotate main grid vectors
            sycl::float2 mvec = rw_vec[item];
            mvec = sycl::float2(
                mvec.x() * rw_rvel[item].x()
                 - mvec.y() * rw_rvel[item].y(),
                mvec.x() * rw_rvel[item].y()
                 + mvec.y() * rw_rvel[item].x()
            );

            // Normalize for good measure
            mvec /= mvec.x()*mvec.x() + mvec.y()*mvec.y();
            rw_vec[item] = mvec;
        });

    });

    if (step_num % 8 == 0) randomize();
}

void NoiseEnv::randomize() {
    step_num += 1;

    // Semi-random initial value
    size_t s = step_num + 17;
    s *= 35317; // large prime number

    // Shuffle
    s ^= (s << 13);
    s ^= (s >> 17);
    s ^= (s <<  5);

    q.submit([&](sycl::handler& cgh) {
        auto w_racc = racc_grid.get_access<mode::write>(cgh);

        cgh.parallel_for(
            sycl::range<2>(vg_width, vg_height),
            [=](sycl::item<2> item)
        {
            // Add locational influence to random seed
            size_t r = s;
            r += 1721*item.get_id(0);
            r += 7919*item.get_id(1);

            // Shuffle
            r ^= (r << 7);
            r ^= (r >> 5);
            r ^= (r << 3);

            // Random integer float using 16 bits of `r`
            float theta = float(r & 0xFFFF);
            theta /= float(0xFFFF); // Scale to [0,1)
            theta *= 2.f*3.1415926535f; // Scale to [0,2pi]

            // Turn random angle direction vector
            w_racc[item] = sycl::float2(
                sycl::cos(theta),
                sycl::sin(theta)
            );
        });
    });
}

const char* NoiseEnv::render() {
    q.submit([&](sycl::handler& cgh) {
        auto r_vec = vec_grid.get_access<mode::read>(cgh);
        auto w_img = img.write().get_access<mode::write>(cgh);

        // Avoid capturing `this`
        auto grid_width   = vg_width;
        auto grid_height  = vg_height;
        auto image_width  = im_width;
        auto image_height = im_height;
        auto gstride      = grid_stride;
        constexpr float PI = 3.1415926535f;
        constexpr float PIPI = 2*PI;

        cgh.parallel_for(
            sycl::range<2>(image_width, image_height),
            [=](sycl::item<2> item)
        {
            size_t ix0 = item.get_id(0) / 12;
            size_t iy0 = item.get_id(1) / 12;
            size_t ix1 = ix0+1;
            size_t iy1 = iy0+1;
            
            // Get values for offset vectors [0,11]
            float xoff0 = item.get_id(0) - 12*ix0;
            float xoff1 = item.get_id(0) - 12*ix1;
            float yoff0 = item.get_id(1) - 12*iy0;
            float yoff1 = item.get_id(1) - 12*iy1;

            // Get gradient vectors
            auto g00 = r_vec[sycl::id<2>(ix0, iy0)];
            auto g01 = r_vec[sycl::id<2>(ix0, iy1)];
            auto g10 = r_vec[sycl::id<2>(ix1, iy0)];
            auto g11 = r_vec[sycl::id<2>(ix1, iy1)];

            // Get dot products
            float d00 = g00.x()*xoff0 + g00.y()*yoff0;
            float d01 = g01.x()*xoff0 + g01.y()*yoff1;
            float d10 = g10.x()*xoff1 + g10.y()*yoff0;
            float d11 = g11.x()*xoff1 + g11.y()*yoff1;

            float lowx = xoff0 * (1.f/12.f);
            float lowy = yoff0 * (1.f/12.f);

            // Smooth step function 3x^2 - 2x^3
            lowx = lowx*lowx*(3 - 2*lowx);
            lowy = lowy*lowy*(3 - 2*lowy);

            float highx = 1.f-lowx;
            float highy = 1.f-lowy;

            // Interpolate dot products using weights given by smooth step function
            float hue = (
                d00*(lowx *lowy ) + 
                d01*(lowx *highy) +
                d10*(highx*lowy ) +
                d11*(highx*highy)
            );

            hue *= 0.1f; // Rescale dot
            hue -= int(hue); // Get fractional portion

            w_img[item] = sycl::uchar4(
                static_cast<unsigned char>(255*(0.5f + 0.5f*sycl::cos(PIPI*hue))),
                static_cast<unsigned char>(255*(0.5f + 0.5f*sycl::cos(PIPI*(hue - (1.f/3.f))))),
                static_cast<unsigned char>(255*(0.5f + 0.5f*sycl::cos(PIPI*(hue - (2.f/3.f))))),
                static_cast<unsigned char>(255*0)
            );
        });
    });

    img.swap();

    auto acc = img.read().get_host_access(sycl::read_only);
    return reinterpret_cast<const char*>(
        acc.get_pointer()
    );
}

