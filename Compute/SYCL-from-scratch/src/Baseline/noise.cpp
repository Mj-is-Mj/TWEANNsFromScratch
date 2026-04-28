#include "noise.hpp"

#include <random>
using sycl::access::mode;


NoiseEnv::NoiseEnv(size_t width, size_t height, size_t vgc)
    : step_num(0),
    grid_stride(vgc),
    vg_width((width)/vgc + 1),
    vg_height((height)/vgc + 1),
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
            w_racc[item] = 0.f;
            w_rvel[item] = 0.f;

            size_t s = 411*item.get_id(0) + 3241*item.get_id(1) + 33;
            s ^= size_t(0xAF63D2910);
            s ^= s << 17;
            s ^= s >> 13;
            s ^= s << 5;
            float r = float(s & 0xFFFF) / float(0x8000); // [0,2]
            r = (r-1.f) * 0.5f;
            float j = s & 0x10000 ? -1.f : 1.f;

            // Pseudorandom inital vector
            w_vec[item]  = sycl::float2(
                r, 
                j * sycl::sqrt(1.f - r*r)
            );
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
    step_num += 1;
    
    q.submit([&](sycl::handler& cgh) {
        auto r_racc  = racc_grid.get_access<mode::read>(cgh);
        auto rw_rvel = rvel_grid.get_access<mode::read_write>(cgh);
        auto rw_vec  = vec_grid.get_access<mode::read_write>(cgh);

        // Avoid capturing `this`
        size_t grid_width  = vg_width;
        size_t grid_height = vg_height;

        // Vector-grid operations
        cgh.parallel_for(
            sycl::range<2>(grid_width, grid_height),
            [=](sycl::item<2> item) 
        {
            // Get rotational acceleration vector
            const float racc = r_racc[item];
            // Get rotational velocity vector
            float rvel  = rw_rvel[item];
            // Rotate further
            rvel += racc;
            // Limit
            rvel /= (1.f + VLS);
            // Reassign
            rw_rvel[item] = rvel;

            // Compute rotation values
            const float rcos = sycl::cos(rvel);
            const float rsin = sycl::sin(rvel);


            // Get main grid vectors
            sycl::float2 mvec = rw_vec[item];
            // Rotate
            mvec = sycl::float2(
                mvec.x()*rcos - mvec.y()*rsin,
                mvec.x()*rsin + mvec.y()*rcos
            );


            // Normalize for good measure
            mvec /= mvec.x()*mvec.x() + mvec.y()*mvec.y();
            // Reassign
            rw_vec[item] = mvec;
        });

    });

    if (step_num % 64 == 0) randomize();
}

void NoiseEnv::randomize() {
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
            theta /= float(0x8000); // Scale to [0,2)
            theta -= 1;
            theta *= MAX_THETA;

            // Turn random angle direction vector
            w_racc[item] = theta;
        });
    });
}

const char* NoiseEnv::render() {
    q.submit([&](sycl::handler& cgh) {
        auto r_vec = vec_grid.get_access<mode::read>(cgh);
        auto w_img = img.write().get_access<mode::write>(cgh);

        // Avoid capturing `this`
        const size_t grid_width   = vg_width;
        const size_t grid_height  = vg_height;
        const size_t image_width  = im_width;
        const size_t image_height = im_height;
        const ssize_t gstride      = grid_stride;
        const float inv_stride    = 1.f / float(gstride);

        cgh.parallel_for(
            sycl::range<2>(image_width, image_height),
            [=](sycl::item<2> item)
        {
            ssize_t ix = item.get_id(0);
            ssize_t iy = item.get_id(1);

            ssize_t gx0 = ix / gstride;
            ssize_t gy0 = iy / gstride;
            ssize_t gx1 = gx0+1;
            ssize_t gy1 = gy0+1;
            
            // Get values for offset vectors : float (-1,1)
            float xoff0 = float(ix - gstride*gx0)*inv_stride;
            float xoff1 = float(ix - gstride*gx1)*inv_stride;
            float yoff0 = float(iy - gstride*gy0)*inv_stride;
            float yoff1 = float(iy - gstride*gy1)*inv_stride;

            // Get gradient vectors : normalized vectors <float,float>
            auto g00 = r_vec[sycl::id<2>(gx0, gy0)];
            auto g01 = r_vec[sycl::id<2>(gx0, gy1)];
            auto g10 = r_vec[sycl::id<2>(gx1, gy0)];
            auto g11 = r_vec[sycl::id<2>(gx1, gy1)];

            // Get dot products : float [-2,2]
            float d00 = g00.x()*xoff0 + g00.y()*yoff0;
            float d01 = g01.x()*xoff0 + g01.y()*yoff1;
            float d10 = g10.x()*xoff1 + g10.y()*yoff0;
            float d11 = g11.x()*xoff1 + g11.y()*yoff1;

            // Testing
            size_t s; float r;

            // Weights for dot products : Smooth step function f(x) = 3x^2 - 2x^3
            float lowx = xoff0*xoff0*(3 - 2*xoff0);
            float lowy = yoff0*yoff0*(3 - 2*yoff0);
            float highx = 1.f-lowx;
            float highy = 1.f-lowy;

            // Interpolate dot products using weights given by smooth step function : float [-2,2]
            float hue = (
                d00*(highx*highy) + 
                d01*(highx*lowy ) +
                d10*(lowx *highy) +
                d11*(lowx *lowy )
            );

            hue += 2.f;
            // hue *= 0.25f; // Rescale hue
            hue -= int(hue); // Get fractional portion

            size_t on = 1; //(ix0 + iy0) % 2;

            w_img[item] = sycl::uchar4(
                static_cast<unsigned char>(on*255*(0.5f + 0.5f*sycl::cos(PIPI*hue))),
                static_cast<unsigned char>(on*255*(0.5f + 0.5f*sycl::cos(PIPI*(hue - (1.f/3.f))))),
                static_cast<unsigned char>(on*255*(0.5f + 0.5f*sycl::cos(PIPI*(hue - (2.f/3.f))))),
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

const char* NoiseEnv::debugRender() {
    q.submit([&](sycl::handler& cgh) {
        auto r_input = vec_grid.get_access<mode::read>(cgh);
        auto w_img   = img.write().get_access<mode::write>(cgh);

        const size_t image_width  = im_width;
        const size_t image_height = im_height;

        const auto grange = r_input.get_range();
        const size_t gwidth  = grange.get(0);
        const size_t gheight = grange.get(1);

        const size_t xstride = ( (image_width -1) / gwidth  ) + 1;
        const size_t ystride = ( (image_height-1) / gheight ) + 1;

        cgh.parallel_for(
            sycl::range<2>(image_width, image_height),
            [=](sycl::item<2> item)
        {
            size_t ix = item.get_id(0);
            size_t iy = item.get_id(1);

            size_t gx = ix / xstride;
            size_t gy = iy / ystride;

            sycl::float2 g = r_input[sycl::id<2>(gx,gy)];
            // g *= INV_PIPI;
            // g += 0.5f;
            // g -= sycl::floor(g);

            // size_t on = ((ix/xstride) + (iy/ystride)) % 2;

            float vx = (g.x()*0.5f) + 0.5f; 
            float vy = (g.y()*0.5f) + 0.5f;

            w_img[item] = sycl::uchar4(
                static_cast<unsigned char>(255*(vx)),
                static_cast<unsigned char>(255*(vy)),
                static_cast<unsigned char>(255*(0)),
                static_cast<unsigned char>(255*(1))
            );
        });
        
    });

    img.swap();

    auto acc = img.read().get_host_access(sycl::read_only);
    return reinterpret_cast<const char*>(
        acc.get_pointer()
    );
}