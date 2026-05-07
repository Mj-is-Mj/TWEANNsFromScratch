#include "cell.hpp"
#include "sycl_util.hpp"
#include "util.hpp"
using sycl::access::mode;

CellEnv::CellEnv(size_t width, size_t height, size_t search_radius, AdvancedParams_t adv_params) 
    : width(width),
    height(height),
    radius(search_radius),
    steps(0),
    adv_params(adv_params),
    life_img(sycl::range<2>(width,height)),
    out_img(sycl::range<2>(width,height)),
    q(sycl::default_selector_v, Q_LAMBDA)
{
    cleanup();
}

void CellEnv::cleanup() {
    q.submit([&](sycl::handler& cgh) {
        auto w_life_0 = life_img.read().get_access<mode::write>(cgh);
        auto w_life_1 = life_img.write().get_access<mode::write>(cgh);
        auto w_out_im = out_img.get_access<mode::write>(cgh);

        // Avoid capturing `this`
        const size_t im_width  = width;
        const size_t im_height = height;

        cgh.parallel_for(
            sycl::range<2>(im_width,im_height),
            [=](sycl::item<2> item)
        {
            w_life_0[item] = sycl::float3(0.f);
            w_life_1[item] = sycl::float3(0.f);
            w_out_im[item] = sycl::uchar4(0);
        });
    });
}

void CellEnv::drawTo(size_t x, size_t y, size_t w, size_t h, float color[3]) {
    q.submit([&](sycl::handler& cgh) {
        auto w_life_0 = life_img.read().get_access<mode::write>(cgh);
        auto w_life_1 = life_img.write().get_access<mode::write>(cgh);

        // Avoid capturing `this`
        const size_t im_width  = width;
        const size_t im_height = height;

        const size_t x0 = MIN(x,im_width -1);
        const size_t y0 = MIN(y,im_height-1);
        const size_t x1 = MIN(x+w,im_width -1);
        const size_t y1 = MIN(y+h,im_height-1);
        const size_t dx = x1-x0;
        const size_t dy = y1-y0;

        float f0 = MAX(adv_params.LIFE_MIN, MIN(adv_params.LIFE_MAX, color[0]));
        float f1 = MAX(adv_params.LIFE_MIN, MIN(adv_params.LIFE_MAX, color[1]));
        float f2 = MAX(adv_params.LIFE_MIN, MIN(adv_params.LIFE_MAX, color[2]));

        // Construct color fill value
        const sycl::float3 COLOR_F3 = sycl::float3(
            f0,
            f1,
            f2
        );

        // Fill color value in both images (technically only need one, but just in case)
        cgh.parallel_for(
            sycl::range<2>(dx,dy),
            [=](sycl::item<2> item)
        {
            const sycl::id<2> offset = sycl::id<2>(
                item.get_id(0) + x0,
                item.get_id(1) + y0
            );

            w_life_0[offset] = COLOR_F3;
            w_life_1[offset] = COLOR_F3;
        });
    });
}

#define scalar(v, a, b) (sycl::min(b, sycl::max(v, a)) - a) / (b-a)

inline static float singleGOL(float life, float near_life, const AdvancedParams_t ap) {
    if (life > ap.LIFE_THRESH) {
        life -= ap.AGE_RATE;
        life = near_life * ap.SPREAD + life *(1.f - ap.SPREAD);
        return life;

        // if (near_life < ap.SAFE_LOWER) {
        //     return life * scalar(life, ap.LIFE_MIN, ap.SAFE_LOWER);
        // }
        // else if (near_life > ap.SAFE_HIGHER) {
        //     return life * (1.f - scalar(life, ap.SAFE_HIGHER, ap.LIFE_MAX));
        // }
        // else {
        //     return life + ap.SAFE_INC;
        // }
    }
    else {
        // if (near_life > ap.REVIVE_LOW && near_life < ap.REVIVE_HIGH) {
        //     if (near_life < ap.REVIVE_MID) {
        //         return ap.REV_RATE * scalar(life, ap.REVIVE_LOW, ap.REVIVE_MID);
        //     }
        //     else {
        //         return ap.REV_RATE * (1.f - scalar(life, ap.REVIVE_MID, ap.REVIVE_HIGH));
        //     }
        // }
        // else {
        //     return 0.f;
        // }
    }
}

#define getPiP(i, j, c, w, h, buff) \
    ( \
        buff[sycl::id<2>( \
            (c.get_id(0) + i) % w, \
            (c.get_id(1) + j) % h \
        )] + \
        buff[sycl::id<2>( \
            (c.get_id(0) + w - i) % w, \
            (c.get_id(1) + h - j) % h \
        )] \
    ) \

void CellEnv::step() {
    typedef ssize_t ind_t; // Type used for indicies within the kernel

    q.submit([&](sycl::handler& cgh) {
        auto r_life = life_img.read().get_access<mode::read>(cgh);
        auto w_life = life_img.write().get_access<mode::write>(cgh);

        // Avoid capturing `this`
        const ind_t im_width  = width;
        const ind_t im_height = height;
        const ind_t rad = radius;
        const AdvancedParams_t ap = adv_params;

        cgh.parallel_for(
            sycl::range<2>(im_width, im_height),
            [=](sycl::item<2> item)
        {
            const sycl::float3 my_life = r_life[item];
            sycl::float3 near_life(0.f, 0.f, 0.f);

            // Sum all nearby life
            for (int r = 1; r <= rad; ++r) {
                // Get cells along local x/y axis
                near_life += getPiP(r, 0, item, im_width, im_height, r_life);
                near_life += getPiP(0, r, item, im_width, im_height, r_life);

                // Get cells at top/bottom of perimeter
                for (int i = 1; i <= rad; ++i) {
                    near_life += getPiP( i, r, item, im_width, im_height, r_life);
                    near_life += getPiP(-i, r, item, im_width, im_height, r_life);
                }

                // Get cells at sides of perimeter
                for (int j = 1; j < rad; ++j) {
                    near_life += getPiP(r,  j, item, im_width, im_height, r_life);
                    near_life += getPiP(r, -j, item, im_width, im_height, r_life);
                }
            }

            // Divide by number of local cells
            near_life = near_life / sycl::pow(2.f*rad+1, 2);

            w_life[item] = sycl::float3(
                singleGOL(my_life.x(), near_life.y(), ap),
                singleGOL(my_life.y(), near_life.z(), ap),
                singleGOL(my_life.z(), near_life.x(), ap)
            );
        });
    });

    life_img.swap();
    ++steps;
}

const uint8_t* const CellEnv::render() {
q.submit([&](sycl::handler& cgh) {
        auto r_life_f = life_img.read().get_access<mode::read>(cgh);
        auto w_out_im = out_img.get_access<mode::write>(cgh);

        // Avoid capturing `this`
        const size_t im_width  = width;
        const size_t im_height = height;

        // Fill color value in both images (technically only need one, but just in case)
        cgh.parallel_for(
            sycl::range<2>(im_width, im_height),
            [=](sycl::item<2> item)
        {
            const sycl::float3 lf = r_life_f[item];

            w_out_im[item] = sycl::uchar4(
                static_cast<unsigned char>(255.f*lf.x()),
                static_cast<unsigned char>(255.f*lf.y()),
                static_cast<unsigned char>(255.f*lf.z()),
                '\255'
            );
        });
    });

    auto acc = out_img.get_host_access(sycl::read_only);
    return reinterpret_cast<const uint8_t* const>(
        acc.get_pointer()
    );
}