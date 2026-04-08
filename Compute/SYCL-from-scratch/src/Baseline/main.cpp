// Standard includes
#include <iostream>
#include <cstdint>
#include <cstdlib>

// Magnum includes
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Trade/MeshData.h>

// SYCL includes
// none

using namespace Magnum;

constexpr PixelFormat PIXELFORMAT{PixelFormat::RGBA8Unorm};

class TriangleExample : public Platform::Application {
    public:
        explicit TriangleExample(const Arguments&);
    
    private:
        void drawEvent() override;
        void tickEvent() override;
        void remakeTexture();

        int _w, _h;
        GL::Mesh _mesh;
        GL::Texture2D _tex;
        Shaders::FlatGL2D _shader;
};

TriangleExample::TriangleExample(const Arguments& args) :
    Platform::Application{
        args,
        Configuration{}.setTitle("Baseline Test")
    },
    _shader{
        Shaders::FlatGL2D::Configuration{}.setFlags(
            Shaders::FlatGL2D::Flag::Textured |
            Shaders::FlatGL2D::Flag::TextureTransformation
        )
    },
    _w(windowSize().x()),
    _h(windowSize().y()),
    _mesh{
        MeshTools::compile(
        Primitives::squareSolid(
        Primitives::SquareFlag::TextureCoordinates))
    }
{
    remakeTexture();
}

void TriangleExample::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    _shader.draw(_mesh);
    redraw();
    swapBuffers();
}

uint64_t rando(uint64_t t) {
    t ^= 0x9fb2b4c97fa7b4c3ull;

    t = t * (t ^ (t >> 3));
    t ^= 0x74c9b4c3fb2b7fa7ull;
    t = t * (t ^ (t >> 7));

    return t;
}

#define INFREQ 8
void TriangleExample::tickEvent() {
    static uint64_t t = 0;
    t += 1;

    uint64_t r = rando(t);
    if (r % (1ul << INFREQ)) return;

    struct Pixel { uint8_t r,g,b,a; };

    r = rando(r);
    Pixel p = {
        (uint8_t)(r & 0x7F),
        (uint8_t)((r >> 8) & 0x7F),
        (uint8_t)((r >> 16) & 0x7F),
        255
    };

    std::cout << (int)p.r << ", " << (int)p.g << ", " << (int)p.b << ", " << (int)p.a << std::endl;

    Pixel* data = (Pixel*)malloc(sizeof(Pixel)*_h*_w);

    // Could use a memset but eh
    for (int i = 0; i < _h*_w; ++i) {
        data[i] = p;
    };

    ImageView2D img{
        PIXELFORMAT,
        {_w, _h},
        Corrade::Containers::ArrayView{
            reinterpret_cast<const char*>(data),
            _w * _h * pixelFormatSize(PIXELFORMAT)
        }
    };

    _tex.setSubImage(0, {0,0}, img);

    free(data);
}

void TriangleExample::remakeTexture() {
    _tex.setWrapping(GL::SamplerWrapping::ClampToEdge)
        .setMagnificationFilter(GL::SamplerFilter::Nearest)
        .setMinificationFilter(GL::SamplerFilter::Nearest)
        .setStorage(
            1, 
            GL::textureFormat(PIXELFORMAT),
            {_w, _h}
        );

    _shader.bindTexture(_tex);
}

MAGNUM_APPLICATION_MAIN(TriangleExample);