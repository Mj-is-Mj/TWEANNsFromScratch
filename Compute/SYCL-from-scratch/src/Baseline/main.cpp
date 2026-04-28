// Standard includes
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <thread>

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
#include "noise.hpp"

using namespace Magnum;

constexpr PixelFormat PIXELFORMAT{PixelFormat::RGBA8Unorm};

class PerlinNoiseApp : public Platform::Application {
    public:
        explicit PerlinNoiseApp(const Arguments&);
    
    private:
        void drawEvent() override;
        void tickEvent() override;
        void remakeTexture();
        void keyPressEvent(KeyEvent& event) override;

        const int _w, _h;
        GL::Mesh _mesh;
        GL::Texture2D _tex;
        Shaders::FlatGL2D _shader;
        NoiseEnv noise;
        bool paused = true;
};

PerlinNoiseApp::PerlinNoiseApp(const Arguments& args) :
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
    },
    noise(_h, _w, 48)
{
    // setWindowSize(Vector2i(_w,_h));
    remakeTexture();
}

void PerlinNoiseApp::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    auto data = noise.render();

    ImageView2D img{
        PIXELFORMAT,
        {_w, _h},
        Corrade::Containers::ArrayView{
            data,
            _w * _h * pixelFormatSize(PIXELFORMAT)
        }
    };

    _tex.setSubImage(0, {0,0}, img);


    _shader.draw(_mesh);
    redraw();
    swapBuffers();

    // ~60fps
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
}

void PerlinNoiseApp::tickEvent() {
    if (paused) { return; }
    noise.step();
}

void PerlinNoiseApp::remakeTexture() {
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

void PerlinNoiseApp::keyPressEvent(KeyEvent& event) {
    // (Un)pause on SPACE
    if (event.key() == Sdl2Application::Key::Space) {
        std::cout << "PAUSED: " << paused << std::endl;
        paused = !paused;
    }
  }

MAGNUM_APPLICATION_MAIN(PerlinNoiseApp);