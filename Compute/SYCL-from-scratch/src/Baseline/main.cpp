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

static constexpr size_t sclTex(float t) {
    return size_t(floor(t*1.f));
}

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
    _w(sclTex(windowSize().x())),
    _h(sclTex(windowSize().y())),
    _mesh{
        MeshTools::compile(
        Primitives::squareSolid(
        Primitives::SquareFlag::TextureCoordinates))
    },
    noise(_h, _w, 32)
{
    // setWindowSize(Vector2i(_w,_h));
    remakeTexture();
}

static int steps = 0;
static int frames = 0;

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


    using Clock = std::chrono::steady_clock;

    static std::chrono::time_point<Clock> start=Clock::now(), end{};

    if (++frames >= 300) {
        end = Clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double frametime = duration / double(frames);
        double steptime  = duration / double(steps);
        double fps = 1000.0 / frametime;
        double sps = 1000.0 / steptime;

        printf(
            "Elapsed =%6.0lfms | Frames =%4d | Steps =%4d\n"
            "\tFrametime = %6.2lfms | FPS=%6.2lf\n"
            "\tSteptime  = %6.2lfms | SPS=%6.2lf\n",
            duration,
            frames,
            steps,
            frametime,
            fps,
            steptime,
            sps
        );

        frames = 0;
        steps = 0;
        start = end;
    }
}

void PerlinNoiseApp::tickEvent() {
    if (paused) { return; }
    noise.step();
    ++steps;
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