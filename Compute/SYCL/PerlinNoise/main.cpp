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

// Local includes
#include "noise.hpp"
#include "util.hpp"

static constexpr size_t sclTex(float t) {
    return size_t(floor(t*0.5f));
}

// #define COUNT_RENDER_AS_STEP // For measuring performance: Just rerender redundantly for each step

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
        int steps = 0;
        int frames = 0;

        int steps_per_frame = 1;
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
    _w(800),
    _h(800),
    _mesh{
        MeshTools::compile(
        Primitives::squareSolid(
        Primitives::SquareFlag::TextureCoordinates))
    },
    noise(_h, _w, 32)
{
    setWindowSize(Vector2i(_w,_h));
    GL::defaultFramebuffer.setViewport({{0,0},{_w,_h}});
    remakeTexture();
}


void PerlinNoiseApp::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    #ifdef COUNT_RENDER_AS_STEP
        for (int i = 1; i < steps_per_frame; ++i)
            auto begone = noise.render();
    #endif
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
    for (int i = 0; i < steps_per_frame; ++i) {
        noise.step();
        ++steps;
    }
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
    switch (event.key()) {
        case Sdl2Application::Key::Space:
            std::cout << "PAUSED: " << paused << std::endl;
            paused = !paused;
            break;
        case Sdl2Application::Key::Up:
            steps_per_frame += 1 + (steps_per_frame >> 4);
            break;
        case Sdl2Application::Key::Down:
            steps_per_frame -= 1 + (steps_per_frame >> 4);
            steps_per_frame = MAX(steps_per_frame,1);
            break;
        default:
            break;
    }
}

MAGNUM_APPLICATION_MAIN(PerlinNoiseApp);