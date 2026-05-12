// Standard includes
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <thread>

// Magnum includes
#include <Magnum/Magnum.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/ImageView.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix3.h>

// Local includes
#include "cell.hpp"


using namespace Magnum;
using namespace Magnum::Math::Literals; // for `0x..._rgbf`

constexpr PixelFormat PIXELFORMAT{PixelFormat::RGBA8Unorm};

class CellAtApp : public Platform::Application {
    public:
        explicit CellAtApp(const Arguments&);
    
    private:
        void drawEvent() override;
        void tickEvent() override;
        void resetBuffers();
        void remakeTexture();
        void keyPressEvent(KeyEvent& event) override;

        const int _w, _h; // Window width/height
        const int  _tw, _th; // Texture width/height
        CellEnv cenv;
        GL::Mesh _mesh;
        GL::Texture2D _tex;
        Shaders::FlatGL2D _shader;
        bool paused = true;
};

CellAtApp::CellAtApp(const Arguments& args) :
    Platform::Application{
        args,
        Configuration{}.setTitle("Baseline Test")
    },
    _w(800),
    _h(_w), // Square window
    _tw(400),
    _th(_tw), // Square texture
    cenv(_th, _tw, 7),
    _shader{
        Shaders::FlatGL2D::Configuration{}.setFlags(
            Shaders::FlatGL2D::Flag::Textured |
            Shaders::FlatGL2D::Flag::TextureTransformation
        )
    },
    _mesh{
        MeshTools::compile(
        Primitives::squareSolid(
        Primitives::SquareFlag::TextureCoordinates))
    }
{
    setWindowSize(Vector2i(_w,_h));
    GL::defaultFramebuffer.setViewport({{0,0},{_w,_h}});
    remakeTexture();
    resetBuffers();
}

static int steps = 0;
static int frames = 0;

void CellAtApp::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    auto data = cenv.render();

    ImageView2D img{
        PIXELFORMAT,
        {_tw, _th},
        Corrade::Containers::ArrayView{
            data,
            _tw * _th * pixelFormatSize(PIXELFORMAT)
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

void CellAtApp::tickEvent() {
    if (paused) { return; }
    cenv.step();
    ++steps;
}

void CellAtApp::resetBuffers() {
    cenv.cleanup();

    cenv.drawTo(
        (_th-30)/4,
        (_tw-30)/3,
        30,
        30,
        (float[3]){1.f, 0.9f, 0.8f}
    );
}

void CellAtApp::remakeTexture() {
    _tex.setWrapping(GL::SamplerWrapping::ClampToEdge)
        .setMagnificationFilter(GL::SamplerFilter::Nearest)
        .setMinificationFilter(GL::SamplerFilter::Nearest)
        .setStorage(
            1, 
            GL::textureFormat(PIXELFORMAT),
            {_tw, _th}
        );

    GL::defaultFramebuffer.bind();

    _shader.bindTexture(_tex);
}

void CellAtApp::keyPressEvent(KeyEvent& event) {
    // (Un)pause on SPACE
    switch(event.key()) {
        case Sdl2Application::Key::Space:
            std::cout << "PAUSED: " << paused << std::endl;
            paused = !paused;
            break;
        case Sdl2Application::Key::Enter:
            resetBuffers();
            break;
        default:
            break;
    }
  }

MAGNUM_APPLICATION_MAIN(CellAtApp);