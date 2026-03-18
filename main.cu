#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>

#define RGFW_OPENGL
#define RGFW_IMPLEMENTATION
#include <rgfw/RGFW.h>

#define RGFW_IMGUI_IMPLEMENTATION
#include <imgui/imgui.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui_impl_rgfw.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <stbi/stbi_wrapper.h>

// tiny-cuda-nn
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

static const int SCR_WIDTH = 800;
static const int SCR_HEIGHT = 600;

const char *kVS = R"(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;
uniform vec2 uScale;
void main() {
    vUV = aUV;
    vec2 p = aPos * uScale;
    gl_Position = vec4(p, 0.0, 1.0);
})";

const char *kFS = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vec2(vUV.x, 1.0 - vUV.y));
})";

static GLuint compileShader(GLenum type, const char *src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    fprintf(stderr, "shader compile failed: %s\n", log);
    exit(1);
  }
  return s;
}

static GLuint createProgram(const char *vs, const char *fs) {
  GLuint p = glCreateProgram();
  GLuint sv = compileShader(GL_VERTEX_SHADER, vs);
  GLuint sf = compileShader(GL_FRAGMENT_SHADER, fs);
  glAttachShader(p, sv);
  glAttachShader(p, sf);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    fprintf(stderr, "program link failed: %s\n", log);
    exit(1);
  }
  glDeleteShader(sv);
  glDeleteShader(sf);
  return p;
}

// ---------- CUDA kernels ----------

__device__ __forceinline__ unsigned char to_u8_gamma(float x) {
  x = fminf(fmaxf(x, 0.0f), 1.0f);
  x = powf(x, 1.0f / 2.2f);
  int v = (int)(x * 255.0f + 0.5f);
  v = v < 0 ? 0 : (v > 255 ? 255 : v);
  return (unsigned char)v;
}

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture,
                           float *__restrict__ xs_and_ys,
                           float *__restrict__ result) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elements)
    return;

  uint32_t output_idx = i * stride;
  uint32_t input_idx = i * 2;

  float4 val =
      tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);

  result[output_idx + 0] = val.x;
  result[output_idx + 1] = val.y;
  result[output_idx + 2] = val.z;

  for (uint32_t c = 3; c < stride; ++c) {
    result[output_idx + c] = 1.0f;
  }
}

__global__ void pack_rgba8_from_prediction(int w, int h,
                                           const float *__restrict__ pred,
                                           int channel_stride,
                                           uchar4 *__restrict__ out_rgba8) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int n = w * h;
  if (idx >= n)
    return;

  int base = idx * channel_stride;
  float r = pred[base + 0];
  float g = pred[base + 1];
  float b = pred[base + 2];

  uchar4 c;
  c.x = to_u8_gamma(r);
  c.y = to_u8_gamma(g);
  c.z = to_u8_gamma(b);
  c.w = 255;
  out_rgba8[idx] = c;
}

// ---------- image loading ----------
using namespace tcnn;
using precision_t = network_precision_t;

static GPUMemory<float> load_image_gpu_rgba(const std::string &filename,
                                            int &width, int &height) {
  float *host = load_stbi(&width, &height, filename.c_str());
  if (!host)
    throw std::runtime_error("load_stbi failed");
  GPUMemory<float> gpu(width * height * 4);
  gpu.copy_from_host(host);
  free(host);
  return gpu;
}

static int choose_cuda_device_for_gl_context() {
  unsigned int cuda_device_count = 0;
  int cuda_devices[8] = {};
  cudaError_t err = cudaGLGetDevices(&cuda_device_count, cuda_devices, 8,
                                     cudaGLDeviceListCurrentFrame);

  if (err == cudaSuccess && cuda_device_count > 0)
    return cuda_devices[0];

  cudaGetLastError();
  return 0;
}

int main(int argc, char **argv) {
  std::string activeImagePath;
  if (argc < 2) {
    const char *exampleImagePath = "../data/fuji.jpg";
    printf("No image specified... using example image: %s\n", exampleImagePath);
    activeImagePath = exampleImagePath;
  } else {
    activeImagePath = argv[2];
  }

  RGFW_glHints *hints = RGFW_getGlobalHints_OpenGL();
  hints->major = 3;
  hints->minor = 3;
  RGFW_setGlobalHints_OpenGL(hints);

  RGFW_windowFlags winflags = RGFW_windowAllowDND | RGFW_windowCenter |
                              RGFW_windowScaleToMonitor | RGFW_windowOpenGL;
  RGFW_window *window = RGFW_createWindow("Neural-SDF", SCR_WIDTH, SCR_HEIGHT,
                                          SCR_WIDTH, SCR_HEIGHT, winflags);
  if (window == NULL) {
    printf("Failed to create RGFW window\n");
    return -1;
  }
  RGFW_window_setExitKey(window, RGFW_escape);
  RGFW_window_makeCurrentContext_OpenGL(window);
  int glad_version = gladLoadGL((GLADloadfunc)RGFW_getProcAddress_OpenGL);
  if (glad_version == 0) {
    printf("Failed to initialize GLAD\n");
    return -1;
  }
  printf("OpenGL Version: %i.%i\n", GLAD_VERSION_MAJOR(glad_version),
         GLAD_VERSION_MINOR(glad_version));
  printf("Device: %s.\n", glGetString(GL_RENDERER));

  GLuint prog = createProgram(kVS, kFS);
  GLint uScaleLoc = glGetUniformLocation(prog, "uScale");
  GLint uTexLoc = glGetUniformLocation(prog, "uTex");

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();

  if (!ImGui_ImplRgfw_InitForOpenGL(window, true)) {
    fprintf(stderr, "ImGui_ImplRgfw_InitForOpenGL failed.\n");
    RGFW_window_close(window);
    return -1;
  }

  if (!ImGui_ImplOpenGL3_Init("#version 330")) {
    fprintf(stderr, "ImGui_ImplOpenGL3_Init failed.\n");
    ImGui_ImplRgfw_Shutdown();
    ImGui::DestroyContext();
    RGFW_window_close(window);
    return -1;
  }

  float verts[] = {
      -1.f, -1.f, 0.f, 0.f, 1.f,  -1.f, 1.f, 0.f,
      1.f,  1.f,  1.f, 1.f, -1.f, 1.f,  0.f, 1.f,
  };
  unsigned int idx[] = {0, 1, 2, 2, 3, 0};

  GLuint vao = 0, vbo = 0, ebo = 0;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));

  glBindVertexArray(0);

  int cuda_device = choose_cuda_device_for_gl_context();
  CUDA_CHECK_THROW(cudaSetDevice(cuda_device));
  printf("Using CUDA device %d for GL interop.\n", cuda_device);

  int imgW = 0;
  int imgH = 0;
  GPUMemory<float> image = load_image_gpu_rgba(activeImagePath, imgW, imgH);

  // CUDA texture object for sampling training targets.
  cudaResourceDesc resDesc{};
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = image.data();
  resDesc.res.pitch2D.desc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  resDesc.res.pitch2D.width = imgW;
  resDesc.res.pitch2D.height = imgH;
  resDesc.res.pitch2D.pitchInBytes = imgW * 4 * sizeof(float);

  cudaTextureDesc texDesc{};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = true;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;

  cudaTextureObject_t image_tex = 0;
  CUDA_CHECK_THROW(
      cudaCreateTextureObject(&image_tex, &resDesc, &texDesc, nullptr));

  // Create display texture (RGBA8).
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgW, imgH, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Single PBO for CUDA->GL interop.
  size_t frame_bytes = (size_t)imgW * (size_t)imgH * sizeof(uchar4);
  GLuint pbo = 0;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_bytes, nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  cudaGraphicsResource *cuda_pbo_res = nullptr;
  CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

  // Training setup.
  const uint32_t batch_size = 1u << 18;
  const uint32_t n_input_dims = 2;
  const uint32_t n_output_dims = 3;

  uint32_t n_coords = (uint32_t)(imgW * imgH);
  uint32_t n_coords_padded = next_multiple(n_coords, BATCH_SIZE_GRANULARITY);

  GPUMemory<float> xs_and_ys(n_coords_padded * 2);
  std::vector<float> host_xs_and_ys(n_coords_padded * 2, 0.0f);
  for (int y = 0; y < imgH; ++y) {
    for (int x = 0; x < imgW; ++x) {
      int idx2 = (y * imgW + x) * 2;
      host_xs_and_ys[idx2 + 0] = (float)(x + 0.5f) / (float)imgW;
      host_xs_and_ys[idx2 + 1] = (float)(y + 0.5f) / (float)imgH;
    }
  }
  xs_and_ys.copy_from_host(host_xs_and_ys.data());

  cudaStream_t stream = nullptr;
  CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  default_rng_t rng{1337};

  GPUMatrix<float> training_target(n_output_dims, batch_size);
  GPUMatrix<float> training_batch(n_input_dims, batch_size);

  GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
  GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims,
                                   n_coords_padded);

  json config = {
      {"loss", {{"otype", "RelativeL2"}}},
      {"optimizer",
       {
           {"otype", "Adam"},
           {"learning_rate", 1e-2},
           {"beta1", 0.9f},
           {"beta2", 0.99f},
           {"epsilon", 1e-15f},
           {"l2_reg", 1e-6f},
       }},
      {"encoding",
       {
           {"otype", "HashGrid"},
           {"n_levels", 16},
           {"n_features_per_level", 2},
           {"log2_hashmap_size", 15},
           {"base_resolution", 16},
           {"per_level_scale", 1.5f},
           {"fixed_point_pos", false},
       }},
      {"network",
       {
           {"otype", "FullyFusedMLP"},
           {"activation", "ReLU"},
           {"output_activation", "None"},
           {"n_neurons", 64},
           {"n_hidden_layers", 2},
       }},
  };

  json encoding_opts = config.value("encoding", json::object());
  json loss_opts = config.value("loss", json::object());
  json optimizer_opts = config.value("optimizer", json::object());
  json network_opts = config.value("network", json::object());

  std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
  std::shared_ptr<Optimizer<precision_t>> optimizer{
      create_optimizer<precision_t>(optimizer_opts)};
  std::shared_ptr<NetworkWithInputEncoding<precision_t>> network =
      std::make_shared<NetworkWithInputEncoding<precision_t>>(
          n_input_dims, n_output_dims, encoding_opts, network_opts);

  network->set_jit_fusion(tcnn::supports_jit_fusion());

  auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
      network, optimizer, loss);

  glUseProgram(prog);
  glUniform1i(uTexLoc, 0);

  bool running = true;
  bool training_enabled = true;
  uint32_t step = 0;
  float current_loss = 0.0f;
  uint32_t current_iteration = 0;
  constexpr int kLossHistorySize = 512;
  std::vector<float> loss_history(kLossHistorySize, 0.0f);
  int loss_history_offset = 0;
  int loss_history_count = 0;

  auto reset_training_state = [&](bool start_training) {
    loss.reset(create_loss<precision_t>(loss_opts));
    optimizer.reset(create_optimizer<precision_t>(optimizer_opts));
    network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        n_input_dims, n_output_dims, encoding_opts, network_opts);
    network->set_jit_fusion(tcnn::supports_jit_fusion());
    trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
        network, optimizer, loss);

    rng = default_rng_t{1337};
    step = 0;
    current_iteration = 0;
    current_loss = 0.0f;
    loss_history_offset = 0;
    loss_history_count = 0;
    for (int i = 0; i < kLossHistorySize; ++i) {
      loss_history[i] = 0.0f;
    }
    training_enabled = start_training;
  };

  auto reload_image = [&](const std::string &image_path) {
    cudaTextureObject_t new_image_tex = 0;
    GLuint new_pbo = 0;
    cudaGraphicsResource *new_cuda_pbo_res = nullptr;
    try {
      int new_img_w = 0;
      int new_img_h = 0;
      GPUMemory<float> new_image =
          load_image_gpu_rgba(image_path, new_img_w, new_img_h);

      cudaResourceDesc new_res_desc{};
      new_res_desc.resType = cudaResourceTypePitch2D;
      new_res_desc.res.pitch2D.devPtr = new_image.data();
      new_res_desc.res.pitch2D.desc =
          cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
      new_res_desc.res.pitch2D.width = new_img_w;
      new_res_desc.res.pitch2D.height = new_img_h;
      new_res_desc.res.pitch2D.pitchInBytes = new_img_w * 4 * sizeof(float);

      CUDA_CHECK_THROW(cudaCreateTextureObject(&new_image_tex, &new_res_desc,
                                               &texDesc, nullptr));

      size_t new_frame_bytes =
          (size_t)new_img_w * (size_t)new_img_h * sizeof(uchar4);
      glGenBuffers(1, &new_pbo);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, new_pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, new_frame_bytes, nullptr,
                   GL_STREAM_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(
          &new_cuda_pbo_res, new_pbo, cudaGraphicsRegisterFlagsWriteDiscard));

      uint32_t new_n_coords = (uint32_t)(new_img_w * new_img_h);
      uint32_t new_n_coords_padded =
          next_multiple(new_n_coords, BATCH_SIZE_GRANULARITY);
      GPUMemory<float> new_xs_and_ys(new_n_coords_padded * 2);
      std::vector<float> new_host_xs_and_ys(new_n_coords_padded * 2, 0.0f);
      for (int y = 0; y < new_img_h; ++y) {
        for (int x = 0; x < new_img_w; ++x) {
          int idx2 = (y * new_img_w + x) * 2;
          new_host_xs_and_ys[idx2 + 0] = (float)(x + 0.5f) / (float)new_img_w;
          new_host_xs_and_ys[idx2 + 1] = (float)(y + 0.5f) / (float)new_img_h;
        }
      }
      new_xs_and_ys.copy_from_host(new_host_xs_and_ys.data());

      GPUMatrix<float> new_prediction(n_output_dims, new_n_coords_padded);
      GPUMatrix<float> new_inference_batch(new_xs_and_ys.data(), n_input_dims,
                                           new_n_coords_padded);

      CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

      if (cuda_pbo_res) {
        CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(cuda_pbo_res));
      }
      if (pbo) {
        glDeleteBuffers(1, &pbo);
      }
      if (image_tex) {
        CUDA_CHECK_THROW(cudaDestroyTextureObject(image_tex));
      }

      image = std::move(new_image);
      image_tex = new_image_tex;
      imgW = new_img_w;
      imgH = new_img_h;
      frame_bytes = new_frame_bytes;
      pbo = new_pbo;
      cuda_pbo_res = new_cuda_pbo_res;
      n_coords = new_n_coords;
      n_coords_padded = new_n_coords_padded;
      xs_and_ys = std::move(new_xs_and_ys);
      prediction = std::move(new_prediction);
      inference_batch = std::move(new_inference_batch);

      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgW, imgH, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, nullptr);
      glBindTexture(GL_TEXTURE_2D, 0);

      activeImagePath = image_path;
      reset_training_state(true);
      return true;
    } catch (const std::exception &e) {
      if (new_cuda_pbo_res) {
        cudaGraphicsUnregisterResource(new_cuda_pbo_res);
      }
      if (new_pbo) {
        glDeleteBuffers(1, &new_pbo);
      }
      if (new_image_tex) {
        cudaDestroyTextureObject(new_image_tex);
      }
      return false;
    }
  };

  while (!RGFW_window_shouldClose(window) && running) {
    RGFW_event event;
    while (RGFW_window_checkEvent(window, &event)) {
      switch (event.type) {
      case RGFW_dataDrop: {
        if (event.drop.count > 0 && event.drop.files && event.drop.files[0])
          reload_image(event.drop.files[0]);
      } break;

      case RGFW_quit: {
        running = false;
      } break;

      case RGFW_keyPressed: {
        if (event.key.value == RGFW_space)
          training_enabled = !training_enabled;
      } break;
      }
    }

    if (!running) {
      break;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplRgfw_NewFrame();
    ImGui::NewFrame();

    if (training_enabled) {
      // Training on CUDA stream
      generate_random_uniform<float>(stream, rng, batch_size * n_input_dims,
                                     training_batch.data());
      linear_kernel(eval_image<n_output_dims>, 0, stream, batch_size, image_tex,
                    training_batch.data(), training_target.data());

      auto ctx =
          trainer->training_step(stream, training_batch, training_target);
      current_loss = trainer->loss(stream, *ctx);
      current_iteration = step + 1;
      loss_history[loss_history_offset] = current_loss;
      loss_history_offset = (loss_history_offset + 1) % kLossHistorySize;
      if (loss_history_count < kLossHistorySize) {
        ++loss_history_count;
      }
      ++step;
    }

    // Inference + CUDA->GL upload on CUDA stream
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_pbo_res, stream));
    void *pbo_dev_ptr = nullptr;
    size_t pbo_bytes = 0;
    CUDA_CHECK_THROW(cudaGraphicsResourceGetMappedPointer(
        &pbo_dev_ptr, &pbo_bytes, cuda_pbo_res));

    if (pbo_bytes < frame_bytes) {
      fprintf(stderr, "Mapped PBO (%zu) smaller than frame (%zu).\n", pbo_bytes,
              frame_bytes);
      CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_pbo_res, stream));
      running = false;
      break;
    }

    network->inference(stream, inference_batch, prediction);

    int n = imgW * imgH;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    pack_rgba8_from_prediction<<<blocks, threads, 0, stream>>>(
        imgW, imgH, prediction.data(), (int)n_output_dims,
        reinterpret_cast<uchar4 *>(pbo_dev_ptr));
    CUDA_CHECK_THROW(cudaGetLastError());

    CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_pbo_res, stream));

    // Set up viewport and clear color.
    int winW = window->w;
    int winH = window->h;
    if (winW <= 0)
      winW = SCR_WIDTH;
    if (winH <= 0)
      winH = SCR_HEIGHT;
    glViewport(0, 0, winW, winH);

    float winAspect = (float)winW / (float)winH;
    float imgAspect = (float)imgW / (float)imgH;

    float scaleX = 1.0f;
    float scaleY = 1.0f;
    if (winAspect > imgAspect) {
      scaleX = imgAspect / winAspect;
      scaleY = 1.0f;
    } else {
      scaleX = 1.0f;
      scaleY = winAspect / imgAspect;
    }

    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Wait for CUDA to finish writing to the PBO before rendering with it.
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    // Render
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgW, imgH, GL_RGBA,
                    GL_UNSIGNED_BYTE, (void *)0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(prog);
    glUniform2f(uScaleLoc, scaleX, scaleY);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    float loss_min = std::numeric_limits<float>::max();
    float loss_max = -std::numeric_limits<float>::max();
    for (int i = 0; i < loss_history_count; ++i) {
      loss_min = loss_history[i] < loss_min ? loss_history[i] : loss_min;
      loss_max = loss_history[i] > loss_max ? loss_history[i] : loss_max;
    }
    if (loss_history_count == 0) {
      loss_min = 0.0f;
      loss_max = 1.0f;
    }
    if (loss_max <= loss_min) {
      loss_max = loss_min + 1e-6f;
    }

    int loss_plot_offset =
        loss_history_count == kLossHistorySize ? loss_history_offset : 0;

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_Once);
    ImGui::Begin("Training");
    if (ImGui::Button(training_enabled ? "Pause Training" : "Start Training")) {
      training_enabled = !training_enabled;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart Training")) {
      reset_training_state(false);
    }
    ImGui::Text("Status: %s", training_enabled ? "Running" : "Paused");
    ImGui::Text("Iteration: %u", current_iteration);
    ImGui::Text("Loss: %.6f", current_loss);
    ImGui::PlotLines("Loss", loss_history.data(), loss_history_count,
                     loss_plot_offset, nullptr, loss_min, loss_max,
                     ImVec2(0.0f, 90.0f));
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    RGFW_window_swapBuffers_OpenGL(window);
  }

  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

  if (cuda_pbo_res) {
    CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(cuda_pbo_res));
    cuda_pbo_res = nullptr;
  }

  if (stream) {
    CUDA_CHECK_THROW(cudaStreamDestroy(stream));
    stream = nullptr;
  }

  if (image_tex) {
    CUDA_CHECK_THROW(cudaDestroyTextureObject(image_tex));
    image_tex = 0;
  }

  free_all_gpu_memory_arenas();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplRgfw_Shutdown();
  ImGui::DestroyContext();

  glDeleteTextures(1, &tex);
  if (pbo)
    glDeleteBuffers(1, &pbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteProgram(prog);

  RGFW_window_close(window);
  return 0;
}
