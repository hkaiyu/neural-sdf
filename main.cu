#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/dependencies/stbi/stb_image_write.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

static const int SCR_WIDTH = 1200;
static const int SCR_HEIGHT = 700;
static const int RENDER_W = 512;
static const int RENDER_H = 512;
static const int RENDER_N = RENDER_W * RENDER_H; // 262144, divisible by 128
static const int N_METRIC_SAMPLES =
    2048; // GT surface samples for Chamfer GT→Learned
static const int METRIC_INTERVAL = 100; // steps between metric evaluations

// ---------- GL shaders ----------

const char *kVS = R"(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;
uniform vec2 uScale;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos * uScale, 0.0, 1.0);
})";

const char *kFS = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vUV);
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

// ---------- Mesh ----------

struct Triangle {
  float3 v0, v1, v2;
  float3 n; // unit outward face normal
};

// Loads triangles from an OBJ file. Handles bare vertex indices and v/vt/vn
// formats; triangulates polygons via fan decomposition.
static bool load_obj(const char *path, std::vector<Triangle> &out,
                     float3 &bbox_min, float3 &bbox_max) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    return false;
  }

  std::vector<float3> verts;
  char line[1024];

  while (fgets(line, sizeof(line), f)) {
    if (line[0] == 'v' && line[1] == ' ') {
      float3 v;
      if (sscanf(line + 2, "%f %f %f", &v.x, &v.y, &v.z) == 3)
        verts.push_back(v);
    } else if (line[0] == 'f' && line[1] == ' ') {
      int idx[16], count = 0;
      const char *p = line + 2;
      while (*p && count < 16) {
        while (*p == ' ' || *p == '\t')
          ++p;
        if (*p == '\n' || *p == '\r' || *p == '\0')
          break;
        int vi = 0;
        if (sscanf(p, "%d", &vi) != 1)
          break;
        idx[count++] = (vi < 0) ? (int)verts.size() + vi : vi - 1;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
          ++p;
      }
      for (int k = 1; k + 1 < count; ++k) {
        if (idx[0] < 0 || idx[0] >= (int)verts.size())
          continue;
        if (idx[k] < 0 || idx[k] >= (int)verts.size())
          continue;
        if (idx[k + 1] < 0 || idx[k + 1] >= (int)verts.size())
          continue;
        Triangle t;
        t.v0 = verts[idx[0]];
        t.v1 = verts[idx[k]];
        t.v2 = verts[idx[k + 1]];
        float ax = t.v1.x - t.v0.x, ay = t.v1.y - t.v0.y, az = t.v1.z - t.v0.z;
        float bx = t.v2.x - t.v0.x, by = t.v2.y - t.v0.y, bz = t.v2.z - t.v0.z;
        float nx = ay * bz - az * by, ny = az * bx - ax * bz,
              nz = ax * by - ay * bx;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len < 1e-12f)
          continue;
        t.n = {nx / len, ny / len, nz / len};
        out.push_back(t);
      }
    }
  }
  fclose(f);
  if (verts.empty() || out.empty())
    return false;

  bbox_min = bbox_max = verts[0];
  for (const auto &v : verts) {
    if (v.x < bbox_min.x)
      bbox_min.x = v.x;
    if (v.y < bbox_min.y)
      bbox_min.y = v.y;
    if (v.z < bbox_min.z)
      bbox_min.z = v.z;
    if (v.x > bbox_max.x)
      bbox_max.x = v.x;
    if (v.y > bbox_max.y)
      bbox_max.y = v.y;
    if (v.z > bbox_max.z)
      bbox_max.z = v.z;
  }
  return true;
}

// Normalizes triangles to fit within [0.05, 0.95]^3.
static void normalize_mesh(std::vector<Triangle> &tris, const float3 &lo,
                           const float3 &hi) {
  float cx = (lo.x + hi.x) * 0.5f;
  float cy = (lo.y + hi.y) * 0.5f;
  float cz = (lo.z + hi.z) * 0.5f;
  float ext = fmaxf(fmaxf(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);
  float s = 0.9f / ext;
  auto xv = [&](float3 v) -> float3 {
    return {(v.x - cx) * s + 0.5f, (v.y - cy) * s + 0.5f,
            (v.z - cz) * s + 0.5f};
  };
  for (auto &t : tris) {
    t.v0 = xv(t.v0);
    t.v1 = xv(t.v1);
    t.v2 = xv(t.v2);
    // normals unchanged by uniform scale + translation
  }
}

// ---------- CUDA device helpers ----------

__device__ __forceinline__ unsigned char to_u8_gamma(float x) {
  x = fminf(fmaxf(x, 0.f), 1.f);
  x = powf(x, 1.f / 2.2f);
  int v = (int)(x * 255.f + 0.5f);
  return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

__device__ __forceinline__ uint32_t pcg32_step(uint32_t &s) {
  s = s * 747796405u + 2891336453u;
  uint32_t x = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (x >> 22u) ^ x;
}
__device__ __forceinline__ float pcg32f(uint32_t &s) {
  return (float)(pcg32_step(s) >> 8) * (1.f / (float)(1 << 24));
}

// Closest point on triangle to p (Ericson, Real-Time Collision Detection
// §5.1.5).
__device__ float3 closest_pt_triangle(float3 p, float3 a, float3 b, float3 c) {
  float3 ab = {b.x - a.x, b.y - a.y, b.z - a.z};
  float3 ac = {c.x - a.x, c.y - a.y, c.z - a.z};
  float3 ap = {p.x - a.x, p.y - a.y, p.z - a.z};
  float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
  float d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
  if (d1 <= 0.f && d2 <= 0.f)
    return a;

  float3 bp = {p.x - b.x, p.y - b.y, p.z - b.z};
  float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
  float d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
  if (d3 >= 0.f && d4 <= d3)
    return b;

  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
    float v = d1 / (d1 - d3);
    return {a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z};
  }

  float3 cp = {p.x - c.x, p.y - c.y, p.z - c.z};
  float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
  float d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
  if (d6 >= 0.f && d5 <= d6)
    return c;

  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
    float w = d2 / (d2 - d6);
    return {a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z};
  }

  float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    float3 bc = {c.x - b.x, c.y - b.y, c.z - b.z};
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return {b.x + w * bc.x, b.y + w * bc.y, b.z + w * bc.z};
  }

  float den = 1.f / (va + vb + vc);
  float v = vb * den, w = vc * den;
  return {a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w,
          a.z + ab.z * v + ac.z * w};
}

// Brute-force signed distance. Sign from closest triangle's outward normal:
// positive = outside, negative = inside.
__device__ float mesh_sdf(float3 p, const Triangle *tris, int n_tris) {
  float min_d2 = 1e30f, sign = 1.f;
  for (int i = 0; i < n_tris; ++i) {
    float3 cp = closest_pt_triangle(p, tris[i].v0, tris[i].v1, tris[i].v2);
    float dx = p.x - cp.x, dy = p.y - cp.y, dz = p.z - cp.z;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < min_d2) {
      min_d2 = d2;
      sign = (dx * tris[i].n.x + dy * tris[i].n.y + dz * tris[i].n.z) >= 0.f
                 ? 1.f
                 : -1.f;
    }
  }
  return sign * sqrtf(min_d2);
}

// Like mesh_sdf but also returns the closest triangle's face normal.
__device__ float mesh_sdf_and_normal(float3 p, const Triangle *tris, int n_tris,
                                     float3 &out_n) {
  float min_d2 = 1e30f, sign = 1.f;
  out_n = {0.f, 1.f, 0.f};
  for (int i = 0; i < n_tris; ++i) {
    float3 cp = closest_pt_triangle(p, tris[i].v0, tris[i].v1, tris[i].v2);
    float dx = p.x - cp.x, dy = p.y - cp.y, dz = p.z - cp.z;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < min_d2) {
      min_d2 = d2;
      sign = (dx * tris[i].n.x + dy * tris[i].n.y + dz * tris[i].n.z) >= 0.f
                 ? 1.f
                 : -1.f;
      out_n = tris[i].n;
    }
  }
  return sign * sqrtf(min_d2);
}

// Area-weighted surface point sampler (uniform barycentric after triangle
// selection).
__device__ float3 sample_surface_pt(const Triangle *tris, int n_tris,
                                    const float *area_cdf, float total_area,
                                    uint32_t &rng) {
  float u = pcg32f(rng) * total_area;
  int lo = 0, hi = n_tris - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (area_cdf[mid] < u)
      lo = mid + 1;
    else
      hi = mid;
  }
  float r1 = sqrtf(pcg32f(rng)), r2 = pcg32f(rng);
  float a = 1.f - r1, b = r1 * (1.f - r2), c = r1 * r2;
  const Triangle &t = tris[lo];
  return {a * t.v0.x + b * t.v1.x + c * t.v2.x,
          a * t.v0.y + b * t.v1.y + c * t.v2.y,
          a * t.v0.z + b * t.v1.z + c * t.v2.z};
}

// ---------- CUDA kernels ----------

// Fills training_coords [3 × batch_size] (col-major: sample i at i*3+dim)
// and training_sdf [1 × batch_size] (sample i at i).
// Proportions from paper appendix C: 1/8 uniform, 4/8 surface, 3/8 perturbed.
__global__ void
generate_sdf_training_batch(int batch_size, int n_uniform, int n_surface,
                            const Triangle *__restrict__ tris, int n_tris,
                            const float *__restrict__ area_cdf,
                            float total_area, float sigma, uint32_t base_seed,
                            float *__restrict__ out_coords,
                            float *__restrict__ out_sdf) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= batch_size)
    return;

  uint32_t rng = base_seed ^ ((uint32_t)i * 2654435761u);
  pcg32_step(rng); // warm up

  float3 p;
  float sdf_val;

  if (i < n_uniform) {
    float px = pcg32f(rng), py = pcg32f(rng), pz = pcg32f(rng);
    p = {px, py, pz};
    sdf_val = mesh_sdf(p, tris, n_tris);
  } else if (i < n_uniform + n_surface) {
    p = sample_surface_pt(tris, n_tris, area_cdf, total_area, rng);
    sdf_val = 0.f;
  } else {
    p = sample_surface_pt(tris, n_tris, area_cdf, total_area, rng);
    // Box-Muller Gaussian perturbation (3 independent samples)
    float u1 = fmaxf(pcg32f(rng), 1e-8f), u2 = pcg32f(rng);
    float u3 = fmaxf(pcg32f(rng), 1e-8f), u4 = pcg32f(rng);
    float u5 = fmaxf(pcg32f(rng), 1e-8f), u6 = pcg32f(rng);
    p.x += sigma * sqrtf(-2.f * logf(u1)) * cosf(6.28318530718f * u2);
    p.y += sigma * sqrtf(-2.f * logf(u3)) * cosf(6.28318530718f * u4);
    p.z += sigma * sqrtf(-2.f * logf(u5)) * cosf(6.28318530718f * u6);
    sdf_val = mesh_sdf(p, tris, n_tris);
  }

  out_coords[i * 3 + 0] = p.x;
  out_coords[i * 3 + 1] = p.y;
  out_coords[i * 3 + 2] = p.z;
  out_sdf[i] = sdf_val;
}

// Initializes per-pixel rays and computes AABB [0,1]^3 entry/exit t values.
__global__ void init_rays(int W, int H, float3 eye, float3 fwd, float3 right,
                          float3 up, float tan_half_fov, float *ray_pos,
                          float *ray_dir, float *ray_t, float *ray_tmax,
                          uint8_t *ray_mask, float *ray_prev_sdf,
                          float *ray_prev_pos) {
  int px = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int py = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (px >= W || py >= H)
    return;
  int i = py * W + px;

  float ux = (2.f * (px + 0.5f) / W - 1.f) * tan_half_fov;
  float uy = (2.f * (py + 0.5f) / H - 1.f) * tan_half_fov;
  float dx = fwd.x + ux * right.x + uy * up.x;
  float dy = fwd.y + ux * right.y + uy * up.y;
  float dz = fwd.z + ux * right.z + uy * up.z;
  float inv = rsqrtf(dx * dx + dy * dy + dz * dz);
  dx *= inv;
  dy *= inv;
  dz *= inv;
  ray_dir[i * 3 + 0] = dx;
  ray_dir[i * 3 + 1] = dy;
  ray_dir[i * 3 + 2] = dz;

  // Slab test against AABB [0,1]^3
  float tx0 = (0.f - eye.x) / dx, tx1 = (1.f - eye.x) / dx;
  if (tx0 > tx1) {
    float t = tx0;
    tx0 = tx1;
    tx1 = t;
  }
  float ty0 = (0.f - eye.y) / dy, ty1 = (1.f - eye.y) / dy;
  if (ty0 > ty1) {
    float t = ty0;
    ty0 = ty1;
    ty1 = t;
  }
  float tz0 = (0.f - eye.z) / dz, tz1 = (1.f - eye.z) / dz;
  if (tz0 > tz1) {
    float t = tz0;
    tz0 = tz1;
    tz1 = t;
  }
  float t_enter = fmaxf(fmaxf(tx0, ty0), tz0);
  float t_exit = fminf(fminf(tx1, ty1), tz1);

  if (t_exit <= t_enter || t_exit <= 0.f) {
    ray_pos[i * 3 + 0] = eye.x;
    ray_pos[i * 3 + 1] = eye.y;
    ray_pos[i * 3 + 2] = eye.z;
    ray_t[i] = 0.f;
    ray_tmax[i] = -1.f;
    ray_mask[i] = 2; // miss
    ray_prev_sdf[i] = 1.0f;
    return;
  }
  float t0 = fmaxf(t_enter, 0.f);
  ray_pos[i * 3 + 0] = eye.x + t0 * dx;
  ray_pos[i * 3 + 1] = eye.y + t0 * dy;
  ray_pos[i * 3 + 2] = eye.z + t0 * dz;
  ray_t[i] = t0;
  ray_tmax[i] = t_exit;
  ray_mask[i] = 0; // active
  ray_prev_sdf[i] = 1.0f;
  ray_prev_pos[i * 3 + 0] = ray_pos[i * 3 + 0];
  ray_prev_pos[i * 3 + 1] = ray_pos[i * 3 + 1];
  ray_prev_pos[i * 3 + 2] = ray_pos[i * 3 + 2];
}

// Sphere-tracing step: advances each active ray by its network-returned SDF
// value. Detects sign changes (overshoot) and interpolates to the zero
// crossing.
__global__ void march_rays(int N, float eps, float min_step,
                           const float *__restrict__ sdf_out,
                           const float *__restrict__ ray_dir, float *ray_pos,
                           float *ray_t, const float *__restrict__ ray_tmax,
                           uint8_t *ray_mask, float *ray_prev_sdf,
                           float *ray_prev_pos) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N || ray_mask[i] != 0)
    return;
  float sdf = sdf_out[i];
  float prev_sdf = ray_prev_sdf[i];

  // Crossed the surface: interpolate to the zero crossing rather than
  // continuing past it.
  if (prev_sdf > 0.f && sdf < 0.f) {
    float alpha = prev_sdf / (prev_sdf - sdf);
    ray_pos[i * 3 + 0] = ray_prev_pos[i * 3 + 0] +
                         alpha * (ray_pos[i * 3 + 0] - ray_prev_pos[i * 3 + 0]);
    ray_pos[i * 3 + 1] = ray_prev_pos[i * 3 + 1] +
                         alpha * (ray_pos[i * 3 + 1] - ray_prev_pos[i * 3 + 1]);
    ray_pos[i * 3 + 2] = ray_prev_pos[i * 3 + 2] +
                         alpha * (ray_pos[i * 3 + 2] - ray_prev_pos[i * 3 + 2]);
    ray_mask[i] = 1;
    return;
  }

  if (fabsf(sdf) < eps) {
    ray_mask[i] = 1;
    return;
  }
  float step = fmaxf(fabsf(sdf) * 0.95f, min_step);
  float t = ray_t[i] + step;
  if (t > ray_tmax[i]) {
    ray_mask[i] = 2;
    return;
  }
  ray_prev_sdf[i] = sdf;
  ray_prev_pos[i * 3 + 0] = ray_pos[i * 3 + 0];
  ray_prev_pos[i * 3 + 1] = ray_pos[i * 3 + 1];
  ray_prev_pos[i * 3 + 2] = ray_pos[i * 3 + 2];
  ray_pos[i * 3 + 0] += step * ray_dir[i * 3 + 0];
  ray_pos[i * 3 + 1] += step * ray_dir[i * 3 + 1];
  ray_pos[i * 3 + 2] += step * ray_dir[i * 3 + 2];
  ray_t[i] = t;
}

// Writes 6 offset positions per pixel into out[3 × 6N] for finite-difference
// normals. Offsets ordered: +x, -x, +y, -y, +z, -z (k=0..5, sample index =
// k*N+i).
__global__ void build_normal_queries(int N, float eps,
                                     const float *__restrict__ ray_pos,
                                     float *__restrict__ out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  float x = ray_pos[i * 3 + 0], y = ray_pos[i * 3 + 1], z = ray_pos[i * 3 + 2];
#define WR(k, ox, oy, oz)                                                      \
  out[((k) * N + i) * 3 + 0] = x + (ox);                                       \
  out[((k) * N + i) * 3 + 1] = y + (oy);                                       \
  out[((k) * N + i) * 3 + 2] = z + (oz);
  WR(0, +eps, 0, 0)
  WR(1, -eps, 0, 0)
  WR(2, 0, +eps, 0) WR(3, 0, -eps, 0) WR(4, 0, 0, +eps) WR(5, 0, 0, -eps)
#undef WR
}

__global__ void fill_ones_half(int N, __half *out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N)
    out[i] = __float2half(1.f);
}

// Writes RGBA8 to PBO using analytic normals [3 × N] AoS (d(sdf)/d(pos)).
// mode 0 = greyscale diffuse (point light), mode 1 = normal-map RGB.
__global__ void shade_and_pack(int N,
                               const float *__restrict__ normals, // [3 × N] AoS
                               const float *__restrict__ ray_pos,
                               const float *__restrict__ ray_dir,
                               const uint8_t *__restrict__ ray_mask,
                               float3 light_pos, int mode,
                               uchar4 *__restrict__ out_rgba) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  uchar4 color;
  if (ray_mask[i] == 1) {
    float nx = normals[i * 3 + 0];
    float ny = normals[i * 3 + 1];
    float nz = normals[i * 3 + 2];
    float inv = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-8f);
    nx *= inv;
    ny *= inv;
    nz *= inv;
    if (mode == 1) {
      color = {(unsigned char)((nx * 0.5f + 0.5f) * 255.f),
               (unsigned char)((ny * 0.5f + 0.5f) * 255.f),
               (unsigned char)((nz * 0.5f + 0.5f) * 255.f), 255};
    } else {
      float hx = ray_pos[i * 3 + 0], hy = ray_pos[i * 3 + 1],
            hz = ray_pos[i * 3 + 2];
      float lx = light_pos.x - hx, ly = light_pos.y - hy, lz = light_pos.z - hz;
      float linv = rsqrtf(lx * lx + ly * ly + lz * lz + 1e-8f);
      lx *= linv;
      ly *= linv;
      lz *= linv;
      float diffuse = fmaxf(0.f, nx * lx + ny * ly + nz * lz);
      unsigned char cv = to_u8_gamma(0.05f + 0.95f * diffuse);
      color = {cv, cv, cv, 255};
    }
  } else {
    color = {0, 0, 0, 255};
  }
  out_rgba[i] = color;
}

// Samples N area-weighted GT surface points into out[3 × N] (col-major).
__global__ void
sample_gt_surface_points(int N, const Triangle *__restrict__ tris, int n_tris,
                         const float *__restrict__ area_cdf, float total_area,
                         uint32_t seed, float *__restrict__ out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  uint32_t rng = seed ^ ((uint32_t)i * 2654435761u);
  pcg32_step(rng);
  float3 p = sample_surface_pt(tris, n_tris, area_cdf, total_area, rng);
  out[i * 3 + 0] = p.x;
  out[i * 3 + 1] = p.y;
  out[i * 3 + 2] = p.z;
}

// For each hit pixel: GT mesh dist (learned→GT) and FD-normal vs GT-normal
// angle. Non-hit pixels write 0.  Hit count accumulated via atomicAdd.
__global__ void
compute_hit_metrics(int N, const float *__restrict__ ray_pos,
                    const uint8_t *__restrict__ ray_mask,
                    const float *__restrict__ normals, // [3 × N] AoS
                    const Triangle *__restrict__ tris, int n_tris,
                    float *__restrict__ dist_buf, float *__restrict__ err_buf,
                    int *__restrict__ hit_count) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  if (ray_mask[i] != 1) {
    dist_buf[i] = 0.f;
    err_buf[i] = 0.f;
    return;
  }
  float3 p = {ray_pos[i * 3 + 0], ray_pos[i * 3 + 1], ray_pos[i * 3 + 2]};
  float3 gt_n;
  float gt_sdf = mesh_sdf_and_normal(p, tris, n_tris, gt_n);
  dist_buf[i] = fabsf(gt_sdf);

  float nx = normals[i * 3 + 0];
  float ny = normals[i * 3 + 1];
  float nz = normals[i * 3 + 2];
  float inv = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-8f);
  nx *= inv;
  ny *= inv;
  nz *= inv;
  float dot = fmaxf(-1.f, fminf(1.f, nx * gt_n.x + ny * gt_n.y + nz * gt_n.z));
  err_buf[i] = acosf(fabsf(dot)) * (180.f / 3.14159265358979f);

  atomicAdd(hit_count, 1);
}

// ---------- Misc ----------

static int choose_cuda_device_for_gl_context() {
  unsigned int cnt = 0;
  int devs[8] = {};
  cudaError_t e = cudaGLGetDevices(&cnt, devs, 8, cudaGLDeviceListCurrentFrame);
  if (e == cudaSuccess && cnt > 0)
    return devs[0];
  cudaGetLastError();
  return 0;
}

// ---------- Hash-grid smoothing ----------

struct HashGridLayout {
  int offsets[16];
  int entry_counts[16];
  int cell_res[16];
  int total;
};

static HashGridLayout compute_hash_grid_layout(int n_levels, int base_res,
                                               float per_level_scale,
                                               int log2_hashmap_size,
                                               int n_features_per_level) {
  HashGridLayout layout{};
  int offset = 0;
  int max_entries = 1 << log2_hashmap_size;
  for (int k = 0; k < n_levels; ++k) {
    double res = std::floor(base_res * std::pow((double)per_level_scale, k));
    long long vol = (long long)res * (long long)res * (long long)res;
    int n_entries = (int)std::min(vol, (long long)max_entries);
    layout.offsets[k] = offset;
    layout.entry_counts[k] = n_entries;
    layout.cell_res[k] = (int)res;
    offset += n_entries * n_features_per_level;
  }
  layout.total = offset;
  return layout;
}

// tcnn hash primes (same as in tiny-cuda-nn/include/tiny-cuda-nn/grid.h).
static constexpr uint32_t kHashPrimes[3] = {1u, 2654435761u, 805459861u};

__device__ __forceinline__ int hash3(int ix, int iy, int iz, int n_entries) {
  uint32_t h = ((uint32_t)ix * kHashPrimes[0]) ^ ((uint32_t)iy * kHashPrimes[1]) ^
               ((uint32_t)iz * kHashPrimes[2]);
  return (int)(h % (uint32_t)n_entries);
}

// 3-D spatial Gaussian blur over one hash-grid level. Iterates over a regular
// grid of res_eff^3 cells (res_eff = min(res, cbrt(n_entries))), maps each to
// its full-resolution position, reads neighbour features from src via the tcnn
// hash function, and writes the smoothed result to dst. src and dst must not
// alias. For levels with hash collisions, write races are benign: all
// contending threads write a valid spatially-averaged value.
__global__ void smooth_hash_level(const __half *__restrict__ src, __half *dst,
                                  int n_entries, int n_features,
                                  int res, int res_eff, float sigma) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int n_cells = res_eff * res_eff * res_eff;
  if (cell >= n_cells)
    return;

  int iz = cell / (res_eff * res_eff);
  int iy = (cell / res_eff) % res_eff;
  int ix = cell % res_eff;

  // Map coarse cell index to full-resolution position.
  int fx = ix * res / res_eff;
  int fy = iy * res / res_eff;
  int fz = iz * res / res_eff;

  int dst_idx = hash3(fx, fy, fz, n_entries);
  int r = (int)ceilf(3.f * sigma);
  float inv2s2 = 0.5f / (sigma * sigma);

  for (int f = 0; f < n_features; ++f) {
    float acc = 0.f, w_sum = 0.f;
    for (int dz = -r; dz <= r; ++dz)
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          int nx = max(0, min(res - 1, fx + dx));
          int ny = max(0, min(res - 1, fy + dy));
          int nz = max(0, min(res - 1, fz + dz));
          int src_idx = hash3(nx, ny, nz, n_entries);
          float dist2 = (float)(dx * dx + dy * dy + dz * dz);
          float w = expf(-dist2 * inv2s2);
          acc += w * __half2float(src[src_idx * n_features + f]);
          w_sum += w;
        }
    dst[dst_idx * n_features + f] = __float2half(acc / w_sum);
  }
}

// ---------- main ----------
using namespace tcnn;
using precision_t = network_precision_t;

int main(int argc, char **argv) {
  const char *mesh_path = "../data/bunny.obj";
  if (argc >= 2)
    mesh_path = argv[1];

  std::vector<Triangle> tris;
  float3 bbox_min{}, bbox_max{};
  if (!load_obj(mesh_path, tris, bbox_min, bbox_max))
    return -1;
  normalize_mesh(tris, bbox_min, bbox_max);
  int n_tris = (int)tris.size();
  printf("Loaded %s: %d triangles\n", mesh_path, n_tris);

  // Window
  RGFW_glHints *hints = RGFW_getGlobalHints_OpenGL();
  hints->major = 3;
  hints->minor = 3;
  RGFW_setGlobalHints_OpenGL(hints);
  RGFW_windowFlags wf = RGFW_windowAllowDND | RGFW_windowCenter |
                        RGFW_windowScaleToMonitor | RGFW_windowOpenGL;
  RGFW_window *window = RGFW_createWindow("Neural-SDF", SCR_WIDTH, SCR_HEIGHT,
                                          SCR_WIDTH, SCR_HEIGHT, wf);
  if (!window) {
    fprintf(stderr, "Failed to create window\n");
    return -1;
  }
  RGFW_window_setExitKey(window, RGFW_escape);
  RGFW_window_makeCurrentContext_OpenGL(window);
  int glad_ver = gladLoadGL((GLADloadfunc)RGFW_getProcAddress_OpenGL);
  if (!glad_ver) {
    fprintf(stderr, "GLAD init failed\n");
    return -1;
  }
  printf("OpenGL %d.%d - %s\n", GLAD_VERSION_MAJOR(glad_ver),
         GLAD_VERSION_MINOR(glad_ver), glGetString(GL_RENDERER));

  GLuint prog = createProgram(kVS, kFS);
  GLint uScaleLoc = glGetUniformLocation(prog, "uScale");
  GLint uTexLoc = glGetUniformLocation(prog, "uTex");

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  if (!ImGui_ImplRgfw_InitForOpenGL(window, true) ||
      !ImGui_ImplOpenGL3_Init("#version 330")) {
    fprintf(stderr, "ImGui init failed\n");
    return -1;
  }

  float verts[] = {-1.f, -1.f, 0.f, 0.f, 1.f,  -1.f, 1.f, 0.f,
                   1.f,  1.f,  1.f, 1.f, -1.f, 1.f,  0.f, 1.f};
  unsigned int idxs[] = {0, 1, 2, 2, 3, 0};
  GLuint vao = 0, vbo = 0, ebo = 0;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxs), idxs, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glBindVertexArray(0);

  int cuda_device = choose_cuda_device_for_gl_context();
  CUDA_CHECK_THROW(cudaSetDevice(cuda_device));
  printf("CUDA device %d\n", cuda_device);

  // GPU mesh data
  GPUMemory<Triangle> gpu_tris(n_tris);
  gpu_tris.copy_from_host(tris.data());

  std::vector<float> area_cdf(n_tris);
  float total_area = 0.f;
  for (int k = 0; k < n_tris; ++k) {
    float ax = tris[k].v1.x - tris[k].v0.x, ay = tris[k].v1.y - tris[k].v0.y,
          az = tris[k].v1.z - tris[k].v0.z;
    float bx = tris[k].v2.x - tris[k].v0.x, by = tris[k].v2.y - tris[k].v0.y,
          bz = tris[k].v2.z - tris[k].v0.z;
    float cx = ay * bz - az * by, cy = az * bx - ax * bz,
          cz = ax * by - ay * bx;
    total_area += 0.5f * sqrtf(cx * cx + cy * cy + cz * cz);
    area_cdf[k] = total_area;
  }
  GPUMemory<float> gpu_area_cdf(n_tris);
  gpu_area_cdf.copy_from_host(area_cdf.data());

  // GL texture + PBO at render resolution
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, RENDER_W, RENDER_H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  GLuint pbo = 0;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)RENDER_N * sizeof(uchar4),
               nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  cudaGraphicsResource *cuda_pbo_res = nullptr;
  CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

  cudaStream_t stream = nullptr;
  CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Ray state buffers
  GPUMemory<float> ray_pos_buf(3 * RENDER_N);
  GPUMemory<float> ray_dir_buf(3 * RENDER_N);
  GPUMemory<float> ray_t_buf(RENDER_N);
  GPUMemory<float> ray_tmax_buf(RENDER_N);
  GPUMemory<uint8_t> ray_mask_buf(RENDER_N);
  GPUMemory<float> ray_prev_sdf_buf(RENDER_N);
  GPUMemory<float> ray_prev_pos_buf(3 * RENDER_N);
  // Non-owning matrix view over ray positions (updated in-place each step)
  GPUMatrix<float> ray_input(ray_pos_buf.data(), 3, RENDER_N);
  GPUMatrix<float> ray_sdf_out(1, RENDER_N);

  // Analytic normals via backward pass: d(sdf)/d(pos) at each hit point.
  GPUMemory<float> analytic_normal_buf(3 * RENDER_N);
  GPUMatrix<float> analytic_normal_mat(analytic_normal_buf.data(), 3, RENDER_N);

  // Metric buffers (Chamfer + normal consistency)
  GPUMemory<float> gt_pts_buf(3 * N_METRIC_SAMPLES);
  GPUMatrix<float> gt_pts_input(gt_pts_buf.data(), 3, N_METRIC_SAMPLES);
  GPUMemory<float> gt_sdf_out_buf(N_METRIC_SAMPLES);
  GPUMatrix<float> gt_sdf_out_mat(gt_sdf_out_buf.data(), 1, N_METRIC_SAMPLES);
  GPUMemory<float> metric_dist_buf(RENDER_N);
  GPUMemory<float> metric_err_buf(RENDER_N);
  GPUMemory<int> metric_hit_count_buf(1);

  // Orbit camera: azimuth around Y, elevation above XZ, distance from target.
  float3 cam_target = {0.5f, 0.5f, 0.5f};
  float cam_azimuth = 0.f;
  float cam_elevation = 0.f;
  float cam_dist = 2.0f;
  float cam_fov = 0.41421356f; // tan(22.5°), i.e. 45° full FOV

  float3 cam_eye{}, cam_fwd{}, cam_right{}, cam_up{};
  auto recompute_camera = [&]() {
    float ce = cosf(cam_elevation), se = sinf(cam_elevation);
    float ca = cosf(cam_azimuth), sa = sinf(cam_azimuth);
    cam_eye = {cam_target.x + cam_dist * ce * sa, cam_target.y + cam_dist * se,
               cam_target.z + cam_dist * ce * ca};
    float fx = cam_target.x - cam_eye.x;
    float fy = cam_target.y - cam_eye.y;
    float fz = cam_target.z - cam_eye.z;
    float fl = rsqrtf(fx * fx + fy * fy + fz * fz);
    cam_fwd = {fx * fl, fy * fl, fz * fl};
    // right = cross(fwd, world_up=(0,1,0)) simplified
    float rx = -cam_fwd.z, rz = cam_fwd.x;
    float rl = rsqrtf(rx * rx + rz * rz + 1e-30f);
    cam_right = {rx * rl, 0.f, rz * rl};
    // up = cross(right, fwd)
    cam_up = {cam_right.y * cam_fwd.z - cam_right.z * cam_fwd.y,
              cam_right.z * cam_fwd.x - cam_right.x * cam_fwd.z,
              cam_right.x * cam_fwd.y - cam_right.y * cam_fwd.x};
  };
  recompute_camera();

  float3 light_pos = {1.5f, 2.5f, 0.5f};

  // Network config: 3D → 1D SDF
  const uint32_t n_input_dims = 3;
  const uint32_t n_output_dims = 1;
  const uint32_t batch_size = 1u << 17; // 131072, matches paper appendix C
  const int n_uniform = (int)batch_size / 8;
  const int n_surface = (int)batch_size / 2; // 4/8

  GPUMatrix<float> training_batch(n_input_dims, batch_size);
  GPUMatrix<float> training_target(n_output_dims, batch_size);

  json config = {
      {"loss", {{"otype", "MAPE"}}},
      {"optimizer",
       {{"otype", "Ema"},
        {"decay", 0.95f},
        {"nested",
         {{"otype", "ExponentialDecay"},
          {"decay_start", 10000},
          {"decay_interval", 5000},
          {"decay_base", 0.33f},
          {"nested",
           {{"otype", "Adam"},
            {"learning_rate", 1e-4f},
            {"beta1", 0.9f},
            {"beta2", 0.99f},
            {"epsilon", 1e-15f},
            {"l2_reg", 1e-6f}}}}}}},
      {"encoding",
       {{"otype", "HashGrid"},
        {"n_levels", 16},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 19},
        {"base_resolution", 16},
        {"per_level_scale", 1.3819f}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", 64},
        {"n_hidden_layers", 2}}},
  };

  json enc_opts = config.value("encoding", json::object());
  json loss_opts = config.value("loss", json::object());
  json opt_opts = config.value("optimizer", json::object());
  json net_opts = config.value("network", json::object());

  std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
  std::shared_ptr<Optimizer<precision_t>> optimizer{
      create_optimizer<precision_t>(opt_opts)};
  std::shared_ptr<NetworkWithInputEncoding<precision_t>> network =
      std::make_shared<NetworkWithInputEncoding<precision_t>>(
          n_input_dims, n_output_dims, enc_opts, net_opts);
  network->set_jit_fusion(tcnn::supports_jit_fusion());
  auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
      network, optimizer, loss);

  // forward() output and dL/doutput must use padded_output_width() rows.
  uint32_t padded_out = network->padded_output_width();
  GPUMemory<precision_t> dL_dsdf_buf(padded_out * RENDER_N);
  GPUMatrix<precision_t> dL_dsdf_mat(dL_dsdf_buf.data(), padded_out, RENDER_N);
  GPUMemory<precision_t> ray_sdf_half_buf(padded_out * RENDER_N);
  GPUMatrix<precision_t> ray_sdf_half_out(ray_sdf_half_buf.data(), padded_out, RENDER_N);

  auto reset_training = [&]() {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    loss.reset(create_loss<precision_t>(loss_opts));
    optimizer.reset(create_optimizer<precision_t>(opt_opts));
    network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        n_input_dims, n_output_dims, enc_opts, net_opts);
    network->set_jit_fusion(tcnn::supports_jit_fusion());
    trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
        network, optimizer, loss);
  };

  glUseProgram(prog);
  glUniform1i(uTexLoc, 0);

  // Hash-grid smoothing state
  constexpr int kHashLevels = 16, kHashFeat = 2, kHashLog2 = 19;
  constexpr int kHashBaseRes = 16;
  constexpr float kHashScale = 1.3819f;
  HashGridLayout hg_layout = compute_hash_grid_layout(
      kHashLevels, kHashBaseRes, kHashScale, kHashLog2, kHashFeat);
  assert(hg_layout.total == (int)network->encoding()->n_params());
  int max_level_entries = *std::max_element(
      hg_layout.entry_counts, hg_layout.entry_counts + kHashLevels);
  GPUMemory<__half> smooth_tmp_buf(max_level_entries * kHashFeat);
  float smooth_sigma = 0.f;

  bool running = true;
  bool training_enabled = true;
  bool mouse_dragging = false;
  uint32_t step = 0;
  float current_loss = 0.f;

  int render_mode = 0;
  float last_chamfer = 0.f, last_gt_to_learned = 0.f, last_learned_to_gt = 0.f;
  float last_normal_err = 0.f;
  int last_n_hits = 0;
  bool metrics_logging = false;
  FILE *metrics_file = nullptr;
  std::vector<float> cpu_gt_sdf(N_METRIC_SAMPLES);
  std::vector<float> cpu_dist(RENDER_N);
  std::vector<float> cpu_err(RENDER_N);
  std::vector<int> cpu_hit_count(1, 0);

  std::vector<uint8_t> render_pixels(RENDER_W * RENDER_H * 4);

  constexpr int kHistSz = 512;
  std::vector<float> loss_history(kHistSz, 0.f);
  int hist_offset = 0, hist_count = 0;

  dim3 ray_block2d(16, 16);
  dim3 ray_grid2d((RENDER_W + 15) / 16, (RENDER_H + 15) / 16);
  int ray_blocks1d = (RENDER_N + 255) / 256;

  fill_ones_half<<<(padded_out * RENDER_N + 255) / 256, 256, 0, stream>>>(
      padded_out * RENDER_N, dL_dsdf_buf.data());
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

  while (!RGFW_window_shouldClose(window) && running) {
    RGFW_event event;
    while (RGFW_window_checkEvent(window, &event)) {
      switch (event.type) {
      case RGFW_quit:
        running = false;
        break;
      case RGFW_keyPressed:
        if (event.key.value == RGFW_space)
          training_enabled = !training_enabled;
        break;
      case RGFW_mouseButtonPressed:
        if (event.button.value == RGFW_mouseLeft)
          mouse_dragging = true;
        break;
      case RGFW_mouseButtonReleased:
        if (event.button.value == RGFW_mouseLeft)
          mouse_dragging = false;
        break;
      case RGFW_mousePosChanged:
        if (mouse_dragging) {
          cam_azimuth -= event.mouse.vecX * 0.005f;
          cam_elevation += event.mouse.vecY * 0.005f;
          cam_elevation = fmaxf(-1.48f, fminf(1.48f, cam_elevation));
        }
        break;
      case RGFW_mouseScroll:
        cam_dist = fmaxf(0.3f, cam_dist - event.scroll.y * 0.15f);
        break;
      }
    }
    if (!running)
      break;

    recompute_camera();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplRgfw_NewFrame();
    ImGui::NewFrame();

    // ----- Training -----
    if (training_enabled) {
      generate_sdf_training_batch<<<((int)batch_size + 255) / 256, 256, 0,
                                    stream>>>(
          (int)batch_size, n_uniform, n_surface, gpu_tris.data(), n_tris,
          gpu_area_cdf.data(), total_area, 0.0005f, step * 1234567u,
          training_batch.data(), training_target.data());
      CUDA_CHECK_THROW(cudaGetLastError());

      auto ctx =
          trainer->training_step(stream, training_batch, training_target);
      current_loss = trainer->loss(stream, *ctx);

      if (smooth_sigma > 0.f) {
        __half *inf = network->encoding()->inference_params();
        for (int k = 0; k < kHashLevels; ++k) {
          int ne = hg_layout.entry_counts[k];
          int res = hg_layout.cell_res[k];
          long long res3 = (long long)res * res * res;
          int res_eff = (res3 <= (long long)ne) ? res : (int)cbrtf((float)ne);
          int n_cells = res_eff * res_eff * res_eff;
          // Init tmp with src so entries not covered by coarse grid keep their values.
          CUDA_CHECK_THROW(cudaMemcpyAsync(smooth_tmp_buf.data(),
                                           inf + hg_layout.offsets[k],
                                           ne * kHashFeat * sizeof(__half),
                                           cudaMemcpyDeviceToDevice, stream));
          smooth_hash_level<<<(n_cells + 255) / 256, 256, 0, stream>>>(
              inf + hg_layout.offsets[k], smooth_tmp_buf.data(),
              ne, kHashFeat, res, res_eff, smooth_sigma);
          CUDA_CHECK_THROW(cudaMemcpyAsync(inf + hg_layout.offsets[k],
                                           smooth_tmp_buf.data(),
                                           ne * kHashFeat * sizeof(__half),
                                           cudaMemcpyDeviceToDevice, stream));
        }
      }

      loss_history[hist_offset] = current_loss;
      hist_offset = (hist_offset + 1) % kHistSz;
      if (hist_count < kHistSz)
        ++hist_count;
      ++step;
    }

    // ----- Sphere-tracing render -----
    init_rays<<<ray_grid2d, ray_block2d, 0, stream>>>(
        RENDER_W, RENDER_H, cam_eye, cam_fwd, cam_right, cam_up, cam_fov,
        ray_pos_buf.data(), ray_dir_buf.data(), ray_t_buf.data(),
        ray_tmax_buf.data(), ray_mask_buf.data(), ray_prev_sdf_buf.data(),
        ray_prev_pos_buf.data());

    for (int r = 0; r < 64; ++r) {
      network->inference(stream, ray_input, ray_sdf_out);
      march_rays<<<ray_blocks1d, 256, 0, stream>>>(
          RENDER_N, 5e-5f, 5e-5f, ray_sdf_out.data(), ray_dir_buf.data(),
          ray_pos_buf.data(), ray_t_buf.data(), ray_tmax_buf.data(),
          ray_mask_buf.data(), ray_prev_sdf_buf.data(),
          ray_prev_pos_buf.data());
    }

    // Analytic normals: d(sdf)/d(pos) via backward pass (no FD smearing).
    auto norm_ctx =
        network->forward(stream, ray_input, &ray_sdf_half_out, true, true);
    network->backward(stream, *norm_ctx, ray_input, ray_sdf_half_out,
                      dL_dsdf_mat, &analytic_normal_mat, true,
                      GradientMode::Ignore);

    // Shade and write to PBO
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_pbo_res, stream));
    void *pbo_ptr = nullptr;
    size_t pbo_sz = 0;
    CUDA_CHECK_THROW(
        cudaGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_sz, cuda_pbo_res));
    shade_and_pack<<<ray_blocks1d, 256, 0, stream>>>(
        RENDER_N, analytic_normal_buf.data(), ray_pos_buf.data(),
        ray_dir_buf.data(), ray_mask_buf.data(), light_pos, render_mode,
        reinterpret_cast<uchar4 *>(pbo_ptr));
    CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_pbo_res, stream));

    // ----- Metrics (every METRIC_INTERVAL training steps) -----
    if (training_enabled && step > 0 && step % METRIC_INTERVAL == 0) {
      sample_gt_surface_points<<<(N_METRIC_SAMPLES + 255) / 256, 256, 0,
                                 stream>>>(
          N_METRIC_SAMPLES, gpu_tris.data(), n_tris, gpu_area_cdf.data(),
          total_area, step * 987654321u, gt_pts_buf.data());
      network->inference(stream, gt_pts_input, gt_sdf_out_mat);

      CUDA_CHECK_THROW(
          cudaMemsetAsync(metric_hit_count_buf.data(), 0, sizeof(int), stream));
      compute_hit_metrics<<<ray_blocks1d, 256, 0, stream>>>(
          RENDER_N, ray_pos_buf.data(), ray_mask_buf.data(),
          analytic_normal_buf.data(), gpu_tris.data(), n_tris,
          metric_dist_buf.data(), metric_err_buf.data(),
          metric_hit_count_buf.data());

      CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

      gt_sdf_out_buf.copy_to_host(cpu_gt_sdf.data(), N_METRIC_SAMPLES);
      metric_dist_buf.copy_to_host(cpu_dist.data(), RENDER_N);
      metric_err_buf.copy_to_host(cpu_err.data(), RENDER_N);
      metric_hit_count_buf.copy_to_host(cpu_hit_count.data(), 1);

      last_n_hits = cpu_hit_count[0];

      last_gt_to_learned = 0.f;
      for (int j = 0; j < N_METRIC_SAMPLES; ++j)
        last_gt_to_learned += fabsf(cpu_gt_sdf[j]);
      last_gt_to_learned /= (float)N_METRIC_SAMPLES;

      last_learned_to_gt = 0.f;
      last_normal_err = 0.f;
      for (int j = 0; j < RENDER_N; ++j) {
        last_learned_to_gt += cpu_dist[j];
        last_normal_err += cpu_err[j];
      }
      if (last_n_hits > 0) {
        last_learned_to_gt /= (float)last_n_hits;
        last_normal_err /= (float)last_n_hits;
      }
      last_chamfer = 0.5f * (last_gt_to_learned + last_learned_to_gt);

      if (metrics_logging && metrics_file) {
        fprintf(metrics_file, "%u,%.4f,%.8f,%.8f,%.8f,%.4f,%d\n", step,
                smooth_sigma, last_chamfer, last_gt_to_learned,
                last_learned_to_gt, last_normal_err, last_n_hits);
        fflush(metrics_file);
      }
    }

    // ----- GL display -----
    int winW = window->w > 0 ? window->w : SCR_WIDTH;
    int winH = window->h > 0 ? window->h : SCR_HEIGHT;
    glViewport(0, 0, winW, winH);

    float winAsp = (float)winW / winH;
    float renAsp = (float)RENDER_W / RENDER_H;
    float sx = 1.f, sy = 1.f;
    if (winAsp > renAsp)
      sx = renAsp / winAsp;
    else
      sy = winAsp / renAsp;

    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RENDER_W, RENDER_H, GL_RGBA,
                    GL_UNSIGNED_BYTE, (void *)0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glUseProgram(prog);
    glUniform2f(uScaleLoc, sx, sy);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // ----- ImGui -----
    float lmin = 0.f, lmax = 1.f;
    if (hist_count > 0) {
      lmin = *std::min_element(loss_history.begin(),
                               loss_history.begin() + hist_count);
      lmax = *std::max_element(loss_history.begin(),
                               loss_history.begin() + hist_count);
      if (lmax <= lmin)
        lmax = lmin + 1e-6f;
    }
    int plot_off = (hist_count == kHistSz) ? hist_offset : 0;

    ImGui::SetNextWindowPos(ImVec2(12.f, 12.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(420.f, 0.f), ImGuiCond_Once);
    ImGui::Begin("Training");
    ImGui::Text("Mesh: %s  (%d tris)", mesh_path, n_tris);
    ImGui::Separator();
    if (ImGui::Button(training_enabled ? "Pause" : "Resume"))
      training_enabled = !training_enabled;
    ImGui::SameLine();
    if (ImGui::Button("Restart")) {
      reset_training();
      step = 0;
      current_loss = 0.f;
      hist_offset = 0;
      hist_count = 0;
      std::fill(loss_history.begin(), loss_history.end(), 0.f);
      last_chamfer = last_gt_to_learned = last_learned_to_gt = 0.f;
      last_normal_err = 0.f;
      last_n_hits = 0;
    }
    ImGui::Text("Status: %s   Step: %u",
                training_enabled ? "Running" : "Paused", step);
    ImGui::Text("Loss: %.6f", current_loss);
    ImGui::PlotLines("##loss", loss_history.data(), hist_count, plot_off,
                     nullptr, lmin, lmax, ImVec2(0.f, 90.f));
    ImGui::Separator();
    ImGui::Text("Render:");
    ImGui::SameLine();
    ImGui::RadioButton("Diffuse", &render_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Normals", &render_mode, 1);
    ImGui::SameLine();
    if (ImGui::Button("Save PNG")) {
      char fname[64];
      snprintf(fname, sizeof(fname), "render_%04u_s%.2f.png", step,
               smooth_sigma);
      glBindTexture(GL_TEXTURE_2D, tex);
      stbi_flip_vertically_on_write(1);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                    render_pixels.data());
      stbi_write_png(fname, RENDER_W, RENDER_H, 4, render_pixels.data(),
                     RENDER_W * 4);
      stbi_flip_vertically_on_write(0);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    ImGui::Separator();
    ImGui::Text("Chamfer:     %.6f", last_chamfer);
    ImGui::Text("  GT->Net:   %.6f   Net->GT: %.6f", last_gt_to_learned,
                last_learned_to_gt);
    ImGui::Text("Normal err:  %.2f deg  (%d hits)", last_normal_err,
                last_n_hits);
    ImGui::TextDisabled("(updated every %d steps)", METRIC_INTERVAL);
    ImGui::Separator();
    ImGui::SliderFloat("Smooth sigma", &smooth_sigma, 0.f, 3.f, "%.2f");
    ImGui::Separator();
    if (!metrics_logging) {
      if (ImGui::Button("Start CSV Logging")) {
        metrics_file = fopen("metrics.csv", "w");
        if (metrics_file) {
          fprintf(metrics_file, "step,sigma,chamfer_dist,gt_to_learned,learned_"
                                "to_gt,normal_err_deg,n_hits\n");
          metrics_logging = true;
        }
      }
    } else {
      if (ImGui::Button("Stop Logging")) {
        fclose(metrics_file);
        metrics_file = nullptr;
        metrics_logging = false;
      }
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "-> metrics.csv");
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    RGFW_window_swapBuffers_OpenGL(window);
  }

  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
  if (cuda_pbo_res)
    CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(cuda_pbo_res));
  if (stream)
    CUDA_CHECK_THROW(cudaStreamDestroy(stream));
  free_all_gpu_memory_arenas();

  if (metrics_file) {
    fclose(metrics_file);
    metrics_file = nullptr;
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplRgfw_Shutdown();
  ImGui::DestroyContext();

  glDeleteTextures(1, &tex);
  glDeleteBuffers(1, &pbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteProgram(prog);

  RGFW_window_close(window);
  return 0;
}
