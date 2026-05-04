#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <direct.h>
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
static const int N_METRIC_SAMPLES = 2048;
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

struct BVHNode {
  float3 lo, hi;
  int left;  // leaf: triangle index; internal: left child index
  int right; // leaf: -1; internal: right child index
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
  }
}

// ---------- BVH stuff ----------

static void bvh_expand_aabb(float3 &lo, float3 &hi, float3 v) {
  if (v.x < lo.x)
    lo.x = v.x;
  if (v.x > hi.x)
    hi.x = v.x;
  if (v.y < lo.y)
    lo.y = v.y;
  if (v.y > hi.y)
    hi.y = v.y;
  if (v.z < lo.z)
    lo.z = v.z;
  if (v.z > hi.z)
    hi.z = v.z;
}

static float tri_centroid_axis(const Triangle &t, int axis) {
  if (axis == 0)
    return (t.v0.x + t.v1.x + t.v2.x) * (1.f / 3.f);
  if (axis == 1)
    return (t.v0.y + t.v1.y + t.v2.y) * (1.f / 3.f);
  return (t.v0.z + t.v1.z + t.v2.z) * (1.f / 3.f);
}

static int build_bvh_recursive(std::vector<BVHNode> &nodes,
                               const std::vector<Triangle> &tris,
                               std::vector<int> &indices, int start, int end) {
  int node_idx = (int)nodes.size();
  nodes.push_back({});

  float3 lo = {1e30f, 1e30f, 1e30f}, hi = {-1e30f, -1e30f, -1e30f};
  for (int i = start; i < end; ++i) {
    const Triangle &t = tris[indices[i]];
    bvh_expand_aabb(lo, hi, t.v0);
    bvh_expand_aabb(lo, hi, t.v1);
    bvh_expand_aabb(lo, hi, t.v2);
  }
  nodes[node_idx].lo = lo;
  nodes[node_idx].hi = hi;

  if (end - start == 1) {
    nodes[node_idx].left = indices[start];
    nodes[node_idx].right = -1;
    return node_idx;
  }

  float dx = hi.x - lo.x, dy = hi.y - lo.y, dz = hi.z - lo.z;
  int axis = (dx >= dy && dx >= dz) ? 0 : (dy >= dz ? 1 : 2);
  int mid = (start + end) / 2;
  std::sort(indices.begin() + start, indices.begin() + end, [&](int a, int b) {
    return tri_centroid_axis(tris[a], axis) < tri_centroid_axis(tris[b], axis);
  });

  int left_child = build_bvh_recursive(nodes, tris, indices, start, mid);
  int right_child = build_bvh_recursive(nodes, tris, indices, mid, end);
  nodes[node_idx].left = left_child;
  nodes[node_idx].right = right_child;
  return node_idx;
}

static std::vector<BVHNode> build_bvh(const std::vector<Triangle> &tris) {
  std::vector<BVHNode> nodes;
  nodes.reserve(2 * tris.size());
  std::vector<int> indices(tris.size());
  for (int i = 0; i < (int)tris.size(); ++i)
    indices[i] = i;
  build_bvh_recursive(nodes, tris, indices, 0, (int)tris.size());
  return nodes;
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

__device__ __forceinline__ float aabb_min_dist2(float3 p, float3 lo,
                                                float3 hi) {
  float dx = fmaxf(0.f, fmaxf(lo.x - p.x, p.x - hi.x));
  float dy = fmaxf(0.f, fmaxf(lo.y - p.y, p.y - hi.y));
  float dz = fmaxf(0.f, fmaxf(lo.z - p.z, p.z - hi.z));
  return dx * dx + dy * dy + dz * dz;
}

// BVH traversal, root is always node 0
__device__ float bvh_sdf_and_normal(float3 p, const Triangle *tris,
                                    const BVHNode *bvh, float3 &out_n) {
  float best_d2 = 1e30f, sign = 1.f;
  out_n = {0.f, 1.f, 0.f};
  int stack[64], top = 0;
  stack[top++] = 0;
  while (top > 0) {
    const BVHNode &node = bvh[stack[--top]];
    if (aabb_min_dist2(p, node.lo, node.hi) >= best_d2)
      continue;
    if (node.right < 0) {
      const Triangle &t = tris[node.left];
      float3 cp = closest_pt_triangle(p, t.v0, t.v1, t.v2);
      float dx = p.x - cp.x, dy = p.y - cp.y, dz = p.z - cp.z;
      float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        sign = (dx * t.n.x + dy * t.n.y + dz * t.n.z) >= 0.f ? 1.f : -1.f;
        out_n = t.n;
      }
    } else {
      stack[top++] = node.left;
      stack[top++] = node.right;
    }
  }
  return sign * sqrtf(best_d2);
}

__device__ float bvh_sdf(float3 p, const Triangle *tris, const BVHNode *bvh) {
  float3 dummy;
  return bvh_sdf_and_normal(p, tris, bvh, dummy);
}

// Area-weighted surface point sampler
__device__ float3 sample_surface_pt(const Triangle *tris, int n_tris,
                                    const float *area_cdf, float total_area,
                                    uint32_t &rng, int *out_tri_idx = nullptr) {
  float u = pcg32f(rng) * total_area;
  int lo = 0, hi = n_tris - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (area_cdf[mid] < u)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (out_tri_idx)
    *out_tri_idx = lo;
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
__global__ void generate_sdf_training_batch(
    int batch_size, int n_uniform, int n_surface,
    const Triangle *__restrict__ tris, const BVHNode *__restrict__ bvh,
    int n_tris, const float *__restrict__ area_cdf, float total_area,
    float sigma, uint32_t base_seed, float *__restrict__ out_coords,
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
    sdf_val = bvh_sdf(p, tris, bvh);
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
    sdf_val = bvh_sdf(p, tris, bvh);
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

// Samples N area-weighted GT surface points into out_pts[3 × N] (col-major).
// Also writes the face normal of the sampled triangle into out_normals[3 × N].
__global__ void
sample_gt_surface_points(int N, const Triangle *__restrict__ tris, int n_tris,
                         const float *__restrict__ area_cdf, float total_area,
                         uint32_t seed, float *__restrict__ out_pts,
                         float *__restrict__ out_normals) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  uint32_t rng = seed ^ ((uint32_t)i * 2654435761u);
  pcg32_step(rng);
  int tri_idx;
  float3 p =
      sample_surface_pt(tris, n_tris, area_cdf, total_area, rng, &tri_idx);
  out_pts[i * 3 + 0] = p.x;
  out_pts[i * 3 + 1] = p.y;
  out_pts[i * 3 + 2] = p.z;
  out_normals[i * 3 + 0] = tris[tri_idx].n.x;
  out_normals[i * 3 + 1] = tris[tri_idx].n.y;
  out_normals[i * 3 + 2] = tris[tri_idx].n.z;
}

// Evaluates the brute-force GT SDF at N query points using the BVH.
// pts: (3, N) col-major.  out_sdf: (N,).
__global__ void eval_gt_sdf_batch(int N, const float *__restrict__ pts,
                                  const Triangle *__restrict__ tris,
                                  const BVHNode *__restrict__ bvh,
                                  float *__restrict__ out_sdf) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  float3 p = {pts[i * 3 + 0], pts[i * 3 + 1], pts[i * 3 + 2]};
  out_sdf[i] = bvh_sdf(p, tris, bvh);
}

// Perturbs N GT surface points by isotropic Gaussian noise (std = sigma).
// out_pts: (3, N) col-major.
__global__ void perturb_surface_pts(int N, float sigma, uint32_t seed,
                                    const float *__restrict__ base_pts,
                                    float *__restrict__ out_pts) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  uint32_t rng = seed ^ ((uint32_t)i * 2654435761u);
  pcg32_step(rng);
  float u1 = fmaxf(pcg32f(rng), 1e-8f), u2 = pcg32f(rng);
  float u3 = fmaxf(pcg32f(rng), 1e-8f), u4 = pcg32f(rng);
  float u5 = fmaxf(pcg32f(rng), 1e-8f), u6 = pcg32f(rng);
  out_pts[i * 3 + 0] = base_pts[i * 3 + 0] + sigma * sqrtf(-2.f * logf(u1)) *
                                                 cosf(6.28318530718f * u2);
  out_pts[i * 3 + 1] = base_pts[i * 3 + 1] + sigma * sqrtf(-2.f * logf(u3)) *
                                                 cosf(6.28318530718f * u4);
  out_pts[i * 3 + 2] = base_pts[i * 3 + 2] + sigma * sqrtf(-2.f * logf(u5)) *
                                                 cosf(6.28318530718f * u6);
}

// ---------- Misc ----------

static int choose_cuda_device_for_gl_context() {
  unsigned int cnt = 0;
  int devs[8] = {}; // surely no more than 8 gpus...
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
  uint32_t h = ((uint32_t)ix * kHashPrimes[0]) ^
               ((uint32_t)iy * kHashPrimes[1]) ^
               ((uint32_t)iz * kHashPrimes[2]);
  return (int)(h % (uint32_t)n_entries);
}

// Accumulate L1-TV gradient for one hash level into a float buffer (must be
// zero-initialized). Skips neighbor pairs that resolve to the same hash entry
// (collision-aware masking) so self-regularization on aliased cells is avoided.
__global__ void accumulate_tv_gradient(const __half *__restrict__ src,
                                       float *tv_acc, int n_entries,
                                       int n_features, int res, int res_eff,
                                       float lambda_tv) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int n_cells = res_eff * res_eff * res_eff;
  if (cell >= n_cells)
    return;

  int iz = cell / (res_eff * res_eff);
  int iy = (cell / res_eff) % res_eff;
  int ix = cell % res_eff;

  int fx = ix * res / res_eff;
  int fy = iy * res / res_eff;
  int fz = iz * res / res_eff;

  int ce = hash3(fx, fy, fz, n_entries);

  const int offx[6] = {1, -1, 0, 0, 0, 0};
  const int offy[6] = {0, 0, 1, -1, 0, 0};
  const int offz[6] = {0, 0, 0, 0, 1, -1};

  for (int d = 0; d < 6; ++d) {
    int nx = max(0, min(res - 1, fx + offx[d]));
    int ny = max(0, min(res - 1, fy + offy[d]));
    int nz = max(0, min(res - 1, fz + offz[d]));
    int ne = hash3(nx, ny, nz, n_entries);
    if (ne == ce)
      continue; // collision-aware: skip aliased pairs

    for (int f = 0; f < n_features; ++f) {
      float vc = __half2float(src[ce * n_features + f]);
      float vn = __half2float(src[ne * n_features + f]);
      float delta = vc - vn;
      float g = lambda_tv * (delta > 0.f ? 1.f : delta < 0.f ? -1.f : 0.f);
      atomicAdd(&tv_acc[ce * n_features + f], g);
    }
  }
}

// Add a float TV gradient buffer into a half gradient buffer element-wise.
__global__ void add_float_to_half_grad(const float *__restrict__ tv_acc,
                                       __half *grad_buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  grad_buf[i] = __float2half(__half2float(grad_buf[i]) + tv_acc[i]);
}

// ---------- Eikonal regularization ----------

// Build 6 FD offset positions for N_eik sample points.
// For sample i, offset j occupies column j*N_eik + i of the output matrix.
__global__ void build_eikonal_offsets(int N_eik, float eps,
                                      const float *__restrict__ pts_in,
                                      float *__restrict__ pts_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_eik)
    return;
  float x = pts_in[i * 3 + 0], y = pts_in[i * 3 + 1], z = pts_in[i * 3 + 2];
  const float ox[6] = {eps, -eps, 0, 0, 0, 0};
  const float oy[6] = {0, 0, eps, -eps, 0, 0};
  const float oz[6] = {0, 0, 0, 0, eps, -eps};
  for (int j = 0; j < 6; ++j) {
    pts_out[(j * N_eik + i) * 3 + 0] = x + ox[j];
    pts_out[(j * N_eik + i) * 3 + 1] = y + oy[j];
    pts_out[(j * N_eik + i) * 3 + 2] = z + oz[j];
  }
}

// Compute dL_eik/dSDF_out for the 6*N_eik offset SDF values.
// The eik_grad_buf must be zeroed before this kernel (only feature 0 is
// written). Loss per sample i: lambda_eik * (||FD_grad_i|| - 1)^2.
__global__ void compute_eikonal_grad(int N_eik, int padded_out, float eps,
                                     float lambda_eik, float loss_scale,
                                     const __half *__restrict__ sdf_vals,
                                     __half *__restrict__ dL_dsdf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_eik)
    return;

  float sv[6];
  for (int j = 0; j < 6; ++j)
    sv[j] = __half2float(sdf_vals[(j * N_eik + i) * padded_out]);

  float gx = (sv[0] - sv[1]) / (2.f * eps);
  float gy = (sv[2] - sv[3]) / (2.f * eps);
  float gz = (sv[4] - sv[5]) / (2.f * eps);
  float norm = sqrtf(gx * gx + gy * gy + gz * gz + 1e-8f);
  float r = norm - 1.f;
  // Clamp denominator so near-zero gradients don't blow up.
  float c = 2.f * lambda_eik * r /
            (fmaxf(norm, 0.01f) * 2.f * eps * (float)N_eik) * loss_scale;

  const float comp[6] = {gx, -gx, gy, -gy, gz, -gz};
  for (int j = 0; j < 6; ++j)
    dL_dsdf[(j * N_eik + i) * padded_out + 0] = __float2half(c * comp[j]);
}

// ---------- 3-D grain roughness metric ----------

// Build 3*6*N probe positions for the grain metric.
// Generates a 6-point central-difference FD stencil around each base point.
// Layout: col = j*N + i  (FD offset j ∈ {0..5}, point i).
// Offsets j: +x, -x, +y, -y, +z, -z at distance fd_eps.
// base_pts: (3, N) col-major.  out_pts: (3, 6*N) col-major.
__global__ void build_fd_stencil(int N, float fd_eps,
                                 const float *__restrict__ base_pts,
                                 float *__restrict__ out_pts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float bx = base_pts[i * 3 + 0];
  float by = base_pts[i * 3 + 1];
  float bz = base_pts[i * 3 + 2];
  const float ox[6] = {fd_eps, -fd_eps, 0.f, 0.f, 0.f, 0.f};
  const float oy[6] = {0.f, 0.f, fd_eps, -fd_eps, 0.f, 0.f};
  const float oz[6] = {0.f, 0.f, 0.f, 0.f, fd_eps, -fd_eps};
  for (int j = 0; j < 6; ++j) {
    int col = j * N + i;
    out_pts[col * 3 + 0] = bx + ox[j];
    out_pts[col * 3 + 1] = by + oy[j];
    out_pts[col * 3 + 2] = bz + oz[j];
  }
}

// ---------- Output-space SDF consistency loss ----------

// Build k random perturbations of N_smooth base points.
// base_pts: (3, batch_size) col-major — sample i at [i*3+0..2].
// out_pts:  (3, k*N_smooth) col-major — perturbation j of sample i at
// [(j*N+i)*3+0..2].
__global__ void build_smooth_offsets(int N_smooth, int k_perturb, float eps,
                                     const float *__restrict__ base_pts,
                                     float *__restrict__ out_pts,
                                     uint32_t seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_smooth)
    return;
  float bx = base_pts[i * 3 + 0];
  float by = base_pts[i * 3 + 1];
  float bz = base_pts[i * 3 + 2];
  uint32_t rng = seed ^ ((uint32_t)i * 2654435761u);
  for (int j = 0; j < k_perturb; ++j) {
    float u1 = pcg32f(rng);
    float u2 = pcg32f(rng);
    float cos_t = 1.f - 2.f * u1;
    float sin_t = sqrtf(fmaxf(0.f, 1.f - cos_t * cos_t));
    float phi = 6.28318530f * u2;
    int out_i = j * N_smooth + i;
    out_pts[out_i * 3 + 0] = bx + eps * sin_t * cosf(phi);
    out_pts[out_i * 3 + 1] = by + eps * sin_t * sinf(phi);
    out_pts[out_i * 3 + 2] = bz + eps * cos_t;
  }
}

// Compute dL/dSDF for k*N_smooth perturbed SDF values.
// Loss per base point i: lambda_smooth/N_smooth * var_j(SDF_{i,j}).
// grad_buf must be zeroed before call; only feature 0 is written.
__global__ void compute_smooth_grad(int N_smooth, int k_perturb, int padded_out,
                                    float lambda_smooth, float loss_scale,
                                    const __half *__restrict__ sdf_vals,
                                    __half *__restrict__ dL_dsdf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_smooth)
    return;
  float mean = 0.f;
  for (int j = 0; j < k_perturb; ++j)
    mean += __half2float(sdf_vals[(j * N_smooth + i) * padded_out]);
  mean /= (float)k_perturb;
  float scale =
      loss_scale * lambda_smooth * 2.f / ((float)k_perturb * (float)N_smooth);
  for (int j = 0; j < k_perturb; ++j) {
    float s = __half2float(sdf_vals[(j * N_smooth + i) * padded_out]);
    dL_dsdf[(j * N_smooth + i) * padded_out + 0] =
        __float2half(scale * (s - mean));
  }
}

// ---------- main ----------
using namespace tcnn;
using precision_t = network_precision_t;

int main(int argc, char **argv) {
  const char *mesh_path = "../data/bunny.obj";
  const char *run_name = "default";

  // most of these we no longer use but still keep in case...
  float lambda_tv = 0.f;
  float lambda_eik = 0.f;
  float eik_eps = 0.001f;
  float lambda_smooth = 0.f;
  float smooth_eps = 1e-3f;
  int tv_level_min = 11, tv_level_max = 15;
  int max_steps = 0;
  int hashmap_size = 19;
  int n_hash_levels = 16;
  float growth_factor = 1.3819f;
  bool no_hash = false;

  for (int ai = 1; ai < argc; ++ai) {
    if (strcmp(argv[ai], "--run") == 0 && ai + 1 < argc)
      run_name = argv[++ai];
    else if (strcmp(argv[ai], "--lambda_tv") == 0 && ai + 1 < argc)
      lambda_tv = (float)atof(argv[++ai]);
    else if (strcmp(argv[ai], "--lambda_eik") == 0 && ai + 1 < argc)
      lambda_eik = (float)atof(argv[++ai]);
    else if (strcmp(argv[ai], "--eik_eps") == 0 && ai + 1 < argc)
      eik_eps = (float)atof(argv[++ai]);
    else if (strcmp(argv[ai], "--tv_min") == 0 && ai + 1 < argc)
      tv_level_min = atoi(argv[++ai]);
    else if (strcmp(argv[ai], "--tv_max") == 0 && ai + 1 < argc)
      tv_level_max = atoi(argv[++ai]);
    else if (strcmp(argv[ai], "--max_steps") == 0 && ai + 1 < argc)
      max_steps = atoi(argv[++ai]);
    else if (strcmp(argv[ai], "--hashmap_size") == 0 && ai + 1 < argc)
      hashmap_size = atoi(argv[++ai]);
    else if (strcmp(argv[ai], "--n_levels") == 0 && ai + 1 < argc)
      n_hash_levels = atoi(argv[++ai]);
    else if (strcmp(argv[ai], "--growth_factor") == 0 && ai + 1 < argc)
      growth_factor = (float)atof(argv[++ai]);
    else if (strcmp(argv[ai], "--no_hash") == 0)
      no_hash = true;
    else if (strcmp(argv[ai], "--lambda_smooth") == 0 && ai + 1 < argc)
      lambda_smooth = (float)atof(argv[++ai]);
    else if (strcmp(argv[ai], "--smooth_eps") == 0 && ai + 1 < argc)
      smooth_eps = (float)atof(argv[++ai]);
    else if (argv[ai][0] != '-')
      mesh_path = argv[ai];
  }

  std::vector<Triangle> tris;
  float3 bbox_min{}, bbox_max{};
  if (!load_obj(mesh_path, tris, bbox_min, bbox_max))
    return -1;
  normalize_mesh(tris, bbox_min, bbox_max);
  int n_tris = (int)tris.size();
  printf("Loaded %s: %d triangles\n", mesh_path, n_tris);
  std::vector<BVHNode> bvh_nodes = build_bvh(tris);
  printf("BVH: %zu nodes\n", bvh_nodes.size());

  // extract stem
  const char *mesh_basename = mesh_path;
  for (const char *p = mesh_path; *p; ++p)
    if (*p == '/' || *p == '\\')
      mesh_basename = p + 1;
  char mesh_stem[256];
  strncpy(mesh_stem, mesh_basename, sizeof(mesh_stem) - 1);
  mesh_stem[sizeof(mesh_stem) - 1] = '\0';
  char *stem_dot = strrchr(mesh_stem, '.');
  if (stem_dot)
    *stem_dot = '\0';

  char results_dir[512];
  snprintf(results_dir, sizeof(results_dir), "../results/%s", mesh_stem);
  _mkdir("../results");
  _mkdir(results_dir);

  char metrics_path[600];
  snprintf(metrics_path, sizeof(metrics_path), "%s/%s_metrics.csv", results_dir,
           run_name);
  FILE *metrics_file = fopen(metrics_path, "w");
  if (metrics_file)
    fprintf(metrics_file, "step,n_levels,hashmap_size,growth_factor,"
                          "gt_to_learned,normal_err_vi_deg,sdf_psnr_db\n");
  printf("Run '%s'  mesh '%s'  output -> %s/\n", run_name, mesh_stem,
         results_dir);

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
  GPUMemory<BVHNode> gpu_bvh(bvh_nodes.size());
  gpu_bvh.copy_from_host(bvh_nodes.data());

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

  // Metric buffers (all view-independent)
  GPUMemory<float> gt_pts_buf(3 * N_METRIC_SAMPLES);
  GPUMatrix<float> gt_pts_input(gt_pts_buf.data(), 3, N_METRIC_SAMPLES);
  GPUMemory<float> gt_normals_buf(3 * N_METRIC_SAMPLES);
  GPUMemory<float> gt_sdf_out_buf(N_METRIC_SAMPLES);
  GPUMatrix<float> gt_sdf_out_mat(gt_sdf_out_buf.data(), 1, N_METRIC_SAMPLES);

  // FD stencil for view-independent normal error: 6 offsets × N surface points.
  constexpr float kFdEps = 1e-3f;
  constexpr int N_FD_TOTAL = 6 * N_METRIC_SAMPLES;
  GPUMemory<float> fd_stencil_buf(3 * N_FD_TOTAL);
  GPUMemory<float> fd_sdf_buf(N_FD_TOTAL);
  GPUMatrix<float> fd_stencil_mat(fd_stencil_buf.data(), 3, N_FD_TOTAL);
  GPUMatrix<float> fd_sdf_mat(fd_sdf_buf.data(), 1, N_FD_TOTAL);

  // SDF PSNR: near-surface perturbed samples (σ = kPsnrSigma).
  constexpr float kPsnrSigma = 0.02f;
  GPUMemory<float> psnr_pts_buf(3 * N_METRIC_SAMPLES);
  GPUMemory<float> psnr_gt_sdf_buf(N_METRIC_SAMPLES);
  GPUMemory<float> psnr_learned_sdf_buf(N_METRIC_SAMPLES);
  GPUMatrix<float> psnr_pts_mat(psnr_pts_buf.data(), 3, N_METRIC_SAMPLES);
  GPUMatrix<float> psnr_learned_sdf_mat(psnr_learned_sdf_buf.data(), 1,
                                        N_METRIC_SAMPLES);

  // Y up/down
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
      {"encoding", no_hash ? json{{"otype", "Frequency"}, {"n_frequencies", 12}}
                           : json{{"otype", "HashGrid"},
                                  {"n_levels", n_hash_levels},
                                  {"n_features_per_level", 2},
                                  {"log2_hashmap_size", hashmap_size},
                                  {"base_resolution", 16},
                                  {"per_level_scale", growth_factor}}},
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
  GPUMatrix<precision_t> ray_sdf_half_out(ray_sdf_half_buf.data(), padded_out,
                                          RENDER_N);

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
  const int kHashLevels = n_hash_levels, kHashFeat = 2,
            kHashLog2 = hashmap_size;
  constexpr int kHashBaseRes = 16;
  float kHashScale = growth_factor;
  HashGridLayout hg_layout{};
  int max_level_entries = 0;
  if (!no_hash) {
    hg_layout = compute_hash_grid_layout(kHashLevels, kHashBaseRes, kHashScale,
                                         kHashLog2, kHashFeat);
    assert(hg_layout.total == (int)network->encoding()->n_params());
    max_level_entries = *std::max_element(hg_layout.entry_counts,
                                          hg_layout.entry_counts + kHashLevels);
  }
  GPUMemory<float> tv_tmp_buf(max_level_entries * kHashFeat);

  constexpr int N_EIK = 1024;
  GPUMemory<float> eik_pts_buf(3 * 6 * N_EIK);
  GPUMemory<precision_t> eik_sdf_buf(padded_out * 6 * N_EIK);
  GPUMemory<precision_t> eik_grad_buf(padded_out * 6 * N_EIK);
  GPUMatrix<float> eik_input_mat(eik_pts_buf.data(), 3, 6 * N_EIK);
  GPUMatrix<precision_t> eik_sdf_mat(eik_sdf_buf.data(), padded_out, 6 * N_EIK);
  GPUMatrix<precision_t> eik_grad_mat(eik_grad_buf.data(), padded_out,
                                      6 * N_EIK);

  constexpr int N_SMOOTH = 16384;
  constexpr int K_SMOOTH = 4;
  GPUMemory<float> smooth_pts_buf(3 * K_SMOOTH * N_SMOOTH);
  GPUMemory<precision_t> smooth_sdf_buf(padded_out * K_SMOOTH * N_SMOOTH);
  GPUMemory<precision_t> smooth_grad_buf(padded_out * K_SMOOTH * N_SMOOTH);
  GPUMatrix<float> smooth_input_mat(smooth_pts_buf.data(), 3,
                                    K_SMOOTH * N_SMOOTH);
  GPUMatrix<precision_t> smooth_sdf_mat(smooth_sdf_buf.data(), padded_out,
                                        K_SMOOTH * N_SMOOTH);
  GPUMatrix<precision_t> smooth_grad_mat(smooth_grad_buf.data(), padded_out,
                                         K_SMOOTH * N_SMOOTH);

  bool running = true;
  bool training_enabled = true;
  bool mouse_dragging = false;
  uint32_t step = 0;
  float current_loss = 0.f;

  int render_mode = 0;
  float last_gt_to_learned = 0.f;
  float last_normal_err_vi = 0.f;
  float last_sdf_psnr = 0.f;
  std::vector<float> cpu_gt_sdf(N_METRIC_SAMPLES);
  std::vector<float> cpu_gt_normals(3 * N_METRIC_SAMPLES);
  std::vector<float> cpu_fd_sdf(N_FD_TOTAL);
  std::vector<float> cpu_psnr_gt_sdf(N_METRIC_SAMPLES);
  std::vector<float> cpu_psnr_learned_sdf(N_METRIC_SAMPLES);

  std::vector<uint8_t> render_pixels(RENDER_W * RENDER_H * 4);
  GPUMemory<uchar4> snap_buf(RENDER_N);

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
          (int)batch_size, n_uniform, n_surface, gpu_tris.data(),
          gpu_bvh.data(), n_tris, gpu_area_cdf.data(), total_area, 0.0005f,
          step * 1234567u, training_batch.data(), training_target.data());
      CUDA_CHECK_THROW(cudaGetLastError());

      auto ctx = trainer->training_step(stream, training_batch, training_target,
                                        nullptr, /*run_optimizer=*/false);
      current_loss = trainer->loss(stream, *ctx);

      if (!no_hash && lambda_tv > 0.f) {
        __half *params = network->encoding()->params();
        __half *grad = trainer->param_gradients();
        int lmin = max(0, min(15, tv_level_min));
        int lmax = max(lmin, min(15, tv_level_max));
        for (int k = lmin; k <= lmax; ++k) {
          int ne = hg_layout.entry_counts[k];
          int res = hg_layout.cell_res[k];
          long long res3 = (long long)res * res * res;
          int res_eff = (res3 <= (long long)ne) ? res : (int)cbrtf((float)ne);
          int n_cells = res_eff * res_eff * res_eff;
          CUDA_CHECK_THROW(cudaMemsetAsync(
              tv_tmp_buf.data(), 0, ne * kHashFeat * sizeof(float), stream));
          accumulate_tv_gradient<<<(n_cells + 255) / 256, 256, 0, stream>>>(
              params + hg_layout.offsets[k], tv_tmp_buf.data(), ne, kHashFeat,
              res, res_eff, lambda_tv);
          add_float_to_half_grad<<<(ne * kHashFeat + 255) / 256, 256, 0,
                                   stream>>>(
              tv_tmp_buf.data(), grad + hg_layout.offsets[k], ne * kHashFeat);
        }
      }
      if (lambda_eik > 0.f) {
        build_eikonal_offsets<<<(N_EIK + 255) / 256, 256, 0, stream>>>(
            N_EIK, eik_eps, training_batch.data(), eik_pts_buf.data());
        auto eik_ctx =
            network->forward(stream, eik_input_mat, &eik_sdf_mat, false, false);
        CUDA_CHECK_THROW(cudaMemsetAsync(
            eik_grad_buf.data(), 0,
            padded_out * 6 * N_EIK * sizeof(precision_t), stream));
        compute_eikonal_grad<<<(N_EIK + 255) / 256, 256, 0, stream>>>(
            N_EIK, padded_out, eik_eps, lambda_eik,
            (float)default_loss_scale<precision_t>(), eik_sdf_buf.data(),
            eik_grad_buf.data());
        network->backward(stream, *eik_ctx, eik_input_mat, eik_sdf_mat,
                          eik_grad_mat, nullptr, false,
                          GradientMode::Accumulate);
      }
      if (lambda_smooth > 0.f) {
        build_smooth_offsets<<<(N_SMOOTH + 255) / 256, 256, 0, stream>>>(
            N_SMOOTH, K_SMOOTH, smooth_eps, training_batch.data(),
            smooth_pts_buf.data(), step * 7654321u);
        auto sm_ctx = network->forward(stream, smooth_input_mat,
                                       &smooth_sdf_mat, false, false);
        CUDA_CHECK_THROW(cudaMemsetAsync(
            smooth_grad_buf.data(), 0,
            padded_out * K_SMOOTH * N_SMOOTH * sizeof(precision_t), stream));
        compute_smooth_grad<<<(N_SMOOTH + 255) / 256, 256, 0, stream>>>(
            N_SMOOTH, K_SMOOTH, padded_out, lambda_smooth,
            (float)default_loss_scale<precision_t>(), smooth_sdf_buf.data(),
            smooth_grad_buf.data());
        network->backward(stream, *sm_ctx, smooth_input_mat, smooth_sdf_mat,
                          smooth_grad_mat, nullptr, false,
                          GradientMode::Accumulate);
      }
      trainer->optimizer_step(stream, default_loss_scale<precision_t>());

      loss_history[hist_offset] = current_loss;
      hist_offset = (hist_offset + 1) % kHistSz;
      if (hist_count < kHistSz)
        ++hist_count;
      ++step;
      if (max_steps > 0 && step >= (uint32_t)max_steps)
        running = false;
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

    // Auto-save normal-map PNG at step 1000 and at the final step.
    if (training_enabled) {
      bool at_milestone = (step == 1000);
      bool at_final = (max_steps > 0 && step == (uint32_t)max_steps);
      if (at_milestone || at_final) {
        shade_and_pack<<<ray_blocks1d, 256, 0, stream>>>(
            RENDER_N, analytic_normal_buf.data(), ray_pos_buf.data(),
            ray_dir_buf.data(), ray_mask_buf.data(), light_pos, 1,
            snap_buf.data());
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        snap_buf.copy_to_host(reinterpret_cast<uchar4 *>(render_pixels.data()));
        char snap_path[600];
        snprintf(snap_path, sizeof(snap_path), "%s/%s_normal_%05u.png",
                 results_dir, run_name, step);
        stbi_flip_vertically_on_write(1);
        stbi_write_png(snap_path, RENDER_W, RENDER_H, 4, render_pixels.data(),
                       RENDER_W * 4);
        stbi_flip_vertically_on_write(0);
        printf("Saved %s\n", snap_path);

        // Per-level ablation and additivity test at the final step.
        if (at_final && !no_hash) {
          int total_fp16 = (int)network->encoding()->n_params();
          precision_t *inf_params = network->encoding()->inference_params();
          std::vector<precision_t> inf_backup(total_fp16);
          CUDA_CHECK_THROW(cudaMemcpy(inf_backup.data(), inf_params,
                                      total_fp16 * sizeof(precision_t),
                                      cudaMemcpyDeviceToHost));

          // --- Per-level ablation: render normal-map with only level k active
          // ---
          for (int lev = 0; lev < kHashLevels; ++lev) {
            CUDA_CHECK_THROW(
                cudaMemset(inf_params, 0, total_fp16 * sizeof(precision_t)));
            int lev_start = hg_layout.offsets[lev];
            int lev_count = hg_layout.entry_counts[lev] * kHashFeat;
            CUDA_CHECK_THROW(cudaMemcpy(
                inf_params + lev_start, inf_backup.data() + lev_start,
                lev_count * sizeof(precision_t), cudaMemcpyHostToDevice));

            // Re-render (sphere-trace + normals + shade).
            init_rays<<<ray_grid2d, ray_block2d, 0, stream>>>(
                RENDER_W, RENDER_H, cam_eye, cam_fwd, cam_right, cam_up,
                cam_fov, ray_pos_buf.data(), ray_dir_buf.data(),
                ray_t_buf.data(), ray_tmax_buf.data(), ray_mask_buf.data(),
                ray_prev_sdf_buf.data(), ray_prev_pos_buf.data());
            for (int r = 0; r < 64; ++r) {
              network->inference(stream, ray_input, ray_sdf_out);
              march_rays<<<ray_blocks1d, 256, 0, stream>>>(
                  RENDER_N, 5e-5f, 5e-5f, ray_sdf_out.data(),
                  ray_dir_buf.data(), ray_pos_buf.data(), ray_t_buf.data(),
                  ray_tmax_buf.data(), ray_mask_buf.data(),
                  ray_prev_sdf_buf.data(), ray_prev_pos_buf.data());
            }
            auto ab_ctx = network->forward(stream, ray_input, &ray_sdf_half_out,
                                           true, true);
            network->backward(stream, *ab_ctx, ray_input, ray_sdf_half_out,
                              dL_dsdf_mat, &analytic_normal_mat, true,
                              GradientMode::Ignore);
            shade_and_pack<<<ray_blocks1d, 256, 0, stream>>>(
                RENDER_N, analytic_normal_buf.data(), ray_pos_buf.data(),
                ray_dir_buf.data(), ray_mask_buf.data(), light_pos, 1,
                snap_buf.data());
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            snap_buf.copy_to_host(
                reinterpret_cast<uchar4 *>(render_pixels.data()));
            char ab_path[600];
            snprintf(ab_path, sizeof(ab_path), "%s/%s_ablation_level_%02d.png",
                     results_dir, run_name, lev);
            stbi_flip_vertically_on_write(1);
            stbi_write_png(ab_path, RENDER_W, RENDER_H, 4, render_pixels.data(),
                           RENDER_W * 4);
            stbi_flip_vertically_on_write(0);
            printf("Saved %s\n", ab_path);
          }

          // Restore full inference_params.
          CUDA_CHECK_THROW(cudaMemcpy(inf_params, inf_backup.data(),
                                      total_fp16 * sizeof(precision_t),
                                      cudaMemcpyHostToDevice));

          // --- Additivity test: does sum_k f(x; level k) ≈ f(x; all levels)?
          // --- Uses psnr_pts_buf (near-surface, σ=0.02 off surface) so f_full
          // ≠ 0. Normalization: mean_abs_error / mean_abs(f_full) to avoid
          // division by near-zero values that arise at exact surface points.
          std::vector<float> f_full(N_METRIC_SAMPLES, 0.f);
          std::vector<float> f_sum(N_METRIC_SAMPLES, 0.f);

          // Full model output at near-surface points.
          network->inference(stream, psnr_pts_mat, psnr_learned_sdf_mat);
          CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
          psnr_learned_sdf_buf.copy_to_host(f_full.data(), N_METRIC_SAMPLES);

          // Per-level outputs at the same near-surface points.
          for (int lev = 0; lev < kHashLevels; ++lev) {
            CUDA_CHECK_THROW(
                cudaMemset(inf_params, 0, total_fp16 * sizeof(precision_t)));
            int lev_start = hg_layout.offsets[lev];
            int lev_count = hg_layout.entry_counts[lev] * kHashFeat;
            CUDA_CHECK_THROW(cudaMemcpy(
                inf_params + lev_start, inf_backup.data() + lev_start,
                lev_count * sizeof(precision_t), cudaMemcpyHostToDevice));
            network->inference(stream, psnr_pts_mat, psnr_learned_sdf_mat);
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            psnr_learned_sdf_buf.copy_to_host(cpu_psnr_learned_sdf.data(),
                                              N_METRIC_SAMPLES);
            for (int j = 0; j < N_METRIC_SAMPLES; ++j)
              f_sum[j] += cpu_psnr_learned_sdf[j];
          }

          // Restore full inference_params again.
          CUDA_CHECK_THROW(cudaMemcpy(inf_params, inf_backup.data(),
                                      total_fp16 * sizeof(precision_t),
                                      cudaMemcpyHostToDevice));

          // Relative additivity error: mean_abs_diff / mean_abs(f_full).
          double mean_abs_diff = 0.0, mean_abs_full = 0.0;
          for (int j = 0; j < N_METRIC_SAMPLES; ++j) {
            mean_abs_diff += (double)fabsf(f_sum[j] - f_full[j]);
            mean_abs_full += (double)fabsf(f_full[j]);
          }
          mean_abs_diff /= N_METRIC_SAMPLES;
          mean_abs_full /= N_METRIC_SAMPLES;
          double add_err = mean_abs_diff / (mean_abs_full + 1e-6);

          char add_path[600];
          snprintf(add_path, sizeof(add_path), "%s/%s_additivity.txt",
                   results_dir, run_name);
          if (FILE *af = fopen(add_path, "w")) {
            fprintf(af, "additivity_error=%.6f\n", (float)add_err);
            fclose(af);
          }
          printf("Additivity error: %.4f  -> %s\n", (float)add_err, add_path);
        }
      }
    }

    // ----- Metrics (every METRIC_INTERVAL training steps, all
    // view-independent) -----
    if (training_enabled && step > 0 && step % METRIC_INTERVAL == 0) {
      // Sample GT surface points with face normals.
      sample_gt_surface_points<<<(N_METRIC_SAMPLES + 255) / 256, 256, 0,
                                 stream>>>(
          N_METRIC_SAMPLES, gpu_tris.data(), n_tris, gpu_area_cdf.data(),
          total_area, step * 987654321u, gt_pts_buf.data(),
          gt_normals_buf.data());

      // gt_to_learned: mean |f(x)| at GT surface samples.
      network->inference(stream, gt_pts_input, gt_sdf_out_mat);

      // View-independent normal error: FD stencil at GT surface points.
      build_fd_stencil<<<(N_METRIC_SAMPLES + 255) / 256, 256, 0, stream>>>(
          N_METRIC_SAMPLES, kFdEps, gt_pts_buf.data(), fd_stencil_buf.data());
      network->inference(stream, fd_stencil_mat, fd_sdf_mat);

      // SDF PSNR: perturb surface pts by σ=kPsnrSigma, evaluate GT SDF
      // (brute-force) and learned SDF at those near-surface points.
      perturb_surface_pts<<<(N_METRIC_SAMPLES + 255) / 256, 256, 0, stream>>>(
          N_METRIC_SAMPLES, kPsnrSigma, step * 123456789u, gt_pts_buf.data(),
          psnr_pts_buf.data());
      eval_gt_sdf_batch<<<(N_METRIC_SAMPLES + 255) / 256, 256, 0, stream>>>(
          N_METRIC_SAMPLES, psnr_pts_buf.data(), gpu_tris.data(),
          gpu_bvh.data(), psnr_gt_sdf_buf.data());
      network->inference(stream, psnr_pts_mat, psnr_learned_sdf_mat);

      CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

      gt_sdf_out_buf.copy_to_host(cpu_gt_sdf.data(), N_METRIC_SAMPLES);
      gt_normals_buf.copy_to_host(cpu_gt_normals.data(), 3 * N_METRIC_SAMPLES);
      fd_sdf_buf.copy_to_host(cpu_fd_sdf.data(), N_FD_TOTAL);
      psnr_gt_sdf_buf.copy_to_host(cpu_psnr_gt_sdf.data(), N_METRIC_SAMPLES);
      psnr_learned_sdf_buf.copy_to_host(cpu_psnr_learned_sdf.data(),
                                        N_METRIC_SAMPLES);

      // gt_to_learned.
      last_gt_to_learned = 0.f;
      for (int j = 0; j < N_METRIC_SAMPLES; ++j)
        last_gt_to_learned += fabsf(cpu_gt_sdf[j]);
      last_gt_to_learned /= (float)N_METRIC_SAMPLES;

      // View-independent normal error: FD normal at GT surface pts vs GT face
      // normal.
      last_normal_err_vi = 0.f;
      for (int i = 0; i < N_METRIC_SAMPLES; ++i) {
        const int N = N_METRIC_SAMPLES;
        float gx =
            (cpu_fd_sdf[0 * N + i] - cpu_fd_sdf[1 * N + i]) / (2.f * kFdEps);
        float gy =
            (cpu_fd_sdf[2 * N + i] - cpu_fd_sdf[3 * N + i]) / (2.f * kFdEps);
        float gz =
            (cpu_fd_sdf[4 * N + i] - cpu_fd_sdf[5 * N + i]) / (2.f * kFdEps);
        float l = sqrtf(gx * gx + gy * gy + gz * gz + 1e-8f);
        float nx = gx / l, ny = gy / l, nz = gz / l;
        float gnx = cpu_gt_normals[i * 3 + 0];
        float gny = cpu_gt_normals[i * 3 + 1];
        float gnz = cpu_gt_normals[i * 3 + 2];
        float d = fmaxf(-1.f, fminf(1.f, nx * gnx + ny * gny + nz * gnz));
        last_normal_err_vi += acosf(fabsf(d));
      }
      last_normal_err_vi =
          last_normal_err_vi / (float)N_METRIC_SAMPLES * (180.f / 3.14159265f);

      // SDF PSNR: signal = mean(gt²), noise = MSE(learned - gt).
      double signal_power = 0.0, mse = 0.0;
      for (int j = 0; j < N_METRIC_SAMPLES; ++j) {
        double gt = (double)cpu_psnr_gt_sdf[j];
        double err = (double)cpu_psnr_learned_sdf[j] - gt;
        signal_power += gt * gt;
        mse += err * err;
      }
      signal_power /= N_METRIC_SAMPLES;
      mse /= N_METRIC_SAMPLES;
      last_sdf_psnr =
          (mse > 1e-12) ? (float)(10.0 * log10(signal_power / mse)) : 999.f;

      if (metrics_file) {
        fprintf(metrics_file, "%u,%d,%d,%.4f,%.8f,%.4f,%.4f\n", step,
                n_hash_levels, hashmap_size, kHashScale, last_gt_to_learned,
                last_normal_err_vi, last_sdf_psnr);
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
      last_gt_to_learned = 0.f;
      last_normal_err_vi = 0.f;
      last_sdf_psnr = 0.f;
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
      char fname[600];
      snprintf(fname, sizeof(fname), "%s/%s_render_%05u.png", results_dir,
               run_name, step);
      glBindTexture(GL_TEXTURE_2D, tex);
      stbi_flip_vertically_on_write(1);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                    render_pixels.data());
      stbi_write_png(fname, RENDER_W, RENDER_H, 4, render_pixels.data(),
                     RENDER_W * 4);
      stbi_flip_vertically_on_write(0);
      glBindTexture(GL_TEXTURE_2D, 0);
      printf("Saved %s\n", fname);
    }
    ImGui::Separator();
    ImGui::Text("GT->Net:     %.6f", last_gt_to_learned);
    ImGui::Text("Normal err:  %.2f deg (VI)", last_normal_err_vi);
    ImGui::Text("SDF PSNR:    %.2f dB", last_sdf_psnr);
    ImGui::TextDisabled("(updated every %d steps)", METRIC_INTERVAL);
    ImGui::Separator();
    ImGui::SliderFloat("TV lambda", &lambda_tv, 0.f, 1e-4f, "%.2e");
    ImGui::SliderInt("TV level min", &tv_level_min, 0, 15);
    ImGui::SliderInt("TV level max", &tv_level_max, 0, 15);
    ImGui::SliderFloat("Eikonal lambda", &lambda_eik, 0.f, 0.1f, "%.4f");
    ImGui::SliderFloat("Eikonal eps", &eik_eps, 1e-4f, 1e-2f, "%.4f");
    ImGui::SliderFloat("Smooth lambda", &lambda_smooth, 0.f, 0.1f, "%.2e");
    ImGui::SliderFloat("Smooth eps", &smooth_eps, 1e-4f, 5e-3f, "%.2e");
    ImGui::Separator();
    ImGui::Text("Run: %s", run_name);
    ImGui::TextDisabled("Output: %s/", results_dir);
    if (max_steps > 0)
      ImGui::Text("Max steps: %d", max_steps);
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
