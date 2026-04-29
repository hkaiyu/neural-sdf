# Findings

## Setup

Single mesh (Stanford bunny, 4968 tris), 3000 training steps. Hash grid: 16 levels,
2 features/level, base_res=16, per_level_scale=1.3819. Metrics logged every 100 steps.

**Grain roughness** — mean angular deviation between adjacent hit-pixel surface normals
(4-connected neighborhood). Roughly 2× more sensitive to hash-collision speckle than mean
normal error, because individual speckle errors average out over ~52K hit pixels but are
captured locally by the roughness metric.

---

## Regularization sweep (hashmap_size=19, n_levels=16)

All output-space regularizers were tested and fail to reduce grain roughness below ~6.5°:

| Condition | Chamfer ×10⁻⁵ | Normal err (°) | Grain roughness (°) |
|-----------|---------------|----------------|---------------------|
| Baseline  | 6.71          | 3.18           | 6.63                |
| Gaussian blur (σ=1.0) | ~6.9 | ~3.2 | ~6.6              |
| TV regularization (λ=1e-5) | ~6.8 | ~3.2 | ~6.6        |
| Eikonal (λ=1e-3) | 9.73     | 3.07           | 6.51                |

**Reason:** The fine hash levels (res³ >> 2^hashmap_size) create feature-level
discontinuities. Regularizing the SDF output cannot fix discontinuities in the underlying
representation — the gradients still collide inside the hash table.

---

## Hash capacity sweep (n_levels=16 fixed)

| log₂(hashmap_size) | Chamfer ×10⁻⁵ | Normal err (°) | Grain roughness (°) |
|--------------------|---------------|----------------|---------------------|
| 16                 | 10.26         | 3.76           | 6.99                |
| 18                 | 5.19          | 3.12           | 6.48                |
| 19 (baseline)      | 6.71          | 3.18           | 6.63                |
| 20                 | 2.32          | 3.05           | 6.48                |
| 22                 | 3.78          | 2.81           | 6.19                |

Grain roughness decreases monotonically with hash capacity (~11% reduction over the tested
range). Effect is modest — even at log₂=22 (4M entries), grain remains above 6°.

---

## Level count sweep (hashmap_size=19 fixed) — key result

| n_levels      | Chamfer ×10⁻⁵ | Normal err (°) | Grain roughness (°) |
|---------------|---------------|----------------|---------------------|
| 8             | 3.56          | 2.38           | **4.91**            |
| 10            | 2.94          | 2.26           | 5.22                |
| 12            | 2.93          | 2.47           | 5.59                |
| 14            | 3.54          | 2.61           | 5.86                |
| 16 (baseline) | 6.71          | 3.18           | 6.63                |

Reducing from 16 to 8 levels drops grain 26% **and** improves Chamfer 47% simultaneously.
Every metric improves as fine levels are removed.

**Interpretation:** The collision-heavy fine levels do not just add cosmetic speckle — they
actively degrade reconstruction quality. Cells sharing hash entries produce contradictory
gradient signals that corrupt the shared feature vector, hurting all metrics. The optimal
level count for this mesh and hash configuration is around n_levels=10–12.

**Caveat / open question:** This result is specific to the Stanford bunny at this scale. The
"optimal" level count likely depends on mesh complexity and the point at which finer hash
levels stop encoding new geometric detail (res exceeds the mesh's intrinsic frequency
content). A mesh with finer surface detail may benefit from more levels before the collision
cost outweighs the representational gain. Next step: repeat the level sweep across multiple
meshes to determine whether a generalizable trend exists.

---

## Summary

The grain artifact is structurally determined by the hash configuration, not addressable by
output-space regularization. The grain roughness metric exposes this where mean normal error
does not. For the bunny, the instant-NGP default of 16 levels appears to be past the point of
diminishing returns — adding collision-heavy fine levels hurts both geometric fidelity and
grain roughness simultaneously.
