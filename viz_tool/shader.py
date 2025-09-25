from __future__ import annotations

from typing import List, Sequence, Tuple


def _warp_function_glsl(mode: str, power: float) -> str:
    mode_norm = (mode or "identity").strip().lower()
    if mode_norm not in {"identity", "sqrt", "log", "power"}:
        raise ValueError(f"Unsupported warp mode '{mode}'.")
    if mode_norm == "power":
        if power <= 0.0:
            raise ValueError("warp_power must be positive when warp_mode='power'.")
        pow_str = ("%.8f" % float(power)).rstrip("0").rstrip(".")
        if not pow_str:
            pow_str = "0.0"
        return (
            f"float warp_value(float v) {{ return pow(max(v, 0.0), {pow_str}); }}\n"
        )
    if mode_norm == "sqrt":
        return "float warp_value(float v) { return sqrt(max(v, 0.0)); }\n"
    if mode_norm == "log":
        return "float warp_value(float v) { return log(max(v, 1e-6)); }\n"
    return "float warp_value(float v) { return v; }\n"


def shader_helpers() -> str:
    # Helper functions: relu overloads, argmax overloads, hsv2rgb, viridis-lite
    return r"""
// Helpers
float relu(float x) { return max(x, 0.0); }
vec2  relu(vec2 v)  { return max(v, vec2(0.0)); }
vec3  relu(vec3 v)  { return max(v, vec3(0.0)); }
vec4  relu(vec4 v)  { return max(v, vec4(0.0)); }

int argmax(vec2 v) { return (v.x >= v.y) ? 0 : 1; }
int argmax(vec3 v) {
  int i = 0; float m = v.x;
  if (v.y > m) { m = v.y; i = 1; }
  if (v.z > m) { i = 2; }
  return i;
}
int argmax(vec4 v) {
  int i = 0; float m = v.x;
  if (v.y > m) { m = v.y; i = 1; }
  if (v.z > m) { m = v.z; i = 2; }
  if (v.w > m) { i = 3; }
  return i;
}

int argmin(vec2 v) { return (v.x <= v.y) ? 0 : 1; }
int argmin(vec3 v) {
  int i = 0; float m = v.x;
  if (v.y < m) { m = v.y; i = 1; }
  if (v.z < m) { i = 2; }
  return i;
}
int argmin(vec4 v) {
  int i = 0; float m = v.x;
  if (v.y < m) { m = v.y; i = 1; }
  if (v.z < m) { m = v.z; i = 2; }
  if (v.w < m) { i = 3; }
  return i;
}

vec3 hsv2rgb(vec3 c){
  vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0)-3.0)-1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

vec3 colormap_viridis(float t) {
  // simple HSV-ish fallback resembling viridis
  t = clamp(t, 0.0, 1.0);
  return hsv2rgb(vec3(0.66*(1.0 - t), 0.9, 0.9));
}

vec3 palette_discrete(int i) {
  vec3 base[16] = vec3[](
    vec3(0.92,0.20,0.23), vec3(0.23,0.75,0.28), vec3(0.12,0.40,0.95), vec3(0.98,0.84,0.20),
    vec3(0.95,0.55,0.18), vec3(0.55,0.30,0.85), vec3(0.20,0.70,0.78), vec3(0.70,0.25,0.55),
    vec3(0.35,0.58,0.35), vec3(0.20,0.45,0.80), vec3(0.65,0.70,0.28), vec3(0.80,0.36,0.25),
    vec3(0.35,0.35,0.35), vec3(0.90,0.65,0.20), vec3(0.45,0.70,0.90), vec3(0.65,0.40,0.20)
  );
  if (i < 16) {
    return base[i];
  }
  float fi = float(i);
  float hue = fract(fi * 0.4567 + 0.37);
  float sat = 0.75;
  float val = 0.88;
  return hsv2rgb(vec3(hue, sat, val));
}
"""


def build_varyings(
    arrays: List[str],
    component_counts: Sequence[int],
    *,
    warp_mode: str = "identity",
    warp_power: float = 1.0,
) -> Tuple[str, str, str]:
    """Return (vdec, vimpl, fdec) shader strings.

    block_sizes[i][b] = number of valid components in block b (<=4).
    """
    vdec: List[str] = []
    vimpl: List[str] = []
    fdec = [shader_helpers()]
    fdec.append(_warp_function_glsl(warp_mode, warp_power))
    fdec.append(
        "// Fallbacks for VTK lighting vars to avoid compile errors\n"
        "const vec3 ambientColor = vec3(0.0);\n"
        "const vec3 diffuseColor = vec3(0.0);\n"
        "const vec3 specularColor = vec3(0.0);\n"
        "const float specularPower = 1.0;\n"
        "const float opacity = 1.0;\n"
        "float diffuse = 0.0; float specular = 0.0; float sf = 0.0;\n"
    )
    helpers: List[str] = []
    for i, total in enumerate(component_counts):
        count = int(total)
        if count <= 0:
            continue
        for j in range(count):
            vdec.append(f"in float attr_f{i}_{j};\n")
            vdec.append(f"smooth out float v_f{i}_{j};\n")
            vimpl.append(f"v_f{i}_{j} = attr_f{i}_{j};\n")
            fdec.append(f"smooth in float v_f{i}_{j};\n")
        helpers.append(f"float f{i}_at(int idx) {{\n")
        helpers.append("  if (idx < 0) { return 1e20; }\n")
        helpers.append(f"  if (idx >= {count}) {{ return 1e20; }}\n")
        helpers.append("  switch(idx) {\n")
        for j in range(count):
            helpers.append(f"    case {j}: return warp_value(v_f{i}_{j});\n")
        helpers.append("    default: return 1e20;\n  }\n}\n")

        helpers.append(f"int f{i}_argmin() {{\n")
        helpers.append("  int lbl = 0;\n")
        helpers.append(f"  float best = f{i}_at(0);\n")
        helpers.append(f"  for (int idx = 1; idx < {count}; ++idx) {{\n")
        helpers.append(f"    float val = f{i}_at(idx);\n")
        helpers.append("    if (val < best) { best = val; lbl = idx; }\n")
        helpers.append("  }\n  return lbl;\n}\n")

        helpers.append(f"int f{i}_argmax() {{\n")
        helpers.append("  int lbl = 0;\n")
        helpers.append(f"  float best = f{i}_at(0);\n")
        helpers.append(f"  for (int idx = 1; idx < {count}; ++idx) {{\n")
        helpers.append(f"    float val = f{i}_at(idx);\n")
        helpers.append("    if (val > best) { best = val; lbl = idx; }\n")
        helpers.append("  }\n  return lbl;\n}\n")

        helpers.append(f"vec4 f{i}_vec4() {{\n")
        helpers.append("  vec4 outv = vec4(1e20);\n")
        for j in range(min(4, count)):
            comp = ["x", "y", "z", "w"][j]
            helpers.append(f"  outv.{comp} = warp_value(v_f{i}_{j});\n")
        helpers.append("  return outv;\n}\n")
        fdec.append(f"#define f{i} f{i}_vec4()\n")
        fdec.append(f"const int F{i}_COMPONENT_COUNT = {count};\n")

    fdec.extend(helpers)
    return ("".join(vdec), "".join(vimpl), "".join(fdec))


def build_impl(expr_glsl: str, mode: str, datamin: float | None, datamax: float | None, num_labels: int | None) -> str:
    impl = []
    if mode == "scalar":
        lo = float(datamin if datamin is not None else 0.0)
        hi = float(datamax if datamax is not None else 1.0)
        impl.append(f"float __mn = {lo:.8g}; float __mx = {hi:.8g};\n")
        impl.append(f"float __v = ({expr_glsl});\n")
        impl.append("float __t = clamp((__v - __mn) / max(1e-12, (__mx - __mn)), 0.0, 1.0);\n")
        impl.append("vec3 outColor = colormap_viridis(__t);\n")
    elif mode == "label":
        nlab = int(num_labels if num_labels is not None else 8)
        impl.append(f"int __k = int({expr_glsl});\n")
        impl.append(f"__k = __k % {max(1, nlab)}; if (__k < 0) __k += {max(1, nlab)};\n")
        impl.append("vec3 outColor = palette_discrete(__k);\n")
    elif mode == "rgb":
        impl.append(f"vec3 outColor = ({expr_glsl});\n")
    else:
        impl.append(f"vec3 outColor = vec3({expr_glsl});\n")
    return "".join(impl)
