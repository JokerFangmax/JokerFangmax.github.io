#!/usr/bin/env python3
"""
Fix 3D viewer rotation from yaw/pitch (gimbal-locked, clamped) to
accumulated 3x3 matrix (free arcball-style rotation).
"""

import math
import re

def mat3_mul(a, b):
    """Multiply two 3x3 matrices (column-major flat list)."""
    out = [0.0] * 9
    for i in range(3):
        for j in range(3):
            out[i * 3 + j] = sum(a[i * 3 + k] * b[k * 3 + j] for k in range(3))
    return out

def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [c, 0, s, 0, 1, 0, -s, 0, c]

def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [1, 0, 0, 0, c, -s, 0, s, c]

def compute_initial_rot(yaw, pitch):
    """Initial rotation = Ry(yaw) * Rx(pitch) to match original view."""
    return mat3_mul(rot_y(yaw), rot_x(pitch))

def format_rot_matrix(rot):
    """Format as JS array literal with 6 decimal places."""
    vals = ", ".join(f"{v:.6f}" for v in rot)
    return f"[{vals}]"

def patch_file(path, yaw, pitch):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Replace state initialization: yaw/pitch -> rot matrix
    old_state = re.compile(
        rf'yaw:\s*{yaw},\s*\n\s*pitch:\s*{pitch},',
        re.MULTILINE
    )
    rot = compute_initial_rot(yaw, pitch)
    new_state = f"rot: {format_rot_matrix(rot)},"
    text = old_state.sub(new_state, text)

    # 2. Replace rotatePoint function
    old_rotate = re.compile(
        r'function rotatePoint\(p\) \{[\s\S]*?return \[x1, cp \* y1 - sp \* z1, sp \* y1 \+ cp \* z1\];\s*\}',
        re.MULTILINE
    )
    new_rotate = '''function rotatePoint(p) {
      const c = scene.bounds.center;
      let x = p[0] - c[0];
      let y = p[1] - c[1];
      let z = p[2] - c[2];
      const m = state.rot;
      return [
        m[0]*x + m[1]*y + m[2]*z,
        m[3]*x + m[4]*y + m[5]*z,
        m[6]*x + m[7]*y + m[8]*z
      ];
    }'''
    text = old_rotate.sub(new_rotate, text)

    # 3. Replace pointermove handler (yaw/pitch clamped update -> free matrix update)
    old_move = re.compile(
        r'state\.yaw \+= dx \* 0\.008;\s*\n\s*state\.pitch = Math\.max\(-1\.45, Math\.min\(1\.45, state\.pitch \+ dy \* 0\.008\)\);'
    )
    new_move = '''const sensitivity = 0.008;
      const ry = rot_y(dx * sensitivity);
      const rx = rot_x(dy * sensitivity);
      state.rot = mat3_mul(ry, mat3_mul(rx, state.rot));'''
    text = old_move.sub(new_move, text)

    # 4. Replace reset handler (yaw/pitch reset -> matrix reset)
    old_reset = re.compile(
        rf'state\.yaw = {yaw};\s*\n\s*state\.pitch = {pitch};'
    )
    new_reset = f"state.rot = {format_rot_matrix(rot)};"
    text = old_reset.sub(new_reset, text)

    # 5. Inject helper functions (mat3_mul, rot_x, rot_y) before rotatePoint
    helpers = '''function mat3_mul(a, b) {
      const out = [0,0,0, 0,0,0, 0,0,0];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          let s = 0;
          for (let k = 0; k < 3; k++) s += a[i*3+k] * b[k*3+j];
          out[i*3+j] = s;
        }
      }
      return out;
    }
    function rot_y(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return [c,0,s, 0,1,0, -s,0,c];
    }
    function rot_x(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return [1,0,0, 0,c,-s, 0,s,c];
    }
    '''
    # Insert before rotatePoint
    text = text.replace('function rotatePoint(p) {', helpers + 'function rotatePoint(p) {')

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Patched: {path}")


if __name__ == "__main__":
    import sys
    # Patch boundary bias detector
    patch_file(
        r"C:\THU\projects\WoSt_Final_project-1\results\boundary_bias_detector_3d.html",
        yaw=-0.75, pitch=0.45
    )
    # Patch live trace viewer
    patch_file(
        r"C:\THU\projects\WoSt_Final_project-1\results\live_trace_interactive_3d.html",
        yaw=-0.75, pitch=0.42
    )
    print("Done.")
