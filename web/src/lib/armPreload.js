import { api } from './api.js';

export const armPreload = {
  model: null,
  urdfText: null,
  urdfUrl: null,
  meshBlobs: {}, // path -> objectURL
  ready: false,
  promise: null,
  error: null,
};

export const ARM_MESH_PRELOAD = [
  "meshes/base_link.STL",
  "meshes/link1.STL",
  "meshes/link2.STL",
  "meshes/link3.STL",
  "meshes/link4.STL",
  "meshes/link5.STL",
  "meshes/link6.STL",
  "meshes/EFG-C65.STL",
  "meshes/EFG-C65.v3.STL",
  "meshes/D405_camera.STL",
  "meshes/EFG-C65_D405_bracket.STL",
  "meshes/EFG-C65_D405_fingertip_l.STL",
  "meshes/EFG-C65_D405_fingertip_r.STL",
  "meshes/EFG-C65_D405_l_clamp.STL",
];

export async function preloadArmAssets() {
  if (armPreload.promise) return armPreload.promise;
  armPreload.promise = (async () => {
    const st = await api("/api/arm/state");
    armPreload.model = st.model;
    const urdfUrl = String((st.model && st.model.urdf) || "/assets/models/cs66/cs66_cell_c.urdf");
    const pathOnly = urdfUrl.split("?")[0];
    const base = pathOnly.replace(/[^/]+$/, "");
    armPreload.urdfUrl = urdfUrl;
    const urdfRes = await fetch(pathOnly);
    if (!urdfRes.ok) throw new Error("URDF HTTP " + urdfRes.status);
    armPreload.urdfText = await urdfRes.text();
    await Promise.all(ARM_MESH_PRELOAD.map(async (rel) => {
      const url = base + rel;
      try {
        const r = await fetch(url);
        if (!r.ok) return;
        const blob = await r.blob();
        const obj = URL.createObjectURL(blob);
        armPreload.meshBlobs[rel] = obj;
        armPreload.meshBlobs[url] = obj;
        // Case-insensitive aliases — URDF often uses .STL while list uses .STL/.stl mix.
        const baseName = rel.split("/").pop() || rel;
        armPreload.meshBlobs[`meshes/${baseName}`] = obj;
        armPreload.meshBlobs[baseName] = obj;
        armPreload.meshBlobs[baseName.toLowerCase()] = obj;
        armPreload.meshBlobs[`meshes/${baseName.toLowerCase()}`] = obj;
      } catch (_) {}
    }));
    armPreload.ready = true;
    return st;
  })().catch((e) => {
    armPreload.error = String(e.message || e);
    armPreload.promise = null;
    throw e;
  });
  return armPreload.promise;
}

