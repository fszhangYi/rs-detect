import { api } from './api.js';

export const armPreload = {
  model: null,
  urdfText: null,
  urdfUrl: null,
  meshVer: null,
  meshBlobs: {}, // path -> objectURL
  ready: false,
  promise: null,
  error: null,
};

const ARM_LINK_MESHES = [
  "meshes/base_link.STL",
  "meshes/link1.STL",
  "meshes/link2.STL",
  "meshes/link3.STL",
  "meshes/link4.STL",
  "meshes/link5.STL",
  "meshes/link6.STL",
];

const EOAT_MESH_NAMES = [
  "efg_c65_base.STL",
  "efg_c65_left_finger_base.STL",
  "efg_c65_left_fingertip.STL",
  "efg_c65_right_finger_base.STL",
  "efg_c65_right_fingertip.STL",
  "efg_c65_camera_bracket.STL",
  "efg_c65_camera_clamp.STL",
  "d405_camera.STL",
];

/** EOAT mesh folder depends on selected URDF (cellc vs cellc_m). */
export function eoatMeshDirFromUrdf(urdfUrl) {
  const path = String(urdfUrl || "").split("?")[0];
  if (path.includes("cell_c_m") || path.includes("cellc_m")) return "meshes_cellc_m";
  return "meshes";
}

export function armMeshPreloadList(urdfUrl) {
  const eoatDir = eoatMeshDirFromUrdf(urdfUrl);
  return [
    ...ARM_LINK_MESHES,
    ...EOAT_MESH_NAMES.map((n) => `${eoatDir}/${n}`),
  ];
}

function urdfMeshVersion(urdfUrl) {
  const q = String(urdfUrl || "").split("?")[1] || "";
  const m = /(?:^|&)v=([^&]+)/.exec(q);
  return (m && m[1]) || "1";
}

function clearMeshBlobs() {
  Object.values(armPreload.meshBlobs).forEach((u) => {
    try {
      if (typeof u === "string" && u.startsWith("blob:")) URL.revokeObjectURL(u);
    } catch (_) { /* ignore */ }
  });
  armPreload.meshBlobs = {};
}

/** Drop in-memory URDF/STL preload so the next call re-fetches (EOAT mesh updates). */
export function invalidateArmPreload() {
  clearMeshBlobs();
  armPreload.model = null;
  armPreload.urdfText = null;
  armPreload.urdfUrl = null;
  armPreload.meshVer = null;
  armPreload.ready = false;
  armPreload.promise = null;
  armPreload.error = null;
}

export async function preloadArmAssets({ force = false, model = null } = {}) {
  if (force) invalidateArmPreload();
  if (armPreload.promise) return armPreload.promise;
  armPreload.promise = (async () => {
    const st = model ? { model } : await api("/api/arm/state");
    const mdl = st.model || model;
    armPreload.model = mdl;
    const urdfUrl = String((mdl && mdl.urdf) || "/assets/models/cs66/cs66_cell_c.urdf");
    const pathOnly = urdfUrl.split("?")[0];
    const base = pathOnly.replace(/[^/]+$/, "");
    const meshVer = urdfMeshVersion(urdfUrl);
    armPreload.urdfUrl = urdfUrl;
    armPreload.meshVer = meshVer;
    const urdfRes = await fetch(`${pathOnly}?v=${encodeURIComponent(meshVer)}`, { cache: "no-store" });
    if (!urdfRes.ok) throw new Error("URDF HTTP " + urdfRes.status);
    armPreload.urdfText = await urdfRes.text();
    const meshList = armMeshPreloadList(urdfUrl);
    await Promise.all(meshList.map(async (rel) => {
      const url = `${base}${rel}?v=${encodeURIComponent(meshVer)}`;
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) return;
        const blob = await r.blob();
        const obj = URL.createObjectURL(blob);
        armPreload.meshBlobs[rel] = obj;
        armPreload.meshBlobs[url] = obj;
        armPreload.meshBlobs[base + rel] = obj;
        const baseName = rel.split("/").pop() || rel;
        armPreload.meshBlobs[`meshes/${baseName}`] = obj;
        armPreload.meshBlobs[`meshes_cellc_m/${baseName}`] = obj;
        armPreload.meshBlobs[baseName] = obj;
        armPreload.meshBlobs[baseName.toLowerCase()] = obj;
        armPreload.meshBlobs[`meshes/${baseName.toLowerCase()}`] = obj;
        armPreload.meshBlobs[`meshes_cellc_m/${baseName.toLowerCase()}`] = obj;
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
