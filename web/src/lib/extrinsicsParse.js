import { IDENTITY, cloneMat4, coerceMat4 } from './matrix.js';

export function parseExtrinsicsJson(text) {
  const data = JSON.parse(text);
  if (!data || typeof data !== "object") throw new Error("JSON 根节点须为对象");
  const out = { d405_T: null, d435_T: null, notes: null, d405_serial: null, d435_serial: null };
  if (data.notes != null) out.notes = String(data.notes);
  // Full store shape
  if (data.d405 && data.d405.T) out.d405_T = coerceMat4(data.d405.T, "d405.T");
  if (data.d435 && data.d435.T) out.d435_T = coerceMat4(data.d435.T, "d435.T");
  if (data.d405 && data.d405.serial != null) out.d405_serial = String(data.d405.serial);
  if (data.d435 && data.d435.serial != null) out.d435_serial = String(data.d435.serial);
  // Flat API shape
  if (data.d405_T) out.d405_T = coerceMat4(data.d405_T, "d405_T");
  if (data.d435_T) out.d435_T = coerceMat4(data.d435_T, "d435_T");
  if (data.d405_serial != null) out.d405_serial = String(data.d405_serial);
  if (data.d435_serial != null) out.d435_serial = String(data.d435_serial);
  // Single-matrix upload
  const which = String(data.which || data.camera || data.role || "").toLowerCase();
  if (data.T && which) {
    const T = coerceMat4(data.T, "T");
    if (which.includes("405") || which === "wrist") out.d405_T = T;
    else if (which.includes("435") || which === "scene") out.d435_T = T;
    else throw new Error("which/camera 须为 d405|wrist 或 d435|scene");
  }
  if (!out.d405_T && !out.d435_T) {
    throw new Error("JSON 未找到 d405/d435 矩阵（支持 d405.T / d405_T / {which,T}）");
  }
  return out;
}


export function parseExtrinsicsCsv(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
  if (!lines.length) throw new Error("CSV 为空");

  const out = { d405_T: null, d435_T: null, notes: null, d405_serial: null, d435_serial: null };

  // Format A: header with camera + 16 floats
  const header = lines[0].toLowerCase();
  if (header.includes("camera") || header.startsWith("name,")) {
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(/[,;\t]/).map((c) => c.trim()).filter((c, idx, arr) => !(idx === arr.length - 1 && c === ""));
      if (cols.length < 17) throw new Error(`CSV 行 ${i + 1} 需要 camera + 16 个数，实际 ${cols.length}`);
      const name = cols[0].toLowerCase();
      const nums = cols.slice(1, 17).map(Number);
      if (nums.some((x) => !Number.isFinite(x))) throw new Error(`CSV 行 ${i + 1} 含非数字`);
      const T = [
        nums.slice(0, 4), nums.slice(4, 8), nums.slice(8, 12), nums.slice(12, 16),
      ];
      if (name.includes("405") || name === "wrist") out.d405_T = T;
      else if (name.includes("435") || name === "scene") out.d435_T = T;
      else throw new Error(`未知 camera 名: ${cols[0]}（用 d405/d435/wrist/scene）`);
    }
    if (!out.d405_T && !out.d435_T) throw new Error("CSV 未解析到任何矩阵行");
    return out;
  }

  // Format B: blocks "# d405" then 4 rows of 4, or unlabeled 4/8 rows
  const mats = [];
  let buf = [];
  let tag = null;
  const flush = () => {
    if (!buf.length) return;
    if (buf.length !== 4) throw new Error("矩阵块须恰好 4 行");
    mats.push({ tag, T: coerceMat4(buf, tag || "block") });
    buf = [];
    tag = null;
  };
  for (const line of String(text || "").split(/\r?\n/)) {
    const raw = line.trim();
    if (!raw) { flush(); continue; }
    if (raw.startsWith("#")) {
      flush();
      const m = raw.slice(1).trim().toLowerCase();
      if (m.includes("405") || m === "wrist") tag = "d405";
      else if (m.includes("435") || m === "scene") tag = "d435";
      else tag = null;
      continue;
    }
    const nums = raw.split(/[,;\t\s]+/).filter(Boolean).map(Number);
    if (nums.length !== 4 || nums.some((x) => !Number.isFinite(x))) {
      throw new Error(`无法解析矩阵行: ${raw}`);
    }
    buf.push(nums);
    if (buf.length === 4) flush();
  }
  flush();
  if (!mats.length) throw new Error("CSV 未找到 4×4 矩阵块");
  if (mats.length === 1) {
    if (mats[0].tag === "d435") out.d435_T = mats[0].T;
    else out.d405_T = mats[0].T; // default / tagged d405
  } else {
    for (const block of mats) {
      if (block.tag === "d435") out.d435_T = block.T;
      else if (block.tag === "d405") out.d405_T = block.T;
    }
    // Untagged pair: first=d405, second=d435
    if (!out.d405_T && !out.d435_T && mats.length >= 2) {
      out.d405_T = mats[0].T;
      out.d435_T = mats[1].T;
    } else {
      const untagged = mats.filter((b) => !b.tag);
      if (!out.d405_T && untagged[0]) out.d405_T = untagged[0].T;
      if (!out.d435_T && untagged[1]) out.d435_T = untagged[1].T;
    }
  }
  if (!out.d405_T && !out.d435_T) throw new Error("CSV 矩阵块未能映射到 d405/d435");
  return out;
}


export function downloadTextFile(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}


export function extrinsicsJsonTemplate(selection) {
  return {
    version: 1,
    units: "meters_radians_row_major_4x4",
    notes: "User-provided extrinsics. D405: T_flange_cam (eye-in-hand). D435: T_base_cam.",
    d405: {
      serial: selection?.d405_serial || selection?.wrist || "",
      frame_from: "flange",
      frame_to: "d405_color_optical",
      T: cloneMat4(IDENTITY),
    },
    d435: {
      serial: selection?.d435_serial || selection?.scene || "",
      frame_from: "base",
      frame_to: "d435_color_optical",
      T: cloneMat4(IDENTITY),
    },
  };
}


export function extrinsicsCsvTemplate() {
  const flat = (T) => T.flat().join(",");
  return [
    "# rs-detect extrinsics template",
    "# units: meters_radians_row_major_4x4",
    "# Format A (recommended): one row per camera, 16 floats after name",
    "camera,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33",
    `d405,${flat(IDENTITY)}`,
    `d435,${flat(IDENTITY)}`,
    "",
    "# Format B (also supported): commented blocks of 4x4",
    "# d405",
    "1,0,0,0",
    "0,1,0,0",
    "0,0,1,0",
    "0,0,0,1",
    "# d435",
    "1,0,0,0",
    "0,1,0,0",
    "0,0,1,0",
    "0,0,0,1",
    "",
  ].join("\n");
}

