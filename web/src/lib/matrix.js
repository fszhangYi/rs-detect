export const IDENTITY = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

export function cloneMat4(m) {
  const src = m && m.length === 4 ? m : IDENTITY;
  return src.map((row) => row.slice());
}

export function coerceMat4(raw, label) {
  if (!raw) throw new Error(`${label || "matrix"} 为空`);
  let rows = raw;
  if (typeof raw === "string") {
    const nums = raw.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
    if (nums.length !== 16 || nums.some((x) => !Number.isFinite(x))) {
      throw new Error(`${label || "matrix"} 需要 16 个有限数字`);
    }
    rows = [nums.slice(0, 4), nums.slice(4, 8), nums.slice(8, 12), nums.slice(12, 16)];
  } else if (Array.isArray(raw) && raw.length === 16 && !Array.isArray(raw[0])) {
    const nums = raw.map(Number);
    if (nums.some((x) => !Number.isFinite(x))) throw new Error(`${label || "matrix"} 含非数字`);
    rows = [nums.slice(0, 4), nums.slice(4, 8), nums.slice(8, 12), nums.slice(12, 16)];
  }
  if (!Array.isArray(rows) || rows.length !== 4) {
    throw new Error(`${label || "matrix"} 必须是 4×4`);
  }
  const out = [];
  for (let r = 0; r < 4; r++) {
    if (!Array.isArray(rows[r]) || rows[r].length !== 4) {
      throw new Error(`${label || "matrix"} 第 ${r} 行不是 4 列`);
    }
    out.push(rows[r].map((x) => {
      const n = Number(x);
      if (!Number.isFinite(n)) throw new Error(`${label || "matrix"}[${r}] 含非数字`);
      return n;
    }));
  }
  return out;
}

