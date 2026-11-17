export function fmtXYZ(p) {
  if (!p || p.length < 3) return "—";
  return `[${Number(p[0]).toFixed(4)}, ${Number(p[1]).toFixed(4)}, ${Number(p[2]).toFixed(4)}] m`;
}

// Shared M2 → Arm Three.js waypoint markers


export function fmtCsv6(arr, digits = 4) {
  if (!arr || arr.length < 6) return "";
  const vals = [];
  for (let i = 0; i < 6; i++) {
    const v = Number(arr[i]);
    if (!Number.isFinite(v)) return "";
    vals.push(v.toFixed(digits));
  }
  return vals.join(",");
}


export function parseCsv6(text) {
  const parts = String(text || "")
    .trim()
    .split(/[,，\s]+/)
    .filter((s) => s.length);
  if (parts.length < 6) return { ok: false, error: "需要 6 个数（逗号或空格分隔）" };
  const vals = [];
  for (let i = 0; i < 6; i++) {
    const v = Number(parts[i]);
    if (!Number.isFinite(v)) return { ok: false, error: `第 ${i + 1} 个数无效: ${parts[i]}` };
    vals.push(v);
  }
  return { ok: true, values: vals };
}


export async function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
}

