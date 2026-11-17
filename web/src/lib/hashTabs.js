export const TAB_KEYS = ["cams", "train", "calib", "detect", "grasp", "arm"];

export function tabFromHash() {
  const h = (location.hash || "").replace(/^#\/?/, "");
  return TAB_KEYS.includes(h) ? h : "cams";
}

export function setHashTab(key) {
  const next = `#/${key}`;
  if (location.hash !== next) location.hash = next;
}
