/** Backend API helper */
export const API = "";

export async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = await res.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail != null ? parsed.detail : detail;
      // FastAPI 422 validation: detail is often an array of {loc,msg,...}
      if (Array.isArray(detail)) {
        detail = detail.map((item) => {
          if (item && typeof item === 'object') {
            const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
            const msg = item.msg || item.message || JSON.stringify(item);
            return loc ? `${loc}: ${msg}` : String(msg);
          }
          return String(item);
        }).join('; ');
      } else if (detail && typeof detail === 'object') {
        detail = JSON.stringify(detail);
      }
    } catch (_) { /* keep text */ }
    throw new Error(detail || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}
