/** Lightweight placeholder while a lazy tab chunk downloads. */
export default function PanelFallback({ label = "面板" }) {
  return (
    <div className="panel-fallback" role="status" aria-live="polite">
      <div className="panel-fallback-spinner" />
      <div>正在加载{label}…</div>
    </div>
  );
}
