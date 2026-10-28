export default function Loader({ label = "Loading…", rows = 0 }: { label?: string; rows?: number }) {
  if (rows > 0) {
    return (
      <div className="space-y-2 py-2" role="status" aria-label={label}>
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="h-3 animate-pulse rounded-sm bg-zinc-100" style={{ width: `${90 - i * 12}%` }} />
        ))}
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 py-6 text-sm text-zinc-500" role="status">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-zinc-200 border-t-zinc-700" />
      {label}
    </div>
  );
}
