export function Spinner() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--color-muted)', fontSize: 13 }}>
      <span
        style={{
          display: 'inline-block',
          width: 14,
          height: 14,
          border: '2px solid var(--color-border)',
          borderTopColor: 'var(--color-blue)',
          borderRadius: '50%',
          animation: 'spin 0.7s linear infinite',
        }}
      />
      Loading…
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}
