interface Props { message: string; onRetry?: () => void }

export function ErrorBox({ message, onRetry }: Props) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      gap: 12, padding: '10px 14px', borderRadius: 6,
      background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)',
      color: 'var(--color-red)', fontSize: 13,
    }}>
      <span>⚠ {message}</span>
      {onRetry && (
        <button onClick={onRetry} style={{
          background: 'none', border: '1px solid rgba(248,81,73,0.4)',
          color: 'var(--color-red)', borderRadius: 4, padding: '2px 10px',
          cursor: 'pointer', fontSize: 12,
        }}>Retry</button>
      )}
    </div>
  )
}
