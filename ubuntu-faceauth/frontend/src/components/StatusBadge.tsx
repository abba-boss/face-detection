interface Props {
  ok: boolean
  labelOk?: string
  labelFail?: string
}

export function StatusBadge({ ok, labelOk = 'OK', labelFail = 'FAIL' }: Props) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: '0.05em',
        background: ok ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)',
        color: ok ? 'var(--color-green)' : 'var(--color-red)',
        border: `1px solid ${ok ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)'}`,
      }}
    >
      <span>{ok ? '●' : '●'}</span>
      {ok ? labelOk : labelFail}
    </span>
  )
}
