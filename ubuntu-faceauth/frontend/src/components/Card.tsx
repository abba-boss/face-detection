import { type ReactNode } from 'react'

interface Props {
  title?: string
  children: ReactNode
  className?: string
}

export function Card({ title, children, className = '' }: Props) {
  return (
    <div
      className={className}
      style={{
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      {title && (
        <div
          style={{
            padding: '10px 16px',
            borderBottom: '1px solid var(--color-border)',
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.08em',
            color: 'var(--color-muted)',
            textTransform: 'uppercase',
          }}
        >
          {title}
        </div>
      )}
      <div style={{ padding: 16 }}>{children}</div>
    </div>
  )
}
