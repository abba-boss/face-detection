import { useLocation, Link } from './router'

interface Props {
  to: string
  icon: string
  label: string
}

export function NavLink({ to, icon, label }: Props) {
  const { pathname } = useLocation()
  const active = to === '/' ? pathname === '/' : pathname.startsWith(to)

  return (
    <Link
      to={to}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '7px 20px',
        fontSize: 13,
        fontWeight: active ? 600 : 400,
        color: active ? 'var(--color-text)' : 'var(--color-muted)',
        background: active ? 'rgba(88,166,255,0.08)' : 'transparent',
        borderLeft: active ? '2px solid var(--color-blue)' : '2px solid transparent',
        textDecoration: 'none',
        transition: 'all 0.15s',
        cursor: 'pointer',
      }}
    >
      <span style={{ fontSize: 14, opacity: 0.8 }}>{icon}</span>
      {label}
    </Link>
  )
}
