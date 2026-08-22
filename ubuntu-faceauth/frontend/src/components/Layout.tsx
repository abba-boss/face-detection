import { type ReactNode } from 'react'
import { NavLink } from './NavLink'

interface Props { children: ReactNode }

export function Layout({ children }: Props) {
  return (
    <div className="flex min-h-screen" style={{ background: 'var(--color-bg)' }}>
      {/* Sidebar */}
      <aside
        className="w-56 flex-shrink-0 flex flex-col border-r"
        style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
      >
        {/* Logo */}
        <div
          className="px-5 py-4 border-b"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div className="flex items-center gap-2">
            <span style={{ color: 'var(--color-green)', fontSize: 18 }}>⬡</span>
            <span style={{ color: 'var(--color-text)', fontWeight: 600, fontSize: 13, letterSpacing: '0.05em' }}>
              FACEAUTH
            </span>
          </div>
          <div style={{ color: 'var(--color-muted)', fontSize: 11, marginTop: 2 }}>
            Ubuntu Security
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3">
          <NavLink to="/"        icon="⊞" label="Dashboard" />
          <NavLink to="/users"   icon="◈" label="Users"     />
          <NavLink to="/logs"    icon="≡" label="Logs"      />
          <NavLink to="/doctor"  icon="✦" label="Doctor"    />
          <NavLink to="/settings" icon="◎" label="Settings" />
        </nav>

        {/* Footer */}
        <div
          className="px-5 py-3 border-t text-xs"
          style={{ borderColor: 'var(--color-border)', color: 'var(--color-muted)' }}
        >
          API: 127.0.0.1:8765
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto" style={{ minWidth: 0 }}>
        {children}
      </main>
    </div>
  )
}
