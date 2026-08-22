/**
 * Minimal hash-based router — no external dependency needed.
 * Uses window.location.hash (#/path) for routing.
 */
import { createContext, useContext, useState, useEffect, type ReactNode, type CSSProperties, type ReactElement } from 'react'

interface RouterCtx { pathname: string }
const Ctx = createContext<RouterCtx>({ pathname: '/' })

export function Router({ children }: { children: ReactNode }) {
  const [pathname, setPathname] = useState(() =>
    window.location.hash.slice(1) || '/'
  )
  useEffect(() => {
    const handler = () => setPathname(window.location.hash.slice(1) || '/')
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])
  return <Ctx.Provider value={{ pathname }}>{children}</Ctx.Provider>
}

export function useLocation() {
  return useContext(Ctx)
}

export function Link({ to, children, style }: { to: string; children: ReactNode; style?: CSSProperties }) {
  return (
    <a href={`#${to}`} style={style}>
      {children}
    </a>
  )
}

interface RouteProps { path: string; element: ReactNode }
interface SwitchProps { children: ReactNode }

export function Route(_props: RouteProps) { return null }

export function Switch({ children }: SwitchProps) {
  const { pathname } = useLocation()
  let match: ReactNode = null
  const arr = Array.isArray(children) ? children : [children]
  for (const child of arr as ReactElement<RouteProps>[]) {
    if (!child?.props?.path) continue
    const p = child.props.path
    if (p === pathname || (p !== '/' && pathname.startsWith(p))) {
      match = child.props.element
      break
    }
    if (p === '/' && pathname === '/') {
      match = child.props.element
      break
    }
  }
  return <>{match}</>
}
