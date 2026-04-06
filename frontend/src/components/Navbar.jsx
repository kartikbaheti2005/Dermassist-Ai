import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Activity, User, LogOut, ChevronDown, Sun, Moon, Shield,
         Bot, Globe, Heart, Users } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'

const Navbar = () => {
  const location  = useLocation()
  const navigate  = useNavigate()
  const { user, logout, isLoggedIn } = useAuth()
  const { isDark, toggle }           = useTheme()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [featuresOpen, setFeaturesOpen] = useState(false)
  const dropdownRef = useRef(null)
  const featuresRef = useRef(null)

  const isActive = (path) => location.pathname === path

  const navLinks = [
    { path: '/',             label: 'Home'         },
    { path: '/how-it-works', label: 'How It Works' },
    { path: '/find-doctors', label: 'Find Doctors' },
    { path: '/about',        label: 'About'        },
  ]

  const featureLinks = [
    { path: '/chat',           label: 'AI Health Chat',  icon: Bot,   desc: 'Ask health questions'      },
    { path: '/outbreak',       label: 'Outbreak Alerts', icon: Globe, desc: 'Live disease intelligence' },
    { path: '/health-records', label: 'Health Records',  icon: Heart, desc: 'Your medical profile'      },
    { path: '/queue',          label: 'My Queue',        icon: Users, desc: "Today's queue position"    },
    { path: '/lesion-tracker', label: 'Lesion Tracker', icon: Activity, desc: 'Track mole changes over time' },
  ]

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setDropdownOpen(false)
      if (featuresRef.current && !featuresRef.current.contains(e.target)) setFeaturesOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleLogout = () => {
    logout(); navigate('/welcome'); setDropdownOpen(false)
  }

  const getInitials = (name) => {
    if (!name) return '?'
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
  }

  return (
    <nav className="bg-white dark:bg-gray-900 shadow-md sticky top-0 z-50 border-b border-gray-100 dark:border-gray-800 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">

          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 flex-shrink-0">
            <Activity className="w-8 h-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">DermAssist AI</span>
          </Link>

          {/* Desktop Nav Links */}
          <div className="hidden md:flex space-x-1 items-center">
            {navLinks.map((link) => (
              <Link key={link.path} to={link.path}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                  isActive(link.path)
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-blue-600'
                }`}
              >
                {link.label}
              </Link>
            ))}

            {/* Features Dropdown */}
            {isLoggedIn && (
              <div className="relative" ref={featuresRef}>
                <button
                  onClick={() => setFeaturesOpen(v => !v)}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                    featureLinks.some(f => isActive(f.path))
                      ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-blue-600'
                  }`}
                >
                  Features
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${featuresOpen ? 'rotate-180' : ''}`} />
                </button>

                {featuresOpen && (
                  <div className="absolute left-0 mt-2 w-60 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700 py-2 z-50">
                    {featureLinks.map(f => (
                      <Link key={f.path} to={f.path}
                        onClick={() => setFeaturesOpen(false)}
                        className={`flex items-center gap-3 px-4 py-2.5 transition-colors ${
                          isActive(f.path)
                            ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600'
                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-blue-600'
                        }`}
                      >
                        <div className="w-7 h-7 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                          <f.icon className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div>
                          <p className="text-sm font-medium leading-none">{f.label}</p>
                          <p className="text-[11px] text-gray-400 mt-0.5">{f.desc}</p>
                        </div>
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Admin Panel */}
            {isLoggedIn && user?.role === 'admin' && (
              <Link to="/admin"
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                  isActive('/admin')
                    ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700'
                    : 'text-purple-700 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20'
                }`}
              >
                <Shield className="w-4 h-4" /> Admin Panel
              </Link>
            )}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Theme toggle */}
            <button onClick={toggle}
              className="w-9 h-9 rounded-xl flex items-center justify-center bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-yellow-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {/* User avatar dropdown */}
            {isLoggedIn && user && (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setDropdownOpen(v => !v)}
                  className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors border border-gray-200 dark:border-gray-700"
                >
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                    {getInitials(user.full_name)}
                  </div>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300 hidden sm:block max-w-[100px] truncate">
                    {user.full_name}
                  </span>
                  <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`} />
                </button>

                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700 py-1 z-50">
                    <div className="px-4 py-2.5 border-b border-gray-100 dark:border-gray-700">
                      <p className="text-sm font-semibold text-gray-800 dark:text-white truncate">{user.full_name}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{user.email}</p>
                      {user.role === 'admin' && (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 mt-1">
                          <Shield className="w-2.5 h-2.5" /> ADMIN
                        </span>
                      )}
                    </div>

                    <Link to="/profile" onClick={() => setDropdownOpen(false)}
                      className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:text-blue-600 transition-colors">
                      <User className="w-4 h-4" /> My Profile
                    </Link>
                    <Link to="/find-doctors" onClick={() => setDropdownOpen(false)}
                      className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:text-blue-600 transition-colors">
                      <Activity className="w-4 h-4" /> Find Doctors
                    </Link>

                    <div className="border-t border-gray-100 dark:border-gray-700 mt-1 pt-1">
                      <p className="px-4 py-1 text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500">Features</p>
                      {featureLinks.map(f => (
                        <Link key={f.path} to={f.path} onClick={() => setDropdownOpen(false)}
                          className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:text-blue-600 transition-colors">
                          <f.icon className="w-4 h-4" /> {f.label}
                        </Link>
                      ))}
                    </div>

                    {user?.role === 'admin' && (
                      <Link to="/admin" onClick={() => setDropdownOpen(false)}
                        className="flex items-center gap-2 px-4 py-2.5 text-sm text-purple-700 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors font-medium border-t border-gray-100 dark:border-gray-700">
                        <Shield className="w-4 h-4" /> Admin Panel
                      </Link>
                    )}

                    <button onClick={handleLogout}
                      className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors border-t border-gray-100 dark:border-gray-700">
                      <LogOut className="w-4 h-4" /> Logout
                    </button>
                  </div>
                )}
              </div>
            )}

            {!isLoggedIn && (
              <Link to="/login"
                className="px-4 py-2 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors">
                Login
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar