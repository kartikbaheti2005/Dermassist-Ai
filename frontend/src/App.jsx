import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import Layout from './layout/Layout'
import Home from './pages/Home'
import HowItWorks from './pages/HowItWorks'
import About from './pages/About'
import Safety from './pages/Safety'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import DoctorRegisterPage from './pages/DoctorRegisterPage'
import RoleSelectPage from './pages/RoleSelectPage'
import ProfilePage from './pages/ProfilePage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import FindDoctorsPage from './pages/FindDoctorsPage'
import BookAppointmentPage from './pages/BookAppointmentPage'
import AdminDashboard from './pages/AdminDashboard'
import DoctorDashboard from './pages/DoctorDashboard'

// ── New PS feature pages ──────────────────────────────────────────────────────
import HealthChatbotPage from './pages/HealthChatbotPage'       // PS-02
import OutbreakPage from './pages/OutbreakPage'                 // PS-05
import HealthRecordsPage from './pages/HealthRecordsPage'       // PS-06
import QueuePage from './pages/QueuePage'                       // PS-07
import LesionTrackerPage from './pages/LesionTrackerPage'           // Lesion Tracker

const Spinner = () => (
  <div className="min-h-screen flex items-center justify-center bg-white dark:bg-gray-900">
    <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
  </div>
)

const RoleRedirect = () => {
  const { user } = useAuth()
  if (user?.role === 'doctor') return <Navigate to="/doctor/dashboard" replace />
  if (user?.role === 'admin')  return <Navigate to="/admin" replace />
  return <Navigate to="/" replace />
}

const ProtectedRoute = ({ children }) => {
  const { isLoggedIn, loading, user } = useAuth()
  if (loading) return <Spinner />
  if (!isLoggedIn) return <Navigate to="/welcome" replace />
  if (user?.role === 'doctor') return <Navigate to="/doctor/dashboard" replace />
  if (user?.role === 'admin')  return <Navigate to="/admin" replace />
  return children
}

const DoctorRoute = ({ children }) => {
  const { isLoggedIn, loading, user } = useAuth()
  if (loading) return <Spinner />
  if (!isLoggedIn) return <Navigate to="/login" replace />
  if (user?.role !== 'doctor') return <RoleRedirect />
  return children
}

const AdminRoute = ({ children }) => {
  const { isLoggedIn, loading, user } = useAuth()
  if (loading) return <Spinner />
  if (!isLoggedIn) return <Navigate to="/login" replace />
  if (user?.role !== 'admin') return <RoleRedirect />
  return children
}

const AuthRoute = ({ children }) => {
  const { isLoggedIn, loading, user } = useAuth()
  if (loading) return <Spinner />
  if (isLoggedIn) return <RoleRedirect />
  return children
}

const CatchAll = () => {
  const { isLoggedIn, user } = useAuth()
  if (!isLoggedIn) return <Navigate to="/welcome" replace />
  if (user?.role === 'doctor') return <Navigate to="/doctor/dashboard" replace />
  if (user?.role === 'admin')  return <Navigate to="/admin" replace />
  return <Navigate to="/" replace />
}

const AppRoutes = () => (
  <Routes>
    {/* Public auth routes */}
    <Route path="/welcome"         element={<AuthRoute><RoleSelectPage /></AuthRoute>} />
    <Route path="/login"           element={<AuthRoute><LoginPage /></AuthRoute>} />
    <Route path="/register"        element={<AuthRoute><RegisterPage /></AuthRoute>} />
    <Route path="/register/doctor" element={<AuthRoute><DoctorRegisterPage /></AuthRoute>} />
    <Route path="/forgot-password" element={<ForgotPasswordPage />} />
    <Route path="/reset-password"  element={<ResetPasswordPage />} />

    {/* Admin — standalone */}
    <Route path="/admin" element={<AdminRoute><AdminDashboard /></AdminRoute>} />

    {/* Doctor dashboard — standalone */}
    <Route path="/doctor/dashboard" element={<DoctorRoute><DoctorDashboard /></DoctorRoute>} />

    {/* Protected user routes */}
    <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
      <Route path="/"                  element={<Home />} />
      <Route path="/how-it-works"      element={<HowItWorks />} />
      <Route path="/about"             element={<About />} />
      <Route path="/safety"            element={<Safety />} />
      <Route path="/profile"           element={<ProfilePage />} />
      <Route path="/find-doctors"      element={<FindDoctorsPage />} />
      <Route path="/appointments/book" element={<BookAppointmentPage />} />

      {/* ── New PS Feature Routes ─────────────────────────────────────────── */}
      <Route path="/chat"             element={<HealthChatbotPage />} />   {/* PS-02 */}
      <Route path="/outbreak"         element={<OutbreakPage />} />        {/* PS-05 */}
      <Route path="/health-records"   element={<HealthRecordsPage />} />   {/* PS-06 */}
      <Route path="/queue"            element={<QueuePage />} />           {/* PS-07 */}
    </Route>

    <Route path="*" element={<CatchAll />} />
  </Routes>
)

const App = () => (
  <ThemeProvider>
    <Router>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </Router>
  </ThemeProvider>
)

export default App