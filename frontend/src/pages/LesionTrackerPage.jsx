import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import { useAuth } from '../context/AuthContext'
import {
  Camera, Upload, TrendingUp, TrendingDown, Minus,
  Calendar, AlertCircle, CheckCircle, ChevronRight,
  Plus, Trash2, Eye, Clock, Activity, X
} from 'lucide-react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const RISK_COLORS = {
  'High Risk':     { bg: 'bg-red-100 dark:bg-red-900/20',     text: 'text-red-700 dark:text-red-400',     dot: 'bg-red-500',     border: 'border-red-200 dark:border-red-800' },
  'Moderate Risk': { bg: 'bg-amber-100 dark:bg-amber-900/20', text: 'text-amber-700 dark:text-amber-400', dot: 'bg-amber-400',   border: 'border-amber-200 dark:border-amber-800' },
  'Low Risk':      { bg: 'bg-emerald-100 dark:bg-emerald-900/20', text: 'text-emerald-700 dark:text-emerald-400', dot: 'bg-emerald-500', border: 'border-emerald-200 dark:border-emerald-800' },
}

const TrendIcon = ({ current, previous }) => {
  if (!previous) return null
  const curr = parseFloat(current)
  const prev = parseFloat(previous)
  if (curr > prev + 5)  return <TrendingUp  className="w-4 h-4 text-red-500" />
  if (curr < prev - 5)  return <TrendingDown className="w-4 h-4 text-emerald-500" />
  return <Minus className="w-4 h-4 text-gray-400" />
}

// ── Scan Card in timeline ──────────────────────────────────────────────────────
const TimelineScan = ({ scan, index, isLast, previousScan }) => {
  const risk   = scan.risk_level || 'Low Risk'
  const colors = RISK_COLORS[risk] || RISK_COLORS['Low Risk']
  const conf   = Math.round((scan.confidence_score || 0) * 100)
  const date   = scan.created_at
    ? new Date(scan.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
    : 'Unknown'
  const prevConf = previousScan ? Math.round((previousScan.confidence_score || 0) * 100) : null

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08, duration: 0.4 }}
      className="flex gap-4"
    >
      {/* Timeline line */}
      <div className="flex flex-col items-center flex-shrink-0">
        <div className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${colors.dot} ring-2 ring-white dark:ring-[#060d1f]`} />
        {!isLast && <div className="w-0.5 flex-1 bg-gray-200 dark:bg-[#1a3260] mt-1" />}
      </div>

      {/* Card */}
      <div className={`flex-1 mb-6 rounded-2xl border p-4 ${colors.bg} ${colors.border}`}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <span className="font-bold text-gray-900 dark:text-[#e8f0ff] text-sm">
                {scan.diagnosis_name || scan.predicted_label}
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold border ${colors.bg} ${colors.text} ${colors.border}`}>
                {risk}
              </span>
              {index === 0 && (
                <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                  Latest
                </span>
              )}
            </div>

            {/* Confidence */}
            <div className="flex items-center gap-2 mb-2">
              <div className="flex-1 bg-white/60 dark:bg-[#060d1f]/60 rounded-full h-1.5 overflow-hidden">
                <div className={`h-1.5 rounded-full ${colors.dot} transition-all`} style={{ width: `${conf}%` }} />
              </div>
              <div className="flex items-center gap-1 flex-shrink-0">
                <span className="text-xs font-bold text-gray-700 dark:text-gray-300">{conf}%</span>
                <TrendIcon current={conf} previous={prevConf} />
              </div>
            </div>

            {/* Date */}
            <div className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-gray-400" />
              <span className="text-xs text-gray-500 dark:text-[#6b8fc2]">{date}</span>
            </div>
          </div>

          {/* Scan image */}
          {scan.image_url && (
            <div className="w-14 h-14 rounded-xl overflow-hidden flex-shrink-0 border-2 border-white dark:border-[#0d1f3c] shadow-md">
              <img src={`${API}${scan.image_url}`} alt="scan"
                className="w-full h-full object-cover"
                onError={e => { e.target.style.display = 'none' }}
              />
            </div>
          )}
        </div>

        {/* Change indicator */}
        {previousScan && (
          <div className="mt-3 pt-3 border-t border-white/40 dark:border-[#1a3260]">
            {scan.risk_level !== previousScan.risk_level ? (
              <div className={`text-xs font-semibold flex items-center gap-1.5 ${
                risk === 'High Risk' ? 'text-red-600 dark:text-red-400' :
                risk === 'Low Risk'  ? 'text-emerald-600 dark:text-emerald-400' :
                'text-amber-600 dark:text-amber-400'
              }`}>
                <AlertCircle className="w-3.5 h-3.5" />
                Risk changed: {previousScan.risk_level} → {risk}
              </div>
            ) : (
              <div className="text-xs text-gray-500 dark:text-[#6b8fc2] flex items-center gap-1.5">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
                Risk level stable since last scan
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
const LesionTrackerPage = () => {
  const { token } = useAuth()
  const [scans,         setScans]         = useState([])
  const [loading,       setLoading]       = useState(true)
  const [selectedGroup, setSelectedGroup] = useState(null)
  const [groups,        setGroups]        = useState([])
  const [showNewGroup,  setShowNewGroup]  = useState(false)
  const [newGroupName,  setNewGroupName]  = useState('')
  const [creating,      setCreating]      = useState(false)
  const fileRef = useRef(null)

  useEffect(() => {
    if (!token) return
    axios.get(`${API}/user/scans`, { headers: { Authorization: `Bearer ${token}` } })
      .then(res => {
        const allScans = res.data.filter(s => s.predicted_label !== '__health_record__')
        setScans(allScans)
        // Auto-group by body location from scan metadata if available
        // For now, group all scans into one default group
        if (allScans.length > 0) {
          const defaultGroup = {
            id: 'all',
            name: 'All Scans',
            location: 'All locations',
            scans: allScans,
            latestRisk: allScans[0]?.risk_level || 'Low Risk',
          }
          setGroups([defaultGroup])
          setSelectedGroup(defaultGroup)
        }
      })
      .catch(() => setScans([]))
      .finally(() => setLoading(false))
  }, [token])

  // ── Stats ──────────────────────────────────────────────────────────────────
  const totalScans   = scans.length
  const highRiskScans = scans.filter(s => s.risk_level === 'High Risk').length
  const lastScan     = scans[0]
  const prevScan     = scans[1]
  const riskChanged  = lastScan && prevScan && lastScan.risk_level !== prevScan.risk_level

  return (
    <div className="min-h-screen py-10 px-4 bg-gradient-to-br from-blue-50/40 via-white to-cyan-50/40 dark:bg-none dark:bg-[#060d1f]">
      <div className="max-w-3xl mx-auto space-y-6">

        {/* ── Header ── */}
        <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-black text-gray-900 dark:text-[#e8f0ff] flex items-center gap-2">
                <Activity className="w-6 h-6 text-blue-500" />
                Lesion Tracker
              </h1>
              <p className="text-gray-500 dark:text-[#6b8fc2] text-sm mt-1">
                Monitor how your skin lesions change over time
              </p>
            </div>
            <motion.a
              href="/"
              whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
              className="flex items-center gap-2 px-4 py-2.5 rounded-2xl text-sm font-bold text-white shadow-lg"
              style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)' }}
            >
              <Camera className="w-4 h-4" /> New Scan
            </motion.a>
          </div>
        </motion.div>

        {/* ── Alert if risk changed ── */}
        {riskChanged && (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
            className={`rounded-2xl p-4 flex gap-3 border ${
              lastScan.risk_level === 'High Risk'
                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800'
            }`}
          >
            <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${lastScan.risk_level === 'High Risk' ? 'text-red-500' : 'text-emerald-500'}`} />
            <div>
              <p className={`font-bold text-sm ${lastScan.risk_level === 'High Risk' ? 'text-red-800 dark:text-red-300' : 'text-emerald-800 dark:text-emerald-300'}`}>
                ⚠️ Risk level changed since your last scan!
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                {prevScan.risk_level} → {lastScan.risk_level}. {lastScan.risk_level === 'High Risk' ? 'Please consult a dermatologist urgently.' : 'Risk has improved — keep monitoring.'}
              </p>
            </div>
          </motion.div>
        )}

        {/* ── Stats row ── */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Total Scans', value: totalScans, icon: Eye, color: '#3b82f6' },
            { label: 'High Risk', value: highRiskScans, icon: AlertCircle, color: '#ef4444' },
            { label: 'Latest Risk', value: lastScan?.risk_level?.replace(' Risk', '') || '—', icon: Activity, color: highRiskScans > 0 ? '#ef4444' : '#10b981' },
          ].map((stat, i) => (
            <motion.div key={stat.label}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
              className="rounded-2xl p-4 border bg-white dark:bg-[#0d1f3c] border-gray-100 dark:border-[#1a3260] shadow-sm text-center"
            >
              <stat.icon className="w-5 h-5 mx-auto mb-1.5" style={{ color: stat.color }} />
              <p className="text-xl font-black text-gray-900 dark:text-[#e8f0ff]">{stat.value}</p>
              <p className="text-xs text-gray-500 dark:text-[#6b8fc2] font-medium">{stat.label}</p>
            </motion.div>
          ))}
        </div>

        {/* ── Timeline ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
          className="rounded-3xl border bg-white dark:bg-[#0d1f3c] border-gray-100 dark:border-[#1a3260] shadow-sm p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-black text-gray-900 dark:text-[#e8f0ff] flex items-center gap-2">
              <Calendar className="w-5 h-5 text-blue-500" /> Scan Timeline
            </h2>
            <span className="text-xs text-gray-400 dark:text-[#2d4a78] bg-gray-100 dark:bg-[#112248] px-3 py-1 rounded-full font-medium">
              {totalScans} scan{totalScans !== 1 ? 's' : ''}
            </span>
          </div>

          {loading ? (
            <div className="flex flex-col items-center py-12 gap-3">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-gray-400 text-sm">Loading your scans...</p>
            </div>
          ) : scans.length === 0 ? (
            <div className="text-center py-14">
              <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}>
                <Activity className="w-14 h-14 text-gray-200 dark:text-[#1a3260] mx-auto mb-4" />
              </motion.div>
              <p className="text-gray-600 dark:text-[#6b8fc2] font-bold text-lg">No scans yet</p>
              <p className="text-gray-400 dark:text-[#2d4a78] text-sm mt-1.5">
                Upload a skin image on the home page to start tracking
              </p>
              <a href="/"
                className="inline-flex items-center gap-2 mt-5 px-5 py-2.5 rounded-2xl text-sm font-bold text-white"
                style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)' }}
              >
                <Camera className="w-4 h-4" /> Take First Scan
              </a>
            </div>
          ) : (
            <div className="relative">
              {scans.map((scan, i) => (
                <TimelineScan
                  key={scan.id}
                  scan={scan}
                  index={i}
                  isLast={i === scans.length - 1}
                  previousScan={i < scans.length - 1 ? scans[i + 1] : null}
                />
              ))}
            </div>
          )}
        </motion.div>

        {/* ── ABCDE reminder ── */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          className="rounded-2xl border border-blue-100 dark:border-[#1a3260] bg-blue-50/50 dark:bg-[#0d1f3c] p-5"
        >
          <p className="text-sm font-bold text-blue-800 dark:text-blue-400 mb-3">
            🔍 ABCDE Self-Check Rule — do this monthly
          </p>
          <div className="grid grid-cols-5 gap-2">
            {[
              { letter: 'A', label: 'Asymmetry', color: '#3b82f6' },
              { letter: 'B', label: 'Border',    color: '#8b5cf6' },
              { letter: 'C', label: 'Colour',    color: '#06b6d4' },
              { letter: 'D', label: 'Diameter',  color: '#10b981' },
              { letter: 'E', label: 'Evolving',  color: '#f59e0b' },
            ].map(item => (
              <div key={item.letter} className="text-center">
                <div className="w-9 h-9 rounded-xl mx-auto mb-1 flex items-center justify-center text-white font-black text-base"
                  style={{ background: item.color }}>
                  {item.letter}
                </div>
                <p className="text-[10px] text-gray-600 dark:text-[#6b8fc2] font-semibold">{item.label}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 dark:text-[#2d4a78] mt-3 leading-relaxed">
            If any of these change between scans, see a dermatologist promptly. Use this tracker to compare photos over time.
          </p>
        </motion.div>

      </div>
    </div>
  )
}

export default LesionTrackerPage