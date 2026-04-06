import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import { ArrowRight, Upload, X, Sparkles, Shield, Zap, Brain, Download, ScanLine, Home as HomeIcon, CheckCircle, Clock, Users, Activity } from 'lucide-react'
import UploadCard from '../components/UploadCard'
import ProcessingLoader from '../components/ProcessingLoader'
import ResultCard from '../components/ResultCard'
import ExplainableAI from '../components/ExplainableAI'
import RecommendationPanel from '../components/RecommendationPanel'
import { useAuth } from '../context/AuthContext'

const LESION_CLASSES = [
  { code:'mel',   name:'Melanoma',             short:'MEL',  risk:'High',     riskColor:'#ef4444', bg:'#fef2f2', darkBg:'rgba(239,68,68,0.12)', border:'#fecaca', icon:'🔴', desc:'Most dangerous form of skin cancer. Arises from pigment-producing melanocytes.' },
  { code:'bcc',   name:'Basal Cell Carcinoma', short:'BCC',  risk:'High',     riskColor:'#ef4444', bg:'#fef2f2', darkBg:'rgba(239,68,68,0.12)', border:'#fecaca', icon:'🟥', desc:'Most common skin cancer. Rarely spreads but requires prompt treatment.' },
  { code:'akiec', name:'Actinic Keratosis',    short:'AK',   risk:'High',     riskColor:'#ef4444', bg:'#fef2f2', darkBg:'rgba(239,68,68,0.12)', border:'#fecaca', icon:'⚠️', desc:'Precancerous lesion caused by UV damage. Can evolve into squamous cell carcinoma.' },
  { code:'bkl',   name:'Benign Keratosis',     short:'BKL',  risk:'Moderate', riskColor:'#f59e0b', bg:'#fffbeb', darkBg:'rgba(245,158,11,0.12)', border:'#fde68a', icon:'🟡', desc:'Non-cancerous skin growth. Includes seborrheic keratoses and solar lentigines.' },
  { code:'df',    name:'Dermatofibroma',        short:'DF',   risk:'Moderate', riskColor:'#f59e0b', bg:'#fffbeb', darkBg:'rgba(245,158,11,0.12)', border:'#fde68a', icon:'🟠', desc:'Benign fibrous nodule in the skin. Generally harmless and rarely needs treatment.' },
  { code:'vasc',  name:'Vascular Lesion',       short:'VASC', risk:'Moderate', riskColor:'#f59e0b', bg:'#fffbeb', darkBg:'rgba(245,158,11,0.12)', border:'#fde68a', icon:'🫀', desc:'Includes cherry angiomas and hemangiomas. Generally benign but should be monitored.' },
  { code:'nv',    name:'Melanocytic Nevi',       short:'NV',   risk:'Low',      riskColor:'#10b981', bg:'#f0fdf4', darkBg:'rgba(16,185,129,0.12)', border:'#bbf7d0', icon:'🟢', desc:'Common moles. Benign but should be monitored for changes in size, shape, or color.' },
]

const downloadReport = async (selectedImage, result) => {
  if (!selectedImage || !result) return
  const canvas = document.createElement('canvas')
  const W = 900, H = 560
  canvas.width = W; canvas.height = H
  const ctx = canvas.getContext('2d')
  const grad = ctx.createLinearGradient(0, 0, W, H)
  grad.addColorStop(0, '#0a1628'); grad.addColorStop(0.5, '#0d1f3c'); grad.addColorStop(1, '#0a1628')
  ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = 'rgba(59,130,246,0.06)'
  for (let x = 0; x < W; x += 32) for (let y = 0; y < H; y += 32) { ctx.beginPath(); ctx.arc(x, y, 1, 0, Math.PI * 2); ctx.fill() }
  await new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      ctx.save(); roundRect(ctx, 30, 30, 340, 340, 16); ctx.fillStyle = '#112248'; ctx.fill()
      ctx.strokeStyle = '#1e3a6e'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.clip()
      const scale = Math.min(340 / img.width, 340 / img.height)
      const iw = img.width * scale, ih = img.height * scale
      ctx.drawImage(img, 30 + (340 - iw) / 2, 30 + (340 - ih) / 2, iw, ih); ctx.restore(); resolve()
    }
    img.onerror = resolve; img.src = URL.createObjectURL(selectedImage)
  })
  const rx = 400
  ctx.fillStyle = 'rgba(59,130,246,0.15)'; roundRect(ctx, rx, 30, W - rx - 30, 60, 12); ctx.fill()
  ctx.fillStyle = '#60a5fa'; ctx.font = 'bold 11px Inter, sans-serif'; ctx.textAlign = 'left'
  ctx.fillText('DERMASSIST AI  •  ANALYSIS REPORT', rx + 16, 55)
  ctx.fillStyle = '#e8f0ff'; ctx.font = 'bold 26px Inter, sans-serif'
  ctx.fillText(result.diagnosis_name || result.diagnosis, rx, 130)
  const riskColors = { 'High Risk': { bg: 'rgba(239,68,68,0.2)', border: '#ef4444', text: '#f87171' }, 'Moderate Risk': { bg: 'rgba(245,158,11,0.2)', border: '#f59e0b', text: '#fbbf24' }, 'Low Risk': { bg: 'rgba(16,185,129,0.2)', border: '#10b981', text: '#34d399' } }
  const rc = riskColors[result.risk_level] || riskColors['Low Risk']
  const riskText = result.risk_level || 'Low Risk'
  const badgeW = ctx.measureText(riskText).width + 24
  ctx.fillStyle = rc.bg; roundRect(ctx, rx, 145, badgeW, 28, 8); ctx.fill()
  ctx.strokeStyle = rc.border; ctx.lineWidth = 1.5; ctx.stroke()
  ctx.fillStyle = rc.text; ctx.font = 'bold 12px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(riskText, rx + badgeW / 2, 164)
  ctx.textAlign = 'left'; ctx.fillStyle = '#6b8fc2'; ctx.font = '13px Inter, sans-serif'; ctx.fillText('CONFIDENCE SCORE', rx, 215)
  const conf = Math.round((result.confidence || 0) * 100)
  ctx.fillStyle = '#e8f0ff'; ctx.font = 'bold 40px Inter, sans-serif'; ctx.fillText(`${conf}%`, rx, 260)
  ctx.fillStyle = '#1a3260'; roundRect(ctx, rx, 270, W - rx - 30, 8, 4); ctx.fill()
  const barW = ((W - rx - 30) * conf) / 100
  const barGrad = ctx.createLinearGradient(rx, 0, rx + barW, 0)
  barGrad.addColorStop(0, '#3b82f6'); barGrad.addColorStop(1, '#06b6d4')
  ctx.fillStyle = barGrad; roundRect(ctx, rx, 270, barW, 8, 4); ctx.fill()
  ctx.fillStyle = '#1a3260'; ctx.fillRect(0, H - 44, W, 44)
  ctx.fillStyle = '#2d4a78'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left'
  ctx.fillText('DermAssist AI  •  Not a substitute for professional medical advice', 24, H - 17)
  const link = document.createElement('a')
  link.download = `dermassist-report-${result.diagnosis}-${Date.now()}.png`
  link.href = canvas.toDataURL('image/png', 1.0); link.click()
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath()
}

// ── Radial Overlay ─────────────────────────────────────────────────────────────
const RadialOverlay = ({ onClose }) => {
  const [hovered, setHovered] = useState(null)
  const SIZE = 500, C = SIZE / 2, R = 185, NS = 80, NH = NS / 2
  const nodes = LESION_CLASSES.map((cls, i) => {
    const deg = -90 + (i * 360) / 7; const rad = (deg * Math.PI) / 180
    return { ...cls, cx: C + R * Math.cos(rad), cy: C + R * Math.sin(rad) }
  })
  useEffect(() => {
    const fn = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', fn); return () => window.removeEventListener('keydown', fn)
  }, [onClose])
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 flex items-center justify-center"
      style={{ zIndex: 9999, backgroundColor: 'rgba(6,13,31,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <motion.div initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.5, opacity: 0 }} transition={{ duration: 0.45, ease: [0.34, 1.56, 0.64, 1] }}
        onClick={e => e.stopPropagation()} style={{ position: 'relative', width: SIZE, height: SIZE }}
      >
        <svg width={SIZE} height={SIZE} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
          <motion.circle cx={C} cy={C} r={R} fill="none" stroke="rgba(147,197,253,0.4)" strokeWidth="1.5" strokeDasharray="8 5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} />
          {nodes.map((n, i) => (<motion.line key={n.code} x1={C} y1={C} x2={n.cx} y2={n.cy} stroke={n.riskColor} strokeWidth="1.5" strokeOpacity={hovered === i ? 0.8 : 0.2} strokeDasharray="5 4" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.15 + i * 0.06, duration: 0.4 }} />))}
        </svg>
        <motion.button onClick={onClose} whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.93 }} style={{ position: 'absolute', top: C - 48, left: C - 48, width: 96, height: 96, borderRadius: '50%', zIndex: 10 }}>
          <div className="w-full h-full rounded-full bg-blue-600 flex flex-col items-center justify-center" style={{ boxShadow: '0 0 0 12px rgba(59,130,246,0.15), 0 0 0 24px rgba(59,130,246,0.07), 0 8px 32px rgba(59,130,246,0.5)' }}>
            <p className="text-white font-black text-2xl leading-none">7</p>
            <p className="text-blue-200 text-[9px] font-bold tracking-widest mt-1">CLASSES</p>
          </div>
        </motion.button>
        {nodes.map((n, i) => (
          <motion.div key={n.code} style={{ position: 'absolute', top: C - NH, left: C - NH, width: NS, height: NS, cursor: 'pointer', zIndex: 5 }}
            initial={{ x: 0, y: 0, opacity: 0, scale: 0 }}
            animate={{ x: n.cx - C, y: n.cy - C, opacity: 1, scale: hovered === i ? 1.18 : 1 }}
            transition={{ x: { delay: 0.12 + i * 0.07, duration: 0.55, type: 'spring', stiffness: 200, damping: 18 }, y: { delay: 0.12 + i * 0.07, duration: 0.55, type: 'spring', stiffness: 200, damping: 18 }, opacity: { delay: 0.12 + i * 0.07, duration: 0.3 }, scale: { duration: 0.2 } }}
            onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}
          >
            <div className="w-full h-full rounded-full flex flex-col items-center justify-center select-none"
              style={{ background: n.darkBg, border: `3px solid ${hovered === i ? n.riskColor : n.riskColor + '50'}`, boxShadow: hovered === i ? `0 0 0 6px ${n.riskColor}25, 0 8px 28px ${n.riskColor}40` : '0 4px 16px rgba(0,0,0,0.3)', transition: 'all 0.15s' }}
            >
              <span style={{ fontSize: 22, lineHeight: 1 }}>{n.icon}</span>
              <span className="text-[10px] font-black mt-1 tracking-wide" style={{ color: n.riskColor }}>{n.short}</span>
            </div>
          </motion.div>
        ))}
        <AnimatePresence>
          {hovered !== null && (
            <motion.div key={LESION_CLASSES[hovered].code} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              style={{ position: 'absolute', bottom: -130, left: '50%', transform: 'translateX(-50%)', width: 320, zIndex: 20, pointerEvents: 'none' }}
            >
              <div className="rounded-2xl p-4 shadow-2xl border-2" style={{ background: LESION_CLASSES[hovered].darkBg, borderColor: LESION_CLASSES[hovered].riskColor, backdropFilter: 'blur(12px)' }}>
                <div className="flex items-center gap-3 mb-2">
                  <span style={{ fontSize: 28 }}>{LESION_CLASSES[hovered].icon}</span>
                  <div className="flex-1"><p className="font-black text-white text-sm">{LESION_CLASSES[hovered].name}</p><p className="text-[10px] text-white/50 font-mono tracking-widest">{LESION_CLASSES[hovered].code.toUpperCase()}</p></div>
                  <span className="text-xs font-bold px-2.5 py-1 rounded-full" style={{ background: LESION_CLASSES[hovered].riskColor + '25', color: LESION_CLASSES[hovered].riskColor }}>{LESION_CLASSES[hovered].risk} Risk</span>
                </div>
                <p className="text-xs text-white/70 leading-relaxed">{LESION_CLASSES[hovered].desc}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <motion.button onClick={onClose} whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} style={{ position: 'absolute', top: -16, right: -16, zIndex: 20 }}>
          <div className="w-10 h-10 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center shadow-xl"><X className="w-5 h-5 text-white" /></div>
        </motion.button>
      </motion.div>
    </motion.div>
  )
}

// ── Stats ───────────────────────────────────────────────────────────────────────
const StatsSection = () => (
  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-12">
    {[
      { icon: Users,    label: 'Patients Screened', value: '12,000+', color: '#3b82f6' },
      { icon: ScanLine, label: 'Scans Completed',   value: '28,500+', color: '#06b6d4' },
      { icon: Shield,   label: 'Accuracy Rate',     value: '94.2%',   color: '#10b981' },
      { icon: Clock,    label: 'Avg Scan Time',     value: '< 2 sec', color: '#8b5cf6' },
    ].map(({ icon: Icon, label, value, color }, i) => (
      <motion.div key={label} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}
        className="bg-white dark:bg-[#0d1f3c] rounded-2xl p-5 text-center border border-blue-50 dark:border-[#1a3260] shadow-md"
      >
        <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-3" style={{ background: color + '18' }}>
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
        <div className="text-2xl font-black text-gray-900 dark:text-white">{value}</div>
        <div className="text-xs text-gray-500 dark:text-[#6b8fc2] mt-1 font-medium">{label}</div>
      </motion.div>
    ))}
  </div>
)

// ── How It Works ────────────────────────────────────────────────────────────────
const HowItWorksSteps = () => (
  <section className="py-16 bg-gray-50 dark:bg-[#070e1c]">
    <div className="max-w-5xl mx-auto px-4">
      <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-12">
        <h2 className="text-3xl font-black text-gray-900 dark:text-[#e8f0ff]">How DermAssist Works</h2>
        <p className="text-gray-500 dark:text-[#6b8fc2] mt-2">Three simple steps to get your AI skin analysis</p>
      </motion.div>
      <div className="grid md:grid-cols-3 gap-6 relative">
        <div className="hidden md:block absolute top-10 left-[22%] right-[22%] h-0.5 bg-gradient-to-r from-blue-300 to-cyan-300 dark:from-blue-800 dark:to-cyan-800" />
        {[
          { step: '01', icon: Upload,      title: 'Upload Image',  desc: 'Take or upload a clear photo of the skin lesion you want analyzed', color: '#3b82f6' },
          { step: '02', icon: Brain,       title: 'AI Analysis',   desc: 'Our EfficientNet model analyzes the image across 7 skin lesion categories', color: '#8b5cf6' },
          { step: '03', icon: CheckCircle, title: 'Get Results',   desc: 'Receive instant diagnosis with risk level, confidence score, and doctor recommendations', color: '#10b981' },
        ].map(({ step, icon: Icon, title, desc, color }, i) => (
          <motion.div key={step} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.15 }}
            className="relative bg-white dark:bg-[#0d1f3c] rounded-2xl p-6 border border-blue-50 dark:border-[#1a3260] shadow-md text-center"
          >
            <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full flex items-center justify-center text-xs font-black text-white" style={{ background: color }}>
              {step}
            </div>
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mt-4 mb-4" style={{ background: color + '18' }}>
              <Icon className="w-7 h-7" style={{ color }} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-[#e8f0ff] mb-2">{title}</h3>
            <p className="text-sm text-gray-500 dark:text-[#6b8fc2] leading-relaxed">{desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  </section>
)

// ── ABCDE ───────────────────────────────────────────────────────────────────────
const ABCDE = [
  { letter: 'A', title: 'Asymmetry',  description: 'One half of the mole does not match the other half', color: '#3b82f6' },
  { letter: 'B', title: 'Border',     description: 'Edges are irregular, ragged, notched, or blurred', color: '#8b5cf6' },
  { letter: 'C', title: 'Color',      description: 'Color is not uniform — shades of brown, black, pink, red, white, or blue', color: '#6366f1' },
  { letter: 'D', title: 'Diameter',   description: 'The spot is larger than 6mm (about the size of a pencil eraser)', color: '#06b6d4' },
  { letter: 'E', title: 'Evolving',   description: 'The mole is changing in size, shape, or color over time', color: '#10b981' },
]

// ── Scan Tab ────────────────────────────────────────────────────────────────────
const ScanTab = ({ token }) => {
  const [selectedImage, setSelectedImage] = useState(null)
  const [isAnalyzing,   setIsAnalyzing]   = useState(false)
  const [result,        setResult]        = useState(null)
  const [error,         setError]         = useState(null)
  const [downloading,   setDownloading]   = useState(false)

  const handleAnalyze = async () => {
    if (!selectedImage) { setError('Please select an image first'); return }
    setIsAnalyzing(true); setError(null); setResult(null)
    const formData = new FormData()
    formData.append('file', selectedImage)
    try {
      const res = await axios.post(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          ...(token && { Authorization: `Bearer ${token}` }),
        },
      })
      setTimeout(() => { setResult(res.data); setIsAnalyzing(false) }, 4800)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze. Please ensure the backend is running.')
      setIsAnalyzing(false)
    }
  }

  const handleDownload = async () => {
    setDownloading(true)
    await downloadReport(selectedImage, result)
    setDownloading(false)
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-4 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
          <ScanLine className="w-4 h-4" /> AI Skin Analysis
        </div>
        <h2 className="text-3xl font-black text-gray-900 dark:text-[#e8f0ff]">Scan Your Skin</h2>
        <p className="text-gray-500 dark:text-[#6b8fc2] mt-2 font-medium">
          Upload or capture a photo — no extra details needed
        </p>
      </motion.div>

      {/* ── Upload Card ── */}
      <UploadCard
        onImageSelect={(file) => { setSelectedImage(file); setResult(null); setError(null) }}
        isLoading={isAnalyzing}
      />

      {/* ── Analyze Button ── */}
      {selectedImage && !isAnalyzing && !result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mt-6">
          <motion.button onClick={handleAnalyze} whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
            className="px-12 py-4 text-white rounded-2xl font-bold text-lg shadow-lg flex items-center gap-2 mx-auto"
            style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)', boxShadow: '0 8px 32px rgba(59,130,246,0.35)' }}
          >
            <Brain className="w-5 h-5" /> Analyze Image with AI
          </motion.button>
        </motion.div>
      )}

      {/* ── Error ── */}
      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-6 p-4 rounded-2xl text-sm font-medium border bg-red-50 border-red-200 text-red-700 dark:bg-red-900/20 dark:border-red-800/50 dark:text-red-400"
        >
          {error}
        </motion.div>
      )}

      {/* ── Processing ── */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-8">
            <ProcessingLoader />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Results ── */}
      <AnimatePresence>
        {result && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mt-8 space-y-6">
            <div className="flex justify-end">
              <motion.button onClick={handleDownload} disabled={downloading} whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.97 }}
                className="flex items-center gap-2.5 px-6 py-3 rounded-2xl text-sm font-bold shadow-lg text-white disabled:opacity-70"
                style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', boxShadow: '0 6px 24px rgba(59,130,246,0.35)' }}
              >
                {downloading
                  ? <><svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>Generating...</>
                  : <><Download className="w-4 h-4" />Download Report</>
                }
              </motion.button>
            </div>
            <ResultCard result={result} />
            <ExplainableAI diagnosis={result.diagnosis} allScores={result.all_scores} heatmapOverlay={result.heatmap_overlay} heatmapOnly={result.heatmap_only} usingStage2={result.using_stage2 && result.stage2_confidence > result.confidence} stage2Diagnosis={result.stage2_diagnosis} />
            <RecommendationPanel riskLevel={result.risk_level} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Tips ── */}
      <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
        className="mt-12 p-5 rounded-2xl border bg-amber-50 border-amber-200 dark:bg-amber-900/15 dark:border-amber-800/40">
        <p className="text-sm font-semibold text-amber-800 dark:text-amber-400 mb-2">📸 Tips for best results:</p>
        <ul className="text-sm text-amber-700 dark:text-amber-500 space-y-1">
          <li>• Use good lighting — natural daylight works best</li>
          <li>• Keep the camera steady and close (5–10 cm from lesion)</li>
          <li>• Make sure the lesion fills most of the frame</li>
          <li>• Avoid blurry or dark photos</li>
        </ul>
      </motion.div>
    </div>
  )
}

// ── Main Home ──────────────────────────────────────────────────────────────────
const Home = () => {
  const { token } = useAuth()
  const [activeTab,   setActiveTab]   = useState('home')
  const [showOverlay, setShowOverlay] = useState(false)

  const tabs = [
    { id: 'home', label: 'Home',    icon: HomeIcon },
    { id: 'scan', label: 'AI Scan', icon: ScanLine },
  ]

  return (
    <div className="min-h-screen bg-white dark:bg-[#0a1628]">
      <AnimatePresence>
        {showOverlay && <RadialOverlay onClose={() => setShowOverlay(false)} />}
      </AnimatePresence>

      {/* Tab Bar */}
      <div className="sticky top-16 z-40 bg-white dark:bg-[#0a1628] border-b border-gray-100 dark:border-[#1a3260] shadow-sm">
        <div className="max-w-7xl mx-auto px-4 flex gap-1 py-2">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 ${
                activeTab === id
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-600 dark:text-[#6b8fc2] hover:bg-gray-100 dark:hover:bg-[#0d1f3c]'
              }`}
            >
              <Icon className="w-4 h-4" />{label}
            </button>
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'home' && (
          <motion.div key="home" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>

            {/* Hero */}
            <section className="relative overflow-hidden py-20">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:hidden" />
              <div className="absolute inset-0 hidden dark:block" style={{ background: 'linear-gradient(135deg, #060d1f 0%, #0a1628 40%, #0d1f3c 70%, #0a1628 100%)' }} />
              <div className="absolute inset-0 opacity-40 dark:opacity-20" style={{ backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(59,130,246,0.15) 1px, transparent 0)', backgroundSize: '32px 32px' }} />

              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }} className="text-center">
                  <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-6 bg-blue-100 text-blue-700 border border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800/50"
                  >
                    <Sparkles className="w-3.5 h-3.5" /> AI-Powered Skin Cancer Detection
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                  </motion.div>

                  <h1 className="text-5xl md:text-6xl font-black text-gray-900 dark:text-white mb-6 leading-tight">
                    Detect Skin Cancer{' '}
                    <span style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Early</span>
                    <span className="block text-blue-600 dark:text-blue-400 mt-2">Using AI Technology</span>
                  </h1>

                  <p className="text-xl text-gray-600 dark:text-[#7a9cc8] max-w-3xl mx-auto mb-10 leading-relaxed font-medium">
                    Advanced artificial intelligence analyzes smartphone images of skin lesions to provide instant risk assessment. Early detection can save lives.
                  </p>

                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <motion.button onClick={() => setActiveTab('scan')}
                      whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.97 }}
                      className="px-8 py-4 text-white rounded-2xl font-bold text-lg flex items-center justify-center gap-2 shadow-lg"
                      style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', boxShadow: '0 8px 32px rgba(59,130,246,0.35)' }}
                    >
                      <ScanLine className="w-5 h-5" /> Start AI Scan
                    </motion.button>
                    <motion.a href="/how-it-works" whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.97 }}
                      className="px-8 py-4 rounded-2xl font-bold text-lg flex items-center justify-center gap-2 border-2 bg-white text-blue-600 border-blue-600 dark:bg-transparent dark:text-blue-400 dark:border-blue-500/60 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors"
                    >
                      Learn More <ArrowRight className="w-5 h-5" />
                    </motion.a>
                  </div>

                  <div className="flex items-center justify-center gap-8 mt-10 flex-wrap">
                    {[{ icon: Shield, text: 'HIPAA-aligned Privacy' }, { icon: Zap, text: 'Instant Results' }, { icon: Brain, text: '7 Cancer Classes' }].map(({ icon: Icon, text }) => (
                      <div key={text} className="flex items-center gap-2 text-sm text-gray-500 dark:text-[#4a6a9a] font-medium">
                        <Icon className="w-4 h-4 text-blue-400" /> {text}
                      </div>
                    ))}
                  </div>
                </motion.div>

                <StatsSection />

                <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.3 }}
                  className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16"
                >
                  {[
                    { title: '10,015 Images',  sub: 'ISIC 2018 Training Dataset',  icon: '🧬', color: '#3b82f6', desc: 'Trained on the gold-standard ISIC benchmark with balanced classes across 7 lesion types' },
                    { title: '7 Lesion Classes', sub: 'Interactive · Click to explore', icon: '🔬', color: '#8b5cf6', desc: 'From low-risk moles to high-risk melanoma — our model covers the full clinical spectrum', action: () => setShowOverlay(true) },
                    { title: '< 2 Seconds',    sub: 'AI Detection Speed',          icon: '⚡', color: '#10b981', desc: 'Preprocessing, inference, and report generation — all done in under 2 seconds on CPU' },
                  ].map(({ title, sub, icon, color, desc, action }, i) => (
                    <motion.div key={title} whileHover={{ y: -4 }} onClick={action}
                      className={`rounded-2xl p-7 border border-blue-50 dark:border-[#1a3260] bg-white dark:bg-[#0d1f3c] shadow-lg hover:shadow-xl transition-all duration-300 ${action ? 'cursor-pointer' : ''}`}
                    >
                      <div className="text-3xl mb-3">{icon}</div>
                      <div className="text-xl font-black text-gray-900 dark:text-[#e8f0ff]">{title}</div>
                      <div className="text-xs font-semibold mt-0.5 mb-3" style={{ color }}>{sub}</div>
                      <p className="text-sm text-gray-500 dark:text-[#6b8fc2] leading-relaxed">{desc}</p>
                      {action && <div className="mt-3 text-xs font-bold" style={{ color }}>Click to explore →</div>}
                    </motion.div>
                  ))}
                </motion.div>
              </div>
            </section>

            <HowItWorksSteps />

            {/* ABCDE */}
            <section className="py-16 bg-white dark:bg-[#0a1628]">
              <div className="max-w-6xl mx-auto px-4">
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-12">
                  <h2 className="text-4xl font-black text-gray-900 dark:text-[#e8f0ff] mb-4">The ABCDE Rule of Melanoma</h2>
                  <p className="text-lg text-gray-600 dark:text-[#6b8fc2] max-w-3xl mx-auto font-medium">Learn the warning signs that may indicate melanoma, the most serious type of skin cancer</p>
                </motion.div>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-5">
                  {ABCDE.map((item, index) => (
                    <motion.div key={item.letter} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.1 }} whileHover={{ y: -6 }}
                      className="rounded-2xl p-6 text-center border bg-white shadow-md hover:shadow-xl transition-all dark:bg-[#0d1f3c] dark:border-[#1a3260]"
                    >
                      <motion.div whileHover={{ scale: 1.1, rotate: 5 }} className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4 mx-auto shadow-md" style={{ background: item.color + '18', border: `2px solid ${item.color}40` }}>
                        <span className="text-3xl font-black" style={{ color: item.color }}>{item.letter}</span>
                      </motion.div>
                      <h3 className="text-lg font-black text-gray-900 dark:text-[#e8f0ff] mb-2">{item.title}</h3>
                      <p className="text-sm text-gray-500 dark:text-[#6b8fc2] leading-relaxed">{item.description}</p>
                    </motion.div>
                  ))}
                </div>
                <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="mt-10 p-6 rounded-2xl border bg-blue-50 border-blue-200 dark:bg-blue-900/15 dark:border-blue-800/40">
                  <p className="text-center text-gray-700 dark:text-[#a8c0e8] font-medium">
                    <strong className="text-gray-900 dark:text-[#e8f0ff]">Important:</strong> If you notice any of these warning signs, consult a dermatologist immediately. Early detection significantly improves treatment outcomes.
                  </p>
                </motion.div>
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mt-12 text-center">
                  <motion.button onClick={() => setActiveTab('scan')} whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.97 }}
                    className="px-10 py-4 text-white rounded-2xl font-bold text-lg shadow-lg inline-flex items-center gap-2"
                    style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', boxShadow: '0 8px 32px rgba(59,130,246,0.35)' }}
                  >
                    <ScanLine className="w-5 h-5" /> Check Your Skin Now
                  </motion.button>
                </motion.div>
              </div>
            </section>
          </motion.div>
        )}

        {activeTab === 'scan' && (
          <motion.div key="scan" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}
            className="bg-white dark:bg-[#0a1628] min-h-screen"
          >
            <ScanTab token={token} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Home