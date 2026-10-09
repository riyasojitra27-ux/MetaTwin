import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Summary = Record<string, number>;
type Curve = Record<string, unknown>[];
type AllData = { summary: Summary; abstention: Curve; calibration: Curve };

const API = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

function metric(v: unknown, digits = 1) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}

function ParticleHuman({ active }: { active: boolean }) {
  const dots = useMemo(() => Array.from({ length: 155 }, (_, i) => {
    const t = i / 154;
    const y = 7 + t * 86;
    const center = 50 + Math.sin(t * Math.PI * 2) * 1.5;
    const width = t < .15 ? 5 + t * 22 : t < .32 ? 9 + (t-.15)*35 : t < .55 ? 20 : t < .78 ? 18 - (t-.55)*18 : 7;
    const x = center + (Math.sin(i * 17.13) * .55 + Math.sin(i * 4.77) * .45) * width;
    return { x, y, s: 1.1 + (i % 4) * .35, delay: (i % 17) * .06 };
  }), []);
  return <div className={`human ${active ? 'human-active' : ''}`} aria-label="Particle representation of the digital twin">
    <div className="human-aura" />
    {dots.map((d, i) => <span key={i} className="particle" style={{ left: `${d.x}%`, top: `${d.y}%`, width: d.s, height: d.s, animationDelay: `${d.delay}s` }} />)}
    <div className="chest-ring" />
  </div>;
}

function App() {
  const [data, setData] = useState<AllData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'overview' | 'trust' | 'about'>('overview');

  const load = async () => {
    setLoading(true); setError('');
    try {
      const r = await fetch(`${API}/api/all`);
      if (!r.ok) throw new Error(`API returned ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to connect to MetaTwin API');
    } finally { setLoading(false); }
  };
  useEffect(() => { load(); }, []);

  const s = data?.summary || {};
  const gain = Number(s.mae_persistence) && Number(s.mae_xgboost) ? ((Number(s.mae_persistence)-Number(s.mae_xgboost))/Number(s.mae_persistence))*100 : NaN;
  const reliability = Number(s.auroc_failure);
  const trust = reliability >= .75 ? 'Forecast' : reliability >= .6 ? 'Caution' : 'Abstain';

  return <main>
    <header className="topbar">
      <div className="brand"><span className="mark">M</span><div><strong>MetaTwin</strong><small>reliability-aware glucose forecasting</small></div></div>
      <div className="status"><span className="status-dot" /> research prototype <button onClick={load}>↻ refresh</button></div>
    </header>

    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">SELF-AWARE DIGITAL TWIN</p>
        <h1>Predict the trajectory.<br/><em>Know when not to trust it.</em></h1>
        <p className="lede">MetaTwin separates two questions: <b>what might happen next?</b> and <b>how reliable is that forecast?</b> When reliability is poor, the system can abstain instead of presenting false confidence.</p>
        <nav className="tabs">
          <button className={tab==='overview'?'active':''} onClick={()=>setTab('overview')}>Overview</button>
          <button className={tab==='trust'?'active':''} onClick={()=>setTab('trust')}>Trust layer</button>
          <button className={tab==='about'?'active':''} onClick={()=>setTab('about')}>How to use</button>
        </nav>
      </div>
      <div className="twin-stage">
        <div className="stage-label">DIGITAL TWIN / LIVE MODEL STATE</div>
        <ParticleHuman active={!loading && !error}/>
        <div className="trust-core"><span>TRUST CORE</span><strong>{loading ? '—' : error ? 'OFFLINE' : trust}</strong><small>failure AUROC {metric(s.auroc_failure, 2)}</small></div>
        <div className="node node-a"><span>FORECAST</span><b>60 min</b></div>
        <div className="node node-b"><span>MODEL MAE</span><b>{metric(s.mae_xgboost)} mg/dL</b></div>
        <div className="node node-c"><span>FAILURE RATE</span><b>{metric(Number(s.failure_prevalence)*100)}%</b></div>
        <div className="node node-d"><span>STATE</span><b>{loading ? 'loading' : error ? 'offline' : trust}</b></div>
      </div>
    </section>

    {error && <section className="error"><b>Model API unavailable.</b> {error}<button onClick={load}>Try again</button><small>Set VITE_API_BASE_URL for a deployed API. No fabricated values are shown.</small></section>}

    {tab === 'overview' && <section className="content">
      <div className="section-head"><div><p className="eyebrow">EVIDENCE</p><h2>Forecast quality, with uncertainty attached.</h2></div><span>Values are read directly from the model results API.</span></div>
      <div className="metrics">
        <article><span>Persistence MAE</span><strong>{metric(s.mae_persistence)} <i>mg/dL</i></strong><small>simple baseline</small></article>
        <article className="featured"><span>XGBoost MAE</span><strong>{metric(s.mae_xgboost)} <i>mg/dL</i></strong><small>{Number.isFinite(gain) ? `${metric(gain)}% lower than persistence` : 'model result'}</small></article>
        <article><span>XGBoost RMSE</span><strong>{metric(s.rmse_xgboost)} <i>mg/dL</i></strong><small>60-minute forecast</small></article>
        <article><span>Failure AUROC</span><strong>{metric(s.auroc_failure, 2)}</strong><small>reliability detector</small></article>
      </div>
      <div className="explain-grid">
        <article className="panel"><p className="eyebrow">WHY ABSTENTION MATTERS</p><h3>Accuracy is not enough.</h3><p>A forecast can look strong on average while being unreliable for particular moments. MetaTwin makes reliability a first-class output so downstream users can distinguish <b>forecast</b>, <b>caution</b>, and <b>abstain</b> states.</p></article>
        <article className="panel"><p className="eyebrow">SAFETY BOUNDARY</p><h3>Research tool, not medical advice.</h3><p>This prototype does not diagnose, dose medication, or replace a clinician. Its useful contribution is a reproducible way to evaluate forecast error, calibration and abstention before a model is trusted.</p></article>
      </div>
    </section>}

    {tab === 'trust' && <section className="content"><div className="section-head"><div><p className="eyebrow">TRUST LAYER</p><h2>Make uncertainty inspectable.</h2></div></div><div className="trust-layout"><div className="big-trust"><span>MODEL RELIABILITY</span><strong>{loading ? '—' : error ? '—' : metric(reliability, 2)}</strong><small>failure-detection AUROC</small></div><div className="steps"><div><b>01 / FORECAST</b><p>Generate the near-term glucose trajectory.</p></div><div><b>02 / SCORE</b><p>Estimate whether the forecast is likely to fail using prediction-time information only.</p></div><div><b>03 / ABSTAIN</b><p>When trust is insufficient, surface uncertainty instead of false precision.</p></div></div></div></section>}

    {tab === 'about' && <section className="content"><div className="section-head"><div><p className="eyebrow">OPEN RESEARCH TOOL</p><h2>Designed to be useful beyond the competition.</h2></div></div><div className="about"><p>MetaTwin is being built as a reproducible research prototype: transparent metrics, explicit failure conditions, and a clear boundary between what the model knows and what it does not.</p><ul><li>Researchers can compare a forecasting model against simple baselines.</li><li>Students can inspect abstention and calibration concepts without a black-box UI.</li><li>Future contributors can plug in external datasets and test whether reliability transfers.</li><li>Clinicians can use the interface as a discussion artifact—not as an autonomous decision maker.</li></ul><div className="callout"><b>Next research milestone:</b> out-of-fold residuals, patient-level bootstrap confidence intervals, then real-data validation. The current dashboard will never silently turn missing evidence into a confident answer.</div></div></section>}

    <footer><span>MetaTwin · research prototype</span><span>Forecast → Reliability → Abstain</span></footer>
  </main>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
