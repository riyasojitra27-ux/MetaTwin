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

function TwinFigure({ active }: { active: boolean }) {
  const dots = useMemo(() => Array.from({ length: 170 }, (_, i) => {
    const t = i / 169;
    const y = 5 + t * 90;
    const center = 50 + Math.sin(t * Math.PI * 2) * 1.2;
    const width = t < .14 ? 7 + t * 18 : t < .3 ? 10 + (t - .14) * 34 : t < .62 ? 20 : t < .8 ? 18 - (t - .62) * 20 : 7;
    const x = center + (Math.sin(i * 17.13) * .52 + Math.sin(i * 4.77) * .48) * width;
    return { x, y, s: 1 + (i % 4) * .35, delay: (i % 19) * .07 };
  }), []);

  return (
    <div className={`twin-figure ${active ? 'is-active' : ''}`} aria-label="Particle representation of the MetaTwin digital twin">
      <div className="twin-glow" />
      {dots.map((d, i) => (
        <span key={i} className="twin-particle" style={{ left: `${d.x}%`, top: `${d.y}%`, width: d.s, height: d.s, animationDelay: `${d.delay}s` }} />
      ))}
      <div className="heart-orbit"><span /></div>
      <div className="center-point" />
    </div>
  );
}

function App() {
  const [data, setData] = useState<AllData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'overview' | 'trust' | 'about'>('overview');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API}/api/all`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setData(await response.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to connect to MetaTwin API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const summary = data?.summary || {};
  const persistence = Number(summary.mae_persistence);
  const xgbMae = Number(summary.mae_xgboost);
  const gain = persistence && xgbMae ? ((persistence - xgbMae) / persistence) * 100 : NaN;
  const reliability = Number(summary.auroc_failure);
  const trust = reliability >= .75 ? 'Forecast' : reliability >= .6 ? 'Caution' : 'Abstain';

  return (
    <main>
      <header className="topbar">
        <div className="wordmark">MetaTwin<span>.</span></div>
        <div className="topbar-right">
          <span className="research-status"><i /> Research prototype</span>
          <button className="refresh" onClick={load}>Refresh data</button>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="kicker">SELF-AWARE GLUCOSE DIGITAL TWIN</p>
          <h1>MetaTwin</h1>
          <h2>Forecast glucose.<br /><span>Measure reliability.</span><br />Know when to abstain.</h2>
          <p className="lede">
            A research prototype that separates <strong>what might happen next</strong> from <strong>how much the forecast should be trusted</strong>.
          </p>
          <div className="hero-actions">
            <button className="primary-action" onClick={() => setTab('overview')}>Explore the model <span>↓</span></button>
            <span className="hero-note">60-minute forecast · reliability-aware</span>
          </div>
        </div>

        <div className="twin-stage">
          <div className="stage-grid" />
          <div className="stage-caption"><span>LIVE MODEL STATE</span><b>{loading ? 'CONNECTING' : error ? 'OFFLINE' : 'ACTIVE'}</b></div>
          <TwinFigure active={!loading && !error} />
          <div className="trust-core">
            <small>TRUST STATE</small>
            <strong>{loading ? '—' : error ? 'Offline' : trust}</strong>
            <span>AUROC {metric(summary.auroc_failure, 2)}</span>
          </div>
          <div className="data-node node-one"><small>FORECAST HORIZON</small><strong>60 min</strong></div>
          <div className="data-node node-two"><small>MODEL MAE</small><strong>{metric(summary.mae_xgboost)} <em>mg/dL</em></strong></div>
          <div className="data-node node-three"><small>FAILURE RATE</small><strong>{metric(Number(summary.failure_prevalence) * 100)}%</strong></div>
          <div className="data-node node-four"><small>RELIABILITY</small><strong>{loading ? '—' : error ? '—' : trust}</strong></div>
        </div>
      </section>

      <nav className="section-nav">
        <button className={tab === 'overview' ? 'active' : ''} onClick={() => setTab('overview')}>Model overview</button>
        <button className={tab === 'trust' ? 'active' : ''} onClick={() => setTab('trust')}>Trust layer</button>
        <button className={tab === 'about' ? 'active' : ''} onClick={() => setTab('about')}>Research scope</button>
      </nav>

      {error && (
        <section className="error-banner">
          <div><strong>Model API unavailable.</strong> {error}</div>
          <button onClick={load}>Try again</button>
        </section>
      )}

      {tab === 'overview' && (
        <section className="content">
          <div className="section-intro">
            <div>
              <p className="kicker">MODEL EVIDENCE</p>
              <h3>Prediction with its limits attached.</h3>
            </div>
            <p>Values below are read directly from the model results API. No placeholder metrics are shown.</p>
          </div>

          <div className="metrics">
            <article><small>PERSISTENCE MAE</small><strong>{metric(summary.mae_persistence)}</strong><span>mg/dL · baseline</span></article>
            <article className="highlight"><small>XGBOOST MAE</small><strong>{metric(summary.mae_xgboost)}</strong><span>mg/dL · {Number.isFinite(gain) ? `${metric(gain)}% below baseline` : 'model result'}</span></article>
            <article><small>XGBOOST RMSE</small><strong>{metric(summary.rmse_xgboost)}</strong><span>mg/dL · 60-minute forecast</span></article>
            <article><small>FAILURE AUROC</small><strong>{metric(summary.auroc_failure, 2)}</strong><span>reliability detector</span></article>
          </div>

          <div className="two-panels">
            <article>
              <p className="kicker">01 · FORECAST</p>
              <h4>What might happen next?</h4>
              <p>The forecasting model estimates the near-term glucose trajectory. Its average accuracy is measured against a simple persistence baseline rather than presented in isolation.</p>
            </article>
            <article>
              <p className="kicker">02 · RELIABILITY</p>
              <h4>Should this prediction be trusted?</h4>
              <p>The reliability layer estimates when a forecast is likely to fail. If evidence is insufficient, MetaTwin can move toward an abstain state instead of implying certainty.</p>
            </article>
          </div>
        </section>
      )}

      {tab === 'trust' && (
        <section className="content">
          <div className="section-intro single">
            <div><p className="kicker">TRUST LAYER</p><h3>Reliability is an output, not a footnote.</h3></div>
          </div>
          <div className="trust-layout">
            <div className="trust-score"><small>FAILURE-DETECTION AUROC</small><strong>{loading || error ? '—' : metric(reliability, 2)}</strong><span>Higher means better separation between likely-successful and likely-failed forecasts.</span></div>
            <div className="trust-steps">
              <div><b>01</b><div><strong>Forecast</strong><p>Generate the next 60 minutes of glucose trajectory.</p></div></div>
              <div><b>02</b><div><strong>Score reliability</strong><p>Use prediction-time information to estimate whether that forecast may fail.</p></div></div>
              <div><b>03</b><div><strong>Abstain</strong><p>Surface uncertainty when the reliability signal is not strong enough.</p></div></div>
            </div>
          </div>
        </section>
      )}

      {tab === 'about' && (
        <section className="content">
          <div className="section-intro single"><div><p className="kicker">RESEARCH SCOPE</p><h3>Built as a research instrument, not a black box.</h3></div></div>
          <div className="research-grid">
            <div><p>MetaTwin is intended to remain useful after the competition: transparent metrics, explicit failure conditions, reproducible evaluation, and a clear boundary between what the model knows and what it does not.</p></div>
            <ul>
              <li>Compare forecasting models against simple baselines.</li>
              <li>Inspect abstention and calibration behaviour.</li>
              <li>Bring external datasets and test whether reliability transfers.</li>
              <li>Use the interface as a discussion artifact, not an autonomous decision maker.</li>
            </ul>
          </div>
          <div className="research-callout"><strong>Next milestone</strong><span>Out-of-fold residuals → patient-level bootstrap confidence intervals → real-data validation.</span></div>
        </section>
      )}

      <footer><span>MetaTwin · self-aware glucose digital twin</span><span>Forecast → Reliability → Abstain</span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
