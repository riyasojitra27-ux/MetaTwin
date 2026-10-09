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
      {dots.map((d, i) => <span key={i} className="twin-particle" style={{ left: `${d.x}%`, top: `${d.y}%`, width: d.s, height: d.s, animationDelay: `${d.delay}s` }} />)}
      <div className="heart-orbit"><span /></div>
      <div className="center-point" />
    </div>
  );
}

function App() {
  const [data, setData] = useState<AllData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'clinician' | 'trust' | 'research'>('clinician');

  const load = async () => {
    setLoading(true); setError('');
    try {
      const response = await fetch(`${API}/api/all`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setData(await response.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to connect to MetaTwin API');
    } finally { setLoading(false); }
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
          <span className="research-status"><i /> Clinician-facing research prototype</span>
          <button className="refresh" onClick={load}>Refresh data</button>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="kicker">GLUCOSE DIGITAL TWIN · CLINICIAN VIEW</p>
          <h1>MetaTwin</h1>
          <h2>Predict glucose events.<br /><span>Measure reliability.</span><br />Know when to abstain.</h2>
          <p className="lede">
            A virtual patient that combines <strong>dynamic glucose, wearable and meal signals</strong> with <strong>static EHR context</strong> to forecast near-term glucose events and show how reliable that forecast is.
          </p>
          <div className="hero-actions">
            <button className="primary-action" onClick={() => setTab('clinician')}>Open clinician view <span>↓</span></button>
            <span className="hero-note">Adverse event window · 60 min · research prototype</span>
          </div>
        </div>

        <div className="twin-stage">
          <div className="stage-grid" />
          <div className="stage-caption"><span>VIRTUAL PATIENT</span><b>{loading ? 'CONNECTING' : error ? 'OFFLINE' : 'ACTIVE'}</b></div>
          <TwinFigure active={!loading && !error} />
          <div className="trust-core">
            <small>FORECAST RELIABILITY</small>
            <strong>{loading ? '—' : error ? 'Offline' : trust}</strong>
            <span>AUROC {metric(summary.auroc_failure, 2)}</span>
          </div>
          <div className="data-node node-one"><small>EVENT WINDOW</small><strong>60 min</strong></div>
          <div className="data-node node-two"><small>FORECAST ERROR</small><strong>{metric(summary.mae_xgboost)} <em>mg/dL MAE</em></strong></div>
          <div className="data-node node-three"><small>ADVERSE EVENT</small><strong>&gt;180 / &lt;70</strong></div>
          <div className="data-node node-four"><small>RELIABILITY</small><strong>{loading || error ? '—' : trust}</strong></div>
        </div>
      </section>

      <nav className="section-nav">
        <button className={tab === 'clinician' ? 'active' : ''} onClick={() => setTab('clinician')}>Clinician view</button>
        <button className={tab === 'trust' ? 'active' : ''} onClick={() => setTab('trust')}>Reliability layer</button>
        <button className={tab === 'research' ? 'active' : ''} onClick={() => setTab('research')}>Research evidence</button>
      </nav>

      {error && <section className="error-banner"><div><strong>Model API unavailable.</strong> {error}</div><button onClick={load}>Try again</button></section>}

      {tab === 'clinician' && (
        <section className="content">
          <div className="section-intro">
            <div><p className="kicker">01 · VIRTUAL PATIENT</p><h3>What should the clinician see?</h3></div>
            <p>The dashboard is centered on the required clinical event: glucose crossing 180 mg/dL or 70 mg/dL within the prediction window.</p>
          </div>

          <div className="patient-shell">
            <div className="patient-header">
              <div><small>VIRTUAL PATIENT</small><strong>Patient 01 · Synthetic cohort</strong><span>Digital twin assembled from dynamic + static patient data</span></div>
              <div className="patient-state"><small>MODEL STATE</small><b>{loading ? 'Connecting' : error ? 'Offline' : 'Active'}</b></div>
            </div>
            <div className="patient-grid">
              <article className="patient-card profile-card">
                <p className="kicker">PATIENT CONTEXT</p>
                <h4>Two-stream patient representation</h4>
                <div className="stream-row"><span>Dynamic</span><b>CGM · HR · activity · meals</b></div>
                <div className="stream-row"><span>Static EHR</span><b>Age · BMI · HbA1c · diagnoses</b></div>
                <div className="stream-row"><span>Twin goal</span><b>Forecast near-term glucose</b></div>
              </article>

              <article className="patient-card event-card">
                <p className="kicker">02 · ADVERSE EVENT LAYER</p>
                <div className="event-title"><h4>What could happen next?</h4><span>60 MIN</span></div>
                <div className="event-thresholds">
                  <div><strong>&gt;180</strong><span>mg/dL · hyperglycemic event</span></div>
                  <div><strong>&lt;70</strong><span>mg/dL · hypoglycemic event</span></div>
                </div>
                <p className="event-note">The event layer evaluates whether the forecast crosses either threshold within 60 minutes. Reliability gating determines whether an alert should be surfaced or withheld.</p>
              </article>
            </div>
          </div>

          <div className="clinical-flow">
            <article><span>01</span><b>Patient data</b><p>CGM + wearable + meals + EHR</p></article>
            <div className="flow-arrow">→</div>
            <article><span>02</span><b>Digital twin</b><p>Personalized glucose trajectory</p></article>
            <div className="flow-arrow">→</div>
            <article><span>03</span><b>Event prediction</b><p>&gt;180 or &lt;70 within 60 min</p></article>
            <div className="flow-arrow">→</div>
            <article><span>04</span><b>Reliability gate</b><p>Forecast · Caution · Abstain</p></article>
          </div>

          <div className="metrics clinical-metrics">
            <article><small>FORECAST MAE</small><strong>{metric(summary.mae_xgboost)}</strong><span>mg/dL · 60-minute model</span></article>
            <article className="highlight"><small>FAILURE AUROC</small><strong>{metric(summary.auroc_failure, 2)}</strong><span>reliability detector</span></article>
            <article><small>FAILURE PREVALENCE</small><strong>{metric(Number(summary.failure_prevalence) * 100)}%</strong><span>current evaluation data</span></article>
            <article><small>BASELINE MAE</small><strong>{metric(summary.mae_persistence)}</strong><span>mg/dL · persistence</span></article>
          </div>

          <div className="clinical-boundary"><strong>Research boundary</strong><span>This prototype predicts and evaluates glucose-event risk; it does not diagnose, prescribe treatment, or replace clinical judgment.</span></div>
        </section>
      )}

      {tab === 'trust' && (
        <section className="content">
          <div className="section-intro single"><div><p className="kicker">RELIABILITY LAYER</p><h3>The event prediction is only surfaced when its reliability is understood.</h3></div></div>
          <div className="trust-layout">
            <div className="trust-score"><small>FAILURE-DETECTION AUROC</small><strong>{loading || error ? '—' : metric(reliability, 2)}</strong><span>Measures separation between likely-successful and likely-failed glucose forecasts.</span></div>
            <div className="trust-steps">
              <div><b>01</b><div><strong>Forecast</strong><p>Estimate the patient's glucose trajectory over the next 60 minutes.</p></div></div>
              <div><b>02</b><div><strong>Predict forecast failure</strong><p>Use information available at prediction time to estimate whether the forecast may be wrong.</p></div></div>
              <div><b>03</b><div><strong>Gate the clinical event</strong><p>High reliability can surface an event alert; low reliability can withhold or downgrade it.</p></div></div>
              <div><b>04</b><div><strong>Abstain</strong><p>When evidence is insufficient, MetaTwin explicitly avoids presenting false certainty.</p></div></div>
            </div>
          </div>
        </section>
      )}

      {tab === 'research' && (
        <section className="content">
          <div className="section-intro single"><div><p className="kicker">RESEARCH EVIDENCE</p><h3>Measure whether the twin deserves to be trusted.</h3></div></div>
          <div className="metrics">
            <article><small>PERSISTENCE MAE</small><strong>{metric(summary.mae_persistence)}</strong><span>mg/dL · baseline</span></article>
            <article className="highlight"><small>XGBOOST MAE</small><strong>{metric(summary.mae_xgboost)}</strong><span>mg/dL · {Number.isFinite(gain) ? `${metric(gain)}% below baseline` : 'model result'}</span></article>
            <article><small>XGBOOST RMSE</small><strong>{metric(summary.rmse_xgboost)}</strong><span>mg/dL · 60-minute forecast</span></article>
            <article><small>FAILURE AUPRC</small><strong>{metric(summary.auprc_failure, 2)}</strong><span>reliability detector</span></article>
          </div>
          <div className="two-panels">
            <article><p className="kicker">FORECASTING</p><h4>What might happen next?</h4><p>The forecaster estimates near-term glucose and is evaluated against a persistence baseline. The primary horizon is 60 minutes.</p></article>
            <article><p className="kicker">ADVERSE EVENT</p><h4>What specific event are we predicting?</h4><p>Glucose above 180 mg/dL or below 70 mg/dL within the prediction window. Reliability gating is used to control whether an event alert should be surfaced.</p></article>
          </div>
          <div className="research-callout"><strong>Validation status</strong><span>These displayed metrics are the current model-results API values. Final submission results should use out-of-fold residuals, patient-level bootstrap confidence intervals, and the completed event-layer evaluation.</span></div>
        </section>
      )}

      <footer><span>MetaTwin · glucose digital twin</span><span>Patient → Event → Reliability → Abstain</span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
