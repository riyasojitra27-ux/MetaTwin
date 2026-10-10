export const access = "public";

const BASE = "https://metatwin-api-production.up.railway.app";
const ROUTES = {
  all: "/api/all",
  cohort: "/api/cohort",
  summary: "/api/summary",
  metrics: "/api/event-metrics",
  abstention: "/api/abstention",
  calibration: "/api/calibration"
};

export default async function (req, res) {
  const route = String(req.query?.route || "all");
  let path = ROUTES[route];
  if (route === "patient") {
    const id = Number(req.query?.id);
    if (!Number.isInteger(id) || id < 1 || id > 9999) return res.status(400).json({error:"Invalid patient id"});
    path = `/api/patient/${id}`;
  }
  if (!path) return res.status(400).json({error:"Unknown MetaTwin backend route"});
  try {
    const upstream = await fetch(BASE + path, {headers:{accept:"application/json"}});
    const text = await upstream.text();
    let body;
    try { body = JSON.parse(text); } catch { body = {error:"Backend returned non-JSON", detail:text.slice(0,300)}; }
    return res.status(upstream.status).json(body);
  } catch (error) {
    return res.status(502).json({error:"MetaTwin backend unavailable", detail:String(error?.message || error)});
  }
}