// Cloudflare Pages worker for baadhi.pages.dev — one address, two modes.
//
//   LIVE  While the owner's laptop is online (scripts/go-live.ps1 has opened a tunnel to the real Baadhi server and stored its
//         address in KV), every request is relayed to it: visitors get the complete dashboard — analyse any area and date,
//         trace flood paths — exactly as it runs locally.
//   DEMO  Otherwise the static files next to this worker are served (saved analyses only), so the link is never dead.
//
// The visitor's address is relayed in X-Baadhi-Client, authenticated by X-Baadhi-Edge (a secret shared with the server), so
// the server can give every visitor a fair allowance. Nothing here stores or logs visitors.

const CACHE_MS = 20000;
let cache = { at: 0, origin: null, edge: "" };

async function liveOrigin(env) {
  const now = Date.now();
  if (now - cache.at < CACHE_MS) return cache;
  let origin = null;
  let edge = "";
  try {
    origin = await env.LIVE.get("origin");
    edge = (await env.LIVE.get("edge_token")) || "";
    if (origin) {
      origin = origin.replace(/\/+$/, "");
      const r = await fetch(origin + "/api/health", { signal: AbortSignal.timeout(3000), headers: { "accept-encoding": "identity" } });
      const j = r.ok ? await r.json() : null;
      if (!(j && j.app === "baadhi" && j.static !== true)) origin = null;
    }
  } catch (e) {
    origin = null;
  }
  cache = { at: now, origin, edge };
  return cache;
}

function relayHeaders(request, edge) {
  const h = new Headers(request.headers);
  h.delete("host");
  h.set("accept-encoding", "identity");                 // Cloudflare compresses for the visitor; keep the relay simple
  h.set("x-baadhi-edge", edge);
  h.set("x-baadhi-client", request.headers.get("cf-connecting-ip") || "");
  return h;
}

export default {
  async fetch(request, env) {
    const live = await liveOrigin(env);
    if (live.origin) {
      try {
        const u = new URL(request.url);
        const hasBody = !["GET", "HEAD"].includes(request.method);
        const resp = await fetch(live.origin + u.pathname + u.search, {
          method: request.method,
          headers: relayHeaders(request, live.edge),
          body: hasBody ? request.body : undefined,
          redirect: "manual",
        });
        const out = new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
        out.headers.set("x-baadhi-mode", "live");
        return out;
      } catch (e) {
        cache.at = 0;                                    // the engine went away: check again next time, serve the demo now
      }
    }
    const resp = await env.ASSETS.fetch(request);
    const out = new Response(resp.body, resp);
    out.headers.set("x-baadhi-mode", "demo");
    return out;
  },
};
