"""Daily football research agent (top-5 leagues).
Finds today's games, researches each with web search, runs the Edge Engine v2
prompt, logs picks, settles yesterday's results, builds docs/index.html.
Paper tracking only - no real money."""
import os, json, re, pathlib, datetime as dt
import anthropic

MODEL = os.environ.get("MODEL", "claude-sonnet-5")
MAX_MATCHES = int(os.environ.get("MAX_MATCHES", "12"))
SHRINK = float(os.environ.get("SHRINK", "0.25"))      # blend toward market (calibration)
MIN_EV = float(os.environ.get("MIN_EV", "0.05"))      # min edge to flag a bet
KELLY_FRAC = float(os.environ.get("KELLY_FRAC", "0.25"))
KELLY_CAP = 0.03
MAX_ODDS = 8.0
LEAGUES = "Premier League, La Liga, Serie A, Bundesliga, Ligue 1"

ROOT = pathlib.Path(__file__).parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
DATA.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
LEDGER = DATA / "ledger.json"
client = anthropic.Anthropic()

# ---------------------------------------------------------------- PROMPTS
# "Pitch Court": two stages. The SCOUT only collects facts (web search).
# The COURT never searches: it judges the dossier through six specialists.
# Rename it, change the weights, add your own specialists - that is what makes it yours.
SCOUT = """You are the Scout of a football analysis desk. You collect FACTS ONLY, no opinions.
Today is {DATE}. Match: {HOME} vs {AWAY} ({LEAGUE}), kickoff {KICKOFF} UTC.
Use web search (prefer fbref, understat, clubelo, transfermarkt, club sites,
Pinnacle/Betfair/oddsportal, reliable local press). Mark anything older than
7 days as stale. Never invent numbers; use null when unknown.

Return ONLY JSON inside <json></json> tags:
<json>{"elo":{"home":0,"away":0,"home_trend_5":0,"away_trend_5":0},
"xg_last8":{"home_for":0,"home_against":0,"away_for":0,"away_against":0,
            "home_at_home_for":0,"home_at_home_against":0,
            "away_away_for":0,"away_away_against":0},
"points_vs_xpoints":{"home":0,"away":0},
"lineups":{"home":"confirmed|probable|unknown + XI/absences","away":"..."},
"key_absences_impact":"who matters most and why, max 40 words",
"rest_days":{"home":0,"away":0},
"midweek_europe":{"home":false,"away":false},
"travel_or_weather":"short",
"situation":"manager change, derby, motivation, table stakes, max 40 words",
"media_narrative":"what press/pundits are saying, max 30 words",
"odds":{"home":0.0,"draw":0.0,"away":0.0},
"sharp_odds":{"home":0.0,"draw":0.0,"away":0.0},
"opening_odds":{"home":0.0,"draw":0.0,"away":0.0},
"line_move":"toward home|draw|away|none|unknown",
"sources":["names only"],"stale_items":["..."]}</json>
odds = best available price. sharp_odds = Pinnacle or Betfair exchange."""

COURT = """You preside over Pitch Court for {HOME} vs {AWAY} ({LEAGUE}), {KICKOFF} UTC.
You have NO web access. Use ONLY the dossier below. If a fact is missing, treat it
as unknown and widen uncertainty; never fill gaps from memory.

DOSSIER: {DOSSIER}

Six specialists each speak, then you judge:
1. STATISTICIAN - Dixon-Coles Poisson from the xG splits (shrink small samples to
   league average, league home advantage, rho about -0.10, full score grid) blended
   80/20 with Elo. Output 1X2 probabilities.
2. REGRESSION HUNTER - compare points vs xPoints. Teams overperforming their xG
   are likely to fall back; underperformers to recover. Re-estimate 1X2 as if
   results-luck reverts by half. Flag "overperforming", "underperforming" or "none".
3. MARKET READER - take sharp no-vig probabilities, then read the line move and
   opening vs current price. What does the market know that the numbers miss?
   Output 1X2 probabilities.
4. FATIGUE AUDITOR - rest days, midweek Europe, travel, rotation risk.
   Give adjustments to home/away win probability, maximum +-3 points.
5. NARRATIVE SKEPTIC - gap between media narrative and the numbers. The market
   already prices narrative, so only adjust (max +-2 points) when the narrative
   contradicts hard evidence.
6. FRAGILITY INSPECTOR - if the single most important player is absent, or the
   lineup differs from expected, how far does the result probability swing?
   Score 0 (robust) to 10 (one team-sheet change flips the pick). Write one
   kill-switch: the exact news that should cancel a bet on this match.

JUDGE: final = 40% Statistician + 25% Regression Hunter + 35% Market Reader,
then apply the Fatigue and Narrative adjustments. If final differs from sharp
no-vig by more than 8 points on any outcome, you need two concrete facts from the
dossier, else move toward the market. Then write the strongest case against your
biggest edge. Probabilities must sum to 1.

Return ONLY JSON inside <json></json> tags:
<json>{"views":{"statistician":{"H":0.0,"D":0.0,"A":0.0},
"regression":{"H":0.0,"D":0.0,"A":0.0},"market":{"H":0.0,"D":0.0,"A":0.0}},
"p_home":0.0,"p_draw":0.0,"p_away":0.0,
"regression_flag":"overperforming|underperforming|none",
"fragility":0,"kill_switch":"max 25 words",
"data_quality":"low|medium|high","confidence":"low|medium|high",
"uncertainty_pts":0,"why":"max 50 words","against":"max 30 words"}</json>"""

# ---------------------------------------------------------------- helpers
def ask(prompt, max_uses=6, search=True):
    kw = {}
    if search:
        kw["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}]
    msgs = [{"role": "user", "content": prompt}]
    text = ""
    for _ in range(16):
        r = client.messages.create(model=MODEL, max_tokens=8000, messages=msgs, **kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        print("  [api] stop=%s blocks=%s text_len=%d" % (r.stop_reason, [b.type for b in r.content][:6], len(text)))
        if r.stop_reason == "pause_turn":
            msgs.append({"role": "assistant", "content": r.content})
            continue
        if text.strip():
            return text
        # research finished but no written answer: ask for it explicitly
        msgs.append({"role": "assistant", "content": r.content or "(no answer yet)"})
        msgs.append({"role": "user", "content": "Stop searching. Write your final answer now, "
                     "in exactly the format requested, using what you found."})
    return text

def parse_json(t):
    m = re.search(r"<json>(.*?)</json>", t, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if not m:
        raise ValueError("no JSON in reply: " + t[:300])
    return json.loads(m.group(1))

def ask_json(prompt, max_uses=6, search=True):
    """Ask, parse JSON. If the reply has no JSON, retry once, then repair without search."""
    text = ""
    for attempt in range(2):
        text = ask(prompt + "\n\nYour FINAL message must contain the JSON inside <json></json> tags.",
                   max_uses=max_uses, search=search)
        try:
            return parse_json(text)
        except Exception as ex:
            print("parse fail (attempt %d): %s | reply start: %r" % (attempt + 1, ex, text[:300]))
    if not text.strip():
        raise RuntimeError("model returned an empty reply twice")
    fixed = ask("Turn the answer below into the JSON requested by the instruction. "
                "Return ONLY the JSON inside <json></json> tags. Use null for anything missing.\n\n"
                "INSTRUCTION:\n" + prompt + "\n\nANSWER:\n" + (text or "(empty)"), search=False)
    return parse_json(fixed)

def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}

def load():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []

def kickoff_dt(e):
    try:
        k = dt.datetime.fromisoformat(str(e["kickoff"]).replace("Z", "+00:00"))
    except Exception:
        k = dt.datetime.fromisoformat(e["date"] + "T23:59:00+00:00")
    return k if k.tzinfo else k.replace(tzinfo=dt.timezone.utc)

# ---------------------------------------------------------------- fixtures
def get_fixtures(today):
    prompt = f"""Today is {today} (UTC). List every football match kicking off in the next 30 hours
in these leagues only: {LEAGUES}. Use web search. Return ONLY JSON inside <json></json>:
<json>[{{"league":"","home":"","away":"","kickoff_utc":"YYYY-MM-DDTHH:MM:00Z"}}]</json>
If there are none, return <json>[]</json>."""
    data = ask_json(prompt, max_uses=8)
    fx, seen = [], set()
    for f in data:
        key = (f["home"], f["away"])
        if key not in seen:
            seen.add(key); fx.append(f)
    return fx

# ---------------------------------------------------------------- analysis
def fill(t, **kw):
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t

def analyse(f, today):
    kw = dict(DATE=today, HOME=f["home"], AWAY=f["away"], LEAGUE=f["league"], KICKOFF=f["kickoff_utc"])
    dossier = ask_json(fill(SCOUT, **kw), max_uses=10)
    j = ask_json(fill(COURT, **kw).replace("{DOSSIER}", json.dumps(dossier)), search=False)
    p = norm({"H": float(j["p_home"]), "D": float(j["p_draw"]), "A": float(j["p_away"])})
    o, so = dossier.get("odds") or {}, dossier.get("sharp_odds") or {}
    dq, fr = j.get("data_quality"), float(j.get("fragility") or 5)
    e = {"date": today, "league": f["league"], "home": f["home"], "away": f["away"],
         "kickoff": f["kickoff_utc"], "status": "pending", "pick": None, "pm": None,
         "odds": None, "ev": None, "kelly": None, "pick_odds": None, "p": p, "tries": 0,
         "conf": j.get("confidence"), "why": j.get("why"), "against": j.get("against"),
         "kill": j.get("kill_switch"), "fr": fr, "dq": dq, "reg": j.get("regression_flag"),
         "move": dossier.get("line_move"), "views": j.get("views"),
         "inj": dossier.get("key_absences_impact"), "lineups": dossier.get("lineups")}
    if all(o.get(k) for k in ("home", "draw", "away")):
        odds = {"H": float(o["home"]), "D": float(o["draw"]), "A": float(o["away"])}
        ref = {"H": float(so["home"]), "D": float(so["draw"]), "A": float(so["away"])} \
            if all(so.get(k) for k in ("home", "draw", "away")) else odds
        mk = norm({k: 1 / v for k, v in ref.items()})          # sharp no-vig market
        shrink = {"high": 0.15, "medium": 0.30}.get(dq, 0.50)
        pf = {k: (1 - shrink) * p[k] + shrink * mk[k] for k in p}
        evs = {k: pf[k] * odds[k] - 1 for k in pf}
        best = max(evs, key=evs.get)
        e.update(p=pf, pm=mk, odds=odds)
        vs = [norm(v)[best] for v in (j.get("views") or {}).values() if v]
        spread = (max(vs) - min(vs)) if vs else 0.0            # disagreement between specialists
        e["spread"] = spread
        if evs[best] >= MIN_EV and odds[best] <= MAX_ODDS and dq in ("medium", "high") and fr < 9:
            k = min(KELLY_CAP, KELLY_FRAC * evs[best] / (odds[best] - 1))
            k *= max(0.25, 1 - fr / 12) * (0.5 if spread > 0.12 else 1.0)
            e.update(pick=best, pick_odds=odds[best], ev=evs[best], kelly=k)
    return e

# ---------------------------------------------------------------- settlement
def settle(ledger, now):
    pend = [e for e in ledger if e["status"] == "pending" and now > kickoff_dt(e) + dt.timedelta(hours=3)]
    for i in range(0, len(pend), 8):
        batch = pend[i:i + 8]
        lst = "\n".join(f'{n}. {e["home"]} vs {e["away"]} ({e["league"]}, {e["kickoff"]})' for n, e in enumerate(batch))
        try:
            res = ask_json(f"""Find the final full-time scores (90 min, no penalties) of these football matches:
{lst}
Return ONLY JSON inside <json></json>:
<json>[{{"n":0,"played":true,"home_goals":0,"away_goals":0}}]</json>
Use played=false if postponed or not finished yet.""", max_uses=8)
        except Exception as ex:
            print("settle error", ex); continue
        for r in res:
            e = batch[int(r["n"])]
            e["tries"] += 1
            if r.get("played"):
                hg, ag = int(r["home_goals"]), int(r["away_goals"])
                out = "H" if hg > ag else "A" if ag > hg else "D"
                e.update(status="settled", score=f"{hg}-{ag}", outcome=out)
                e["brier"] = sum((e["p"][k] - (k == out)) ** 2 for k in "HDA")
                e["brier_mkt"] = sum((e["pm"][k] - (k == out)) ** 2 for k in "HDA") if e["pm"] else None
                if e["pick"]:
                    e["profit"] = e["pick_odds"] - 1 if e["pick"] == out else -1
            elif e["tries"] >= 3:
                e["status"] = "void"

# ---------------------------------------------------------------- stats + page
def stats(L):
    S = [e for e in L if e["status"] == "settled"]
    B = [e for e in S if e.get("pick")]
    bank = 100.0
    for e in sorted(B, key=lambda e: e["kickoff"]):
        bank += bank * e["kelly"] * e["profit"]
    M = [e for e in S if e.get("brier_mkt") is not None]
    avg = lambda xs: sum(xs) / len(xs) if xs else None
    by = {}
    for e in B:
        d = by.setdefault(e["league"], {"bets": 0, "flat": 0.0})
        d["bets"] += 1; d["flat"] += e["profit"]
    return {"settled": len(S), "bets": len(B), "wins": sum(e["profit"] > 0 for e in B),
            "flat_profit": sum(e["profit"] for e in B),
            "roi": (sum(e["profit"] for e in B) / len(B)) if B else None,
            "bankroll": bank, "brier_model": avg([e["brier"] for e in M]),
            "brier_market": avg([e["brier_mkt"] for e in M]), "by_league": by}

HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Agent</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--card:#f3f4f6;--g:#16a34a;--r:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#eee;--mut:#9aa;--card:#1a1d24}}
body{font:15px system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:14px;max-width:720px;margin:auto}
h1{font-size:20px}h2{font-size:16px;margin:22px 0 8px}
.g{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.c{background:var(--card);border-radius:10px;padding:10px;margin-bottom:8px}
.k{font-size:12px;color:var(--mut)}.v{font-size:18px;font-weight:600}
.bet{color:var(--g);font-weight:600}.pass{color:var(--mut)}.w{color:var(--g)}.l{color:var(--r)}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:5px 3px;border-bottom:1px solid var(--card)}
</style></head><body><h1>⚽ Football agent</h1><div id=app></div><script>
const D=__DATA__;const f=(x,d=1)=>x==null?'–':x.toFixed(d);const pc=x=>Math.round(x*100)+'%';
const s=D.stats;let h='<h2>Track record (paper)</h2><div class=g>';
const box=(k,v)=>h+='<div class=c><div class=k>'+k+'</div><div class=v>'+v+'</div></div>';
box('Bets settled',s.bets+' ('+s.wins+' won)');box('Flat ROI',s.roi==null?'–':f(s.roi*100)+'%');box('Kelly bankroll',f(s.bankroll));
box('Flat P/L (units)',f(s.flat_profit,2));box('Brier model',f(s.brier_model,3));box('Brier market',f(s.brier_market,3));
h+='</div><div class=k>Lower Brier is better. If model is not below market over 100+ games, the model has no edge.</div>';
const today=D.ledger.filter(e=>e.date==D.today);
h+='<h2>Today ('+D.today+') - '+today.length+' matches</h2>';
today.forEach(e=>{h+='<div class=c><div class=k>'+e.league+' - '+e.kickoff.replace('T',' ').slice(0,16)+' UTC</div><b>'+e.home+' vs '+e.away+'</b><br>'+
'H '+pc(e.p.H)+' D '+pc(e.p.D)+' A '+pc(e.p.A)+(e.odds?' | odds '+e.odds.H+' / '+e.odds.D+' / '+e.odds.A:' | no odds')+'<br>'+
(e.pick?'<span class=bet>BET '+({H:e.home,D:'Draw',A:e.away})[e.pick]+' @'+e.pick_odds+' - EV +'+f(e.ev*100)+'% - stake '+f(e.kelly*100)+'%</span>':'<span class=pass>Pass</span>')+
'<div class=k>'+(e.conf||'')+' conf. '+(e.why||'')+(e.inj?' Injuries: '+e.inj:'')+(e.against?' Risk: '+e.against:'')+(e.pick&&e.kill?' Cancel if: '+e.kill:'')+(e.fr!=null?' Fragility '+e.fr+'/10.':'')+'</div></div>'});
const hist=D.ledger.filter(e=>e.status=='settled'&&e.pick).slice(-60).reverse();
h+='<h2>Settled bets</h2><table>'+hist.map(e=>'<tr><td>'+e.date.slice(5)+'</td><td>'+e.home+' - '+e.away+'</td><td>'+e.score+'</td><td>'+e.pick+'@'+e.pick_odds+'</td><td class='+(e.profit>0?'w':'l')+'>'+(e.profit>0?'+':'')+f(e.profit,2)+'</td></tr>').join('')+'</table>';
document.getElementById('app').innerHTML=h;</script></body></html>"""

def build_page(ledger, today):
    payload = json.dumps({"today": today, "ledger": ledger[-600:], "stats": stats(ledger)}).replace("</", "<\\/")
    (DOCS / "index.html").write_text(HTML.replace("__DATA__", payload))

# ---------------------------------------------------------------- main
def main():
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date().isoformat()
    ledger = load()
    settle(ledger, now)
    if not any(e["date"] == today for e in ledger):
        fixtures = get_fixtures(today)
        print("fixtures found:", len(fixtures))
        for f in fixtures[:MAX_MATCHES]:
            try:
                ledger.append(analyse(f, today))
                print("done", f["home"], f["away"])
            except Exception as ex:
                print("skip", f, ex)
    LEDGER.write_text(json.dumps(ledger, indent=1))
    build_page(ledger, today)

if __name__ == "__main__":
    main()
