/* Match statistics, computed in the browser from whatever events exist.
 *
 * Deliberately independent of where the events came from: hand-tagged and
 * AI-detected events are the same objects, so the numbers update the moment
 * either changes. That also means the analyst can correct a misread pass and
 * watch the pass accuracy move.
 */

import { eventTime, store } from './store.js';

const PASS_LIKE = new Set(['Pass', 'Cross', 'Shot Assist']);

export function computeStats(events, teamNames) {
  const blank = () => ({
    events: 0, passes: 0, passesCompleted: 0, crosses: 0,
    shots: 0, shotsOnTarget: 0, goals: 0, shotAssists: 0,
    dribbles: 0, dribblesCompleted: 0,
    tackles: 0, interceptions: 0, blocks: 0, clearances: 0,
  });
  const teams = { 1: blank(), 2: blank() };
  const players = new Map();

  for (const event of events) {
    const teamIndex = store.teamIndex(event.Team);
    if (!teamIndex) continue;
    const team = teams[teamIndex];
    team.events += 1;

    const key = `${teamIndex}::${event.Player || '—'}`;
    if (!players.has(key)) {
      players.set(key, {
        name: event.Player || '—', team: teamIndex, teamName: teamNames[teamIndex],
        ...blank(),
      });
    }
    const player = players.get(key);
    player.events += 1;

    const completed = event.Outcome === 'Successful';

    if (PASS_LIKE.has(event.Event)) {
      const isAssist = event.Event === 'Shot Assist';
      for (const bucket of [team, player]) {
        bucket.passes += 1;
        // A shot assist is by definition a pass that found its man; it has no
        // "unsuccessful" variant to check.
        if (completed || isAssist) bucket.passesCompleted += 1;
        if (event.Event === 'Cross') bucket.crosses += 1;
        if (isAssist) bucket.shotAssists += 1;
      }
    } else if (event.Event === 'Shot') {
      for (const bucket of [team, player]) {
        bucket.shots += 1;
        if (event.Outcome === 'Goal' || event.Outcome === 'Saved') bucket.shotsOnTarget += 1;
        if (event.Outcome === 'Goal') bucket.goals += 1;
      }
    } else if (event.Event === 'Dribble') {
      for (const bucket of [team, player]) {
        bucket.dribbles += 1;
        if (completed) bucket.dribblesCompleted += 1;
      }
    } else if (event.Event === 'Tackle') {
      for (const bucket of [team, player]) bucket.tackles += 1;
    } else if (event.Event === 'Interception') {
      for (const bucket of [team, player]) bucket.interceptions += 1;
    } else if (event.Event === 'Block') {
      for (const bucket of [team, player]) bucket.blocks += 1;
    } else if (event.Event === 'Clearance') {
      for (const bucket of [team, player]) bucket.clearances += 1;
    }
  }

  // Possession, from the share of on-ball events. Hand-tagged data has no
  // clock on the ball, so this is a proxy — it is labelled as one in the UI.
  const onBall = (t) => t.passes + t.shots + t.dribbles + t.clearances;
  const total = onBall(teams[1]) + onBall(teams[2]);
  for (const index of [1, 2]) {
    teams[index].possession = total ? (100 * onBall(teams[index])) / total : 0;
    teams[index].passAccuracy = teams[index].passes
      ? (100 * teams[index].passesCompleted) / teams[index].passes : 0;
  }

  return {
    teams,
    players: [...players.values()].sort((a, b) => b.events - a.events),
    hasData: events.length > 0,
  };
}

const pct = (n) => `${Math.round(n)}%`;

/* A count that both teams contribute to: one bar split by their share.
 *
 * When neither team has any, the bar stays empty. A 50/50 split of nothing
 * would read as "evenly matched" when the truth is "this has not happened". */
function versusRow(label, left, right, format = (v) => v) {
  const l = Number(left) || 0;
  const r = Number(right) || 0;
  const total = l + r;
  const leftShare = total ? (l / total) * 100 : 0;
  const rightShare = total ? 100 - leftShare : 0;
  return `
    <div class="versus-row">
      <span class="v-num left"${total ? '' : ' style="color: var(--chalk-faint)"'}>${format(l)}</span>
      <span class="versus-bar">
        <i style="width:${leftShare}%; background: var(--team-1)"></i>
        ${total && leftShare > 0 && rightShare > 0 ? '<i style="width:2px; background: var(--surface)"></i>' : ''}
        <i style="width:${rightShare}%; background: var(--team-2)"></i>
      </span>
      <span class="v-num right"${total ? '' : ' style="color: var(--chalk-faint)"'}>${format(r)}</span>
      <span class="versus-label eyebrow">${label}</span>
    </div>`;
}

/* A rate each team has independently — accuracy, conversion.
 *
 * These do not sum to anything, so they get two meters on a common 0-100
 * scale rather than one split bar. Two teams both passing at 100% is not a
 * 50/50 split of accuracy, and a split bar would say it was. */
function rateRow(label, left, right) {
  const l = Math.max(0, Math.min(100, Number(left) || 0));
  const r = Math.max(0, Math.min(100, Number(right) || 0));
  return `
    <div class="versus-row">
      <span class="v-num left">${Math.round(l)}%</span>
      <span class="rate-pair">
        <span class="rate-track"><i style="width:${l}%; background: var(--team-1)"></i></span>
        <span class="rate-track"><i style="width:${r}%; background: var(--team-2)"></i></span>
      </span>
      <span class="v-num right">${Math.round(r)}%</span>
      <span class="versus-label eyebrow">${label} — each out of 100%</span>
    </div>`;
}

function statTile(figure, label, accent = 'var(--chalk)') {
  return `
    <div class="stat-card">
      <div class="figure" style="color:${accent}">${figure}</div>
      <div class="eyebrow">${label}</div>
    </div>`;
}

function playerRows(players) {
  if (!players.length) return '';
  return players.map((p) => `
    <tr>
      <td><span class="team-dot" style="background: var(--team-${p.team})"></span>${escapeHtml(p.name)}</td>
      <td class="num">${p.events}</td>
      <td class="num">${p.passes}</td>
      <td class="num">${p.passes ? Math.round((100 * p.passesCompleted) / p.passes) : 0}%</td>
      <td class="num">${p.shots}</td>
      <td class="num">${p.goals || ''}</td>
      <td class="num">${p.shotAssists || ''}</td>
      <td class="num">${p.dribbles}</td>
      <td class="num">${p.tackles + p.interceptions + p.blocks + p.clearances}</td>
    </tr>`).join('');
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* The heatmap is drawn on a canvas rather than as a grid of divs.
 *
 * A coarse grid of hard squares reads as a chart of bins; what the analyst
 * wants to see is where the action was, over a pitch they can orient
 * themselves on. So: real markings underneath, and a smooth single-hue
 * density on top — light to dark amber, the sequential ramp. */
function drawHeatmap(canvas, events, teamIndex) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const pad = 10;
  const iw = W - 2 * pad;
  const ih = H - 2 * pad;
  const px = (x, y) => [pad + (x / 120) * iw, pad + (y / 80) * ih];

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#22492f';
  ctx.fillRect(0, 0, W, H);

  const points = events.filter((e) => store.teamIndex(e.Team) === teamIndex);

  if (points.length) {
    // Each event is a soft radial blob; overlapping blobs accumulate, which is
    // what makes a busy area darker without any binning.
    const layer = document.createElement('canvas');
    layer.width = W;
    layer.height = H;
    const lc = layer.getContext('2d');
    const radius = Math.max(24, iw * 0.075);
    lc.globalCompositeOperation = 'lighter';
    for (const event of points) {
      const [cx, cy] = px(Number(event.X) || 0, Number(event.Y) || 0);
      const gradient = lc.createRadialGradient(cx, cy, 0, cx, cy, radius);
      gradient.addColorStop(0, 'rgba(255,176,32,0.55)');
      gradient.addColorStop(1, 'rgba(255,176,32,0)');
      lc.fillStyle = gradient;
      lc.beginPath();
      lc.arc(cx, cy, radius, 0, Math.PI * 2);
      lc.fill();
    }
    ctx.drawImage(layer, 0, 0);
  }

  // Markings last, so they stay legible through the density.
  ctx.strokeStyle = 'rgba(223,232,242,.62)';
  ctx.lineWidth = 1.5;
  const rect = (x1, y1, x2, y2) => {
    const [ax, ay] = px(x1, y1);
    const [bx, by] = px(x2, y2);
    ctx.strokeRect(ax, ay, bx - ax, by - ay);
  };
  rect(0, 0, 120, 80);
  rect(0, 18, 18, 62);
  rect(0, 30, 6, 50);
  rect(102, 18, 120, 62);
  rect(114, 30, 120, 50);
  const [mx1, my1] = px(60, 0);
  const [mx2, my2] = px(60, 80);
  ctx.beginPath(); ctx.moveTo(mx1, my1); ctx.lineTo(mx2, my2); ctx.stroke();
  const [cx, cy] = px(60, 40);
  ctx.beginPath(); ctx.arc(cx, cy, (10 / 80) * ih, 0, Math.PI * 2); ctx.stroke();

  // Direction of attack, so the map cannot be read the wrong way round.
  ctx.fillStyle = 'rgba(223,232,242,.5)';
  ctx.font = '600 11px "IBM Plex Sans", sans-serif';
  ctx.fillText('attacking →', W - 84, H - 14);
}

export function renderStats(container) {
  const { events, team1Name, team2Name } = store.state;
  const names = { 1: team1Name, 2: team2Name };
  const stats = computeStats(events, names);

  if (!stats.hasData) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="figure">Nothing to count yet</div>
        <p>Tag an event on the pitch, or run an AI analysis, and the numbers appear here.</p>
      </div>`;
    return;
  }

  const t1 = stats.teams[1];
  const t2 = stats.teams[2];
  const duration = events.reduce((m, e) => Math.max(m, eventTime(e)), 0);

  container.innerHTML = `
    <div class="stat-grid" style="margin-bottom: 18px">
      ${statTile(`${t1.goals}–${t2.goals}`, 'Score', 'var(--amber)')}
      ${statTile(t1.shots + t2.shots, 'Shots')}
      ${statTile(t1.passes + t2.passes, 'Passes')}
      ${statTile(events.length, 'Events recorded')}
      ${statTile(`${Math.floor(duration / 60)}′`, 'Last event at')}
    </div>

    <div class="row" style="justify-content: space-between; margin-bottom: 10px">
      <span class="panel-title" style="color: var(--team-1)">${escapeHtml(team1Name)}</span>
      <span class="eyebrow">Team comparison</span>
      <span class="panel-title" style="color: var(--team-2)">${escapeHtml(team2Name)}</span>
    </div>

    <div class="versus">
      ${versusRow('Possession — share of on-ball events', t1.possession, t2.possession, pct)}
      ${versusRow('Passes', t1.passes, t2.passes)}
      ${rateRow('Pass accuracy', t1.passAccuracy, t2.passAccuracy)}
      ${versusRow('Shots', t1.shots, t2.shots)}
      ${versusRow('Shots on target', t1.shotsOnTarget, t2.shotsOnTarget)}
      ${versusRow('Crosses', t1.crosses, t2.crosses)}
      ${versusRow('Dribbles', t1.dribbles, t2.dribbles)}
      ${versusRow('Tackles', t1.tackles, t2.tackles)}
      ${versusRow('Interceptions', t1.interceptions, t2.interceptions)}
      ${versusRow('Clearances', t1.clearances, t2.clearances)}
    </div>

    <div class="row" style="align-items: flex-start; gap: 16px; margin-top: 22px">
      <div style="flex: 1 1 260px; min-width: 0">
        <div class="eyebrow" style="margin-bottom: 7px">
          <span class="team-dot" style="background: var(--team-1)"></span>${escapeHtml(team1Name)} — where they acted
        </div>
        <canvas class="heatmap" id="heat1" width="600" height="400"></canvas>
      </div>
      <div style="flex: 1 1 260px; min-width: 0">
        <div class="eyebrow" style="margin-bottom: 7px">
          <span class="team-dot" style="background: var(--team-2)"></span>${escapeHtml(team2Name)} — where they acted
        </div>
        <canvas class="heatmap" id="heat2" width="600" height="400"></canvas>
      </div>
    </div>
    <p class="faint" style="font-size: 11.5px; margin: 8px 0 0">
      Both maps are drawn left-to-right as attacking. Second-half events are rotated to match.
    </p>

    <div class="eyebrow" style="margin: 22px 0 8px">Players</div>
    <div class="table-scroll" style="max-height: 320px">
      <table class="data">
        <thead>
          <tr>
            <th>Player</th><th>Events</th><th>Passes</th><th>Acc.</th>
            <th>Shots</th><th>Goals</th><th>Assists</th><th>Dribbles</th><th>Defensive</th>
          </tr>
        </thead>
        <tbody>${playerRows(stats.players)}</tbody>
      </table>
    </div>`;

  drawHeatmap(container.querySelector('#heat1'), events, 1);
  drawHeatmap(container.querySelector('#heat2'), events, 2);
}
