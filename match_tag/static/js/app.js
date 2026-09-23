/* Wiring.
 *
 * Everything the page can do is set up here: the video, the pitch, the
 * timeline, the tables, the keyboard, and the AI panel. Each view owns its own
 * rendering and reads the store; this module only connects them.
 */

// Report a failure in this page back to the server.
//
// The interface is the half of this program that cannot be debugged from the
// machine that wrote it. A script that throws during start-up leaves a page
// that looks entirely broken — no pitch markings, no menus — while saying
// nothing to anybody who is not standing in front of it with a console open,
// which is never the person who hits it. Installed first, before anything
// that could throw.
(() => {
  const seen = new Set();
  const report = (payload) => {
    const key = `${payload.message}@${payload.line}`;
    if (seen.has(key)) return;
    seen.add(key);
    try {
      fetch('/api/client-error', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).catch(() => { /* reporting a failure must not cause one */ });
    } catch { /* ditto */ }
  };

  // And say it on the page. A server log helps whoever runs the server; the
  // person looking at a blank pitch needs to be told what happened without
  // opening a console, and can then repeat it to somebody who can fix it.
  const show = (text) => {
    let banner = document.getElementById('clientErrorBanner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'clientErrorBanner';
      banner.style.cssText = [
        'position:fixed', 'left:0', 'right:0', 'bottom:0', 'z-index:9999',
        'background:#2a1414', 'color:#ffb0b0', 'border-top:2px solid #ff5c5c',
        'padding:10px 14px', 'font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace',
        'white-space:pre-wrap', 'max-height:35vh', 'overflow:auto',
      ].join(';');
      document.body.appendChild(banner);
    }
    banner.textContent = `This page hit an error and may be partly broken:\n${text}`;
  };

  window.addEventListener('error', (e) => show(
    `${e.message}\n  ${e.filename}:${e.lineno}:${e.colno}`,
  ));
  window.addEventListener('error', (e) => report({
    message: String(e.message || e.error || 'unknown error'),
    source: String(e.filename || ''),
    line: e.lineno, column: e.colno,
    stack: String(e.error?.stack || ''),
  }));
  window.addEventListener('unhandledrejection', (e) => report({
    message: `unhandled rejection: ${e.reason?.message || e.reason}`,
    source: '', line: 0, column: 0,
    stack: String(e.reason?.stack || ''),
  }));
})();

import { api } from './api.js';
import { AnalysisPanel } from './ai.js';
import { Pitch } from './pitch.js';
import { renderStats, escapeHtml } from './stats.js';
import {
  DEFAULT_EVENT_ORDER, EVENT_TYPES, eventTime, formatTime, isGoal, store, typeInfo,
} from './store.js';
import { Timeline } from './timeline.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

// Broadcast footage is 25 fps almost everywhere; used only for frame stepping.
const ASSUMED_FPS = 25;

/* ---------------------------------------------------------------- toasts */

function toast(message, kind = 'info') {
  const element = document.createElement('div');
  element.className = `toast toast-${kind}`;
  element.textContent = message;
  $('#toasts').append(element);
  setTimeout(() => {
    element.style.opacity = '0';
    element.style.transition = 'opacity .25s ease';
    setTimeout(() => element.remove(), 250);
  }, kind === 'bad' ? 6000 : 3500);
}

/* ----------------------------------------------------------------- video */

const video = $('#video');
let objectUrl = null;
// The File the analyst loaded, kept so the AI panel can reuse it instead of
// asking them to find the same video on disk a second time.
let loadedFile = null;

function loadVideoFile(file) {
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  loadedFile = file;
  video.src = objectUrl;
  $('#videoEmpty').style.display = 'none';
  store.set({ videoName: file.name }, 'video');
  video.play().catch(() => { /* autoplay is often blocked; the button still works */ });
  analysisPanel?.useFile(file);
}

function seek(seconds) {
  if (!video.duration) return;
  video.currentTime = Math.max(0, Math.min(video.duration, seconds));
}

function setupVideo() {
  bind('#btnPickVideo', 'click', () => $('#fileInput').click());
  bind('#btnChangeVideo', 'click', () => $('#fileInput').click());
  bind('#fileInput', 'change', (e) => loadVideoFile(e.target.files[0]));

  bind('#btnPlay', 'click', togglePlay);
  bind('#btnBack5', 'click', () => seek(video.currentTime - 5));
  bind('#btnFwd5', 'click', () => seek(video.currentTime + 5));
  bind('#btnFrameBack', 'click', () => seek(video.currentTime - 1 / ASSUMED_FPS));
  bind('#btnFrameFwd', 'click', () => seek(video.currentTime + 1 / ASSUMED_FPS));
  bind('#speed', 'change', (e) => { video.playbackRate = Number(e.target.value); });

  video.addEventListener('loadedmetadata', () => {
    store.set({ videoDuration: video.duration || 0 }, 'video', 'time');
    $('#clockTotal').textContent = formatTime(video.duration);
  });
  video.addEventListener('timeupdate', () => {
    store.state.currentTime = video.currentTime;
    $('#clockNow').textContent = formatTime(video.currentTime, true);
    store.emit('time');
  });
  video.addEventListener('play', () => { $('#btnPlay').textContent = '❚❚'; });
  video.addEventListener('pause', () => { $('#btnPlay').textContent = '▶'; });
  video.addEventListener('error', () => {
    if (video.src) toast('That video could not be played in this browser.', 'bad');
  });
}

function togglePlay() {
  if (!video.src) { $('#fileInput').click(); return; }
  if (video.paused) video.play().catch(() => {}); else video.pause();
}

/* ------------------------------------------------------------ tag controls */

function currentPlayer() {
  const typed = $('#playerFree').value.trim();
  return typed || $('#playerSel').value || '';
}

function currentOutcome() {
  const outcomes = typeInfo(store.state.currentType).outcomes;
  if (!outcomes.length) return '';
  return $('#outcome').value || outcomes[0];
}

let warnedAboutNoVideo = false;

function placeEvent({ x, y, x2, y2 }) {
  const teamIndex = store.state.currentTeam;
  const time = video.currentTime || 0;

  // Without a video every event is stamped 0:00, which looks like a bug when
  // the analyst later wonders why nothing is on the timeline.
  if (!video.src && !warnedAboutNoVideo) {
    warnedAboutNoVideo = true;
    toast('No video loaded, so this event is recorded at 0:00. Load the match video first to place events in time.', 'bad');
  }
  const event = {
    Team: store.teamName(teamIndex),
    Player: currentPlayer(),
    Event: store.state.currentType,
    Outcome: currentOutcome(),
    Mins: Math.floor(time / 60),
    Secs: Math.floor(time % 60),
    X: x, Y: y,
    X2: x2 ?? '', Y2: y2 ?? '',
    half: Number(store.state.half),
    source: 'manual',
    t: time,
  };
  store.addEvent(event);
  store.set({ selectedEvent: store.state.events.length - 1 }, 'selection');
}

function renderPalette() {
  const palette = $('#palette');
  palette.innerHTML = '';
  store.state.eventTypes.forEach((name, index) => {
    const button = document.createElement('button');
    button.className = `evt${name === store.state.currentType ? ' is-active' : ''}`;
    button.type = 'button';
    const shortcut = index < 9 ? `<kbd>${index + 1}</kbd>` : '';
    const info = typeInfo(name);
    button.innerHTML = `${escapeHtml(name)}${shortcut}`;
    button.title = info.arrow
      ? `${name} — two clicks on the pitch: where it started, then where it ended`
      : `${name} — one click on the pitch`;
    button.addEventListener('click', () => selectType(name));

    if (!DEFAULT_EVENT_ORDER.includes(name)) {
      const kill = document.createElement('button');
      kill.className = 'kill';
      kill.type = 'button';
      kill.textContent = '×';
      kill.title = `Remove ${name}`;
      kill.addEventListener('click', (e) => {
        e.stopPropagation();
        const remaining = store.state.eventTypes.filter((n) => n !== name);
        store.set({
          eventTypes: remaining,
          currentType: store.state.currentType === name ? remaining[0] : store.state.currentType,
        }, 'types');
      });
      button.append(kill);
    }
    palette.append(button);
  });
}

function selectType(name) {
  store.set({ currentType: name, pendingStart: null }, 'types', 'pending');
}

function renderOutcomes() {
  const outcomes = typeInfo(store.state.currentType).outcomes;
  const field = $('#outcomeField');
  const select = $('#outcome');
  field.style.visibility = outcomes.length ? 'visible' : 'hidden';
  select.innerHTML = outcomes.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join('');
  $('#pitchMode').textContent = typeInfo(store.state.currentType).arrow
    ? 'Two clicks: start, then end'
    : 'One click to place';
}

function renderPlayers() {
  const select = $('#playerSel');
  const roster = store.roster(store.state.currentTeam);
  const previous = select.value;

  // An empty menu reading "no player" looks like a missing feature rather than
  // an empty squad, so it says where the squad is entered.
  select.innerHTML = roster.length
    ? '<option value="">— no player —</option>'
      + roster.map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join('')
    : '<option value="">— add a squad in the Squads tab —</option>';
  select.disabled = roster.length === 0;
  if (roster.includes(previous)) select.value = previous;
}

function renderTeams() {
  const { team1Name, team2Name, team1Colour, team2Colour } = store.state;
  document.documentElement.style.setProperty('--team-1', team1Colour);
  document.documentElement.style.setProperty('--team-2', team2Colour);

  $('#scoreName1').textContent = team1Name;
  $('#scoreName2').textContent = team2Name;
  $('#toggleName1').textContent = team1Name;
  $('#toggleName2').textContent = team2Name;
  $('#team1Name').value = team1Name;
  $('#team2Name').value = team2Name;
  $('#team1Colour').value = team1Colour;
  $('#team2Colour').value = team2Colour;

  $$('#teamToggle button').forEach((b) => {
    b.classList.toggle('is-active', Number(b.dataset.team) === store.state.currentTeam);
  });

  const filter = $('#eventTeamFilter');
  const chosen = filter.value;
  filter.innerHTML = `<option value="">Both teams</option>
    <option value="${escapeHtml(team1Name)}">${escapeHtml(team1Name)}</option>
    <option value="${escapeHtml(team2Name)}">${escapeHtml(team2Name)}</option>`;
  filter.value = chosen;
}

function renderScore() {
  const goals = { 1: 0, 2: 0 };
  for (const event of store.state.events) {
    if (!isGoal(event)) continue;
    const index = store.teamIndex(event.Team);
    if (index) goals[index] += 1;
  }
  $('#score1').textContent = goals[1];
  $('#score2').textContent = goals[2];
}

/* ----------------------------------------------------------- events table */

let sortKey = 'index';
let sortDir = 'asc';

function filteredEvents() {
  const query = $('#eventSearch').value.trim().toLowerCase();
  const team = $('#eventTeamFilter').value;
  const type = $('#eventTypeFilter').value;
  const source = $('#eventSourceFilter').value;

  const rows = store.state.events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => {
      if (team && event.Team !== team) return false;
      if (type && event.Event !== type) return false;
      if (source && (event.source || 'manual') !== source) return false;
      if (!query) return true;
      return [event.Player, event.Event, event.Outcome, event.Team]
        .some((v) => String(v ?? '').toLowerCase().includes(query));
    });

  const direction = sortDir === 'asc' ? 1 : -1;
  rows.sort((a, b) => {
    let av;
    let bv;
    if (sortKey === 'index') { av = a.index; bv = b.index; }
    else if (sortKey === 'time') { av = eventTime(a.event); bv = eventTime(b.event); }
    else { av = String(a.event[sortKey] ?? ''); bv = String(b.event[sortKey] ?? ''); }
    if (av < bv) return -direction;
    if (av > bv) return direction;
    return 0;
  });
  return rows;
}

// Which row is open for editing, or -1. Kept here rather than in the store:
// it is a property of this table on this screen, not of the match.
let editingIndex = -1;

function renderEventsTable() {
  const body = $('#eventsTable tbody');
  const rows = filteredEvents();
  $('#countEvents').textContent = store.state.events.length;
  $('#eventsShown').textContent = rows.length === store.state.events.length
    ? `${rows.length} events`
    : `${rows.length} of ${store.state.events.length} events`;

  const typeFilter = $('#eventTypeFilter');
  const chosenType = typeFilter.value;
  const present = [...new Set(store.state.events.map((e) => e.Event))].sort();
  typeFilter.innerHTML = '<option value="">All event types</option>'
    + present.map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
  typeFilter.value = chosenType;

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="10">
      <div class="empty-state">
        <div class="figure">${store.state.events.length ? 'Nothing matches that filter' : 'No events yet'}</div>
        <p>${store.state.events.length
          ? 'Clear the filters to see the rest.'
          : 'Pick an event type, then click the pitch. Or open the AI analysis tab and let the models do a first pass.'}</p>
      </div></td></tr>`;
    return;
  }

  body.innerHTML = rows.map(({ event, index }) => {
    if (index === editingIndex) return editingRow(event, index);
    const teamIndex = store.teamIndex(event.Team);
    const outcomeClass = event.Outcome === 'Goal' ? 'tag-goal'
      : ['Successful'].includes(event.Outcome) ? 'tag-good'
      : ['Unsuccessful', 'Failed', 'Off Target', 'Blocked'].includes(event.Outcome) ? 'tag-bad' : '';
    const hasEnd = event.X2 !== '' && event.X2 !== null && event.X2 !== undefined;
    const confidence = typeof event.confidence === 'number'
      ? ` <span class="faint">${Math.round(event.confidence * 100)}%</span>` : '';

    return `<tr data-index="${index}" class="${index === store.state.selectedEvent ? 'is-selected' : ''}">
      <td class="num faint">${index + 1}</td>
      <td class="num">${formatTime(eventTime(event))}</td>
      <td>${teamIndex ? `<span class="team-dot" style="background: var(--team-${teamIndex})"></span>` : ''}${escapeHtml(event.Team)}</td>
      <td>${escapeHtml(event.Player || '—')}</td>
      <td>${escapeHtml(event.Event)}</td>
      <td>${event.Outcome ? `<span class="tag ${outcomeClass}">${escapeHtml(event.Outcome)}</span>` : ''}</td>
      <td class="num faint">${Number(event.X).toFixed(0)}, ${Number(event.Y).toFixed(0)}</td>
      <td class="num faint">${hasEnd ? `${Number(event.X2).toFixed(0)}, ${Number(event.Y2).toFixed(0)}` : '—'}</td>
      <td>${(event.source === 'ai')
        ? `<span class="tag tag-ai">AI${confidence}</span>`
        : '<span class="tag">Tagged</span>'}</td>
      <td class="nowrap">
        <button class="btn btn-sm btn-ghost" data-edit="${index}" title="Edit this event">✎</button>
        <button class="btn btn-sm btn-ghost btn-danger" data-delete="${index}" title="Delete this event">×</button>
      </td>
    </tr>`;
  }).join('');
}

/** The same row, opened for editing.
 *
 * Everything typed while watching is worth being able to correct — the team,
 * the player, what the event was, how it ended, and when. The pitch positions
 * are deliberately not editable here: they came from clicking the pitch, which
 * is a better way to say where something happened than any number entry, and
 * they are the one part of a tagged event that survives being wrong about
 * everything else.
 */
function editingRow(event, index) {
  const teams = [store.state.team1Name, store.state.team2Name];
  const outcomes = [
    '', 'Successful', 'Unsuccessful', 'Failed', 'Goal', 'Saved',
    'Off Target', 'Blocked',
  ];
  const seconds = Math.round(eventTime(event));
  const option = (value, current, label) =>
    `<option value="${escapeHtml(value)}"${value === current ? ' selected' : ''}>${escapeHtml(label ?? (value || '—'))}</option>`;

  return `<tr data-editing="${index}" class="is-editing">
    <td class="num faint">${index + 1}</td>
    <td><input class="input input-sm" type="number" min="0" step="1"
               data-field="seconds" value="${seconds}" style="width: 68px"
               title="Seconds from the start"></td>
    <td><select class="input input-sm" data-field="Team">
          ${teams.map((t) => option(t, event.Team)).join('')}
        </select></td>
    <td><input class="input input-sm" data-field="Player"
               value="${escapeHtml(event.Player || '')}" placeholder="e.g. left back"></td>
    <td><select class="input input-sm" data-field="Event">
          ${store.state.eventTypes.map((t) => option(t, event.Event)).join('')}
        </select></td>
    <td><select class="input input-sm" data-field="Outcome">
          ${outcomes.map((o) => option(o, event.Outcome || '')).join('')}
        </select></td>
    <td class="num faint">${Number(event.X).toFixed(0)}, ${Number(event.Y).toFixed(0)}</td>
    <td class="num faint">${event.X2 !== '' && event.X2 != null
      ? `${Number(event.X2).toFixed(0)}, ${Number(event.Y2).toFixed(0)}` : '—'}</td>
    <td class="faint">editing</td>
    <td class="nowrap">
      <button class="btn btn-sm btn-primary" data-save="${index}">Save</button>
      <button class="btn btn-sm btn-ghost" data-cancel="1">Cancel</button>
    </td>
  </tr>`;
}

function setupEventsTable() {
  const table = $('#eventsTable');

  table.addEventListener('click', (e) => {
    const editButton = e.target.closest('[data-edit]');
    if (editButton) {
      editingIndex = Number(editButton.dataset.edit);
      renderEventsTable();
      return;
    }
    if (e.target.closest('[data-cancel]')) {
      editingIndex = -1;
      renderEventsTable();
      return;
    }
    const saveButton = e.target.closest('[data-save]');
    if (saveButton) {
      const index = Number(saveButton.dataset.save);
      const row = saveButton.closest('tr');
      const value = (field) => row.querySelector(`[data-field="${field}"]`)?.value ?? '';
      const seconds = Math.max(0, Number(value('seconds')) || 0);
      store.updateEvent(index, {
        Team: value('Team'),
        Player: value('Player').trim(),
        Event: value('Event'),
        Outcome: value('Outcome'),
        Mins: Math.floor(seconds / 60),
        Secs: Math.round(seconds % 60),
        // The precise timestamp has to move with the clock the user just
        // corrected, or the timeline would keep drawing the tick where the
        // mistake was.
        t: seconds,
      });
      editingIndex = -1;
      toast('Event updated.', 'good');
      return;
    }
    const deleteButton = e.target.closest('[data-delete]');
    if (deleteButton) {
      if (editingIndex === Number(deleteButton.dataset.delete)) editingIndex = -1;
      store.removeEvent(Number(deleteButton.dataset.delete));
      return;
    }
    // A click inside the row being edited must not re-seek the video under it.
    if (e.target.closest('tr[data-editing]')) return;
    const row = e.target.closest('tr[data-index]');
    if (!row) return;
    const index = Number(row.dataset.index);
    store.set({ selectedEvent: index }, 'selection');
    seek(eventTime(store.state.events[index]));
  });

  $$('#eventsTable th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      sortDir = sortKey === key && sortDir === 'asc' ? 'desc' : 'asc';
      sortKey = key;
      $$('#eventsTable th.sortable').forEach((other) => other.removeAttribute('data-dir'));
      th.dataset.dir = sortDir;
      renderEventsTable();
    });
  });

  ['#eventSearch', '#eventTeamFilter', '#eventTypeFilter', '#eventSourceFilter']
    .forEach((selector) => $(selector).addEventListener('input', renderEventsTable));
}

/* -------------------------------------------------------------- live feed */

/* The most recent events, beside the video. During tagging this is the
 * confirmation that the click landed on what you meant — checking the full
 * table below would mean looking away from the match. */
function renderFeed() {
  const feed = $('#feed');
  const { events, selectedEvent } = store.state;
  $('#feedCount').textContent = events.length ? `${events.length} total` : '';

  if (!events.length) {
    feed.innerHTML = `<div class="empty-state" style="padding: 22px 16px">
      <p>Events appear here as you tag them.</p>
    </div>`;
    return;
  }

  const recent = events.map((event, index) => ({ event, index })).slice(-40).reverse();
  feed.innerHTML = recent.map(({ event, index }) => {
    const teamIndex = store.teamIndex(event.Team);
    const outcome = event.Outcome ? ` · ${escapeHtml(event.Outcome)}` : '';
    return `<div class="feed-item ${index === selectedEvent ? 'is-selected' : ''}" data-feed="${index}">
      <span class="at">${formatTime(eventTime(event))}</span>
      <span class="what">
        ${teamIndex ? `<span class="team-dot" style="background: var(--team-${teamIndex})"></span>` : ''}
        ${escapeHtml(event.Event)}<span class="who">${outcome}</span>
        ${event.Player ? `<span class="who"> — ${escapeHtml(event.Player)}</span>` : ''}
      </span>
      ${event.source === 'ai' ? '<span class="tag tag-ai">AI</span>' : ''}
    </div>`;
  }).join('');
}

function setupFeed() {
  bind('#feed', 'click', (e) => {
    const item = e.target.closest('[data-feed]');
    if (!item) return;
    const index = Number(item.dataset.feed);
    store.set({ selectedEvent: index }, 'selection');
    seek(eventTime(store.state.events[index]));
  });
}

/* ---------------------------------------------------------- saved matches */

async function refreshSavedMatches() {
  const body = $('#savedTable tbody');
  try {
    const matches = await api.listMatches();
    if (!matches.length) {
      body.innerHTML = `<tr><td colspan="7"><div class="empty-state">
        <div class="figure">No saved matches</div>
        <p>Save the match you are working on and it appears here.</p>
      </div></td></tr>`;
      return;
    }
    body.innerHTML = matches.map((m, i) => `
      <tr>
        <td class="num faint">${i + 1}</td>
        <td>${escapeHtml(m.match_name)}</td>
        <td class="muted">${escapeHtml(m.team1)} v ${escapeHtml(m.team2)}</td>
        <td class="num">${m.events_count}</td>
        <td>${m.source === 'ai' ? '<span class="tag tag-ai">AI</span>'
              : m.source === 'mixed' ? '<span class="tag">Mixed</span>'
              : '<span class="tag">Tagged</span>'}</td>
        <td class="num faint">${new Date(m.date_created).toLocaleString()}</td>
        <td>
          <button class="btn btn-sm" data-load="${m.match_id}">Open</button>
          <a class="btn btn-sm" href="${api.csvUrl(m.match_id)}">CSV</a>
          <button class="btn btn-sm btn-ghost btn-danger" data-drop="${m.match_id}">Delete</button>
        </td>
      </tr>`).join('');
  } catch (error) {
    body.innerHTML = `<tr><td colspan="7"><div class="empty-state">
      <div class="figure">Could not reach the server</div><p>${escapeHtml(error.message)}</p>
    </div></td></tr>`;
  }
}

async function loadMatch(matchId) {
  try {
    const match = await api.getMatch(matchId);
    store.set({
      team1Name: match.team1, team2Name: match.team2,
      matchName: match.match_name, matchId: match.match_id,
    }, 'teams');
    store.clearEvents();
    store.addEvents(match.events.map((e) => ({ ...e, t: (e.Mins || 0) * 60 + (e.Secs || 0) })));
    toast(`Opened “${match.match_name}” — ${match.events.length} events.`, 'good');
    switchTab('events');
  } catch (error) {
    toast(error.message, 'bad');
  }
}

async function saveMatch() {
  const { events, team1Name, team2Name, matchName, videoName, matchId } = store.state;
  if (!events.length) { toast('There is nothing to save yet.', 'bad'); return; }

  const name = prompt('Name this match', matchName || `${team1Name} v ${team2Name}`);
  if (!name) return;

  const body = { matchName: name, team1: team1Name, team2: team2Name, videoName, events };
  try {
    // An already-saved match is updated in place, so repeated saves do not
    // leave a trail of near-duplicates on the server.
    const result = matchId
      ? await api.updateMatch(matchId, body)
      : await api.saveMatch(body);
    store.set({ matchName: name, matchId: result.match_id }, 'teams');
    toast(`Saved “${name}”.`, 'good');
    refreshSavedMatches();
  } catch (error) {
    toast(error.message, 'bad');
  }
}

function setupSaved() {
  bind('#btnRefreshMatches', 'click', refreshSavedMatches);
  bind('#savedTable', 'click', async (e) => {
    const load = e.target.closest('[data-load]');
    if (load) { loadMatch(Number(load.dataset.load)); return; }
    const drop = e.target.closest('[data-drop]');
    if (drop && confirm('Delete this saved match and all of its events?')) {
      try {
        await api.deleteMatch(Number(drop.dataset.drop));
        toast('Match deleted.', 'good');
        refreshSavedMatches();
      } catch (error) { toast(error.message, 'bad'); }
    }
  });
}

/* ------------------------------------------------------------------ export */

function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function exportCsv() {
  const { events } = store.state;
  if (!events.length) { toast('There is nothing to export yet.', 'bad'); return; }
  const header = ['Team', 'Player', 'Event', 'Outcome', 'Mins', 'Secs', 'X', 'Y', 'X2', 'Y2', 'Half', 'Source'];
  const lines = [header.join(',')];
  for (const e of events) {
    lines.push([
      e.Team, e.Player, e.Event, e.Outcome, e.Mins, e.Secs,
      e.X, e.Y, e.X2, e.Y2, e.half ?? 1, e.source ?? 'manual',
    ].map(csvCell).join(','));
  }
  download(`${store.state.matchName || 'match'}.csv`, lines.join('\n'), 'text/csv;charset=utf-8');
}

function exportJson() {
  const { events, team1Name, team2Name, matchName } = store.state;
  if (!events.length) { toast('There is nothing to export yet.', 'bad'); return; }
  download(
    `${matchName || 'match'}.json`,
    JSON.stringify({ matchName, team1: team1Name, team2: team2Name, events }, null, 2),
    'application/json',
  );
}

/* -------------------------------------------------------------------- tabs */

function switchTab(name, { scroll = false } = {}) {
  $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === name));
  $$('.tabpane').forEach((p) => p.classList.toggle('is-active', p.dataset.pane === name));
  if (name === 'stats') renderStats($('#statsPane'));
  if (name === 'saved') refreshSavedMatches();
  if (name === 'squads') renderTrackMapping();
  // The tab strip sits below the fold on a laptop screen. Switching to a tab
  // without bringing it into view looks exactly like the button doing nothing.
  if (scroll) {
    $('.tab[data-tab="' + name + '"]').closest('.panel')
      .scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ------------------------------------------------ AI track name mapping */

function renderTrackMapping() {
  const container = $('#trackMapping');
  const tracked = [...new Set(
    store.state.events.filter((e) => e.source === 'ai' && e.track_id).map((e) => e.track_id),
  )].sort((a, b) => a - b);

  if (!tracked.length) { container.innerHTML = ''; return; }

  const counts = new Map();
  for (const event of store.state.events) {
    if (event.track_id) counts.set(event.track_id, (counts.get(event.track_id) || 0) + 1);
  }

  container.innerHTML = `
    <div class="eyebrow" style="margin-bottom: 8px">Name the players the AI found</div>
    <p class="muted" style="font-size: 12.5px; margin: 0 0 10px">
      The models track players but cannot read shirt names. Give each tracked player a name
      and it replaces <span class="mono">#id</span> everywhere.
    </p>
    <div class="row" style="gap: 8px">
      ${tracked.map((id) => `
        <div class="field" style="flex: 0 0 150px">
          <label for="track-${id}">#${id} · ${counts.get(id) || 0} events</label>
          <input class="input" id="track-${id}" data-track="${id}"
                 value="${escapeHtml(store.state.trackNames[id] || '')}" placeholder="Player name">
        </div>`).join('')}
    </div>
    <button class="btn btn-primary btn-sm" id="btnApplyTracks" style="margin-top: 10px">Apply names</button>`;

  bind('#btnApplyTracks', 'click', () => {
    const names = { ...store.state.trackNames };
    container.querySelectorAll('[data-track]').forEach((input) => {
      const id = Number(input.dataset.track);
      const value = input.value.trim();
      if (value) names[id] = value; else delete names[id];
    });
    store.set({ trackNames: names }, 'teams');
    let renamed = 0;
    for (const event of store.state.events) {
      if (event.track_id && names[event.track_id]) {
        event.Player = names[event.track_id];
        renamed += 1;
      }
    }
    store.emit('events');
    toast(renamed ? `Renamed ${renamed} events.` : 'No names to apply.', renamed ? 'good' : 'info');
  });
}

/* --------------------------------------------------------------- keyboard */

function setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    const inField = ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
    if (e.key === 'Escape') {
      pitch.cancelPending();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      saveMatch();
      return;
    }
    if (inField || e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key === ' ') { e.preventDefault(); togglePlay(); return; }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); seek(video.currentTime - (e.shiftKey ? 5 : 1)); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); seek(video.currentTime + (e.shiftKey ? 5 : 1)); return; }
    if (e.key === ',') { seek(video.currentTime - 1 / ASSUMED_FPS); return; }
    if (e.key === '.') { seek(video.currentTime + 1 / ASSUMED_FPS); return; }
    if (e.key === '-' || e.key === '_') { stepSpeed(-1); return; }
    if (e.key === '+' || e.key === '=') { stepSpeed(1); return; }

    if (e.key >= '1' && e.key <= '9') {
      const type = store.state.eventTypes[Number(e.key) - 1];
      if (type) selectType(type);
      return;
    }
    if (e.key.toLowerCase() === 'q') { store.set({ currentTeam: 1 }, 'teams'); return; }
    if (e.key.toLowerCase() === 'w') { store.set({ currentTeam: 2 }, 'teams'); return; }
    if (e.key.toLowerCase() === 'z') {
      const removed = store.undoLast();
      if (removed) toast(`Removed the ${removed.Event.toLowerCase()}.`, 'info');
      return;
    }
    if (e.key === '?') { $('#keysDialog').showModal(); }
  });
}

function stepSpeed(direction) {
  const options = [0.25, 0.5, 1, 1.5, 2];
  const current = options.indexOf(Number($('#speed').value));
  const next = Math.max(0, Math.min(options.length - 1, current + direction));
  $('#speed').value = String(options[next]);
  video.playbackRate = options[next];
  toast(`Playback ${options[next]}×`, 'info');
}

/* ----------------------------------------------------------------- squads */

function setupSquads() {
  $('#t1Roster').value = store.state.roster1;
  $('#t2Roster').value = store.state.roster2;

  bind('#btnApplySquads', 'click', () => {
    const oldNames = [store.state.team1Name, store.state.team2Name];
    const team1Name = $('#team1Name').value.trim() || 'Team 1';
    const team2Name = $('#team2Name').value.trim() || 'Team 2';

    store.renameTeamInEvents(oldNames[0], team1Name);
    store.renameTeamInEvents(oldNames[1], team2Name);
    store.set({
      team1Name, team2Name,
      team1Colour: $('#team1Colour').value,
      team2Colour: $('#team2Colour').value,
      roster1: $('#t1Roster').value,
      roster2: $('#t2Roster').value,
    }, 'teams', 'events');
    toast('Squads updated.', 'good');
  });
}

/* ------------------------------------------------------------------- boot */

const pitch = new Pitch($('#pitch'), $('#pitchHint'));
const timeline = new Timeline($('#timeline'), $('#timelineCanvas'));
let analysisPanel = null;

function renderTimelineLegend() {
  const { team1Name, team2Name } = store.state;
  $('#timelineLegend').innerHTML = `
    <span class="legend-chip"><i style="background: var(--team-1)"></i>${escapeHtml(team1Name)} above</span>
    <span class="legend-chip"><i style="background: var(--team-2)"></i>${escapeHtml(team2Name)} below</span>
    <span class="legend-chip"><i style="background: #3fd98a; border-radius: 50%"></i>Goal</span>
    <span class="legend-chip faint">Taller tick = shot · fainter tick = possession lost</span>
    <span class="legend-chip faint">Click to seek, or click a tick to jump to that event</span>`;
}

/** Bind a handler, tolerating an element that is not there.
 *
 * Start-up is a single straight line: one `$('#thing').addEventListener` on a
 * selector that matches nothing throws, and every line after it — the pitch
 * markings, the player list, the outcome menu — silently never runs. The page
 * then looks comprehensively broken because of one button.
 *
 * A missing control should cost only that control. It is still worth knowing
 * about, so it says so rather than passing in silence.
 */
function bind(selector, type, handler) {
  const element = $(selector);
  if (!element) {
    console.warn(`no element for ${selector} — its handler was not attached`);
    return null;
  }
  element.addEventListener(type, handler);
  return element;
}

function init() {
  pitch.onPlace = placeEvent;
  timeline.onSeek = seek;
  timeline.onSelect = (index) => store.set({ selectedEvent: index }, 'selection');

  setupVideo();
  setupFeed();
  setupEventsTable();
  setupSaved();
  setupSquads();
  setupKeyboard();

  $$('#teamToggle button').forEach((button) => {
    button.addEventListener('click', () => store.set({ currentTeam: Number(button.dataset.team) }, 'teams'));
  });
  $$('.tab').forEach((tab) => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));
  $$('[data-close-dialog]').forEach((b) => b.addEventListener('click', () => b.closest('dialog').close()));

  bind('#half', 'change', (e) => store.set({ half: Number(e.target.value) }, 'pending'));
  bind('#btnUndo', 'click', () => store.undoLast());
  bind('#btnClearEvents', 'click', () => {
    if (store.state.events.length && confirm('Remove every event from this match?')) store.clearEvents();
  });
  bind('#btnSave', 'click', saveMatch);
  bind('#btnExportCsv', 'click', exportCsv);
  bind('#btnExportJson', 'click', exportJson);
  bind('#btnKeys', 'click', () => $('#keysDialog').showModal());
  bind('#linkSquads', 'click', () => {
    switchTab('squads', { scroll: true });
    $(`#team${store.state.currentTeam}Name`).focus();
  });
  // The same destination, reached from the team switch rather than the player
  // field, because that is where somebody wondering what "Team 1" is called
  // is actually looking.
  bind('#linkTeamSetup', 'click', () => {
    switchTab('squads', { scroll: true });
    $(`#team${store.state.currentTeam}Name`).focus();
    $(`#team${store.state.currentTeam}Name`).select();
  });
  bind('#btnAnalyse', 'click', () => {
    switchTab('ai', { scroll: true });
    if (loadedFile) analysisPanel?.useFile(loadedFile);
  });
  bind('#btnReset', 'click', () => {
    if (!confirm('Clear the events, the video and the squads on this page?')) return;
    store.clearEvents();
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    video.removeAttribute('src');
    video.load();
    $('#videoEmpty').style.display = '';
    store.set({ videoDuration: 0, currentTime: 0, matchId: null, matchName: '' }, 'video', 'time');
    toast('Cleared.', 'info');
  });

  bind('#btnAddEvent', 'click', () => {
    const name = $('#newEvent').value.trim();
    if (!name) return;
    if (store.state.eventTypes.includes(name)) { toast('That event type already exists.', 'bad'); return; }
    store.set({ eventTypes: [...store.state.eventTypes, name] }, 'types');
    $('#newEvent').value = '';
    toast(`Added “${name}”. It takes one click on the pitch.`, 'good');
  });
  bind('#newEvent', 'keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $('#btnAddEvent').click(); }
  });

  // Re-render on the channels each view cares about.
  store.on('events', () => {
    renderEventsTable();
    renderFeed();
    renderScore();
    if ($('.tabpane[data-pane="stats"]').classList.contains('is-active')) renderStats($('#statsPane'));
  });
  store.on('selection', () => { renderEventsTable(); renderFeed(); });
  store.on('teams', () => { renderTeams(); renderPlayers(); renderEventsTable(); renderScore(); renderTimelineLegend(); });
  store.on('types', () => { renderPalette(); renderOutcomes(); });

  renderTeams();
  renderPlayers();
  renderPalette();
  renderOutcomes();
  renderEventsTable();
  renderFeed();
  renderScore();
  renderTimelineLegend();
  timeline.resize();
  pitch.draw();

  analysisPanel = new AnalysisPanel($('#aiPane'), {
    toast,
    onImport: (count, replaced) => {
      toast(`${replaced ? 'Replaced with' : 'Added'} ${count} detected events.`, 'good');
      switchTab('events', { scroll: true });
    },
  });
  if (loadedFile) analysisPanel.useFile(loadedFile);
}

init();
