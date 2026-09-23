/* Application state.
 *
 * One object holds everything the page knows, and every mutation goes through
 * a setter that notifies subscribers. Views never read each other; they read
 * the store and re-render. That is what lets the pitch, the table, the
 * timeline and the statistics all stay in step when an event is added from any
 * of them — or by the AI.
 */

const STORAGE_KEY = 'matchtag.v3';

/* Event vocabulary.
 *
 * `arrow` marks the events that describe a movement of the ball and therefore
 * need two clicks — a start and an end. `outcomes` is what the outcome menu
 * offers when that event is selected; an empty list hides the menu entirely,
 * because an interception has no outcome and a menu offering none is noise.
 */
export const EVENT_TYPES = {
  'Pass':         { arrow: true,  outcomes: ['Successful', 'Unsuccessful'], colour: '#4fd6a0' },
  'Cross':        { arrow: true,  outcomes: ['Successful', 'Unsuccessful'], colour: '#7ab8ff' },
  'Shot':         { arrow: true,  outcomes: ['Goal', 'Saved', 'Off Target', 'Blocked'], colour: '#ffb020' },
  'Shot Assist':  { arrow: true,  outcomes: ['Successful'], colour: '#ffd88a' },
  'Dribble':      { arrow: true,  outcomes: ['Successful', 'Failed'], colour: '#c58cff' },
  'Tackle':       { arrow: false, outcomes: ['Successful', 'Unsuccessful'], colour: '#ff8a5c' },
  'Interception': { arrow: false, outcomes: [], colour: '#ffc861' },
  'Block':        { arrow: false, outcomes: [], colour: '#9fb0cc' },
  'Clearance':    { arrow: true,  outcomes: [], colour: '#6f8fbf' },
};

export const DEFAULT_EVENT_ORDER = Object.keys(EVENT_TYPES);

/** A custom event type the analyst added: one click, no outcomes. */
export function typeInfo(name) {
  return EVENT_TYPES[name] || { arrow: false, outcomes: [], colour: '#8fa3c4' };
}

export function isGoal(event) {
  return event.Event === 'Shot' && event.Outcome === 'Goal';
}

/** Seconds -> "12:04.28" for the clock, "12:04" for tables. */
export function formatTime(seconds, withFrames = false) {
  const s = Math.max(0, seconds || 0);
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  const base = `${mins}:${String(secs).padStart(2, '0')}`;
  if (!withFrames) return base;
  return `${base}.${String(Math.floor((s % 1) * 100)).padStart(2, '0')}`;
}

export function eventTime(event) {
  // `t` carries sub-second precision for events created in this session. Events
  // loaded from the database only have whole seconds, which is all they ever
  // stored.
  return typeof event.t === 'number' ? event.t : (event.Mins || 0) * 60 + (event.Secs || 0);
}

const DEFAULT_STATE = {
  team1Name: 'Team 1',
  team2Name: 'Team 2',
  team1Colour: '#e8503a',
  team2Colour: '#3d8bfd',
  roster1: '',
  roster2: '',
  eventTypes: [...DEFAULT_EVENT_ORDER],
  matchName: '',
  trackNames: {},           // AI track id -> the analyst's name for that player
};

function loadPersisted() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

class Store {
  constructor() {
    this.state = {
      ...DEFAULT_STATE,
      ...loadPersisted(),
      // Never persisted: these belong to the session, not the browser.
      events: [],
      currentTeam: 1,
      currentType: 'Pass',
      currentOutcome: '',
      pendingStart: null,
      half: 1,
      videoName: null,
      videoDuration: 0,
      currentTime: 0,
      selectedEvent: -1,
      matchId: null,
    };
    this.listeners = new Map();
  }

  /** Subscribe to a channel; returns an unsubscribe function. */
  on(channel, fn) {
    if (!this.listeners.has(channel)) this.listeners.set(channel, new Set());
    this.listeners.get(channel).add(fn);
    return () => this.listeners.get(channel).delete(fn);
  }

  emit(...channels) {
    for (const channel of channels) {
      for (const fn of this.listeners.get(channel) || []) fn(this.state);
    }
  }

  set(patch, ...channels) {
    Object.assign(this.state, patch);
    this.persist();
    this.emit(...channels);
  }

  persist() {
    const { team1Name, team2Name, team1Colour, team2Colour, roster1, roster2,
            eventTypes, matchName, trackNames } = this.state;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        team1Name, team2Name, team1Colour, team2Colour,
        roster1, roster2, eventTypes, matchName, trackNames,
      }));
    } catch {
      // A full or disabled localStorage must not break tagging.
    }
  }

  // ---- events ------------------------------------------------------------

  teamName(index) {
    return index === 1 ? this.state.team1Name : this.state.team2Name;
  }

  teamColour(index) {
    return index === 1 ? this.state.team1Colour : this.state.team2Colour;
  }

  /** 1 or 2 for a team name, or 0 when the name matches neither. */
  teamIndex(name) {
    if (name === this.state.team1Name) return 1;
    if (name === this.state.team2Name) return 2;
    return 0;
  }

  roster(index) {
    const raw = index === 1 ? this.state.roster1 : this.state.roster2;
    return (raw || '').split('\n').map((s) => s.trim()).filter(Boolean);
  }

  addEvent(event) {
    this.state.events.push(event);
    this.emit('events');
  }

  addEvents(events) {
    this.state.events.push(...events);
    this.emit('events');
  }

  /** Change one already-tagged event.
   *
   * Tagging live means getting things wrong: the wrong team pressed, a pass
   * that turned out to be a cross, a name typed before the player was
   * recognised. Without this the only remedy is to delete and re-tag, which
   * loses the pitch coordinates that were clicked — the part that cannot be
   * retyped.
   */
  updateEvent(index, patch) {
    const event = this.state.events[index];
    if (!event) return null;
    Object.assign(event, patch);
    this.emit('events');
    return event;
  }

  removeEvent(index) {
    if (index < 0 || index >= this.state.events.length) return;
    this.state.events.splice(index, 1);
    if (this.state.selectedEvent === index) this.state.selectedEvent = -1;
    this.emit('events');
  }

  undoLast() {
    const removed = this.state.events.pop();
    this.emit('events');
    return removed;
  }

  clearEvents() {
    this.state.events = [];
    this.state.selectedEvent = -1;
    this.state.pendingStart = null;
    this.emit('events');
  }

  /** Rename team names inside already-tagged events, so the table stays true. */
  renameTeamInEvents(oldName, newName) {
    if (!oldName || oldName === newName) return;
    for (const event of this.state.events) {
      if (event.Team === oldName) event.Team = newName;
    }
  }
}

export const store = new Store();
