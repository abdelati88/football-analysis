/* The match timeline.
 *
 * Every event is a tick on a strip the width of the match: above the centre
 * axis for the home team, below it for the away team. Read left to right it
 * shows the shape of the game — a run of ticks on one side is a spell of
 * pressure, a gap is a stoppage — and clicking anywhere on it seeks the video.
 *
 * This is the piece that ties the three views together: the video knows time,
 * the pitch knows space, and the timeline is where they meet.
 */

import { eventTime, isGoal, store, typeInfo } from './store.js';

const AXIS_COLOUR = '#24344f';
const PLAYHEAD = '#ffb020';

export class Timeline {
  constructor(container, canvas) {
    this.container = container;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSeek = () => {};
    this.onSelect = () => {};
    this.hoverIndex = -1;

    container.addEventListener('click', (e) => this.handleClick(e));
    container.addEventListener('mousemove', (e) => this.handleMove(e));
    container.addEventListener('mouseleave', () => {
      this.hoverIndex = -1;
      this.container.title = '';
      this.draw();
    });
    container.addEventListener('keydown', (e) => this.handleKey(e));

    const observer = new ResizeObserver(() => this.resize());
    observer.observe(container);

    store.on('events', () => this.draw());
    store.on('time', () => this.draw());
    store.on('teams', () => this.draw());
    store.on('selection', () => this.draw());
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = width;
    this.height = height;
    this.draw();
  }

  get duration() {
    // Before a video is loaded the timeline still has to span the events that
    // exist, or hand-tagged work would be invisible until a video appears.
    const fromVideo = store.state.videoDuration || 0;
    const fromEvents = store.state.events.reduce((max, e) => Math.max(max, eventTime(e)), 0);
    return Math.max(fromVideo, fromEvents + 10, 60);
  }

  timeToX(t) {
    return (t / this.duration) * this.width;
  }

  xToTime(x) {
    return Math.max(0, Math.min(this.duration, (x / this.width) * this.duration));
  }

  eventAt(x, y) {
    const axis = this.height / 2;
    let best = -1;
    let bestDistance = 9;
    store.state.events.forEach((event, index) => {
      const ex = this.timeToX(eventTime(event));
      const team = store.teamIndex(event.Team);
      const ey = team === 2 ? axis + 15 : axis - 15;
      const distance = Math.abs(ex - x) + Math.abs(ey - y) * 0.25;
      if (distance < bestDistance) { bestDistance = distance; best = index; }
    });
    return best;
  }

  handleMove(e) {
    const rect = this.container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const index = this.eventAt(x, y);
    if (index !== this.hoverIndex) {
      this.hoverIndex = index;
      const event = store.state.events[index];
      this.container.title = event
        ? `${event.Event}${event.Outcome ? ` · ${event.Outcome}` : ''} — ${event.Player || '—'} (${event.Team})`
        : 'Click to seek';
      this.draw();
    }
  }

  handleClick(e) {
    const rect = this.container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const index = this.eventAt(x, y);
    if (index >= 0) {
      this.onSelect(index);
      this.onSeek(eventTime(store.state.events[index]));
    } else {
      this.onSeek(this.xToTime(x));
    }
  }

  handleKey(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const step = e.shiftKey ? 10 : 2;
    this.onSeek(store.state.currentTime + (e.key === 'ArrowRight' ? step : -step));
  }

  draw() {
    if (!this.width) { this.resize(); return; }
    const ctx = this.ctx;
    const { width: w, height: h } = this;
    const axis = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Without a video every event sits at 0:00, so the strip would be a single
    // stack of ticks at the left edge pretending to be a timeline.
    if (!store.state.videoDuration) {
      ctx.fillStyle = '#6c7f9d';
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        store.state.events.length
          ? 'Load the match video to place these events in time'
          : 'The match timeline appears here once a video is loaded',
        w / 2, h / 2 + 4,
      );
      ctx.textAlign = 'left';
      return;
    }

    // Minute rules, spaced so they stay legible however long the match is.
    const duration = this.duration;
    const step = duration > 3600 ? 600 : duration > 900 ? 300 : duration > 240 ? 60 : 30;
    ctx.strokeStyle = 'rgba(255,255,255,.05)';
    ctx.fillStyle = '#6c7f9d';
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.lineWidth = 1;
    for (let t = 0; t <= duration; t += step) {
      const x = Math.round(this.timeToX(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, h - 12); ctx.stroke();
      if (t > 0) ctx.fillText(`${Math.round(t / 60)}'`, x + 4, h - 4);
    }

    ctx.strokeStyle = AXIS_COLOUR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(axis) + 0.5);
    ctx.lineTo(w, Math.round(axis) + 0.5);
    ctx.stroke();

    // Ticks. Goals get a full-height marker, because a goal is not just
    // another event in the list.
    store.state.events.forEach((event, index) => {
      const x = this.timeToX(eventTime(event));
      const team = store.teamIndex(event.Team);
      const up = team !== 2;
      const info = typeInfo(event.Event);
      const selected = index === store.state.selectedEvent;
      const hovered = index === this.hoverIndex;

      // Colour is team, here as on the pitch. A lost ball is a fainter tick,
      // a shot is a taller one, a goal reaches the top.
      const colour = team ? store.teamColour(team) : '#8fa3c4';
      const failed = event.Outcome === 'Unsuccessful' || event.Outcome === 'Failed';

      const length = isGoal(event) ? axis - 8 : (event.Event === 'Shot' ? 20 : 13);
      const thickness = selected || hovered ? 3 : 2;

      ctx.strokeStyle = colour;
      ctx.globalAlpha = selected || hovered ? 1 : (failed ? 0.42 : 0.82);
      ctx.lineWidth = thickness;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x, axis + (up ? -2 : 2));
      ctx.lineTo(x, axis + (up ? -length : length));
      ctx.stroke();

      if (isGoal(event)) {
        ctx.fillStyle = '#3fd98a';
        ctx.beginPath();
        ctx.arc(x, axis + (up ? -length - 5 : length + 5), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });

    // Playhead.
    const px = this.timeToX(store.state.currentTime || 0);
    ctx.strokeStyle = PLAYHEAD;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    ctx.fillStyle = PLAYHEAD;
    ctx.beginPath();
    ctx.moveTo(px - 5, 0); ctx.lineTo(px + 5, 0); ctx.lineTo(px, 7);
    ctx.closePath(); ctx.fill();
  }
}
