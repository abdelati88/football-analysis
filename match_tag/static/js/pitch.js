/* The pitch canvas: drawing, and placing events on it by clicking.
 *
 * The coordinate system is the tagger's own 120 x 80 grid, which every saved
 * match is already stored in. Second-half events are recorded rotated 180
 * degrees about the centre spot, so that a team's attacks all point the same
 * way in the data whichever end they were actually kicking towards.
 */

import { eventTime, isGoal, store, typeInfo } from './store.js';

const PAD = 26;                 // canvas padding around the playing area, in px
const LENGTH = 120;
const WIDTH = 80;

const TURF = '#2c6540';
const TURF_STRIPE = '#31704697';
const LINE = '#dfe8f2';

export class Pitch {
  constructor(canvas, hintElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.hint = hintElement;
    this.hover = null;
    this.onPlace = () => {};

    canvas.addEventListener('click', (e) => this.handleClick(e));
    canvas.addEventListener('mousemove', (e) => this.handleMove(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.draw(); this.setHint(''); });

    store.on('events', () => this.draw());
    store.on('teams', () => this.draw());
    store.on('selection', () => this.draw());
    store.on('pending', () => this.draw());
  }

  // ---- geometry ----------------------------------------------------------

  get area() {
    return {
      w: this.canvas.width - 2 * PAD,
      h: this.canvas.height - 2 * PAD,
    };
  }

  toCanvas(x, y) {
    const { w, h } = this.area;
    return [PAD + (x / LENGTH) * w, PAD + (y / WIDTH) * h];
  }

  fromEvent(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const cy = (e.clientY - rect.top) * (this.canvas.height / rect.height);
    const { w, h } = this.area;
    const x = ((cx - PAD) / w) * LENGTH;
    const y = ((cy - PAD) / h) * WIDTH;
    return [
      Math.round(Math.min(Math.max(x, 0), LENGTH) * 10) / 10,
      Math.round(Math.min(Math.max(y, 0), WIDTH) * 10) / 10,
    ];
  }

  /** Second-half coordinates are stored rotated, so both halves compare. */
  normalise(x, y) {
    if (Number(store.state.half) !== 2) return [x, y];
    return [Math.round((LENGTH - x) * 10) / 10, Math.round((WIDTH - y) * 10) / 10];
  }

  // ---- interaction -------------------------------------------------------

  setHint(text) {
    if (!this.hint) return;
    this.hint.textContent = text;
    this.hint.classList.toggle('is-visible', Boolean(text));
  }

  handleMove(e) {
    this.hover = this.fromEvent(e);
    const info = typeInfo(store.state.currentType);
    const [x, y] = this.hover;
    const stage = store.state.pendingStart
      ? 'Click where it ended'
      : (info.arrow ? 'Click where it started' : 'Click to place');
    this.setHint(`${stage} · ${x.toFixed(0)}, ${y.toFixed(0)}`);
    this.draw();
  }

  handleClick(e) {
    const [x, y] = this.fromEvent(e);
    const [nx, ny] = this.normalise(x, y);
    const info = typeInfo(store.state.currentType);

    if (!info.arrow) {
      this.onPlace({ x: nx, y: ny, x2: null, y2: null });
      return;
    }

    if (!store.state.pendingStart) {
      store.set({ pendingStart: { x: nx, y: ny, rawX: x, rawY: y } }, 'pending');
      return;
    }

    const start = store.state.pendingStart;
    store.set({ pendingStart: null }, 'pending');
    this.onPlace({ x: start.x, y: start.y, x2: nx, y2: ny });
  }

  cancelPending() {
    if (store.state.pendingStart) store.set({ pendingStart: null }, 'pending');
  }

  // ---- drawing -----------------------------------------------------------

  drawMarkings() {
    const ctx = this.ctx;
    const { w, h } = this.area;
    const W = this.canvas.width;
    const H = this.canvas.height;

    ctx.fillStyle = TURF;
    ctx.fillRect(0, 0, W, H);

    // Mown stripes run the length of the pitch, as they do on a real one.
    const stripes = 12;
    for (let i = 0; i < stripes; i += 1) {
      if (i % 2 === 0) continue;
      ctx.fillStyle = TURF_STRIPE;
      ctx.fillRect(PAD + (i * w) / stripes, PAD, w / stripes, h);
    }

    ctx.strokeStyle = LINE;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.85;

    const px = (x, y) => this.toCanvas(x, y);
    const rect = (x1, y1, x2, y2) => {
      const [ax, ay] = px(x1, y1);
      const [bx, by] = px(x2, y2);
      ctx.strokeRect(ax, ay, bx - ax, by - ay);
    };

    rect(0, 0, LENGTH, WIDTH);                       // touchlines and goal lines

    const [mx1, my1] = px(60, 0);
    const [mx2, my2] = px(60, WIDTH);
    ctx.beginPath(); ctx.moveTo(mx1, my1); ctx.lineTo(mx2, my2); ctx.stroke();

    const [cx, cy] = px(60, 40);
    const radius = (10 / WIDTH) * h;
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();

    // Penalty and six-yard areas, proportioned to the 120 x 80 grid.
    rect(0, 18, 18, 62);
    rect(0, 30, 6, 50);
    rect(LENGTH - 18, 18, LENGTH, 62);
    rect(LENGTH - 6, 30, LENGTH, 50);

    ctx.fillStyle = LINE;
    for (const [sx, sy] of [[12, 40], [LENGTH - 12, 40], [60, 40]]) {
      const [dx, dy] = px(sx, sy);
      ctx.beginPath(); ctx.arc(dx, dy, 3.5, 0, Math.PI * 2); ctx.fill();
    }

    // Goals, drawn just outside the goal line.
    ctx.lineWidth = 4;
    for (const side of [0, LENGTH]) {
      const [gx, gy1] = px(side, 36);
      const [, gy2] = px(side, 44);
      const depth = side === 0 ? -10 : 10;
      ctx.beginPath();
      ctx.moveTo(gx, gy1);
      ctx.lineTo(gx + depth, gy1);
      ctx.lineTo(gx + depth, gy2);
      ctx.lineTo(gx, gy2);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
  }

  drawArrow(x1, y1, x2, y2, colour, options = {}) {
    const ctx = this.ctx;
    const [ax, ay] = this.toCanvas(x1, y1);
    const [bx, by] = this.toCanvas(x2, y2);
    const width = options.emphasis ? 5 : 3;

    ctx.save();
    ctx.globalAlpha = options.alpha ?? 1;
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    if (options.failed) ctx.setLineDash([9, 7]);

    ctx.beginPath();
    if (options.curved) {
      // A carry is a run, not a kick: bow the line so it reads differently
      // from a pass at a glance.
      const mx = (ax + bx) / 2;
      const my = (ay + by) / 2;
      const nx = -(by - ay);
      const ny = bx - ax;
      const len = Math.hypot(nx, ny) || 1;
      ctx.moveTo(ax, ay);
      ctx.quadraticCurveTo(mx + (nx / len) * 26, my + (ny / len) * 26, bx, by);
    } else {
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    const angle = Math.atan2(by - ay, bx - ax);
    const head = options.emphasis ? 17 : 13;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - head * Math.cos(angle - Math.PI / 6), by - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(bx - head * Math.cos(angle + Math.PI / 6), by - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    if (options.failed) {
      // Hollow head: the ball set off but never arrived.
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = colour;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(ax, ay, options.emphasis ? 7 : 5, 0, Math.PI * 2);
    ctx.fillStyle = '#0a1120';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = colour;
    ctx.stroke();

    if (options.goal) {
      ctx.beginPath();
      ctx.arc(bx, by, 13, 0, Math.PI * 2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#3fd98a';
      ctx.stroke();
    }
    ctx.restore();
  }

  drawMarker(x, y, colour, shape, options = {}) {
    const ctx = this.ctx;
    const [cx, cy] = this.toCanvas(x, y);
    const size = options.emphasis ? 12 : 9;

    ctx.save();
    ctx.globalAlpha = options.alpha ?? 1;
    ctx.fillStyle = options.failed ? 'transparent' : colour;
    ctx.strokeStyle = options.failed ? colour : '#0a1120';
    ctx.lineWidth = 2;

    if (shape === 'square') {
      if (!options.failed) ctx.fillRect(cx - size / 2, cy - size / 2, size, size);
      ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
    } else if (shape === 'cross') {
      ctx.strokeStyle = colour;
      ctx.lineWidth = options.emphasis ? 4.5 : 3.5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx - size / 2, cy - size / 2); ctx.lineTo(cx + size / 2, cy + size / 2);
      ctx.moveTo(cx + size / 2, cy - size / 2); ctx.lineTo(cx - size / 2, cy + size / 2);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, size / 2, 0, Math.PI * 2);
      if (!options.failed) ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  draw() {
    this.drawMarkings();

    const { events, selectedEvent, pendingStart, currentTime } = store.state;

    events.forEach((event, index) => {
      const selected = index === selectedEvent;
      const info = typeInfo(event.Event);
      const teamIdx = store.teamIndex(event.Team);
      const teamColour = teamIdx ? store.teamColour(teamIdx) : '#9fb0cc';

      // Colour always means team, never outcome. A red "failed" would be
      // indistinguishable from a team that plays in red, which is most of
      // them. Outcome is carried by line style instead: a lost ball is dashed
      // with a hollow head, a goal gets a bright ring.
      const colour = teamColour;
      const failed = event.Outcome === 'Unsuccessful' || event.Outcome === 'Failed'
        || event.Outcome === 'Off Target' || event.Outcome === 'Blocked';
      let alpha = selected ? 1 : (failed ? 0.62 : 0.86);

      // Events far from the playhead recede, so the pitch shows the passage of
      // play being watched rather than the whole match at once.
      if (!selected && currentTime > 0) {
        const gap = Math.abs(eventTime(event) - currentTime);
        if (gap > 25) alpha = 0.3;
        else if (gap > 8) alpha = 0.55;
      }

      const options = { emphasis: selected, alpha, failed, goal: isGoal(event) };

      if (event.X2 !== '' && event.X2 !== null && event.X2 !== undefined) {
        this.drawArrow(
          event.X, event.Y, event.X2, event.Y2, colour,
          { ...options, curved: event.Event === 'Dribble' },
        );
      } else if (event.Event === 'Tackle') {
        this.drawMarker(event.X, event.Y, colour, 'square', options);
      } else if (event.Event === 'Interception' || event.Event === 'Block') {
        this.drawMarker(event.X, event.Y, colour, 'cross', options);
      } else {
        this.drawMarker(event.X, event.Y, colour, 'dot', options);
      }
    });

    // The pending start point, shown where it was clicked rather than where it
    // will be stored, so the analyst sees their own click.
    if (pendingStart) {
      const ctx = this.ctx;
      const [cx, cy] = this.toCanvas(pendingStart.rawX, pendingStart.rawY);
      ctx.save();
      ctx.strokeStyle = '#ffb020';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ffb020';
      ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, Math.PI * 2); ctx.fill();

      if (this.hover) {
        const [hx, hy] = this.toCanvas(this.hover[0], this.hover[1]);
        ctx.globalAlpha = 0.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(hx, hy); ctx.stroke();
      }
      ctx.restore();
    }
  }
}
