/* The AI analysis panel.
 *
 * One button, one video, and a few minutes later the same kind of events the
 * analyst would have clicked in by hand — placed on the same pitch, in the
 * same table, ready to be corrected. The machine does the first pass; the
 * person keeps the last word.
 */

import { api } from './api.js';
import { store } from './store.js';
import { escapeHtml } from './stats.js';

const STAGES = [
  ['survey', 'Reading kit colours'],
  ['analyse', 'Tracking players and the ball'],
  ['derive', 'Working out possession and events'],
  ['render', 'Drawing the annotated video'],
];

const POLL_MS = 1200;

/** May a newly chosen video take over the panel from `job`?
 *
 * Only a run actually in flight says no. Refusing for a *finished* run as well
 * used to be harmless, because nothing was finished when the page opened; once
 * a completed run started being recovered on load, that same rule swallowed
 * every subsequent upload in silence — choosing a video and having the page do
 * nothing, which is indistinguishable from a broken button.
 *
 * A finished result is history. Choosing a new video says the user is done
 * with it.
 */
export function mayReplaceRun(job) {
  if (!job) return true;
  return job.status !== 'queued' && job.status !== 'running';
}

/** How well supported one event is, as a word rather than a percentage.
 *
 * The underlying number blends how reliable the rule that produced the event
 * is with how well the ball was actually seen while it happened. It is not a
 * calibrated probability, and printing it as "55%" invited exactly the reading
 * it cannot support — a coin flip — when what it means is "derived from a
 * sound rule, on a stretch where the ball was only intermittently visible".
 * Three words claim precisely as much as the number can carry.
 */
function evidenceLabel(value) {
  const v = Number(value) || 0;
  if (v >= 0.6) return '<span class="tag tag-good">Strong</span>';
  if (v >= 0.4) return '<span class="tag">Fair</span>';
  return '<span class="tag tag-warn">Weak</span>';
}

/** How the re-linking stage did, in one line.
 *
 * Worth showing because it is the difference between per-player numbers that
 * mean something and per-player numbers split across three phantom players.
 * The tracker labels the same person afresh every time it loses sight of them
 * — behind a crowd, or when the camera pans — and this says how many of those
 * strangers were put back together afterwards.
 */
function identityDetail(identities) {
  if (!identities || !identities.tracks_before) return '—';
  const { tracks_before: before, tracks_after: after, merges } = identities;
  if (!merges) return `${after} identities, none needed re-linking`;
  const pct = Math.round((merges / before) * 100);
  return `${before} tracked fragments joined into ${after} players (${pct}% were duplicates)`;
}

export class AnalysisPanel {
  constructor(container, { toast, onImport }) {
    this.container = container;
    this.toast = toast;
    this.onImport = onImport;
    this.engine = null;
    this.job = null;
    this.file = null;
    this.timer = null;
    this.imported = false;
    this.render();
    this.refreshEngine();
  }

  /** Adopt the video already loaded in the tagger. */
  useFile(file) {
    if (!file || this.file === file) return;
    if (!mayReplaceRun(this.job)) return;
    this.job = null;
    this.recovered = false;
    this.imported = false;
    this.file = file;
    this.render();
  }

  async refreshEngine() {
    try {
      this.engine = await api.engineStatus();
    } catch (error) {
      this.engine = { available: false, reason: error.message, weights: {} };
    }
    this.render();
    // Pick up a run this page is not currently watching — one started before
    // it was opened, and equally one that finished while it was not looking.
    // An analysis costs minutes of the user's time; losing the result because
    // the tab was closed, reloaded, or knocked back to the idle form would
    // throw that away when the answer is sitting on the server.
    if (!this.job) {
      try {
        const jobs = await api.listJobs();
        const live = jobs.find((j) => j.status === 'running' || j.status === 'queued');
        if (live) {
          this.job = live;
          this.startPolling();
        } else {
          const recent = jobs.find((j) => j.status === 'done');
          if (recent) {
            // Fetched with its result, so the events are ready to import
            // rather than merely announced.
            this.job = await api.jobStatus(recent.job_id, true);
            this.recovered = true;
          }
        }
        this.render();
      } catch { /* the job list is a convenience, not a requirement */ }
    }
  }

  // ---- running ------------------------------------------------------------

  async start(file, options) {
    const body = new FormData();
    body.append('video', file);
    body.append('team1', options.team1);
    body.append('team2', options.team2);
    body.append('half', options.half);
    body.append('render_video', options.renderVideo ? 'true' : 'false');
    if (options.maxSeconds) body.append('max_seconds', options.maxSeconds);
    if (options.sampleFps) body.append('sample_fps', options.sampleFps);

    this.imported = false;
    this.recovered = false;
    this.job = { status: 'queued', stage: 'queued', progress: 0, message: 'Uploading the video' };
    this.render();

    try {
      this.job = await api.startAnalysis(body);
      this.startPolling();
      this.toast('Analysis started. You can keep tagging while it runs.', 'info');
    } catch (error) {
      // "Something is already running" is the one refusal that is really an
      // answer: the server hands back the job it is busy with. Discarding it
      // and showing the idle form again was the worst possible response — the
      // run carried on to completion with nobody watching it, so a finished
      // analysis looked exactly like one that never started.
      const running = error.status === 409 ? error.payload?.current_job : null;
      if (running) {
        this.job = typeof running === 'string' ? { job_id: running, status: 'running' } : running;
        this.startPolling();
        this.toast('That analysis is already running — showing its progress.', 'info');
      } else {
        this.job = null;
        this.toast(error.message, 'bad');
      }
    }
    this.render();
  }

  startPolling() {
    this.stopPolling();
    this.timer = setInterval(() => this.poll(), POLL_MS);
  }

  stopPolling() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  async poll() {
    if (!this.job?.job_id) return this.stopPolling();
    try {
      const finished = ['done', 'failed', 'cancelled'];
      const next = await api.jobStatus(this.job.job_id, true);
      this.job = next;
      if (finished.includes(next.status)) {
        this.stopPolling();
        if (next.status === 'done') {
          const found = next.result?.events?.length || 0;
          // With no hand-tagged work to protect, put the events straight into
          // the table. Waiting minutes for an analysis and then finding an
          // empty table with a button you have to notice is not a result.
          if (found && store.state.events.length === 0) {
            this.imported = true;
            this.importEvents(false);
          } else if (found) {
            this.toast(`Analysis finished — ${found} events found. Choose how to add them.`, 'good');
          } else {
            this.toast('Analysis finished, but no events could be recognised.', 'bad');
          }
        } else if (next.status === 'failed') {
          this.toast(`Analysis failed: ${next.message}`, 'bad');
        }
      }
      this.render();
    } catch (error) {
      this.stopPolling();
      this.toast(error.message, 'bad');
    }
  }

  async cancel() {
    if (!this.job?.job_id) return;
    try {
      await api.cancelJob(this.job.job_id);
      this.toast('Stopping the analysis…', 'info');
    } catch (error) {
      this.toast(error.message, 'bad');
    }
  }

  importEvents(replace) {
    const events = this.job?.result?.events || [];
    if (!events.length) return;
    // The pipeline emits `Mins`/`Secs`; give each event a precise `t` from its
    // frame so the timeline can place it exactly.
    const fps = this.job.result?.diagnostics?.video?.fps || 25;
    const prepared = events.map((e) => ({
      ...e,
      t: typeof e.frame === 'number' ? e.frame / fps : (e.Mins || 0) * 60 + (e.Secs || 0),
    }));

    if (replace) store.clearEvents();
    store.addEvents(prepared);
    this.onImport?.(prepared.length, replace);
  }

  // ---- rendering ----------------------------------------------------------

  engineNotice() {
    if (!this.engine) return '<div class="notice">Checking the vision engine…</div>';
    if (this.engine.available) {
      const device = { cuda: 'an NVIDIA GPU', mps: 'the Apple GPU', cpu: 'the CPU' }[this.engine.device]
        || this.engine.device;
      return `<div class="notice">
        <span>Ready. Analysis will run on <b>${escapeHtml(device)}</b>.
        Expect roughly one to three minutes of processing per minute of video.</span>
      </div>`;
    }
    return `<div class="notice notice-warn">
      <span><b>The engine is not ready.</b><br>${escapeHtml(this.engine.reason || 'Unknown reason')}</span>
    </div>`;
  }

  progressView() {
    const job = this.job;
    const percent = Math.round((job.progress || 0) * 100);
    const stageIndex = STAGES.findIndex(([key]) => key === job.stage);
    const running = job.status === 'running' || job.status === 'queued';

    return `
      <div class="row" style="justify-content: space-between; margin-bottom: 8px">
        <span class="panel-title">${percent}%</span>
        <span class="muted">${escapeHtml(job.message || '')}</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:${percent}%"></div></div>
      <ul class="stage-list">
        ${STAGES.map(([key, label], i) => {
          const state = stageIndex > i || job.status === 'done' ? 'is-done'
            : stageIndex === i ? 'is-active' : '';
          return `<li class="stage-item ${state}"><span class="dot"></span>${label}</li>`;
        }).join('')}
      </ul>
      ${running ? `
        <div class="row" style="margin-top: 12px">
          <button class="btn btn-danger btn-sm" data-action="cancel">Stop the analysis</button>
          <span class="faint" style="font-size: 12px">Tagging by hand still works while this runs.</span>
        </div>` : ''}`;
  }

  resultView() {
    const job = this.job;
    const result = job.result || {};
    const stats = result.stats || {};
    const diagnostics = result.diagnostics || {};
    const events = result.events || [];
    const teams = stats.teams || {};
    const calibration = diagnostics.calibration || {};

    const t1 = teams['1'] || teams[1] || {};
    const t2 = teams['2'] || teams[2] || {};

    // Calibration coverage decides whether the pitch coordinates mean anything,
    // so it is stated plainly rather than buried in a diagnostics blob.
    const coverage = Math.round((calibration.coverage || 0) * 100);
    const coverageNotice = coverage >= 70
      ? ''
      : `<div class="notice notice-warn" style="margin-bottom: 12px">
           <span>The pitch was only located in <b>${coverage}%</b> of the frames checked.
           Events still carry times and players, but their positions on the pitch are
           less reliable. A wider camera angle that shows more pitch markings helps.</span>
         </div>`;

    // A recovered run is somebody else's work as far as this page is
    // concerned — it finished while nothing was watching. Say so, or the
    // numbers read as the result of whatever the user did most recently.
    const recoveredNotice = this.recovered
      ? `<div class="notice" style="margin-bottom: 12px">
           <span>This is the <b>last analysis that finished</b> on this server, recovered
           because the page was not watching when it completed. Its results are complete
           and ready to add.</span>
         </div>`
      : '';

    return `
      ${recoveredNotice}
      ${coverageNotice}
      <div class="stat-grid" style="margin-bottom: 14px">
        <div class="stat-card">
          <div class="figure" style="color: var(--amber)">${events.length}</div>
          <div class="eyebrow">Events found</div>
        </div>
        <div class="stat-card">
          <div class="figure">${(t1.goals || 0)}–${(t2.goals || 0)}</div>
          <div class="eyebrow">Goals detected</div>
        </div>
        <div class="stat-card">
          <div class="figure">${Math.round(t1.possession || 0)}<span style="font-size:16px">%</span></div>
          <div class="eyebrow">${escapeHtml(t1.name || 'Team 1')} possession</div>
        </div>
        <div class="stat-card">
          <div class="figure">${stats.totals?.tracked_players ?? '—'}</div>
          <div class="eyebrow">Players tracked</div>
        </div>
        <div class="stat-card">
          <div class="figure">${diagnostics.runtime_s ? Math.round(diagnostics.runtime_s / 60) : '—'}<span style="font-size:16px">m</span></div>
          <div class="eyebrow">Processing time</div>
        </div>
      </div>

      <div class="row" style="margin-bottom: 14px">
        ${this.imported
          ? `<span class="tag tag-good">✓ Added to the match — see the Events tab</span>
             <button class="btn btn-sm" data-action="import">Add them again</button>`
          : `<button class="btn btn-primary" data-action="import">Add these ${events.length} events to the match</button>
             <button class="btn" data-action="import-replace">Replace what I have with these</button>`}
        ${job.has_video ? `<a class="btn" href="${api.jobVideoUrl(job.job_id)}" target="_blank" rel="noopener">Watch the annotated video</a>` : ''}
        <button class="btn btn-sm btn-ghost" data-action="new">Analyse another video</button>
      </div>

      ${events.length ? `
      <div class="eyebrow" style="margin-bottom: 7px">What it found</div>
      <div class="table-scroll" style="max-height: 260px; margin-bottom: 14px">
        <table class="data">
          <thead><tr><th>Time</th><th>Team</th><th>Player</th><th>Event</th><th>Outcome</th><th>Evidence</th></tr></thead>
          <tbody>
            ${events.map((e) => `<tr>
              <td class="num">${e.Mins}:${String(e.Secs).padStart(2, '0')}</td>
              <td>${escapeHtml(e.Team)}</td>
              <td>${escapeHtml(e.Player || '—')}</td>
              <td>${escapeHtml(e.Event)}</td>
              <td>${escapeHtml(e.Outcome || '')}</td>
              <td class="faint">${evidenceLabel(e.confidence)}</td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
      <p class="muted" style="font-size: 12px; margin: -6px 0 14px">
        <b>Evidence</b> is how well this particular event is supported: how reliably
        its kind can be told apart at all, and how clearly the ball was actually
        seen while it happened. <b>Fair</b> does not mean the event is doubtful —
        a dribble is read from a player's run rather than a ball flight, so it is
        never as directly observed as a pass. Anything marked <b>Weak</b> is worth
        checking against the video.
      </p>` : `
      <div class="notice notice-warn" style="margin-bottom: 14px">
        <span>No events could be recognised. That usually means the ball was not
        tracked for long enough — check the pitch coverage below, and try footage
        from a single wide camera with unbroken play.</span>
      </div>`}

      <details open>
        <summary class="eyebrow" style="cursor: pointer">Technical detail</summary>
        <div class="table-scroll" style="max-height: 240px; margin-top: 10px">
          <table class="data">
            <tbody>
              ${row('Video', `${diagnostics.video?.width}×${diagnostics.video?.height} at ${diagnostics.video?.fps} fps, ${diagnostics.video?.duration_s}s`)}
              ${row('Frames analysed', `every ${diagnostics.sampling?.stride} frame (${diagnostics.sampling?.effective_fps} fps)`)}
              ${row('Pitch located in', `${calibration.solved || 0} frames · ${coverage}% coverage · ${calibration.mean_error_px ?? '—'} px mean error`)}
              ${row('Identities re-linked', identityDetail(diagnostics.identities))}
              ${row('Kit samples used', diagnostics.teams?.fit_samples ?? '—')}
              ${row('Team assignment confidence', diagnostics.teams?.mean_confidence ?? '—')}
              ${row('Ran on', diagnostics.device ?? '—')}
            </tbody>
          </table>
        </div>
      </details>`;

    function row(label, value) {
      return `<tr><td class="muted">${escapeHtml(label)}</td><td class="num">${escapeHtml(value)}</td></tr>`;
    }
  }

  formView() {
    const disabled = !this.engine?.available ? 'disabled' : '';
    return `
      <div class="notice" style="margin-bottom: 14px">
        <span><b>What this reads well.</b> Continuous footage of a match from a
        single wide camera — a broadcast feed, a tactical camera, a phone on a
        tripod. It needs to see the pitch markings to work out where players are,
        and it needs unbroken play to follow them.<br>
        <b>What it reads badly.</b> Highlight reels and edited compilations: every
        cut restarts the tracking, close-ups and replays show no pitch lines, and
        the result will be a handful of disconnected events rather than a match.</span>
      </div>

      <div class="dropzone" id="aiDrop" tabindex="0" role="button">
        <div class="figure">${this.file ? escapeHtml(this.file.name) : 'Drop a match video here'}</div>
        <p class="muted" style="margin: 6px 0 0; font-size: 13px">
          ${this.file
            ? `${(this.file.size / 1024 / 1024).toFixed(0)} MB — click to choose a different file`
            : 'or click to choose one. MP4, MOV, MKV, AVI or WebM.'}
        </p>
        <input type="file" id="aiFile" accept="video/*" hidden>
      </div>

      <div class="row" style="margin-top: 14px; align-items: flex-end; gap: 10px">
        <div class="field" style="flex: 1 1 160px">
          <label for="aiTeam1">Home team</label>
          <input class="input" id="aiTeam1" value="${escapeHtml(store.state.team1Name)}">
        </div>
        <div class="field" style="flex: 1 1 160px">
          <label for="aiTeam2">Away team</label>
          <input class="input" id="aiTeam2" value="${escapeHtml(store.state.team2Name)}">
        </div>
        <div class="field" style="flex: 0 0 110px">
          <label for="aiHalf">Half</label>
          <select class="select" id="aiHalf">
            <option value="1" ${store.state.half == 1 ? 'selected' : ''}>First</option>
            <option value="2" ${store.state.half == 2 ? 'selected' : ''}>Second</option>
          </select>
        </div>
        <div class="field" style="flex: 0 0 140px">
          <label for="aiLimit">Analyse only</label>
          <select class="select" id="aiLimit">
            <option value="">The whole video</option>
            <option value="60">First minute</option>
            <option value="300">First 5 minutes</option>
            <option value="900">First 15 minutes</option>
          </select>
        </div>
      </div>

      <label class="row" style="margin-top: 12px; gap: 7px; font-size: 13px; cursor: pointer">
        <input type="checkbox" id="aiRender" checked>
        <span>Also produce an annotated video with the tactical radar</span>
      </label>

      <div class="row" style="margin-top: 14px">
        <button class="btn btn-primary" data-action="run" ${disabled}>Run the analysis</button>
        <span class="faint" style="font-size: 12px">
          The video is uploaded to the server running this page so the models can read it.
        </span>
      </div>`;
  }

  render() {
    const job = this.job;
    let body;

    if (job && (job.status === 'running' || job.status === 'queued')) {
      body = this.progressView();
    } else if (job && job.status === 'done') {
      body = this.resultView();
    } else if (job && (job.status === 'failed' || job.status === 'cancelled')) {
      body = `
        <div class="notice ${job.status === 'failed' ? 'notice-bad' : ''}" style="margin-bottom: 12px">
          <span>${job.status === 'failed' ? 'The analysis failed.' : 'The analysis was stopped.'}
          ${escapeHtml(job.message || '')}</span>
        </div>
        <button class="btn" data-action="new">Try again</button>`;
    } else {
      body = this.formView();
    }

    this.container.innerHTML = `${this.engineNotice()}<div style="margin-top: 14px">${body}</div>`;
    this.bind();
  }

  bind() {
    const root = this.container;

    const drop = root.querySelector('#aiDrop');
    const input = root.querySelector('#aiFile');
    if (drop && input) {
      drop.addEventListener('click', () => input.click());
      drop.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
      });
      drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('is-over'); });
      drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
      // Both routes go through useFile, so there is one rule about when a
      // chosen video may replace what the panel is showing rather than three
      // places quietly disagreeing about it.
      drop.addEventListener('drop', (e) => {
        e.preventDefault();
        drop.classList.remove('is-over');
        const file = e.dataTransfer?.files?.[0];
        if (file) this.useFile(file);
      });
      input.addEventListener('change', () => {
        if (input.files?.[0]) this.useFile(input.files[0]);
      });
    }

    root.querySelectorAll('[data-action]').forEach((element) => {
      element.addEventListener('click', (e) => {
        const action = e.currentTarget.dataset.action;
        if (action === 'run') {
          if (!this.file) { this.toast('Choose a video first.', 'bad'); return; }
          this.start(this.file, {
            team1: root.querySelector('#aiTeam1').value.trim() || 'Team 1',
            team2: root.querySelector('#aiTeam2').value.trim() || 'Team 2',
            half: root.querySelector('#aiHalf').value,
            maxSeconds: root.querySelector('#aiLimit').value,
            renderVideo: root.querySelector('#aiRender').checked,
          });
        } else if (action === 'cancel') {
          this.cancel();
        } else if (action === 'import') {
          this.imported = true;
          this.importEvents(false);
          this.render();
        } else if (action === 'import-replace') {
          this.imported = true;
          this.importEvents(true);
          this.render();
        } else if (action === 'new') {
          this.job = null;
          this.file = null;
          this.render();
          this.refreshEngine();
        }
      });
    });
  }
}
