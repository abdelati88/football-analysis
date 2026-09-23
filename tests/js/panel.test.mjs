/* When a newly chosen video may take over the analysis panel.
 *
 * This rule broke once in a way worth guarding: it refused a new file whenever
 * *any* job was present, including a finished one. That was harmless while
 * nothing was finished at page load, and became invisible breakage the moment
 * completed runs started being recovered — choosing a video did nothing at
 * all, with no message, which reads exactly like a dead button.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { mayReplaceRun } from '../../match_tag/static/js/ai.js';

test('a fresh panel accepts a video', () => {
  assert.equal(mayReplaceRun(null), true);
  assert.equal(mayReplaceRun(undefined), true);
});

test('a run in flight keeps the panel', () => {
  assert.equal(mayReplaceRun({ status: 'queued' }), false);
  assert.equal(mayReplaceRun({ status: 'running' }), false);
});

test('a finished run steps aside for a new video', () => {
  // The regression: this returned false, so the upload vanished in silence.
  assert.equal(mayReplaceRun({ status: 'done' }), true);
});

test('a recovered run is history, not work in progress', () => {
  // Recovery puts a completed job on the panel at load. It must not lock it.
  assert.equal(mayReplaceRun({ status: 'done', job_id: 'recovered' }), true);
});

test('a failed or cancelled run never blocks a retry', () => {
  assert.equal(mayReplaceRun({ status: 'failed' }), true);
  assert.equal(mayReplaceRun({ status: 'cancelled' }), true);
});
