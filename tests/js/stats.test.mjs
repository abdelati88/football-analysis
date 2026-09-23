/* Checks on the statistics the browser computes.
 *
 * These are the numbers on screen, and they are derived in the browser so that
 * they update the instant an analyst corrects a misread event. Run with:
 *     node --test tests/js/
 */
import assert from 'node:assert/strict';
import test from 'node:test';

const { store } = await import('../../match_tag/static/js/store.js');
const { computeStats } = await import('../../match_tag/static/js/stats.js');

const NAMES = { 1: 'Home', 2: 'Away' };

function reset() {
  store.state.team1Name = 'Home';
  store.state.team2Name = 'Away';
}

const event = (team, type, outcome = '', player = 'p', x = 60, y = 40) => ({
  Team: team, Player: player, Event: type, Outcome: outcome,
  Mins: 0, Secs: 0, X: x, Y: y, X2: '', Y2: '', half: 1, source: 'manual',
});

test('an empty match reports no data', () => {
  reset();
  const s = computeStats([], NAMES);
  assert.equal(s.hasData, false);
  assert.equal(s.teams[1].passes, 0);
});

test('passes and accuracy count per team', () => {
  reset();
  const events = [
    event('Home', 'Pass', 'Successful'),
    event('Home', 'Pass', 'Successful'),
    event('Home', 'Pass', 'Unsuccessful'),
    event('Away', 'Pass', 'Successful'),
  ];
  const s = computeStats(events, NAMES);
  assert.equal(s.teams[1].passes, 3);
  assert.equal(s.teams[1].passesCompleted, 2);
  assert.equal(Math.round(s.teams[1].passAccuracy), 67);
  assert.equal(s.teams[2].passes, 1);
});

test('a cross counts as both a cross and a pass', () => {
  reset();
  const s = computeStats([event('Home', 'Cross', 'Successful')], NAMES);
  assert.equal(s.teams[1].crosses, 1);
  assert.equal(s.teams[1].passes, 1);
  assert.equal(s.teams[1].passesCompleted, 1);
});

test('a shot assist is a completed pass with no unsuccessful variant', () => {
  reset();
  const s = computeStats([event('Home', 'Shot Assist', 'Successful')], NAMES);
  assert.equal(s.teams[1].shotAssists, 1);
  assert.equal(s.teams[1].passes, 1);
  assert.equal(s.teams[1].passesCompleted, 1);
});

test('shots on target include goals and saves but not misses', () => {
  reset();
  const events = [
    event('Home', 'Shot', 'Goal'),
    event('Home', 'Shot', 'Saved'),
    event('Home', 'Shot', 'Off Target'),
    event('Home', 'Shot', 'Blocked'),
  ];
  const s = computeStats(events, NAMES);
  assert.equal(s.teams[1].shots, 4);
  assert.equal(s.teams[1].shotsOnTarget, 2);
  assert.equal(s.teams[1].goals, 1);
});

test('possession is the share of on-ball events and sums to 100', () => {
  reset();
  const events = [
    event('Home', 'Pass', 'Successful'),
    event('Home', 'Pass', 'Successful'),
    event('Home', 'Shot', 'Saved'),
    event('Away', 'Pass', 'Successful'),
    // Defensive actions are not on-ball possession and must not count.
    event('Away', 'Tackle', 'Successful'),
    event('Away', 'Interception'),
  ];
  const s = computeStats(events, NAMES);
  assert.equal(Math.round(s.teams[1].possession), 75);
  assert.equal(Math.round(s.teams[2].possession), 25);
  assert.equal(Math.round(s.teams[1].possession + s.teams[2].possession), 100);
});

test('events for an unknown team are ignored rather than miscounted', () => {
  reset();
  const s = computeStats([event('Someone Else', 'Pass', 'Successful')], NAMES);
  assert.equal(s.teams[1].passes, 0);
  assert.equal(s.teams[2].passes, 0);
  assert.equal(s.players.length, 0);
});

test('players are listed per team and ranked by involvement', () => {
  reset();
  const events = [
    event('Home', 'Pass', 'Successful', 'Ali'),
    event('Home', 'Pass', 'Successful', 'Ali'),
    event('Home', 'Shot', 'Goal', 'Omar'),
    event('Away', 'Tackle', 'Successful', 'Ali'),   // same name, other team
  ];
  const s = computeStats(events, NAMES);
  assert.equal(s.players[0].name, 'Ali');
  assert.equal(s.players[0].team, 1);
  assert.equal(s.players[0].events, 2);
  // A shared name across teams must stay two separate players.
  const alis = s.players.filter((p) => p.name === 'Ali');
  assert.equal(alis.length, 2);
  assert.deepEqual(alis.map((p) => p.team).sort(), [1, 2]);
  const omar = s.players.find((p) => p.name === 'Omar');
  assert.equal(omar.goals, 1);
});

test('defensive actions are counted separately from possession events', () => {
  reset();
  const events = [
    event('Away', 'Tackle', 'Successful'),
    event('Away', 'Interception'),
    event('Away', 'Block'),
    event('Away', 'Clearance'),
  ];
  const s = computeStats(events, NAMES);
  const t = s.teams[2];
  assert.equal(t.tackles, 1);
  assert.equal(t.interceptions, 1);
  assert.equal(t.blocks, 1);
  assert.equal(t.clearances, 1);
  assert.equal(t.passes, 0);
});
