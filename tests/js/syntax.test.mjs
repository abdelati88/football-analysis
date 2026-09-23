/* Every browser module must actually parse.
 *
 * This exists because a syntax error shipped. `label ?? value || '—'` is a
 * SyntaxError by the letter of the specification — mixing `??` with `||`
 * without parentheses is forbidden precisely because the precedence reads
 * ambiguously — and one of them in a template literal stopped the whole file
 * from being parsed. Not one line of the interface ran: no pitch markings, no
 * menus, no event table. The page looked comprehensively broken for a reason
 * that had nothing to do with anything visible.
 *
 * It shipped because the check used to wave it through. `node --check` on a
 * file containing `import` statements exits zero without parsing it as a
 * module — it reports success on a file it has not read. Reading from stdin
 * with `--input-type=module` does parse it, and rejects this file in the same
 * place Chrome does.
 *
 * So the check runs here, over every module, on every test run.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const JS_DIR = join(
  dirname(fileURLToPath(import.meta.url)), '..', '..', 'match_tag', 'static', 'js',
);

const modules = readdirSync(JS_DIR).filter((f) => f.endsWith('.js'));

test('there are modules to check', () => {
  assert.ok(modules.length >= 5, `found only ${modules.length} modules in ${JS_DIR}`);
});

for (const name of modules) {
  test(`${name} parses as an ES module`, () => {
    const source = readFileSync(join(JS_DIR, name));
    try {
      execFileSync(process.execPath, ['--input-type=module', '--check'], {
        input: source,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch (error) {
      const detail = String(error.stderr || error.message)
        .split('\n').slice(0, 6).join('\n');
      assert.fail(`${name} does not parse:\n${detail}`);
    }
  });
}

test('the check would catch a nullish/or mix, which node --check does not', () => {
  const offending = 'export const f = (a, b) => a ?? b || "x";\n';
  assert.throws(() => {
    execFileSync(process.execPath, ['--input-type=module', '--check'], {
      input: offending,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  }, 'the module check must reject `a ?? b || c`');
});
