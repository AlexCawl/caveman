// Codex always-on (#573): `--only codex` installs an owned SessionStart hook
// payload under $CODEX_HOME/caveman/ and merges one entry into
// $CODEX_HOME/hooks.json. Uninstall removes only that entry and the payload.
// Every run is sandboxed: HOME, XDG, CODEX_HOME and PATH point at a temp dir,
// and a recording `npx` stub stands in for the upstream skills CLI.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { nodeStub } from '../../packages/cli/tests/harness/stub-bin.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const INSTALLER = path.join(ROOT, 'bin', 'install.js');
const HOOK_REL = 'caveman/hooks/codex-sessionstart.js';

function sandbox(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'caveman codex hooks '));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const bin = path.join(dir, 'bin');
  nodeStub(bin, 'npx', 'process.exit(0);');
  const codexHome = path.join(dir, 'codex home');
  fs.mkdirSync(codexHome);
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    key.toLowerCase() !== 'path' && !key.startsWith('CAVEMAN_') && key !== 'CLAUDE_PLUGIN_ROOT'));
  Object.assign(env, {
    HOME: dir, USERPROFILE: dir, APPDATA: dir, LOCALAPPDATA: dir, XDG_CONFIG_HOME: dir,
    CODEX_HOME: codexHome, NO_COLOR: '1',
    PATH: process.platform === 'win32'
      ? `${bin};${path.dirname(process.execPath)};${process.env.SystemRoot || 'C:\\Windows'}\\System32`
      : `${bin}:${path.dirname(process.execPath)}:/usr/bin:/bin`,
  });
  const run = (...args) => spawnSync(process.execPath, [
    INSTALLER, ...args, '--config-dir', path.join(dir, 'claude'), '--non-interactive', '--no-mcp-shrink',
  ], { encoding: 'utf8', cwd: dir, env });
  const hooksPath = path.join(codexHome, 'hooks.json');
  const readHooks = () => JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  return { dir, env, codexHome, hooksPath, readHooks, run };
}

const isOurs = (entry) => (entry.hooks || []).some((h) => String(h.command).replace(/\\/g, '/').includes(HOOK_REL));

test('codex install merges exactly one SessionStart hook, even when run twice', (t) => {
  const { codexHome, readHooks, run } = sandbox(t);
  for (let i = 0; i < 2; i++) {
    const r = run('--only', 'codex');
    assert.equal(r.status, 0, r.stdout + r.stderr);
  }
  const ours = readHooks().hooks.SessionStart.filter(isOurs);
  assert.equal(ours.length, 1, JSON.stringify(readHooks()));
  assert.equal(ours[0].hooks[0].type, 'command');
  assert.match(ours[0].matcher, /startup/);
  for (const rel of [HOOK_REL, 'caveman/hooks/caveman-config.js', 'caveman/hooks/package.json',
    'caveman/skills/caveman/SKILL.md', 'caveman/skills/ultracave/SKILL.md', 'caveman/skills/megacave/SKILL.md']) {
    assert.ok(fs.existsSync(path.join(codexHome, ...rel.split('/'))), `${rel} missing`);
  }
});

test('foreign and caveman CLI native-hook entries survive install and uninstall', (t) => {
  const { hooksPath, readHooks, codexHome, run } = sandbox(t);
  const foreign = { matcher: 'startup', hooks: [{ type: 'command', command: 'echo user hook' }] };
  const native = { hooks: [{ type: 'command', command: 'caveman native-hook codex' }] };
  const before = { theme: 'kept', hooks: { SessionStart: [foreign, native], PreToolUse: [native] } };
  fs.writeFileSync(hooksPath, JSON.stringify(before, null, 2) + '\n');

  assert.equal(run('--only', 'codex').status, 0);
  const merged = readHooks();
  assert.deepEqual(merged.hooks.SessionStart.slice(0, 2), [foreign, native]);
  assert.equal(merged.hooks.SessionStart.filter(isOurs).length, 1);
  assert.deepEqual(merged.hooks.PreToolUse, [native]);

  const r = run('--uninstall');
  assert.equal(r.status, 0, r.stdout + r.stderr);
  assert.deepEqual(readHooks(), before);
  assert.equal(fs.existsSync(path.join(codexHome, 'caveman')), false, 'owned payload must be removed');
});

test('uninstall deletes a hooks.json that only held the caveman entry', (t) => {
  const { hooksPath, codexHome, run } = sandbox(t);
  assert.equal(run('--only', 'codex').status, 0);
  assert.ok(fs.existsSync(hooksPath));
  assert.equal(run('--uninstall').status, 0);
  assert.equal(fs.existsSync(hooksPath), false);
  assert.deepEqual(fs.readdirSync(codexHome), []);
});

test('installed hook follows the configured default mode', (t) => {
  const { dir, env, codexHome, run } = sandbox(t);
  assert.equal(run('--only', 'codex').status, 0);
  const hook = path.join(codexHome, ...HOOK_REL.split('/'));
  const exec = (mode) => spawnSync(process.execPath, [hook], {
    encoding: 'utf8', cwd: dir, env: { ...env, CAVEMAN_DEFAULT_MODE: mode },
  });
  const off = exec('off');
  assert.equal(off.status, 0);
  assert.equal(off.stdout, '');
  const ultra = exec('ultracave');
  assert.equal(ultra.status, 0);
  const body = fs.readFileSync(path.join(ROOT, 'skills', 'ultracave', 'SKILL.md'), 'utf8').replace(/^---[\s\S]*?---\s*/, '');
  assert.ok(ultra.stdout.startsWith('CAVEMAN MODE ACTIVE — mode: ultracave\n\n'), ultra.stdout.slice(0, 200));
  assert.ok(ultra.stdout.endsWith(body), 'full ultracave ruleset expected');
});

test('--no-hooks and --dry-run leave CODEX_HOME untouched', (t) => {
  const { codexHome, run } = sandbox(t);
  assert.equal(run('--only', 'codex', '--no-hooks').status, 0);
  const dry = run('--only', 'codex', '--dry-run');
  assert.equal(dry.status, 0);
  assert.match(dry.stdout, /would merge .*hooks\.json/);
  assert.deepEqual(fs.readdirSync(codexHome), []);
});

test('unparseable hooks.json is left alone and nothing is installed', (t) => {
  const { hooksPath, codexHome, run } = sandbox(t);
  fs.writeFileSync(hooksPath, '{ not json');
  const r = run('--only', 'codex');
  assert.match(r.stdout + r.stderr, /codex-hooks/);
  assert.equal(fs.readFileSync(hooksPath, 'utf8'), '{ not json');
  assert.deepEqual(fs.readdirSync(codexHome), ['hooks.json']);
});
