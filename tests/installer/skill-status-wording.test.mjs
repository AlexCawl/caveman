// `/caveman status` on a host with no mode hook (Copilot CLI, Cursor, ...) must
// answer from the conversation with an honest marker, never a bare `unknown`
// that reads as a broken install (#1185).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (relative) => readFileSync(new URL(`../../${relative}`, import.meta.url), "utf8");

for (const skill of ["caveman", "ultracave", "megacave", "caveman-help"]) {
  test(`${skill} SKILL.md: hook-less status is marked, not unknown`, () => {
    const body = read(`skills/${skill}/SKILL.md`);
    assert.ok(body.includes("(not tracked by this host)"), "names the no-hook marker");
    assert.doesNotMatch(body, /Caveman mode: unknown|report `unknown`/);
  });
}
