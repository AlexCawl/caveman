import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash, generateKeyPairSync } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  releaseArtifactName,
  releaseArtifactNames,
} from "../../scripts/build-release-binaries.mjs";
import {
  assertManifestNamesRelease,
  checksumSignatureBundle,
  verifyChecksumSignatureBundle,
} from "../../scripts/sign-binary-checksums.mjs";

test("release matrix contains six binaries for six OS/architecture targets", () => {
  const names = releaseArtifactNames();
  assert.equal(names.length, 36);
  assert.equal(new Set(names).size, 36);
  assert.ok(names.includes("caveman-proxy_win32_amd64"));
  assert.ok(names.includes("caveman-shrink_win32_arm64"));
  assert.equal(releaseArtifactName("cavemem", "windows", "amd64"), "cavemem_win32_amd64");
});

// Released binaries reported version "dev": the build passed no -ldflags, so
// `var version` in main.go kept its default. A stub `go` records each build.
test("release binaries are stamped with the pinned release tag", { skip: process.platform === "win32" }, () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
  const dir = mkdtempSync(join(tmpdir(), "release-stamp-"));
  try {
    const log = join(dir, "go-args.log");
    writeFileSync(join(dir, "go"), `#!/bin/sh
printf '%s\\n' "$*" >> "${log}"
while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then shift; : > "$1"; fi; shift; done
`, { mode: 0o755 });
    const result = spawnSync(process.execPath, ["scripts/build-release-binaries.mjs", "--target", "linux/amd64", "--out", join(dir, "out")], {
      cwd: root,
      env: { ...process.env, PATH: `${dir}${delimiter}${process.env.PATH}` },
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    const tag = readFileSync(join(root, "packages", "cli", "BINARY_RELEASE"), "utf8").trim();
    const builds = readFileSync(log, "utf8").trim().split("\n");
    assert.equal(builds.length, 6);
    for (const args of builds) assert.ok(args.includes(`-ldflags -X main.version=${tag} `), args);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("checksum signer emits bundle accepted by pinned-key verifier contract", () => {
  const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  const privatePEM = privateKey.export({ type: "pkcs8", format: "pem" });
  const publicPEM = publicKey.export({ type: "spki", format: "pem" });
  const checksums = Buffer.from(`${"a".repeat(64)}  caveman-proxy_win32_amd64\n`);
  const bundle = checksumSignatureBundle(checksums, privatePEM);
  assert.equal(verifyChecksumSignatureBundle(checksums, bundle, publicPEM), true);
  assert.equal(verifyChecksumSignatureBundle(Buffer.from("changed"), bundle, publicPEM), false);
});

test("checksum signer refuses a manifest that does not name its release", () => {
  const entry = (tag) => `${createHash("sha256").update(`${tag}\n`).digest("hex")}  RELEASE\n`;
  const binaries = `${"a".repeat(64)}  caveman-proxy_win32_amd64\n`;
  assertManifestNamesRelease(binaries + entry("bin-v2.0.0"), "bin-v2.0.0");
  assert.throws(() => assertManifestNamesRelease(binaries, "bin-v2.0.0"), /exactly one RELEASE entry/);
  assert.throws(() => assertManifestNamesRelease(binaries + entry("bin-v1.9.9"), "bin-v2.0.0"), /exactly one RELEASE entry/);
  assert.throws(() => assertManifestNamesRelease(binaries + entry("bin-v2.0.0").repeat(2), "bin-v2.0.0"), /exactly one RELEASE entry/);
});
