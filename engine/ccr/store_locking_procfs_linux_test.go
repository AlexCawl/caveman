//go:build linux && !js

package ccr

import (
	"os"
	"path/filepath"
	"testing"
)

// The claim under test is not "chmod works" but "chmod without an ordinary
// descriptor keeps SQLite's POSIX locks": a chmod through the pinned
// /proc/self/fd link performs no open and no close of the database inode, so a
// short-lived consumer cannot decide it is the last connection and unlink the
// live writer's WAL.
func TestProcfsChmodFallbackPreservesLiveWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ccr.db")
	writer, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()

	// The parent is what matters: the consumer subprocess does not inherit the
	// seam, which mirrors the real mixed fleet (an old proxy, a new CLI).
	request := lockingConsumerRequest{Path: path}
	for round := range 3 {
		data := []byte("procfs fallback round " + string(rune('0'+round)) + ": exact bytes\x00\xff\r\n")
		handle, err := writer.Put(Recovery{ContentType: "text", Compressor: "text", Original: data})
		if err != nil {
			t.Fatalf("round %d: write after consumer closed: %v", round, err)
		}
		object, err := writer.PutObject(Object{Type: ObjectCommandResult, SessionID: "procfs-regression", Data: data})
		if err != nil {
			t.Fatalf("round %d: write object after consumer closed: %v", round, err)
		}
		request.Recoveries = append(request.Recoveries, lockingRecovery{Handle: handle, Object: object, Data: data})
		runLockingConsumer(t, request)
	}
}

// The fallback must still do the job it exists for. A lock-preserving chmod
// that silently fails to tighten the mode would leave a database of prompts,
// credentials and tool results at the umask default.
func TestProcfsChmodFallbackTightensMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ccr.db")
	if err := os.WriteFile(path, []byte("not yet secured"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := PrepareSQLitePath(path); err != nil {
		t.Fatalf("prepare via procfs fallback: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("procfs fallback left mode %v, want -rw-------", perm)
	}
}

// A symlink must not be followed on the fallback path either. chmodSQLiteFile
// is only reached after the caller's Lstat rejects a symlink, so this pins the
// caller's guard rather than the fallback's own O_NOFOLLOW.
func TestProcfsChmodFallbackRefusesSymlink(t *testing.T) {
	if os.Getenv("ANDROID_ROOT") != "" {
		t.Skip("Android O_PATH symlink semantics differ; covered on Linux")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "victim")
	if err := os.WriteFile(target, []byte("someone else's file"), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "ccr.db")
	if err := os.Symlink(target, path); err != nil {
		t.Fatal(err)
	}
	if err := PrepareSQLitePath(path); err == nil {
		t.Fatal("PrepareSQLitePath followed a symlink; want refusal")
	}
	info, err := os.Lstat(target)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o644 {
		t.Fatalf("symlink target was chmodded to %v; want it untouched at -rw-r--r--", perm)
	}
}
