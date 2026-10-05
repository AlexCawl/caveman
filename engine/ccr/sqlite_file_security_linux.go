//go:build linux && !js

package ccr

import (
	"fmt"
	"os"
	"strconv"

	"golang.org/x/sys/unix"
)

func chmodSQLiteFile(path string, info os.FileInfo) error {
	// Closing an ordinary descriptor for the database or -shm drops ALL POSIX
	// locks held by SQLite in this process. O_PATH pins the inode without opening
	// it for I/O; closing this metadata-only descriptor does not release locks.
	fd, err := unix.Open(path, unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	file := os.NewFile(uintptr(fd), path)
	defer file.Close()
	opened, err := file.Stat()
	if err != nil {
		return err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return fmt.Errorf("file changed while opening")
	}
	// Use procfs rather than fchmodat2: Android/Termux blocks that syscall with
	// SIGSYS, and procfs works on both old and new Linux kernels. This is the pinned
	// descriptor's kernel-controlled link, NOT the swappable database pathname.
	// chmod performs no open/close, so SQLite's locks remain intact. Fail closed
	// if procfs is unavailable; never fall back to opening the database for I/O.
	return unix.Chmod("/proc/self/fd/"+strconv.Itoa(fd), 0o600)
}
