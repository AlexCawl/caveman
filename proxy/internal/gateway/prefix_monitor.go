package gateway

import (
	"bytes"
	"crypto/sha256"
	"sync"

	"github.com/JuliusBrussee/caveman/proxy/providers"
)

// prefixMonitor is the runtime tripwire for the cache invariant that
// cache_prefix_invariant_test.go proves: when a request's client bytes repeat
// a prefix an earlier accepted request of the session cached, the bytes
// forwarded over that prefix must repeat what was forwarded then. Each anchor
// keeps both views of an accepted request's cached prefix — what the client
// sent and what went upstream — and a request that does not extend what the
// session cached is classified by who changed the bytes:
//
//   - client: the client's own bytes changed (an edit, a rewind, a compaction).
//     Flagged on the row and logged at DEBUG; nearly every old warning was one.
//   - caveman: the client's bytes are unchanged and the forwarded bytes are not.
//     That is a caveman bug: logged at ERROR, counted by stats and status.
//   - raw_retry: the provider rejected the transformed request and accepted the
//     original bytes, the invariant's one exception. Requests after it are held
//     to the raw request (see raw_pin.go).
//
// It is OBSERVE-ONLY: it never blocks or modifies traffic. A session is not one
// conversation — a header-less Claude Code process correlates its main thread,
// subagents and side requests to one session (#1094) — so it keeps several
// anchors per session and compares a request only with those it extends.
type prefixMonitor struct {
	mu sync.Mutex
	// last holds the anchors of each session, least recently written first.
	last map[string][]prefixAnchor
	// order tracks session insertion order for oldest-first eviction so a
	// long-running proxy's per-session state stays bounded (mirrors cacheguard).
	order []string
	cap   int
}

// prefixAnchor is one accepted request's cached prefix as component digests.
type prefixAnchor struct {
	client, forwarded [][32]byte
	// raw: the provider holds this prefix as the client's original bytes after
	// a raw retry, so a request extending it is held to it, not to compressed
	// requests it covers.
	raw bool
}

const (
	bustCauseClient   = "client"
	bustCauseCaveman  = "caveman"
	bustCauseRawRetry = "raw_retry"
)

// defaultPrefixMonitorCap bounds retained sessions. An evicted session's next
// request is treated as a fresh first observation (no prior → no bust), which is
// the safe direction: eviction can only drop a warning, never fabricate one.
const defaultPrefixMonitorCap = 8192

// maxAnchorsPerSession bounds the anchors remembered per session; the least
// recently written one is dropped first. Dropping can only turn a later drift
// into a fresh first observation, never fabricate a bust.
// ponytail: 16 anchors × 8192 sessions × up to 1024 digest pairs is gigabytes
// on a saturated hosted gateway; lower the session cap if that deployment ever
// exists.
const maxAnchorsPerSession = 16

// conversationComponents is how many leading components identify a conversation
// rather than a turn of one: system, tools and the first message (see
// anthropic.CachedPrefixComponents). A request that already differs inside those
// is another conversation sharing the session — a subagent, a side request — so
// it is never a bust.
const conversationComponents = 3

func newPrefixMonitor() *prefixMonitor {
	return &prefixMonitor{last: map[string][]prefixAnchor{}, cap: defaultPrefixMonitorCap}
}

// observe checks one accepted request against the anchors of its session and
// records it as an anchor. client and forwarded are its cached-prefix
// components as the client sent them and as they went upstream (see
// cachedPrefix), cached how many of them it caches, and rawRetry whether the
// provider accepted it only as the client's original bytes.
//
// Every anchor whose cached client prefix the request repeats must find its
// forwarded prefix repeated too; the first component that is not is a caveman
// bust (raw_retry for the retry itself), unless a raw request that covers the
// anchor took it over. Otherwise, a request whose closest anchor shares the
// conversation identity but is not repeated is a client bust at the first
// differing component. "" and -1 mean the request extends what was cached, or
// there was nothing to compare: no session, or no comparable components.
func (m *prefixMonitor) observe(session string, client, forwarded [][]byte, cached int, rawRetry bool) (cause string, index int) {
	if m == nil || session == "" || len(client) == 0 || len(forwarded) != len(client) || cached < 0 || cached > len(client) {
		return "", -1
	}
	cur := prefixAnchor{client: digestEach(client), forwarded: digestEach(forwarded), raw: rawRetry}
	m.mu.Lock()
	defer m.mu.Unlock()
	anchors := m.last[session]
	cause, index = "", -1
	closest, closestRepeated := -1, false
	var kept []prefixAnchor
	for i, a := range anchors {
		n := commonPrefixLen(a.client, cur.client)
		repeats := n == len(a.client)
		if n > closest || (n == closest && repeats && !closestRepeated) {
			closest, closestRepeated = n, repeats
		}
		if !repeats {
			kept = append(kept, a)
			continue
		}
		preserved := commonPrefixLen(a.forwarded, cur.forwarded)
		if preserved < len(a.forwarded) && !rebased(anchors, i, cur) && (index < 0 || preserved < index) {
			cause, index = bustCauseCaveman, preserved
			if rawRetry {
				cause = bustCauseRawRetry
			}
		}
		// The request takes over an anchor it repeats when it caches at least as
		// much — it is the provider's latest entry for those bytes — and always
		// when it is a raw retry, whose raw bytes the provider now holds there.
		if rawRetry || cached >= len(a.client) {
			cur.raw = cur.raw || (a.raw && preserved == len(a.forwarded))
			continue
		}
		kept = append(kept, a)
	}
	if cause == "" && !closestRepeated && closest >= conversationComponents {
		cause, index = bustCauseClient, closest
	}
	if cached > 0 {
		cur.client, cur.forwarded = cur.client[:cached], cur.forwarded[:cached]
		kept = append(kept, cur)
	}
	if len(kept) > maxAnchorsPerSession {
		kept = kept[len(kept)-maxAnchorsPerSession:]
	}
	m.put(session, kept)
	return cause, index
}

// observeCachedPrefix runs the tripwire on one accepted request — body as the
// client sent it, accepted as it went upstream — logs what it found and returns
// the cause. A provider caches per model, so the model is part of the key.
func (s *Server) observeCachedPrefix(adapter providers.Adapter, meta providers.RequestMetadata, body, accepted []byte, rawRetry bool, sessionID, requestID string) string {
	if sessionID == "" {
		return ""
	}
	client, cached, ok := cachedPrefix(adapter, meta, body)
	if !ok {
		return ""
	}
	forwarded := client
	if !bytes.Equal(accepted, body) {
		if forwarded, _, ok = cachedPrefix(adapter, meta, accepted); !ok {
			return ""
		}
	}
	cause, index := s.prefixMonitor.observe(sessionID+"\x00"+meta.Model, client, forwarded, cached, rawRetry)
	if s.logger != nil {
		attrs := []any{"request_id", requestID, "session_id", sessionID, "index", index}
		switch cause {
		case bustCauseCaveman:
			s.logger.Error("caveman changed bytes the provider already cached", attrs...)
		case bustCauseRawRetry:
			s.logger.Warn("provider accepted only the original bytes; its cached prefix restarts here", attrs...)
		case bustCauseClient:
			s.logger.Debug("client changed bytes the provider already cached", attrs...)
		}
	}
	return cause
}

// rebased reports whether a raw anchor other than anchors[i] covers anchor i's
// cached client prefix and cur repeats that raw anchor: cur is then held to the
// raw request the provider re-cached those bytes as, not to anchor i.
func rebased(anchors []prefixAnchor, i int, cur prefixAnchor) bool {
	for j, x := range anchors {
		if j != i && x.raw &&
			commonPrefixLen(x.client, anchors[i].client) == len(anchors[i].client) &&
			commonPrefixLen(x.client, cur.client) == len(x.client) {
			return true
		}
	}
	return false
}

func digestEach(components [][]byte) [][32]byte {
	out := make([][32]byte, len(components))
	for i, c := range components {
		out[i] = sha256.Sum256(c)
	}
	return out
}

func commonPrefixLen[T comparable](a, b []T) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// put records a session's anchors, tracking insertion order and evicting the
// oldest session once the cap is exceeded. Callers must hold m.mu.
func (m *prefixMonitor) put(session string, anchors []prefixAnchor) {
	if _, exists := m.last[session]; !exists {
		m.order = append(m.order, session)
		for len(m.order) > m.cap {
			oldest := m.order[0]
			m.order = m.order[1:]
			delete(m.last, oldest)
		}
	}
	m.last[session] = anchors
}
