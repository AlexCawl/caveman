package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"sync"

	"github.com/JuliusBrussee/caveman/proxy/providers"
)

// A raw pin records that the provider rejected a request in transformed form
// and accepted its original bytes. What the provider cached is then the raw
// request, so every later request that extends it must go out raw as well —
// re-substituting earlier replacements would bust that entry on the next turn.
//
// The pin follows the conversation, not the session: it matches requests whose
// client bytes extend the pinned request's cached prefix, so a subagent or
// side request on the same session keeps compressing. Pins live in the
// PrefixCache (they survive restarts) and in memory (a failing store cannot
// unpin a conversation this process pinned).
const (
	rawPinScope  = "rawpin"
	rawPinHandle = "rawpin"
	// rawPinSlots bounds the pins kept per conversation identity.
	rawPinSlots = 64
	// rawPinIdentities bounds the in-memory copy; the store keeps the rest.
	rawPinIdentities = 4096
)

type rawPin struct {
	n      uint32
	digest [32]byte
}

func (p rawPin) encode() []byte {
	return append(binary.BigEndian.AppendUint32(nil, p.n), p.digest[:]...)
}

func decodeRawPin(b []byte) (rawPin, bool) {
	if len(b) != 4+32 {
		return rawPin{}, false
	}
	p := rawPin{n: binary.BigEndian.Uint32(b)}
	copy(p.digest[:], b[4:])
	return p, true
}

func (p rawPin) matches(components [][]byte) bool {
	return int(p.n) <= len(components) && digestComponents(components[:p.n]) == p.digest
}

type rawPins struct {
	mu   sync.Mutex
	pins map[[32]byte][]rawPin
}

func (r *rawPins) add(identity [32]byte, p rawPin) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pins == nil || len(r.pins) >= rawPinIdentities {
		r.pins = map[[32]byte][]rawPin{}
	}
	for _, have := range r.pins[identity] {
		if have == p {
			return
		}
	}
	r.pins[identity] = append(r.pins[identity], p)
}

// longest returns the length of the longest pin components extend, 0 if none.
func (r *rawPins) longest(identity [32]byte, components [][]byte) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, p := range r.pins[identity] {
		if p.matches(components) {
			n = max(n, int(p.n))
		}
	}
	return n
}

// pinRaw records that the provider accepted body, the client's original bytes,
// after rejecting transformed. It pins only when the two differ inside what
// body caches; otherwise the next turn's substitutions already extend it.
func (s *Server) pinRaw(adapter providers.Adapter, meta providers.RequestMetadata, body, transformed []byte) {
	if s.prefixCache == nil {
		return // nothing was ever substituted, so raw already extends raw
	}
	components, cached, ok := cachedPrefix(adapter, meta, body)
	if !ok || cached == 0 {
		return
	}
	if sent, _, ok := cachedPrefix(adapter, meta, transformed); ok && len(sent) >= cached && digestComponents(sent[:cached]) == digestComponents(components[:cached]) {
		return
	}
	identity := sha256.Sum256(components[0])
	pin := rawPin{n: uint32(cached), digest: digestComponents(components[:cached])}
	s.rawPins.add(identity, pin)
	value := pin.encode()
	for slot := 0; slot < rawPinSlots; slot++ {
		stored, err := s.prefixCache.RememberReplacement(rawPinScope, rawPinKey(identity, slot), value, rawPinHandle)
		if err != nil || bytes.Equal(stored, value) {
			return
		}
	}
}

// rawPinned reports whether body extends a request pinned raw.
func (s *Server) rawPinned(adapter providers.Adapter, meta providers.RequestMetadata, body []byte) bool {
	components, _, ok := cachedPrefix(adapter, meta, body)
	return ok && s.rawPinCoverage(components) > 0
}

// rawPinCoverage returns how many leading components the longest raw pin a
// request extends covers, 0 when it extends none. Pins this process has not
// seen yet (a restart) are loaded from the store on first use.
func (s *Server) rawPinCoverage(components [][]byte) int {
	if s.prefixCache == nil || len(components) == 0 {
		return 0
	}
	identity := sha256.Sum256(components[0])
	if n := s.rawPins.longest(identity, components); n > 0 {
		return n
	}
	n := 0
	for slot := 0; slot < rawPinSlots; slot++ {
		value, _, hit := s.prefixCache.LookupReplacement(rawPinScope, rawPinKey(identity, slot))
		if !hit {
			break
		}
		if p, ok := decodeRawPin(value); ok && p.matches(components) {
			s.rawPins.add(identity, p)
			n = max(n, int(p.n))
		}
	}
	return n
}

func rawPinKey(identity [32]byte, slot int) []byte {
	return append(identity[:len(identity):len(identity)], byte(slot))
}

// cachedPrefix splits a request into the components a provider prompt cache
// keys on, plus how many of them this request caches. Adapters that know their
// cache markers say so (Anthropic); the rest cache the whole prompt
// implicitly, so every component counts.
func cachedPrefix(adapter providers.Adapter, meta providers.RequestMetadata, body []byte) ([][]byte, int, bool) {
	if inspector, ok := adapter.(CachedPrefixInspector); ok {
		return inspector.CachedPrefixComponents(body, meta)
	}
	return wholePromptComponents(body)
}

// wholePromptComponents is the provider-agnostic split: everything outside the
// conversation array as one component, then each conversation item.
func wholePromptComponents(body []byte) ([][]byte, int, bool) {
	root, ok := gatewayRootObjectSpan(body)
	if !ok {
		return nil, 0, false
	}
	for _, field := range []string{"messages", "input", "contents"} {
		list, found := gatewayFindObjectField(body, root, field)
		if !found || list.start >= list.end || body[list.start] != '[' {
			continue
		}
		items, ok := gatewayArrayElements(body, list)
		if !ok {
			return nil, 0, false
		}
		rest := append(append([]byte(nil), body[:list.start]...), body[list.end:]...)
		out := [][]byte{frameComponent(rest)}
		for _, item := range items {
			out = append(out, frameComponent(body[item.start:item.end]))
		}
		return out, len(out), true
	}
	return [][]byte{frameComponent(body)}, 1, true
}

func frameComponent(b []byte) []byte {
	return append(binary.BigEndian.AppendUint64(nil, uint64(len(b))), b...)
}

// digestComponents hashes length-framed components, so no two different
// component lists share a digest.
func digestComponents(components [][]byte) [32]byte {
	h := sha256.New()
	for _, c := range components {
		_, _ = h.Write(c)
	}
	var out [32]byte
	h.Sum(out[:0])
	return out
}
