package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/JuliusBrussee/caveman/engine/compressors"
	"github.com/JuliusBrussee/caveman/proxy/providers"
	"github.com/JuliusBrussee/caveman/proxy/providers/anthropic"
)

// The cache-prefix invariant (#1105): caveman must never change bytes the
// provider already cached.
//
// For accepted requests P then R, let k be P's last cache_control-marked
// message. If R's client bytes over system, tools and messages[0..k]
// (cache_control stripped) equal P's, R's FORWARDED bytes over that range must
// equal P's forwarded bytes. The one exception is a request the provider
// rejected in transformed form and accepted through the raw retry: that request
// may differ from P, and every later request that extends it is then held to
// the raw request instead of to P.
//
// The scenarios below drive Claude Code shaped traffic through Server.Handler
// and check every pair of accepted requests against that rule.

// exchange is one client request as the provider accepted it.
type exchange struct {
	client    []byte
	forwarded []byte
	// rawRetry: the transformed attempt was rejected and the raw retry accepted.
	rawRetry bool
	// group > 0 marks requests that were in flight together. The provider may
	// have processed either one first, so the rule is checked both ways.
	group int
}

type prefixView struct {
	client, forwarded [][]byte
	cached            int
}

func cachedPrefixView(t testing.TB, x exchange) prefixView {
	t.Helper()
	client, cached, ok := anthropic.CachedPrefixComponents(x.client)
	if !ok {
		t.Fatalf("client body has no cached prefix: %.300s", x.client)
	}
	forwarded, _, ok := anthropic.CachedPrefixComponents(x.forwarded)
	if !ok || len(forwarded) != len(client) {
		t.Fatalf("forwarded body does not mirror the client's components (ok=%v %d vs %d): %.300s", ok, len(forwarded), len(client), x.forwarded)
	}
	return prefixView{client: client, forwarded: forwarded, cached: cached}
}

// extendsRange reports whether components starts with prefix[:n].
func extendsRange(components, prefix [][]byte, n int) bool {
	if len(components) < n || len(prefix) < n {
		return false
	}
	for i := 0; i < n; i++ {
		if !bytes.Equal(components[i], prefix[i]) {
			return false
		}
	}
	return true
}

func assertCachedPrefixPreserved(t testing.TB, xs []exchange) {
	t.Helper()
	views := make([]prefixView, len(xs))
	for i, x := range xs {
		views[i] = cachedPrefixView(t, x)
	}
	for j := range xs {
		for i := range xs {
			concurrent := i != j && xs[i].group > 0 && xs[i].group == xs[j].group
			if i >= j && !concurrent {
				continue
			}
			p, r := views[i], views[j]
			if !extendsRange(r.client, p.client, p.cached) || rebasedByRawRetry(xs, views, i, j) {
				continue
			}
			for c := 0; c < p.cached; c++ {
				if !bytes.Equal(r.forwarded[c], p.forwarded[c]) {
					t.Errorf("request %d changed bytes request %d cached: component %d of %d differs although the client sent the same bytes\n  cached:  %.240q\n  re-sent: %.240q",
						j, i, c, p.cached, p.forwarded[c], r.forwarded[c])
					break
				}
			}
		}
	}
}

// rebasedByRawRetry reports whether a raw retry took over P's (i) range for R
// (j): the retry covers P's cached range, and R is that retry or extends the
// retry's whole cached range. R is then held to the raw retry instead — which
// the provider cached over at least P's range — whether P came before the
// retry or re-cached a shorter compressed prefix after it.
func rebasedByRawRetry(xs []exchange, views []prefixView, i, j int) bool {
	for k := 0; k <= j; k++ {
		if k == i || !xs[k].rawRetry || !extendsRange(views[k].client, views[i].client, views[i].cached) {
			continue
		}
		if k == j || extendsRange(views[j].client, views[k].client, views[k].cached) {
			return true
		}
	}
	return false
}

// --- the simulated client ---------------------------------------------------------

func jsonText(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func textBlock(s string) string { return `{"type":"text","text":` + jsonText(s) + `}` }

func toolUseBlock(id string) string {
	return `{"type":"tool_use","id":"` + id + `","name":"Read","input":{"path":"f"}}`
}

func toolResultBlock(id, content string) string {
	return `{"type":"tool_result","tool_use_id":"` + id + `","content":` + jsonText(content) + `}`
}

// filler is a block big enough to be a compression candidate.
func filler(label string) string { return strings.Repeat(label+" ", 600/(len(label)+1)+1) }

type ccMessage struct {
	role   string
	blocks []string
}

func (m ccMessage) json(marked bool) string {
	blocks := append([]string(nil), m.blocks...)
	if marked {
		last := blocks[len(blocks)-1]
		blocks[len(blocks)-1] = last[:len(last)-1] + `,"cache_control":{"type":"ephemeral"}}`
	}
	return `{"role":"` + m.role + `","content":[` + strings.Join(blocks, ",") + `]}`
}

// ccConversation builds Claude Code shaped requests: a marked system prompt and
// a marker on the newest message, which moves forward every turn.
type ccConversation struct {
	system   string
	model    string
	session  string
	messages []ccMessage
	tools    int
}

func newCCConversation(system, session string) *ccConversation {
	return &ccConversation{system: system, model: "claude-sonnet-4-6", session: session}
}

func (c *ccConversation) clone() *ccConversation {
	out := *c
	out.messages = append([]ccMessage(nil), c.messages...)
	return &out
}

func (c *ccConversation) reply() {
	if n := len(c.messages); n > 0 && c.messages[n-1].role == "user" {
		c.messages = append(c.messages, ccMessage{role: "assistant", blocks: []string{textBlock("answer " + strconv.Itoa(n))}})
	}
}

// user appends a user turn.
func (c *ccConversation) user(text string) *ccConversation {
	c.reply()
	c.messages = append(c.messages, ccMessage{role: "user", blocks: []string{textBlock(text)}})
	return c
}

// toolResult appends a tool call and its result.
func (c *ccConversation) toolResult(content string) *ccConversation {
	c.reply()
	c.tools++
	id := "toolu_" + strconv.Itoa(c.tools)
	c.messages = append(c.messages,
		ccMessage{role: "assistant", blocks: []string{toolUseBlock(id)}},
		ccMessage{role: "user", blocks: []string{toolResultBlock(id, content)}})
	return c
}

func (c *ccConversation) body() []byte {
	parts := make([]string, len(c.messages))
	for i, m := range c.messages {
		parts[i] = m.json(i == len(c.messages)-1)
	}
	return []byte(`{"model":"` + c.model + `","max_tokens":1024,` +
		`"system":[{"type":"text","text":` + jsonText(c.system) + `,"cache_control":{"type":"ephemeral"}}],` +
		`"tools":[{"name":"Read","description":"Read a file","input_schema":{"type":"object","title":"Read args"}}],` +
		`"messages":[` + strings.Join(parts, ",") + `]}`)
}

// --- the harness ------------------------------------------------------------------

// invariantCompressor compresses deterministically by content. With
// declineFirst it turns a block down the first time it sees it and compresses
// it afterwards, the way a query-aware compressor or a failed recovery write
// can; failStore makes the recovery write fail.
type invariantCompressor struct {
	mu           sync.Mutex
	nonce        string
	declineFirst bool
	failStore    bool
	seen         map[string]bool
}

func (c *invariantCompressor) CompressSegment(seg []byte) ([]byte, int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := contentHandle(seg)
	if c.declineFirst && !c.seen[key] {
		if c.seen == nil {
			c.seen = map[string]bool{}
		}
		c.seen[key] = true
		return nil, 0, 0
	}
	return []byte("CMP" + c.nonce + ":" + key), 100, 40
}

func (c *invariantCompressor) StoreOriginal(body []byte) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.failStore {
		return "", errTestPrefixCacheDown
	}
	return contentHandle(body), nil
}

func (c *invariantCompressor) StripToolSchema(tools []byte) ([]byte, bool) {
	return compressors.StripToolSchemaAnnotations(tools)
}

type invariantTagKey struct{}

type upstreamAttempt struct {
	body   []byte
	status int
}

// respondFunc decides one upstream attempt's status and headers.
type respondFunc func(attempt int, body []byte) (int, http.Header)

type invariantTransport struct {
	mu       sync.Mutex
	attempts map[int][]upstreamAttempt
	respond  map[int]respondFunc
}

func (t *invariantTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(r.Body)
	tag, _ := r.Context().Value(invariantTagKey{}).(int)
	t.mu.Lock()
	attempt := len(t.attempts[tag])
	status, header := http.StatusOK, http.Header{}
	if f := t.respond[tag]; f != nil {
		status, header = f(attempt, body)
		if header == nil {
			header = http.Header{}
		}
	}
	t.attempts[tag] = append(t.attempts[tag], upstreamAttempt{body: body, status: status})
	t.mu.Unlock()
	header.Set("Content-Type", "application/json")
	payload := subMessageRespBody
	if status >= 400 {
		payload = `{"type":"error","error":{"type":"rate_limit_error","message":"Error"}}`
	}
	return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: header, Body: io.NopCloser(strings.NewReader(payload)), Request: r}, nil
}

type invariantHarness struct {
	t     testing.TB
	comp  *invariantCompressor
	cache *testPrefixCache
	rt    *invariantTransport
	srv   *Server
	// strip turns on the tool-schema annotation strip.
	strip bool

	mu   sync.Mutex
	tag  int
	xs   []exchange
	sent int
}

func newInvariantHarness(t testing.TB) *invariantHarness {
	h := &invariantHarness{
		t:     t,
		comp:  &invariantCompressor{},
		cache: newTestPrefixCache(),
		rt:    &invariantTransport{attempts: map[int][]upstreamAttempt{}, respond: map[int]respondFunc{}},
	}
	h.restart("")
	return h
}

// restart replaces the proxy process: in-memory state is gone, the durable
// replacement cache survives, and the engine may now compress differently.
func (h *invariantHarness) restart(nonce string) {
	h.comp.mu.Lock()
	h.comp.nonce = nonce
	h.comp.mu.Unlock()
	strip := ""
	if h.strip {
		strip = toolSchemaStripMode
	}
	h.srv = New(Config{
		Adapters:        []providers.Adapter{anthropic.New("https://upstream.test")},
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Creds:           passthroughTestCreds{},
		Compressor:      h.comp,
		PrefixCache:     h.cache,
		HTTPClient:      &http.Client{Transport: h.rt},
		RecoveryViaMCP:  true,
		ToolSchemaStrip: strip,
	})
}

type sendOpts struct {
	respond respondFunc
	group   int
}

// send serves one client request and records it if the provider accepted it.
func (h *invariantHarness) send(c *ccConversation, o sendOpts) (int, []upstreamAttempt) {
	return h.sendBody(c.body(), c.session, o)
}

func (h *invariantHarness) sendBody(body []byte, session string, o sendOpts) (int, []upstreamAttempt) {
	h.mu.Lock()
	h.tag++
	tag := h.tag
	srv := h.srv
	h.mu.Unlock()
	if o.respond != nil {
		h.rt.mu.Lock()
		h.rt.respond[tag] = o.respond
		h.rt.mu.Unlock()
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	req = req.WithContext(context.WithValue(req.Context(), invariantTagKey{}, tag))
	for k, v := range subscriptionAgentHeaders {
		req.Header.Set(k, v)
	}
	if session != "" {
		req.Header.Set("x-cave-session", session)
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	h.rt.mu.Lock()
	attempts := h.rt.attempts[tag]
	h.rt.mu.Unlock()
	h.mu.Lock()
	defer h.mu.Unlock()
	h.sent++
	if rec.Code < 400 && len(attempts) > 0 && attempts[len(attempts)-1].status < 400 {
		h.xs = append(h.xs, exchange{
			client:    body,
			forwarded: attempts[len(attempts)-1].body,
			rawRetry:  len(attempts) > 1,
			group:     o.group,
		})
	}
	return rec.Code, attempts
}

func (h *invariantHarness) assert() {
	h.t.Helper()
	assertCachedPrefixPreserved(h.t, h.xs)
}

// compressedSomething guards against a scenario passing vacuously.
func (h *invariantHarness) compressedSomething() {
	h.t.Helper()
	for _, x := range h.xs {
		if bytes.Contains(x.forwarded, []byte("<<ccr:")) {
			return
		}
	}
	h.t.Fatal("scenario never compressed anything, so it proves nothing")
}

func rejectTransformed(status int, header http.Header) respondFunc {
	return func(attempt int, body []byte) (int, http.Header) {
		if attempt == 0 && bytes.Contains(body, []byte("<<ccr:")) {
			return status, header.Clone()
		}
		return http.StatusOK, nil
	}
}

func rateLimitResponse(attempt int, body []byte) (int, http.Header) {
	if attempt == 0 {
		return http.StatusTooManyRequests, http.Header{"Retry-After": {"7"}, "Anthropic-Ratelimit-Requests-Remaining": {"0"}}
	}
	return http.StatusOK, nil
}

// --- scenarios --------------------------------------------------------------------

func TestCachePrefixInvariant(t *testing.T) {
	const session = "sess-1105"
	for _, tc := range []struct {
		name string
		run  func(t *testing.T, h *invariantHarness)
	}{
		{"ten turns with a moving marker", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			for i := 1; i <= 10; i++ {
				h.send(main.user(filler("main turn "+strconv.Itoa(i))), sendOpts{})
			}
		}},
		{"main, subagents, side request and fork under one session", func(t *testing.T, h *invariantHarness) {
			interleaved(h, session)
		}},
		{"main, subagents, side request and fork uncorrelated", func(t *testing.T, h *invariantHarness) {
			interleaved(h, "")
		}},
		{"rewind and edit", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("u1")), sendOpts{})
			h.send(main.user(filler("u2")), sendOpts{})
			rewindPoint := main.clone()
			h.send(main.user(filler("u3")), sendOpts{})
			// Esc-Esc: back to after u1's answer, a different second prompt.
			rewound := rewindPoint
			rewound.messages = rewound.messages[:len(rewound.messages)-1]
			h.send(rewound.user(filler("u2 edited")), sendOpts{})
			h.send(rewound.user(filler("u4")), sendOpts{})
		}},
		{"client clears an old tool_result", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("read the files")), sendOpts{})
			for i := 1; i <= 3; i++ {
				h.send(main.toolResult(filler("file "+strconv.Itoa(i))), sendOpts{})
			}
			// Microcompaction rewrites the first tool_result in place.
			for i, m := range main.messages {
				if strings.Contains(m.blocks[0], `"tool_result"`) {
					main.messages[i].blocks[0] = toolResultBlock("toolu_1", "[Old tool result content cleared]")
					break
				}
			}
			h.send(main.user(filler("keep going")), sendOpts{})
			h.send(main.toolResult(filler("file 4")), sendOpts{})
		}},
		{"compact then continue", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			for i := 1; i <= 3; i++ {
				h.send(main.toolResult(filler("before compact "+strconv.Itoa(i))), sendOpts{})
			}
			h.send(main.clone().user(filler("summarize the conversation so far")), sendOpts{})
			after := newCCConversation("You are Claude Code.", session)
			h.send(after.user(filler("summary of the earlier conversation")), sendOpts{})
			h.send(after.toolResult(filler("after compact")), sendOpts{})
		}},
		{"model switch", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("sonnet 1")), sendOpts{})
			h.send(main.user(filler("sonnet 2")), sendOpts{})
			main.model = "claude-opus-4-1"
			h.send(main.user(filler("opus 3")), sendOpts{})
			main.model = "claude-sonnet-4-6"
			h.send(main.user(filler("sonnet 4")), sendOpts{})
		}},
		{"proxy restart with the same cache", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("before restart 1")), sendOpts{})
			h.send(main.toolResult(filler("before restart 2")), sendOpts{})
			h.restart("v2")
			h.send(main.user(filler("after restart 1")), sendOpts{})
			h.send(main.toolResult(filler("after restart 2")), sendOpts{})
		}},
		{"first-sight decline then accept", func(t *testing.T, h *invariantHarness) {
			shared := filler("the same file read by two conversations")
			b := newCCConversation("You are subagent B.", session)
			h.comp.declineFirst = true // B's first sight of the file is declined
			h.send(b.toolResult(shared), sendOpts{})
			h.comp.declineFirst = false
			a := newCCConversation("You are subagent A.", session)
			h.send(a.toolResult(shared), sendOpts{})
			h.send(b.user(filler("b next")), sendOpts{})
			h.send(a.user(filler("a next")), sendOpts{})
			h.send(b.user(filler("b last")), sendOpts{})
		}},
		{"memo write failure on one turn", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("w1")), sendOpts{})
			h.send(main.user(filler("w2")), sendOpts{})
			h.cache.mu.Lock()
			h.cache.failWrites = true
			h.cache.mu.Unlock()
			h.send(main.user(filler("w3 while the store is down")), sendOpts{})
			h.cache.mu.Lock()
			h.cache.failWrites = false
			h.cache.mu.Unlock()
			// The client re-sends the accepted turn (an aborted stream, a retry):
			// the block it carried raw is live again and must stay raw.
			h.send(main, sendOpts{})
			h.send(main.user(filler("w4")), sendOpts{})
			h.send(main.user(filler("w5")), sendOpts{})
		}},
		{"memo lookup error on one turn", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("l1")), sendOpts{})
			h.send(main.user(filler("l2")), sendOpts{})
			h.cache.mu.Lock()
			h.cache.failLookups = true
			h.cache.mu.Unlock()
			h.send(main.user(filler("l3 while lookups fail")), sendOpts{})
			h.cache.mu.Lock()
			h.cache.failLookups = false
			h.cache.mu.Unlock()
			h.send(main.user(filler("l4")), sendOpts{})
		}},
		{"recovery store failure on one turn", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("s1")), sendOpts{})
			h.comp.mu.Lock()
			h.comp.failStore = true
			h.comp.mu.Unlock()
			h.send(main.user(filler("s2 while CCR is down")), sendOpts{})
			h.comp.mu.Lock()
			h.comp.failStore = false
			h.comp.mu.Unlock()
			h.send(main, sendOpts{}) // the client re-sends the accepted turn
			h.send(main.user(filler("s3")), sendOpts{})
			h.send(main.user(filler("s4")), sendOpts{})
		}},
		{"rate limit 429 with retry-after", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("r1")), sendOpts{})
			main.user(filler("r2"))
			status, attempts := h.send(main, sendOpts{respond: rateLimitResponse})
			if status != http.StatusTooManyRequests || len(attempts) != 1 {
				t.Fatalf("a rate limit must reach the client unreplayed: status=%d upstream attempts=%d", status, len(attempts))
			}
			_, again := h.send(main, sendOpts{})
			if len(again) != 1 || !bytes.Equal(again[0].body, attempts[0].body) {
				t.Fatal("the client's retry must reproduce the rate-limited request's transformed bytes")
			}
			h.send(main.user(filler("r3")), sendOpts{})
		}},
		{"opaque 429 accepted raw", func(t *testing.T, h *invariantHarness) {
			rawRetryScenario(t, h, session, http.StatusTooManyRequests)
		}},
		{"400 accepted raw", func(t *testing.T, h *invariantHarness) {
			rawRetryScenario(t, h, session, http.StatusBadRequest)
		}},
		{"401 accepted raw", func(t *testing.T, h *invariantHarness) {
			rawRetryScenario(t, h, session, http.StatusUnauthorized)
		}},
		{"raw retry in a subagent leaves the main thread alone", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			sub := newCCConversation("You are a subagent.", session)
			h.send(main.user(filler("main 1")), sendOpts{})
			h.send(main.toolResult(filler("main 2")), sendOpts{})
			h.send(sub.user(filler("sub 1")), sendOpts{})
			if _, attempts := h.send(sub.toolResult(filler("sub 2")), sendOpts{respond: rejectTransformed(http.StatusBadRequest, nil)}); len(attempts) != 2 {
				t.Fatalf("test setup: the subagent turn should have taken the raw retry, attempts=%d", len(attempts))
			}
			h.send(main.user(filler("main 3")), sendOpts{})
			h.send(sub.user(filler("sub 3")), sendOpts{})
			h.send(main.toolResult(filler("main 4")), sendOpts{})
		}},
		{"fork and main concurrently carry the same new live block", func(t *testing.T, h *invariantHarness) {
			main := newCCConversation("You are Claude Code.", session)
			h.send(main.user(filler("c1")), sendOpts{})
			h.send(main.toolResult(filler("c2")), sendOpts{})
			for round := 0; round < 4; round++ {
				main.toolResult(filler("concurrent new block " + strconv.Itoa(round)))
				fork := main.clone().user(filler("suggest the next prompt"))
				var wg sync.WaitGroup
				for _, c := range []*ccConversation{main, fork} {
					wg.Add(1)
					go func(c *ccConversation) {
						defer wg.Done()
						h.send(c, sendOpts{group: round + 1})
					}(c)
				}
				wg.Wait()
			}
			h.send(main.user(filler("after the race")), sendOpts{})
		}},
		{"subagents with identical first messages", func(t *testing.T, h *invariantHarness) {
			task := filler("explore the repository and report")
			a := newCCConversation("You are an Explore subagent.", session).user(task)
			b := a.clone()
			h.send(a, sendOpts{})
			h.send(b, sendOpts{})
			h.send(a.toolResult(filler("a reads x")), sendOpts{})
			h.send(b.toolResult(filler("b reads y")), sendOpts{})
			h.send(a.toolResult(filler("a reads z")), sendOpts{})
			h.send(b.user(filler("b wraps up")), sendOpts{})
		}},
		{"tool-schema strip through a store failure and a raw retry", func(t *testing.T, h *invariantHarness) {
			h.strip = true
			h.restart("")
			main := newCCConversation("You are Claude Code.", session)
			sub := newCCConversation("You are Claude Code.", session) // same catalog, own thread
			h.send(main.user(filler("strip 1")), sendOpts{})
			h.comp.mu.Lock()
			h.comp.failStore = true
			h.comp.mu.Unlock()
			h.send(main.user(filler("strip 2 while CCR is down")), sendOpts{})
			h.comp.mu.Lock()
			h.comp.failStore = false
			h.comp.mu.Unlock()
			h.send(sub.user(filler("sub 1")), sendOpts{})
			h.send(main.user(filler("strip 3")), sendOpts{respond: rejectTransformed(http.StatusBadRequest, nil)})
			h.send(main.user(filler("strip 4")), sendOpts{})
			h.send(sub.user(filler("sub 2")), sendOpts{})
			if !bytes.Contains(h.xs[0].forwarded, []byte(`"input_schema":{"type":"object"}`)) {
				t.Fatalf("test setup: the catalog was not stripped:\n%.400s", h.xs[0].forwarded)
			}
		}},
		{"two sessions sharing file contents", func(t *testing.T, h *invariantHarness) {
			file := filler("package main shared file contents")
			one := newCCConversation("You are Claude Code in repo one.", "sess-one")
			two := newCCConversation("You are Claude Code in repo two.", "sess-two")
			h.send(one.user(filler("one starts")), sendOpts{})
			h.send(two.toolResult(file), sendOpts{})
			h.send(one.toolResult(file), sendOpts{})
			h.send(two.user(filler("two continues")), sendOpts{})
			h.send(one.user(filler("one continues")), sendOpts{})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newInvariantHarness(t)
			tc.run(t, h)
			h.compressedSomething()
			h.assert()
		})
	}
}

// interleaved is a Claude Code process: the main thread, two subagents with
// their own system prompts, a haiku side request and the prompt-suggestion fork,
// all on one session (or none).
func interleaved(h *invariantHarness, session string) {
	main := newCCConversation("You are Claude Code.", session)
	explore := newCCConversation("You are an Explore subagent.", session)
	plan := newCCConversation("You are a Plan subagent.", session)
	h.send(main.user(filler("main 1")), sendOpts{})
	h.send(newCCConversation("Write a 5-word title.", session).user(filler("main 1")), sendOpts{})
	h.send(main.toolResult(filler("main reads a")), sendOpts{})
	h.send(explore.user(filler("explore task")), sendOpts{})
	h.send(main.clone().user(filler("suggest the next prompt")), sendOpts{})
	h.send(plan.user(filler("plan task")), sendOpts{})
	h.send(explore.toolResult(filler("explore reads b")), sendOpts{})
	h.send(main.toolResult(filler("main reads c")), sendOpts{})
	h.send(plan.toolResult(filler("main reads a")), sendOpts{})
	h.send(explore.toolResult(filler("main reads c")), sendOpts{})
	h.send(main.clone().user(filler("suggest the next prompt")), sendOpts{})
	h.send(main.user(filler("main 2")), sendOpts{})
	h.send(plan.user(filler("plan wraps up")), sendOpts{})
	h.send(main.toolResult(filler("main reads d")), sendOpts{})
}

// rawRetryScenario: the provider rejects one transformed turn with status and
// accepts the original bytes. Every later request must extend that raw request.
func rawRetryScenario(t *testing.T, h *invariantHarness, session string, status int) {
	main := newCCConversation("You are Claude Code.", session)
	h.send(main.user(filler("t1")), sendOpts{})
	h.send(main.toolResult(filler("t2")), sendOpts{})
	_, attempts := h.send(main.user(filler("t3")), sendOpts{respond: rejectTransformed(status, nil)})
	if len(attempts) != 2 {
		t.Fatalf("the rejected transformed turn must be retried once with the original bytes, attempts=%d", len(attempts))
	}
	h.send(main.toolResult(filler("t4")), sendOpts{})
	h.send(main.clone().user(filler("suggest the next prompt")), sendOpts{})
	h.send(main.user(filler("t5")), sendOpts{})
}

// FuzzCachePrefixInvariant walks random sequences of the same operations —
// turns, tool results, side requests, forks, rewinds, client edits, compaction,
// model switches, restarts, provider rejections, store failures and declines —
// and checks the invariant over everything the provider accepted.
func FuzzCachePrefixInvariant(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 0, 1, 0, 0, 1, 1, 1, 3, 0, 0, 0},
		{0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 8, 0, 1, 0},
		{0, 1, 1, 2, 2, 0, 9, 1, 1, 1, 4, 1, 0, 1},
		{1, 0, 1, 0, 5, 0, 0, 0, 10, 0, 0, 0, 11, 0, 1, 0},
		{12, 0, 1, 3, 1, 4, 1, 3, 0, 0, 6, 0, 0, 0},
		{0, 0, 9, 2, 0, 0, 8, 0, 0, 0, 3, 0, 0, 0},
		{0, 2, 9, 3, 0, 2, 0, 2, 13, 0, 0, 0, 7, 0, 0, 0},
		{0, 0, 0, 1, 9, 0, 1, 1, 0, 0, 0, 1, 14, 0, 0, 1},
		{0, 0, 10, 0, 15, 0, 0, 0, 13, 0, 15, 0, 0, 0},
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, script []byte) {
		if len(script) > 80 {
			script = script[:80]
		}
		h := newInvariantHarness(t)
		if len(script) > 0 && script[0]&1 == 1 {
			h.strip = true
			h.restart("")
		}
		w := newFuzzWorld()
		for i := 0; i+1 < len(script); i += 2 {
			w.step(h, script[i], script[i+1])
		}
		h.assert()
	})
}

// fuzzWorld holds the conversations a fuzz script drives. Conversation 3
// shares conversation 0's system prompt and first message, like two subagents
// started with the same task.
type fuzzWorld struct {
	convs   []*ccConversation
	next    respondFunc
	retryOn bool
	turn    int
	// unpersisted: a store write failed, so some raw decisions live only in
	// this process (rawMemory). Restarting on top of that is a double fault the
	// proxy cannot see through, so the walk stops restarting from then on.
	unpersisted bool
}

func newFuzzWorld() *fuzzWorld {
	w := &fuzzWorld{convs: []*ccConversation{
		newCCConversation("You are Claude Code.", "sess-fuzz"),
		newCCConversation("You are an Explore subagent.", "sess-fuzz"),
		newCCConversation("You are Claude Code in another repo.", "sess-other"),
		newCCConversation("You are Claude Code.", "sess-fuzz"),
	}}
	opening := filler("open the task")
	w.convs[0].user(opening)
	w.convs[3].user(opening)
	w.convs[1].user(filler("explore the code"))
	w.convs[2].user(filler("other repo task"))
	return w
}

// content draws from a small pool so conversations share file contents.
func (w *fuzzWorld) content(arg byte) string {
	if arg&0x80 != 0 {
		return filler("shared file " + strconv.Itoa(int(arg&0x7)))
	}
	w.turn++
	return filler("unique content " + strconv.Itoa(w.turn))
}

func (w *fuzzWorld) send(h *invariantHarness, c *ccConversation) {
	o := sendOpts{respond: w.next}
	retry := w.retryOn
	w.next, w.retryOn = nil, false
	status, _ := h.send(c, o)
	if retry && status == http.StatusTooManyRequests {
		h.send(c, sendOpts{}) // the client honors retry-after and re-sends
	}
}

func (w *fuzzWorld) step(h *invariantHarness, op, arg byte) {
	c := w.convs[int(arg)%len(w.convs)]
	switch op % 16 {
	case 0:
		w.send(h, c.user(w.content(arg)))
	case 1:
		w.send(h, c.toolResult(w.content(arg)))
	case 2: // haiku side request
		h.send(newCCConversation("Summarize in five words.", c.session).user(w.content(arg)), sendOpts{})
	case 3: // prompt-suggestion fork
		w.send(h, c.clone().user(filler("suggest the next prompt")))
	case 4: // rewind to an earlier user turn and edit it
		if users := userIndexes(c); len(users) > 1 {
			c.messages = c.messages[:users[int(arg/4)%(len(users)-1)+1]]
			w.send(h, c.user(w.content(arg)))
		}
	case 5: // the client clears an old tool_result in place
		for i, m := range c.messages {
			if i < len(c.messages)-1 && strings.Contains(m.blocks[0], `"tool_result"`) && !strings.Contains(m.blocks[0], "cleared") {
				id := m.blocks[0][strings.Index(m.blocks[0], "toolu_"):]
				id = id[:strings.Index(id, `"`)]
				c.messages[i].blocks[0] = toolResultBlock(id, "[Old tool result content cleared]")
				w.send(h, c.user(w.content(arg)))
				break
			}
		}
	case 6: // compaction, then the conversation restarts from a summary
		w.send(h, c.clone().user(filler("summarize the conversation")))
		c.messages = nil
		w.send(h, c.user(filler("summary "+strconv.Itoa(int(arg)))))
	case 7:
		if c.model == "claude-sonnet-4-6" {
			c.model = "claude-opus-4-1"
		} else {
			c.model = "claude-sonnet-4-6"
		}
		w.send(h, c.user(w.content(arg)))
	case 8:
		if !w.unpersisted {
			h.restart("r" + strconv.Itoa(int(arg)))
		}
	case 9: // the provider rejects the next transformed request
		switch arg % 4 {
		case 0:
			w.next = rejectTransformed(http.StatusTooManyRequests, nil)
		case 1:
			w.next = rejectTransformed(http.StatusBadRequest, nil)
		case 2:
			w.next = rejectTransformed(http.StatusUnauthorized, nil)
		case 3:
			w.next, w.retryOn = rateLimitResponse, true
		}
	case 10:
		w.unpersisted = true
		h.cache.mu.Lock()
		h.cache.failWrites = true
		h.cache.mu.Unlock()
		w.send(h, c.user(w.content(arg)))
		h.cache.mu.Lock()
		h.cache.failWrites = false
		h.cache.mu.Unlock()
	case 11:
		h.cache.mu.Lock()
		h.cache.failLookups = true
		h.cache.mu.Unlock()
		w.send(h, c.user(w.content(arg)))
		h.cache.mu.Lock()
		h.cache.failLookups = false
		h.cache.mu.Unlock()
	case 12:
		h.comp.mu.Lock()
		h.comp.declineFirst = !h.comp.declineFirst
		h.comp.mu.Unlock()
	case 13:
		h.comp.mu.Lock()
		h.comp.failStore = true
		h.comp.mu.Unlock()
		w.send(h, c.toolResult(w.content(arg)))
		h.comp.mu.Lock()
		h.comp.failStore = false
		h.comp.mu.Unlock()
	case 15: // the client re-sends its last request unchanged
		if len(c.messages) > 0 {
			w.send(h, c)
		}
	case 14: // a turn that carries the session id, or drops it
		if c.session == "" {
			c.session = "sess-fuzz"
		} else {
			c.session = ""
		}
		w.send(h, c.user(w.content(arg)))
	}
}

func userIndexes(c *ccConversation) []int {
	var out []int
	for i, m := range c.messages {
		if m.role == "user" && !strings.Contains(m.blocks[0], `"tool_result"`) {
			out = append(out, i)
		}
	}
	return out
}

// newestMarkedConversation is the Claude Code shape: only the newest message
// carries cache_control, so the marker moves forward every turn.
func newestMarkedConversation(userTexts ...string) string {
	msgs := make([]string, 0, len(userTexts)*2)
	for i, text := range userTexts {
		if i > 0 {
			msgs = append(msgs, `{"role":"assistant","content":[`+subBlock("assistant "+strconv.Itoa(i))+`]}`)
		}
		block := subBlock(text)
		if i == len(userTexts)-1 {
			block = subCachedBlock(text)
		}
		msgs = append(msgs, `{"role":"user","content":[`+block+`]}`)
	}
	return `{"model":"claude-sonnet-4-6","max_tokens":1024,` +
		`"system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` + strings.Join(msgs, ",") + `]}`
}

func withHeaders(base map[string]string, extra ...string) map[string]string {
	out := map[string]string{}
	for k, v := range base {
		out[k] = v
	}
	for i := 0; i+1 < len(extra); i += 2 {
		out[extra[i]] = extra[i+1]
	}
	return out
}

// TestGateTripKeepsEarlierSubstitutions: a rewind makes the client's own
// prefix diverge from the previous turn. Whatever the proxy decides about NEW
// content, it must keep re-sending the replacement turn 1 was cached with.
func TestGateTripKeepsEarlierSubstitutions(t *testing.T) {
	t1, t2, t3, t2b := turnText(1), turnText(2), turnText(3), strings.Repeat("turn two rewritten ", 40)
	rt := prefixStableTransport(4)
	comp := &stableCompressor{}
	srv, _ := newPrefixStableServer(comp, newTestPrefixCache(), rt)
	headers := withHeaders(subscriptionAgentHeaders, "x-cave-session", "sess-rewind")

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1), headers)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2), headers)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2, t3), headers)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2b), headers)

	if !strings.Contains(string(rt.bodies[3]), comp.expectedReplacement(t, t1)) {
		t.Fatalf("the rewound request dropped turn 1's cached replacement:\n%s", rt.bodies[3])
	}
}

// TestExplicitEpochVetoKeepsSubstitutions: a framework's declared epoch may veto
// NEW compression, but the replacement it accepted on an earlier turn is already
// in the provider cache, so it is re-sent either way.
func TestExplicitEpochVetoKeepsSubstitutions(t *testing.T) {
	t1, t2 := turnText(1), turnText(2)
	rt := prefixStableTransport(2)
	comp := &stableCompressor{}
	srv, sink := newPrefixStableServer(comp, newTestPrefixCache(), rt)
	epoch := func(digest string) map[string]string {
		return withHeaders(subscriptionAgentHeaders, "x-cave-cache-epoch", "epoch-1", "x-cave-cache-prefix-sha256", strings.Repeat(digest, 64))
	}

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1), epoch("a"))
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2), epoch("b")) // drift: veto

	second := string(rt.bodies[1])
	if !strings.Contains(second, comp.expectedReplacement(t, t1)) {
		t.Fatalf("an epoch veto dropped turn 1's cached replacement:\n%s", second)
	}
	if !strings.Contains(second, t2) {
		t.Fatalf("an epoch veto must still stop NEW compression:\n%s", second)
	}
	if sink.last(t).CompressionEligible {
		t.Fatal("a vetoed request is not a compression candidate")
	}
}

// TestToolSchemaStripIgnoresEpochVeto: the strip is a pure function of the tool
// catalog at the head of the prefix. Gating it per request flipped the catalog
// between stripped and original, busting the whole prefix.
func TestToolSchemaStripIgnoresEpochVeto(t *testing.T) {
	rt := &captureTransport{responses: []string{subMessageRespBody, subMessageRespBody}}
	srv, _ := newToolSchemaStripServer(t, &toolSchemaStripCompressor{}, rt, Config{RecoveryViaMCP: true, ToolSchemaStrip: toolSchemaStripMode})
	epoch := func(digest string) map[string]string {
		return withHeaders(subscriptionAgentHeaders, "x-cave-cache-epoch", "epoch-1", "x-cave-cache-prefix-sha256", strings.Repeat(digest, 64))
	}

	serveBody(t, srv, "/v1/messages", toolCatalogRequest("first turn"), epoch("a"))
	serveBody(t, srv, "/v1/messages", toolCatalogRequest("second turn"), epoch("b"))

	stripped := strippedToolCatalog(t)
	for i, body := range rt.bodies {
		if !strings.Contains(string(body), `"tools":`+stripped) {
			t.Fatalf("request %d did not carry the stripped catalog:\n%s", i+1, body)
		}
	}
}

func conversationWithSystem(system string, userTexts ...string) string {
	body := newestMarkedConversation(userTexts...)
	return strings.Replace(body, `"text":"You are Claude Code."`, `"text":`+jsonText(system), 1)
}

// TestMemoNeverFlipsABlockSentRaw: the replacement memo is keyed by content
// across conversations. A block one conversation already sent raw must never be
// substituted later because another conversation compressed the same bytes —
// whatever went out first is what the provider cached.
func TestMemoNeverFlipsABlockSentRaw(t *testing.T) {
	x, y := strings.Repeat("the same tool output ", 40), turnText(9)
	comp := &declineOnceCompressor{target: x}
	rt := prefixStableTransport(3)
	srv, _ := newPrefixStableServer(comp, newTestPrefixCache(), rt)

	serveBody(t, srv, "/v1/messages", conversationWithSystem("You are B.", x), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", conversationWithSystem("You are A.", x), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", conversationWithSystem("You are B.", x, y), subscriptionAgentHeaders)

	for i, body := range rt.bodies {
		if !strings.Contains(string(body), x) {
			t.Fatalf("request %d did not send x raw although B sent it raw first:\n%s", i+1, body)
		}
	}
	if !strings.Contains(string(rt.bodies[2]), (&stableCompressor{}).expectedReplacement(t, y)) {
		t.Fatalf("new content must still compress:\n%s", rt.bodies[2])
	}
}

// declineOnceCompressor turns target down the first time it sees it, the way a
// query-aware compressor or a failed recovery write can, and compresses it (and
// everything else) afterwards.
type declineOnceCompressor struct {
	stableCompressor
	target   string
	declined bool
}

func (c *declineOnceCompressor) CompressSegment(seg []byte) ([]byte, int, int) {
	c.mu.Lock()
	decline := string(seg) == c.target && !c.declined
	c.declined = c.declined || decline
	c.mu.Unlock()
	if decline {
		return nil, 0, 0
	}
	return c.stableCompressor.CompressSegment(seg)
}

// TestResumedHistoryNeverFlips: a block first seen below the cache floor (a
// --resume history built without the proxy) went out raw, so raw is its
// decision even when the same bytes later arrive as live content.
func TestResumedHistoryNeverFlips(t *testing.T) {
	history, live, next := strings.Repeat("history from before the proxy ", 30), turnText(2), turnText(3)
	rt := prefixStableTransport(3)
	comp := &stableCompressor{}
	srv, _ := newPrefixStableServer(comp, newTestPrefixCache(), rt)

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(history, live), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", conversationWithSystem("You are a subagent.", history), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(history, live, next), subscriptionAgentHeaders)

	for i, body := range rt.bodies {
		if !strings.Contains(string(body), history) {
			t.Fatalf("request %d compressed history the resumed conversation sent raw:\n%s", i+1, body)
		}
	}
	if !strings.Contains(string(rt.bodies[2]), comp.expectedReplacement(t, live)) {
		t.Fatalf("the resumed conversation's own live turn must stay replaced:\n%s", rt.bodies[2])
	}
}

// TestUnpersistedRawDecisionHolds: when the replacement store cannot take a
// write, the turn's new block goes out raw and that decision cannot be stored.
// If the client then re-sends the accepted turn (an aborted stream, a retry),
// the block is live again and must still go out raw: the provider cached it so.
func TestUnpersistedRawDecisionHolds(t *testing.T) {
	t1, t2 := turnText(1), turnText(2)
	cache := newTestPrefixCache()
	rt := prefixStableTransport(3)
	srv, _ := newPrefixStableServer(&stableCompressor{}, cache, rt)

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1), subscriptionAgentHeaders)
	cache.mu.Lock()
	cache.failWrites = true
	cache.mu.Unlock()
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2), subscriptionAgentHeaders)
	cache.mu.Lock()
	cache.failWrites = false
	cache.mu.Unlock()
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(t1, t2), subscriptionAgentHeaders)

	if !strings.Contains(string(rt.bodies[1]), t2) || !bytes.Equal(rt.bodies[1], rt.bodies[2]) {
		t.Fatalf("the re-sent turn must reproduce the bytes the provider accepted:\n%s\n%s", rt.bodies[1], rt.bodies[2])
	}
}

// TestRateLimit429IsReturnedNotReplayed: a 429 that says it is a rate limit
// goes back to the client. Replaying it raw cannot beat the limit, and a raw
// replay accepted once the window frees caches bytes the next turn will not
// send. The client's own retry reproduces the transformed request exactly.
func TestRateLimit429IsReturnedNotReplayed(t *testing.T) {
	h := newInvariantHarness(t)
	main := newCCConversation("You are Claude Code.", "sess-429")
	t1 := filler("turn one")
	h.send(main.user(t1), sendOpts{})
	main.user(filler("turn two"))

	status, attempts := h.send(main, sendOpts{respond: rateLimitResponse})
	if status != http.StatusTooManyRequests || len(attempts) != 1 {
		t.Fatalf("rate limit: client status %d after %d upstream attempts, want 429 after 1", status, len(attempts))
	}
	status, again := h.send(main, sendOpts{})
	if status != http.StatusOK || len(again) != 1 || !bytes.Equal(again[0].body, attempts[0].body) {
		t.Fatalf("the client's retry must reproduce the rate-limited bytes (status %d)", status)
	}
	if !bytes.Contains(again[0].body, []byte("CMP:"+contentHandle([]byte(t1)))) {
		t.Fatalf("the retried turn lost turn one's replacement:\n%s", again[0].body)
	}
}

// TestAcceptedRawRetryPinsTheConversationRaw: once the provider accepted a
// request only in its original form, that is what it cached, so every later
// request extending it goes out raw too. The pin follows the conversation, not
// the session: a subagent on the same session keeps compressing.
func TestAcceptedRawRetryPinsTheConversationRaw(t *testing.T) {
	h := newInvariantHarness(t)
	main := newCCConversation("You are Claude Code.", "sess-pin")
	sub := newCCConversation("You are a subagent.", "sess-pin")
	h.send(main.user(filler("one")), sendOpts{})
	h.send(sub.user(filler("sub one")), sendOpts{})

	main.user(filler("two"))
	rawBody := main.body()
	_, attempts := h.send(main, sendOpts{respond: rejectTransformed(http.StatusBadRequest, nil)})
	if len(attempts) != 2 || !bytes.Equal(attempts[1].body, rawBody) {
		t.Fatalf("test setup: want a rejected transformed attempt and an accepted raw retry, got %d attempts", len(attempts))
	}
	_, next := h.send(main.user(filler("three")), sendOpts{})
	if !bytes.Equal(next[0].body, main.body()) {
		t.Fatalf("the turn after an accepted raw retry must go out raw:\n%s", next[0].body)
	}
	_, subNext := h.send(sub.user(filler("sub two")), sendOpts{})
	if !bytes.Contains(subNext[0].body, []byte("<<ccr:")) {
		t.Fatalf("a different conversation on the same session must keep compressing:\n%s", subNext[0].body)
	}
	h.restart("")
	_, afterRestart := h.send(main.user(filler("four")), sendOpts{})
	if !bytes.Equal(afterRestart[0].body, main.body()) {
		t.Fatalf("the pin must survive a proxy restart:\n%s", afterRestart[0].body)
	}
	h.assert()
}

// TestPrefixMonitorAnchorsAcceptedBytes: the monitor must remember what the
// provider accepted. Anchoring the rejected transformed attempt of a raw retry
// made the next (correctly raw) turn read as a bust.
func TestPrefixMonitorAnchorsAcceptedBytes(t *testing.T) {
	short := "a first message too short to compress"
	m2, m3, m4 := turnText(2), turnText(3), turnText(4)
	rt := &captureTransport{
		statuses:  []int{200, 200, http.StatusBadRequest, 200, 200},
		responses: []string{subMessageRespBody, subMessageRespBody, subMessageRespBody, subMessageRespBody, subMessageRespBody},
	}
	srv, sink := newPrefixStableServer(&stableCompressor{}, newTestPrefixCache(), rt)
	headers := withHeaders(subscriptionAgentHeaders, "x-cave-session", "sess-monitor")

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(short), headers)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(short, m2), headers)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(short, m2, m3), headers) // 400, then raw
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(short, m2, m3, m4), headers)

	if len(rt.bodies) != 5 || !bytes.Equal(rt.bodies[4], []byte(newestMarkedConversation(short, m2, m3, m4))) {
		t.Fatalf("test setup: want the turn after the raw retry pinned raw, got %d upstream calls", len(rt.bodies))
	}
	if row := sink.last(t); row.CacheBust {
		t.Fatal("the turn that extends the accepted raw request was flagged as a cache bust")
	}
}

// poisonSpliceAdapter fails reassembly whenever the poison block is replaced.
type poisonSpliceAdapter struct {
	anthropic.Adapter
	poison string
}

func (a poisonSpliceAdapter) ExtractStabilizable(body []byte, meta providers.RequestMetadata) ([]providers.RewritableBlock, func([][]byte) ([]byte, error), bool) {
	blocks, reassemble, ok := a.Adapter.ExtractStabilizable(body, meta)
	return blocks, func(reps [][]byte) ([]byte, error) {
		for i, rep := range reps {
			if rep != nil && string(blocks[i].Content) == a.poison {
				return nil, errTestPrefixCacheDown
			}
		}
		return reassemble(reps)
	}, ok
}

// TestLateSpliceFailureKeepsSubstitutions: when assembling this turn's new
// compression fails, the request must still re-send the replacements earlier
// turns were cached with, and the new block — which goes out raw — must not be
// left on record as compressed.
func TestLateSpliceFailureKeepsSubstitutions(t *testing.T) {
	x, y, z := turnText(1), turnText(2), turnText(3)
	rt := prefixStableTransport(3)
	comp := &stableCompressor{}
	srv := New(Config{
		Adapters:       []providers.Adapter{poisonSpliceAdapter{Adapter: anthropic.New("https://upstream.test").(anthropic.Adapter), poison: y}},
		Auth:           stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Creds:          passthroughTestCreds{},
		Compressor:     comp,
		PrefixCache:    newTestPrefixCache(),
		HTTPClient:     &http.Client{Transport: rt},
		RecoveryViaMCP: true,
	})

	serveBody(t, srv, "/v1/messages", newestMarkedConversation(x), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(x, y), subscriptionAgentHeaders)
	serveBody(t, srv, "/v1/messages", newestMarkedConversation(x, y, z), subscriptionAgentHeaders)

	for i := 0; i < 3; i++ {
		body := string(rt.bodies[i])
		if !strings.Contains(body, comp.expectedReplacement(t, x)) {
			t.Fatalf("request %d dropped x's cached replacement:\n%s", i+1, body)
		}
		if i > 0 && !strings.Contains(body, y) {
			t.Fatalf("request %d did not keep y raw as it first went out:\n%s", i+1, body)
		}
	}
}
