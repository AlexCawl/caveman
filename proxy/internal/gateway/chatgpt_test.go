package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/JuliusBrussee/caveman/engine"
	"github.com/JuliusBrussee/caveman/engine/ccr"
)

type chatGPTRealEngineCompressor struct {
	eng   *engine.Engine
	store *ccr.Store
}

func (c *chatGPTRealEngineCompressor) CompressSegment(segment []byte) ([]byte, int, int) {
	result, err := c.eng.Compress(segment, engine.Options{Mode: engine.ModeCompress})
	if err != nil {
		return segment, 0, 0
	}
	return result.Output, result.TokensBefore, result.TokensAfter
}

func (c *chatGPTRealEngineCompressor) CompressSegmentQuery(segment []byte, query string) ([]byte, int, int) {
	result, err := c.eng.Compress(segment, engine.Options{Mode: engine.ModeCompress, Query: query})
	if err != nil {
		return segment, 0, 0
	}
	return result.Output, result.TokensBefore, result.TokensAfter
}

func (c *chatGPTRealEngineCompressor) StoreOriginal(body []byte) (string, error) {
	return c.store.Put(ccr.Recovery{ContentType: "block", Compressor: "proxy-content", Original: body})
}

// chatgptTestServer wires a Server whose /chatgpt upstream is a local stub.
func chatgptTestServer(t *testing.T, upstream string) (*Server, *captureSink, *bytes.Buffer) {
	t.Helper()
	sink := &captureSink{}
	logs := &bytes.Buffer{}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            sink,
		ChatGPTUpstream: upstream,
		HTTPClient:      &http.Client{},
		Logger:          slog.New(slog.NewJSONHandler(logs, nil)),
	})
	return srv, sink, logs
}

func TestChatGPTPathAndQueryPreserved(t *testing.T) {
	var gotPath, gotQuery, gotMethod string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotQuery, gotMethod = r.URL.Path, r.URL.RawQuery, r.Method
		w.Write([]byte(`{"data":[]}`))
	}))
	defer upstream.Close()
	srv, _, _ := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(http.MethodGet, "/chatgpt/models?client_version=0.142.4", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotMethod != http.MethodGet || gotPath != "/models" || gotQuery != "client_version=0.142.4" {
		t.Fatalf("upstream saw %s %s?%s, want GET /models?client_version=0.142.4", gotMethod, gotPath, gotQuery)
	}
}

func TestChatGPTHeadersForwardedByteExactExceptHost(t *testing.T) {
	const bearer = "Bearer oauth-token-sekret-123"
	const acct = "acct-e5f6"
	var got http.Header
	var gotHost string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		gotHost = r.Host
		w.Header().Set("Connection", "X-Upstream-Private")
		w.Header().Set("X-Upstream-Private", "must-not-leak")
		w.Header().Set("X-Cave-Upstream", "must-not-leak")
		w.Header().Set("X-Upstream-Note", "hello")
		w.Write([]byte(`ok`))
	}))
	defer upstream.Close()
	srv, _, logs := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(`{"model":"gpt-5.5"}`))
	req.Header.Set("Authorization", bearer)
	req.Header.Set("ChatGPT-Account-ID", acct)
	req.Header.Set("X-Custom-Header", "custom-v")
	req.Header.Set("Connection", "X-Connection-Private")
	req.Header.Set("X-Connection-Private", "must-not-forward")
	req.Header.Set("X-Cave-Session", "must-not-forward")
	req.Header.Set("Proxy-Authorization", "Basic must-not-forward")
	req.Host = "127.0.0.1:8787"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if got.Get("Authorization") != bearer {
		t.Fatalf("Authorization not byte-exact: %q", got.Get("Authorization"))
	}
	if got.Get("ChatGPT-Account-ID") != acct {
		t.Fatalf("ChatGPT-Account-ID not byte-exact: %q", got.Get("ChatGPT-Account-ID"))
	}
	if got.Get("X-Custom-Header") != "custom-v" {
		t.Fatalf("arbitrary header dropped: %q", got.Get("X-Custom-Header"))
	}
	for _, name := range []string{"Connection", "X-Connection-Private", "X-Cave-Session", "Proxy-Authorization"} {
		if got.Get(name) != "" {
			t.Fatalf("unsafe request header %s reached upstream: %q", name, got.Get(name))
		}
	}
	if gotHost == "127.0.0.1:8787" {
		t.Fatal("Host must be rewritten to the upstream, not the proxy's")
	}
	if rec.Header().Get("X-Upstream-Note") != "hello" {
		t.Fatal("response headers must pass through")
	}
	for _, name := range []string{"Connection", "X-Upstream-Private", "X-Cave-Upstream"} {
		if rec.Header().Get(name) != "" {
			t.Fatalf("unsafe response header %s reached client: %q", name, rec.Header().Get(name))
		}
	}
	// The credential values must never reach the logs.
	for _, secret := range []string{"sekret-123", acct} {
		if strings.Contains(logs.String(), secret) {
			t.Fatalf("credential %q leaked into logs: %s", secret, logs.String())
		}
	}
}

func TestChatGPTSSEStreamsThroughAndMetersUsage(t *testing.T) {
	sse := "event: response.created\ndata: {\"type\":\"response.created\"}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":111,\"output_tokens\":22}}}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, chunk := range strings.SplitAfter(sse, "\n\n") {
			if chunk == "" {
				continue
			}
			w.Write([]byte(chunk))
			f.Flush()
		}
	}))
	defer upstream.Close()
	srv, sink, _ := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(`{"model":"gpt-5.5","stream":true}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if body := rec.Body.String(); body != sse {
		t.Fatalf("SSE body not byte-exact:\n got %q\nwant %q", body, sse)
	}
	row := sink.last(t)
	if row.InputTokens != 111 || row.OutputTokens != 22 {
		t.Fatalf("usage not metered: %+v", row)
	}
	if row.TotalCostUSD != 0 || row.SavingsUSD != 0 {
		t.Fatalf("subscription traffic must record zero dollars, got cost=%v savings=%v", row.TotalCostUSD, row.SavingsUSD)
	}
	if row.Model != "gpt-5.5" || row.Provider != "chatgpt-subscription" {
		t.Fatalf("row attribution wrong: %+v", row)
	}
	if row.Basis != "inferred" {
		t.Fatalf("basis = %q, want inferred", row.Basis)
	}
	if row.RawRequestSHA256 != row.TransformedRequestSHA256 {
		t.Fatal("passthrough hashes must match — nothing may be transformed")
	}
}

func TestChatGPTCodexSubscriptionCompressesLiveZoneAndRecordsTokensOnly(t *testing.T) {
	live := strings.Repeat("codex oauth live tool output ", 40)
	body := `{"model":"gpt-5.5","stream":true,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"` + live + `"}]}]}`
	rt := &captureTransport{responses: []string{`{"id":"resp","usage":{"input_tokens":90,"output_tokens":10}}`}}
	sink := &captureSink{}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            sink,
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer codex-oauth-secret")
	req.Header.Set("ChatGPT-Account-ID", "acct_test")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if len(rt.bodies) != 1 {
		t.Fatalf("upstream calls = %d, want 1", len(rt.bodies))
	}
	upstream := string(rt.bodies[0])
	if strings.Contains(upstream, live) || !strings.Contains(upstream, "<<ccr:") {
		t.Fatalf("Codex live zone was not compressed: %s", upstream)
	}
	if got := rt.headers[0].Get("Authorization"); got != "Bearer codex-oauth-secret" {
		t.Fatalf("OAuth authorization changed: %q", got)
	}
	if rec.Header().Get("x-caveman-recovery-handle") == "" || rec.Header().Get("x-caveman-tokens-before") == "" {
		t.Fatalf("compression disclosure headers missing: %v", rec.Header())
	}
	row := sink.last(t)
	if row.AuthMode != "subscription" || row.RuntimeMode != "compress" {
		t.Fatalf("auth/runtime = %q/%q, want subscription/compress", row.AuthMode, row.RuntimeMode)
	}
	if row.CompressionTokensBefore <= row.CompressionTokensAfter || row.CompressionTokenCountBasis != "estimated_engine_o200k" || row.RecoveryHandle == "" {
		t.Fatalf("compression accounting missing: %+v", row)
	}
	if row.TotalCostUSD != 0 || row.SavingsUSD != 0 {
		t.Fatalf("subscription compression must remain tokens-only: %+v", row)
	}
	if row.RawRequestSHA256 == row.TransformedRequestSHA256 {
		t.Fatal("compressed request hashes must differ")
	}
}

func TestChatGPTCodexSubscriptionCompressesOnlyExactResponsesRoute(t *testing.T) {
	body := `{"model":"gpt-5.5","input":"` + strings.Repeat("must stay exact ", 40) + `"}`
	rt := &captureTransport{responses: []string{`{"id":"resp"}`}}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            &captureSink{},
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses/batch", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || len(rt.bodies) != 1 {
		t.Fatalf("status/calls = %d/%d, want 200/1", rec.Code, len(rt.bodies))
	}
	if !bytes.Equal(rt.bodies[0], []byte(body)) {
		t.Fatalf("non-/responses route changed bytes: %s", rt.bodies[0])
	}
	if rec.Header().Get("x-caveman-recovery-handle") != "" {
		t.Fatal("non-/responses route disclosed compression")
	}
}

func TestChatGPTCompressionHonorsRequestWidePassThrough(t *testing.T) {
	body := `{"model":"gpt-5.5","input":"` + strings.Repeat("must stay exact ", 40) + `"}`
	rt := &captureTransport{responses: []string{`{"id":"resp"}`}}
	srv := New(Config{
		Auth: stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}}, Sink: &captureSink{},
		Compressor: &liveZoneCompressor{}, PrefixCache: newTestPrefixCache(), RecoveryViaMCP: true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex", HTTPClient: &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	req.Header.Set("x-cave-transforms", "caveman.pass-through.v1")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || len(rt.bodies) != 1 || !bytes.Equal(rt.bodies[0], []byte(body)) {
		t.Fatalf("pass-through contract changed bytes: status=%d body=%s", rec.Code, rt.bodies[0])
	}
	if rec.Header().Get("x-caveman-recovery-handle") != "" {
		t.Fatal("pass-through contract disclosed compression")
	}
}

func TestChatGPTCompressionHonorsCompiledPlanAndCacheEpochGates(t *testing.T) {
	body := `{"model":"gpt-5.5","input":"` + strings.Repeat("must stay exact ", 40) + `"}`
	for _, headers := range []http.Header{
		{"x-cave-agent-build": []string{strings.Repeat("a", 64)}},
		{"x-cave-transform-location": []string{"local"}},
		{"x-cave-cache-epoch": []string{"epoch-without-digest"}},
	} {
		t.Run(fmt.Sprint(headers), func(t *testing.T) {
			rt := &captureTransport{responses: []string{`{"id":"resp"}`}}
			srv := New(Config{
				Auth: stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}}, Sink: &captureSink{},
				Compressor: &liveZoneCompressor{}, PrefixCache: newTestPrefixCache(), RecoveryViaMCP: true,
				ChatGPTUpstream: "https://chatgpt.test/backend-api/codex", HTTPClient: &http.Client{Transport: rt},
			})
			req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
			for name, values := range headers {
				for _, value := range values {
					req.Header.Add(name, value)
				}
			}
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, req)
			if rec.Code != http.StatusOK || len(rt.bodies) != 1 || !bytes.Equal(rt.bodies[0], []byte(body)) {
				t.Fatalf("closed gate changed bytes: status=%d body=%s", rec.Code, rt.bodies[0])
			}
		})
	}
}

func TestChatGPTCodexSubscriptionRealEngineShrinksAndRecoversExactOriginal(t *testing.T) {
	t.Setenv("CAVE_ENGINE_TOON", "")
	recovery, err := ccr.OpenMemory()
	if err != nil {
		t.Fatalf("open ccr: %v", err)
	}
	defer recovery.Close()
	compressor := &chatGPTRealEngineCompressor{eng: engine.New(recovery, nil), store: recovery}

	items := make([]map[string]any, 80)
	for i := range items {
		items[i] = map[string]any{"id": i + 1, "name": "very repetitive fixture row", "city": "boulder"}
	}
	original, _ := json.Marshal(map[string]any{"items": items})
	reqBody, _ := json.Marshal(map[string]any{
		"model":  "gpt-5.5",
		"stream": true,
		"input": []any{map[string]any{
			"type": "message", "role": "user",
			"content": []any{map[string]any{"type": "input_text", "text": string(original)}},
		}},
	})
	rt := &captureTransport{responses: []string{`{"id":"resp","usage":{"input_tokens":90,"output_tokens":10}}`}}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            &captureSink{},
		Compressor:      compressor,
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", bytes.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer codex-oauth-secret")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	before, _ := strconv.Atoi(rec.Header().Get("x-caveman-tokens-before"))
	after, _ := strconv.Atoi(rec.Header().Get("x-caveman-tokens-after"))
	if before <= after || before == 0 {
		t.Fatalf("real engine did not reduce tokens: before=%d after=%d body=%s", before, after, rt.bodies[0])
	}
	if bytes.Equal(rt.bodies[0], reqBody) || !bytes.Contains(rt.bodies[0], []byte("<<ccr:")) {
		t.Fatalf("real engine request was not transformed with CCR marker: %s", rt.bodies[0])
	}
	handle := rec.Header().Get("x-caveman-recovery-handle")
	recovered, err := engine.New(recovery, nil).Retrieve(handle)
	if err != nil {
		t.Fatalf("recover original: %v", err)
	}
	if !bytes.Equal(recovered, original) {
		t.Fatalf("recovered block differs from exact original:\n got %s\nwant %s", recovered, original)
	}
}

func TestChatGPTCodexSubscriptionTransformed4xxRetriesOriginal(t *testing.T) {
	live := strings.Repeat("codex oauth fallback content ", 40)
	body := `{"model":"gpt-5.5","input":"` + live + `"}`
	rt := &captureTransport{statuses: []int{http.StatusTooManyRequests, http.StatusOK}}
	sink := &captureSink{}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            sink,
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer codex-oauth-secret")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || len(rt.bodies) != 2 {
		t.Fatalf("status/calls = %d/%d, want 200/2", rec.Code, len(rt.bodies))
	}
	if bytes.Equal(rt.bodies[0], []byte(body)) {
		t.Fatal("first request should exercise transformed path")
	}
	if !bytes.Equal(rt.bodies[1], []byte(body)) {
		t.Fatalf("retry must use exact original bytes:\n got %s\nwant %s", rt.bodies[1], body)
	}
	row := sink.last(t)
	if row.RecoveryHandle != "" || row.CompressionTokensBefore != 0 || row.RawRequestSHA256 != row.TransformedRequestSHA256 {
		t.Fatalf("fallback row must claim no compression: %+v", row)
	}
}

// The ChatGPT route's OAuth 4xx retry is the documented real-world fallback, so
// its capture record needs its own coverage: without it, a regression that drops
// or mislabels the retry record ships green while the on-disk evidence claims
// the transformed bytes served a request the upstream actually rejected.
func TestChatGPTCaptureRecordsRetryWithOriginalBytes(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CAVE_CAPTURE_DIR", dir)
	live := strings.Repeat("codex oauth fallback content ", 40)
	body := `{"model":"gpt-5.5","input":"` + live + `"}`
	rt := &captureTransport{statuses: []int{http.StatusTooManyRequests, http.StatusOK}}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            &captureSink{},
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer codex-oauth-secret")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	srv.capture.flush()

	if rec.Code != http.StatusOK || len(rt.bodies) != 2 {
		t.Fatalf("status/calls = %d/%d, want 200/2", rec.Code, len(rt.bodies))
	}
	got := readCaptures(t, dir)
	if len(got) != 2 {
		t.Fatalf("want 2 captures (attempt + retry), got %d", len(got))
	}
	first, retry := got[0], got[1]
	if first.RetryOriginal || !first.Transformed {
		t.Errorf("first record must be the transformed attempt: retry=%v transformed=%v", first.RetryOriginal, first.Transformed)
	}
	if !bytes.Equal(first.UpstreamBody, rt.bodies[0]) {
		t.Errorf("first record must hold the rejected transformed bytes")
	}
	if !retry.RetryOriginal {
		t.Error("second record must be flagged retry_original")
	}
	if retry.Transformed {
		t.Error("retry record must not claim a transform: both sides are the original bytes")
	}
	if !bytes.Equal(retry.ClientBody, []byte(body)) {
		t.Errorf("retry record must hold the exact original bytes:\n got %q\nwant %q", retry.ClientBody, body)
	}
	if first.RequestID == "" || first.RequestID != retry.RequestID {
		t.Errorf("both records must share one request id: %q vs %q", first.RequestID, retry.RequestID)
	}
}

func TestChatGPTMalformedBodyForwardedByteExactNothingInvented(t *testing.T) {
	garbage := "\x00\x01 not json or sse at all \xff"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(garbage))
	}))
	defer upstream.Close()
	srv, sink, _ := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader("also \x00 garbage"))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Body.String() != garbage {
		t.Fatalf("malformed body must forward byte-exact, got %q", rec.Body.String())
	}
	row := sink.last(t)
	if row.InputTokens != 0 || row.OutputTokens != 0 || row.TotalCostUSD != 0 {
		t.Fatalf("nothing parseable must record nothing: %+v", row)
	}
}

func TestChatGPTUpstreamUnreachableFailsWithCaveCode(t *testing.T) {
	srv, sink, _ := chatgptTestServer(t, "http://127.0.0.1:1") // nothing listens
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "cave_upstream_unreachable") {
		t.Fatalf("error must carry cave_snake_code, got %s", rec.Body.String())
	}
	row := sink.last(t)
	if row.ErrorCode != "cave_upstream_unreachable" {
		t.Fatalf("row error code = %q", row.ErrorCode)
	}
	if row.RequestHashComplete || row.RawRequestSHA256 != "" || row.TransformedRequestSHA256 != "" {
		t.Fatalf("transport-before-EOF must not record a partial prefix as exact hashes: %+v", row)
	}
}

func TestChatGPTNoCredentialFallback(t *testing.T) {
	// No Authorization at all: the route must forward as-is and surface the
	// upstream's own 401 — never inject a credential from env.
	t.Setenv("OPENAI_API_KEY", "sk-env-must-not-be-used")
	var gotAuth string
	var had bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, had = r.Header["Authorization"]
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":"missing bearer"}`)
	}))
	defer upstream.Close()
	srv, _, _ := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if had || gotAuth != "" {
		t.Fatalf("no credential may be injected, upstream saw %q", gotAuth)
	}
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("upstream 401 must pass through, got %d", rec.Code)
	}
}

// TestChatGPTCodexArrayToolOutputCompresses pins the Codex 0.149+ wire shape:
// custom_tool_call_output.output is an ARRAY of input_text parts, not a string.
// The extractor only accepted the string form, so every real Codex tool turn
// had an empty live zone and the subscription route compressed nothing
// (measured 2026-09-04: 75-116 KB requests, zero compression, on real traffic).
func TestChatGPTCodexArrayToolOutputCompresses(t *testing.T) {
	live := strings.Repeat("codex shell output line that is long enough to matter ", 40)
	body := `{"model":"gpt-5.6","stream":true,"store":false,"input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"Read the file."}]},` +
		`{"type":"custom_tool_call","call_id":"call_1","name":"shell_command","input":"sed -n 1,200p README.md"},` +
		`{"type":"custom_tool_call_output","call_id":"call_1","output":[{"type":"input_text","text":"Chunk ID: 1 exit code 0"},{"type":"input_text","text":"` + live + `"}]}]}`
	rt := &captureTransport{responses: []string{`{"id":"resp","usage":{"input_tokens":900,"output_tokens":10}}`}}
	sink := &captureSink{}
	comp := &liveZoneCompressor{}
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            sink,
		Compressor:      comp,
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer codex-oauth-secret")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	upstream := string(rt.bodies[0])
	if strings.Contains(upstream, live) || !strings.Contains(upstream, "<<ccr:") {
		t.Fatalf("array-shaped Codex tool output was not compressed: %s", upstream)
	}
	if !strings.Contains(upstream, "Chunk ID: 1 exit code 0") {
		t.Fatalf("the short header part must survive untouched: %s", upstream)
	}
	if comp.calls == 0 {
		t.Fatal("compressor was never called")
	}
	row := sink.last(t)
	if !row.CompressionEligible {
		t.Fatalf("the ChatGPT route must record eligibility so the CLI can tell routing-never-applied from nothing-to-compress: %+v", row)
	}
	if row.CompressionTokensBefore <= row.CompressionTokensAfter {
		t.Fatalf("compression accounting missing: %+v", row)
	}
}
func TestOpenCodeChatGPTOAuthReroutesToSubscriptionUpstream(t *testing.T) {
	const bearer = "Bearer opencode-oauth-secret"
	const account = "acct-opencode"

	var gotPath string
	var gotAuth string
	var gotAccount string
	var gotAgent string

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotAccount = r.Header.Get("ChatGPT-Account-ID")
		gotAgent = r.Header.Get("x-cave-agent")
		w.Write([]byte(`{"id":"resp"}`))
	}))
	defer upstream.Close()

	srv, _, _ := chatgptTestServer(t, upstream.URL)

	req := httptest.NewRequest(
		http.MethodPost,
		"/w/opencode/openai/v1/responses",
		strings.NewReader(`{"model":"gpt-5.5"}`),
	)
	req.Header.Set("Authorization", bearer)
	req.Header.Set("ChatGPT-Account-ID", account)

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if gotPath != "/responses" {
		t.Fatalf("upstream path = %q, want /responses", gotPath)
	}
	if gotAuth != bearer {
		t.Fatalf("Authorization changed: %q", gotAuth)
	}
	if gotAccount != account {
		t.Fatalf("ChatGPT-Account-ID changed: %q", gotAccount)
	}
	if gotAgent != "" {
		t.Fatalf("internal agent header reached upstream: %q", gotAgent)
	}
}

func TestOpenCodeChatGPTSubscriptionDetection(t *testing.T) {
	tests := []struct {
		name    string
		method  string
		path    string
		agent   string
		account string
		want    bool
	}{
		{
			name:    "opencode responses with account",
			method:  http.MethodPost,
			path:    "/openai/v1/responses",
			agent:   "opencode",
			account: "acct",
			want:    true,
		},
		{
			name:   "missing account",
			method: http.MethodPost,
			path:   "/openai/v1/responses",
			agent:  "opencode",
		},
		{
			name:    "different agent",
			method:  http.MethodPost,
			path:    "/openai/v1/responses",
			agent:   "other",
			account: "acct",
		},
		{
			name:    "different route",
			method:  http.MethodPost,
			path:    "/openai/v1/chat/completions",
			agent:   "opencode",
			account: "acct",
		},
		{
			name:    "different method",
			method:  http.MethodGet,
			path:    "/openai/v1/responses",
			agent:   "opencode",
			account: "acct",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			req.Header.Set("x-cave-agent", tt.agent)
			if tt.account != "" {
				req.Header.Set("ChatGPT-Account-ID", tt.account)
			}

			if got := isOpenCodeChatGPTSubscription(req); got != tt.want {
				t.Fatalf("isOpenCodeChatGPTSubscription() = %v, want %v", got, tt.want)
			}
		})
	}
}

func chatGPTCompressServer(transport http.RoundTripper) *Server {
	return New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            &captureSink{},
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     newTestPrefixCache(),
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: transport},
	})
}

func chatGPTTurn(texts ...string) string {
	items := make([]string, 0, len(texts))
	for _, text := range texts {
		items = append(items, `{"type":"message","role":"user","content":[{"type":"input_text","text":"`+text+`"}]}`)
	}
	return `{"model":"gpt-5.5","input":[` + strings.Join(items, ",") + `]}`
}

func serveChatGPT(srv *Server, body string, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/chatgpt/responses", strings.NewReader(body))
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

// TestChatGPTRateLimit429IsReturnedNotReplayed: raw bytes cannot beat a rate
// limit, and replaying them sends the provider a raw prefix in place of the
// compressed one it cached.
func TestChatGPTRateLimit429IsReturnedNotReplayed(t *testing.T) {
	var calls int
	srv := chatGPTCompressServer(roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": {"30"}},
			Body: io.NopCloser(strings.NewReader(`{"error":"rate limited"}`)), Request: r}, nil
	}))
	rec := serveChatGPT(srv, chatGPTTurn(strings.Repeat("codex tool output ", 40)))
	if rec.Code != http.StatusTooManyRequests || calls != 1 {
		t.Fatalf("status/calls = %d/%d, want the rate limit returned after one call", rec.Code, calls)
	}
}

// TestChatGPTAcceptedRawRetryPinsTheConversation: once the backend accepted a
// conversation only as raw bytes, that is what it cached, so the next turn goes
// out raw instead of re-substituting the replacements.
func TestChatGPTAcceptedRawRetryPinsTheConversation(t *testing.T) {
	first, second, third := strings.Repeat("codex first output ", 40), strings.Repeat("codex second output ", 40), strings.Repeat("codex third output ", 40)
	rt := &captureTransport{statuses: []int{http.StatusOK, http.StatusBadRequest, http.StatusOK, http.StatusOK}}
	srv := chatGPTCompressServer(rt)
	serveChatGPT(srv, chatGPTTurn(first))
	serveChatGPT(srv, chatGPTTurn(first, second))
	serveChatGPT(srv, chatGPTTurn(first, second, third))

	if len(rt.bodies) != 4 || !bytes.Equal(rt.bodies[2], []byte(chatGPTTurn(first, second))) {
		t.Fatalf("test setup: want a rejected transformed turn 2 and its accepted raw retry, got %d calls", len(rt.bodies))
	}
	if !bytes.Equal(rt.bodies[3], []byte(chatGPTTurn(first, second, third))) {
		t.Fatalf("the turn after an accepted raw retry must go out raw:\n%s", rt.bodies[3])
	}
}

// TestChatGPTEpochVetoKeepsSubstitutions: a declared epoch that drifted may
// veto new compression only; the replacement an earlier turn was cached with
// is still re-sent.
func TestChatGPTEpochVetoKeepsSubstitutions(t *testing.T) {
	first, second := strings.Repeat("codex first output ", 40), strings.Repeat("codex second output ", 40)
	rt := &captureTransport{}
	srv := chatGPTCompressServer(rt)
	epoch := func(digest string) []string {
		return []string{"x-cave-cache-epoch", "epoch-1", "x-cave-cache-prefix-sha256", strings.Repeat(digest, 64)}
	}
	serveChatGPT(srv, chatGPTTurn(first), epoch("a")...)
	serveChatGPT(srv, chatGPTTurn(first, second), epoch("b")...)

	if len(rt.bodies) != 2 || strings.Contains(string(rt.bodies[0]), first) {
		t.Fatal("test setup: want turn 1 compressed")
	}
	if strings.Contains(string(rt.bodies[1]), first) {
		t.Fatalf("an epoch veto dropped turn 1's cached replacement:\n%s", rt.bodies[1])
	}
	if !strings.Contains(string(rt.bodies[1]), second) {
		t.Fatalf("an epoch veto must still stop new compression:\n%s", rt.bodies[1])
	}
}

// TestChatGPTTripwireRecordsCavemansBust: the /chatgpt route takes the same
// compression path as the generic one, so it is held to the same rule on live
// traffic: a turn that re-sends bytes the backend cached replaced as raw (here
// the memo row was lost between turns) is caveman's bust, on the row too.
func TestChatGPTTripwireRecordsCavemansBust(t *testing.T) {
	first, second := strings.Repeat("codex first output ", 40), strings.Repeat("codex second output ", 40)
	rt := &captureTransport{}
	sink := &captureSink{}
	cache := newTestPrefixCache()
	srv := New(Config{
		Auth:            stubAuth{rc: RequestContext{Label: "local", RuntimeMode: "compress"}},
		Sink:            sink,
		Compressor:      &liveZoneCompressor{},
		PrefixCache:     cache,
		RecoveryViaMCP:  true,
		ChatGPTUpstream: "https://chatgpt.test/backend-api/codex",
		HTTPClient:      &http.Client{Transport: rt},
	})
	serveChatGPT(srv, chatGPTTurn(first), "x-cave-session", "sess-codex")
	cache.mu.Lock()
	cache.entries = map[string]testReplacement{}
	cache.mu.Unlock()
	serveChatGPT(srv, chatGPTTurn(first, second), "x-cave-session", "sess-codex")

	if len(rt.bodies) != 2 || strings.Contains(string(rt.bodies[0]), first) || !strings.Contains(string(rt.bodies[1]), first) {
		t.Fatal("test setup: want turn 1 compressed and turn 2 re-sending it raw")
	}
	if row := sink.last(t); !row.CacheBust || row.CacheBustCause != bustCauseCaveman {
		t.Fatalf("the /chatgpt row of a caveman bust: bust=%v cause=%q", row.CacheBust, row.CacheBustCause)
	}
}

// TestChatGPTOverCaptureLimitKeepsSubstitutions: a Codex session's body
// crosses the 4 MiB capture limit as screenshots and reasoning items pile up.
// That turn still extends the prefix earlier turns cached replaced, so it is
// still read whole (up to the route's request limit) and substituted.
func TestChatGPTOverCaptureLimitKeepsSubstitutions(t *testing.T) {
	first := strings.Repeat("codex first output ", 40)
	huge := strings.Repeat("x", chatGPTCaptureLimit)
	rt := &captureTransport{}
	srv := chatGPTCompressServer(rt)
	serveChatGPT(srv, chatGPTTurn(first))
	serveChatGPT(srv, chatGPTTurn(first, huge))

	if len(rt.bodies) != 2 || strings.Contains(string(rt.bodies[0]), first) {
		t.Fatal("test setup: want turn 1 compressed")
	}
	if strings.Contains(string(rt.bodies[1]), first) {
		t.Fatalf("turn 2 re-sent turn 1's message raw although turn 1 cached it compressed (body %d bytes)", len(rt.bodies[1]))
	}
}
