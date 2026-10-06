package openai

import (
	"context"
	"net/http"
	"strings"

	"github.com/JuliusBrussee/caveman/proxy/providers"
)

func New(baseURL string) providers.Adapter {
	return Adapter{Base: providers.Base{Provider: "openai", BaseURL: baseURL, Routes: []string{
		"/v1/chat/completions", "/v1/responses", "/v1/responses/input_tokens", "/v1/embeddings",
		"/openai/v1/chat/completions", "/openai/v1/responses", "/openai/v1/responses/input_tokens", "/openai/v1/embeddings",
	}, MetadataRoutes: []string{"/v1/models", "/openai/v1/models"}}}
}

// MatchMetadataRequest is the OpenAI half of the bare /v1/models split (see
// anthropic.Adapter.MatchMetadataRequest): /openai/v1/models is always OpenAI,
// the bare spelling only when the caller speaks neither Anthropic's protocol
// nor Google's, so their keys never reach this upstream.
func (a Adapter) MatchMetadataRequest(r *http.Request) bool {
	return a.MatchMetadataRoute(r.Method, r.URL.Path) &&
		(strings.HasPrefix(r.URL.Path, "/openai/") ||
			r.Header.Get("anthropic-version") == "" && r.Header.Get("x-goog-api-key") == "")
}

func isInputTokenCountEndpoint(endpoint string) bool {
	return strings.HasSuffix(endpoint, "/responses/input_tokens")
}

func (a Adapter) InspectRequest(ctx context.Context, body providers.BodyReader, headers http.Header) (providers.RequestMetadata, error) {
	meta, err := a.Base.InspectRequest(ctx, body, headers)
	if path := headers.Get("x-cave-route-path"); path != "" {
		meta.Endpoint = path
	}
	return meta, err
}
