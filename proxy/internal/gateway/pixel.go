package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/JuliusBrussee/caveman/engine/pixel"
	"github.com/JuliusBrussee/caveman/proxy/providers"
	anthropicprovider "github.com/JuliusBrussee/caveman/proxy/providers/anthropic"
	"github.com/JuliusBrussee/caveman/shared/platform/redact"
)

const pixelOptimizerID = "pixel-render"

// pixelRequest applies S4 text-to-PNG compression to provider wire formats. It
// gates by measured model allowlist, stores the original before publishing lossy
// bytes, and reports inferred estimates only.
func (s *Server) pixelRequest(body []byte, meta providers.RequestMetadata, transform *providers.TransformResult, requestID string) *compressionOutcome {
	if s.compressor == nil || !pixel.Allowed(meta.Model) {
		return nil
	}

	opts := pixel.DefaultTransformOptions(meta.Model)
	out, info, err := transformPixelLiveZone(meta, body, pixelDecider{s: s, opts: opts})
	if err != nil || len(out) == 0 {
		return nil
	}

	handle, err := s.compressor.StoreOriginal(body)
	if err != nil {
		if s.logger != nil {
			s.logger.Warn("pixel recovery store failed; forwarding original bytes unchanged", "error", redact.Error(err), "request_id", requestID)
		}
		return nil
	}

	transform.Body = out
	transform.OptimizerIDs = append(transform.OptimizerIDs, pixelOptimizerID)
	// A turn that only re-sends earlier renderings earned nothing new: like a
	// substitution-only compress turn, it books an honest zero.
	ratio := 0.0
	if before := info.TextTokensEstimate; before > 0 {
		ratio = float64(before-info.ImageTokensEstimate) / float64(before)
	}
	return &compressionOutcome{
		handle:      handle,
		before:      info.TextTokensEstimate,
		after:       info.ImageTokensEstimate,
		ratio:       ratio,
		bookSavings: false,
	}
}

type pixelReplacement struct {
	span       gatewayJSONSpan
	raw        []byte
	before     int
	after      int
	imageCount int
	imageBytes int
}

// pixelHandle marks a PrefixCache row holding rendered image parts.
const pixelHandle = "pixel"

// pixelFamily is one provider's image wire shape: render turns text into the
// comma-joined image parts that provider reads.
type pixelFamily struct {
	scope  string
	render func(text string, opts pixel.TransformOptions) (parts []byte, before, after, imageBytes, imageCount int, ok bool)
}

var (
	anthropicPixel = pixelFamily{scope: "pixel:anthropic:", render: func(text string, opts pixel.TransformOptions) ([]byte, int, int, int, int, bool) {
		blocks, before, after, imageBytes, ok := anthropicImageBlocks(text, opts)
		return blocks, before, after, imageBytes, bytes.Count(blocks, []byte(`"type":"image"`)), ok
	}}
	openAIChatPixel = pixelFamily{scope: "pixel:openai-chat:", render: func(text string, opts pixel.TransformOptions) ([]byte, int, int, int, int, bool) {
		parts, before, after, imageBytes, ok := openAIImageParts(text, opts)
		return parts, before, after, imageBytes, bytes.Count(parts, []byte(`"type":"image_url"`)), ok
	}}
	openAIResponsesPixel = pixelFamily{scope: "pixel:openai-responses:", render: openAIResponsesImageParts}
)

func bracketed(parts []byte) []byte { return append(append([]byte("["), parts...), ']') }

func asIs(parts []byte) []byte { return parts }

// pixelDecider makes pixel rendering first-decision-wins across turns, the way
// compressRequest does for text. Pixel used to render only the live message
// and forget it, so the next turn re-sent that message as text and busted the
// prefix the provider had just cached with the images in it. Now a text the
// live turn rendered is re-sent as the same image bytes on every later turn,
// and one that went out as text stays text. Without a PrefixCache (embedders)
// it renders the live message only, as before.
type pixelDecider struct {
	s    *Server
	opts pixel.TransformOptions
}

// add decides the JSON string at text. replace is the span its rendering
// replaces and wrap shapes the image parts for that span; only a live text may
// be rendered for the first time.
func (d pixelDecider) add(reps *[]pixelReplacement, body []byte, text, replace gatewayJSONSpan, family pixelFamily, live bool, minChars int, wrap func([]byte) []byte) {
	render := func() (pixelReplacement, bool) {
		value, ok := gatewayDecodeJSONString(body[text.start:text.end])
		if !ok || len(value) < minChars {
			return pixelReplacement{}, false
		}
		parts, before, after, imageBytes, imageCount, ok := family.render(value, d.opts)
		if !ok {
			return pixelReplacement{}, false
		}
		return pixelReplacement{span: replace, raw: parts, before: before, after: after, imageCount: imageCount, imageBytes: imageBytes}, true
	}
	cache := d.s.prefixCache
	if cache == nil {
		if !live {
			return
		}
		if rep, ok := render(); ok {
			rep.raw = wrap(rep.raw)
			*reps = append(*reps, rep)
		}
		return
	}
	// The model is in the scope: geometry is resolved per reader model, and a
	// model switch starts a new provider cache anyway.
	scope := family.scope + d.opts.Model
	original := body[text.start:text.end]
	if d.s.unpersistedRaw.has(scope, original) {
		return
	}
	if stored, handle, hit := cache.LookupReplacement(scope, original); hit {
		if handle != RawDecisionHandle {
			*reps = append(*reps, pixelReplacement{span: replace, raw: wrap(stored)})
		}
		return
	}
	rep, ok := pixelReplacement{}, false
	if live {
		rep, ok = render()
	}
	parts, handle := rep.raw, pixelHandle
	if !ok {
		parts, handle = nil, RawDecisionHandle // going out as text: that is its decision
	}
	stored, err := cache.RememberReplacement(scope, original, parts, handle)
	switch {
	case err != nil:
		d.s.unpersistedRaw.add(scope, original)
	case len(stored) == 0:
	case ok && bytes.Equal(stored, parts):
		rep.raw = wrap(stored) // rendered here: books its saving
		*reps = append(*reps, rep)
	default:
		*reps = append(*reps, pixelReplacement{span: replace, raw: wrap(stored)})
	}
}

func transformPixelLiveZone(meta providers.RequestMetadata, body []byte, d pixelDecider) ([]byte, pixel.TransformInfo, error) {
	// A token count must measure the caller's exact prompt: neither a new
	// rendering nor an earlier one may alter it.
	if strings.Contains(meta.Endpoint, "count_tokens") || strings.HasSuffix(meta.Endpoint, "/input_tokens") {
		return nil, pixel.TransformInfo{Reason: "count_endpoint"}, nil
	}
	switch meta.Provider {
	case "anthropic":
		return transformAnthropicPixelLiveZone(body, d)
	case "openai", "azure_openai", "openai_compatible":
		return transformOpenAIPixelLiveZone(body, d)
	default:
		return nil, pixel.TransformInfo{Reason: "unsupported_live_zone_pixel"}, nil
	}
}

func transformAnthropicPixelLiveZone(body []byte, d pixelDecider) ([]byte, pixel.TransformInfo, error) {
	root, ok := gatewayRootObjectSpan(body)
	if !ok {
		return nil, pixel.TransformInfo{Reason: "parse_error"}, nil
	}
	messagesSpan, ok := gatewayFindObjectField(body, root, "messages")
	if !ok || messagesSpan.start >= len(body) || body[messagesSpan.start] != '[' {
		return nil, pixel.TransformInfo{Reason: "no_messages"}, nil
	}
	messageSpans, ok := gatewayArrayElements(body, messagesSpan)
	if !ok {
		return nil, pixel.TransformInfo{Reason: "parse_error"}, nil
	}
	rawMessages := make([]json.RawMessage, 0, len(messageSpans))
	for _, span := range messageSpans {
		rawMessages = append(rawMessages, append(json.RawMessage(nil), body[span.start:span.end]...))
	}
	floor := anthropicprovider.ComputeFrozenCount(rawMessages)
	target := -1
	for i := len(messageSpans) - 1; i >= floor; i-- {
		if gatewayObjectStringField(body, messageSpans[i], "role") == "user" {
			target = i
			break
		}
	}
	var reps []pixelReplacement
	for i, msg := range messageSpans {
		if gatewayObjectStringField(body, msg, "role") == "user" {
			collectAnthropicPixelCandidates(body, msg, d, i == target, &reps)
		}
	}
	return applyPixelReplacements(body, reps)
}

func collectAnthropicPixelCandidates(body []byte, msg gatewayJSONSpan, d pixelDecider, live bool, reps *[]pixelReplacement) {
	content, ok := gatewayFindObjectField(body, msg, "content")
	if !ok {
		return
	}
	switch {
	case gatewayIsJSONString(body, content):
		d.add(reps, body, content, content, anthropicPixel, live, d.opts.MinCompressChars, bracketed)
	case content.start < content.end && body[content.start] == '[':
		collectAnthropicPixelBlocks(body, content, d, live, reps)
	}
}

func collectAnthropicPixelBlocks(body []byte, blocksSpan gatewayJSONSpan, d pixelDecider, live bool, reps *[]pixelReplacement) {
	blocks, ok := gatewayArrayElements(body, blocksSpan)
	if !ok {
		return
	}
	for _, block := range blocks {
		if block.start >= block.end || body[block.start] != '{' {
			continue
		}
		switch gatewayObjectStringField(body, block, "type") {
		case "text":
			textSpan, ok := gatewayFindObjectField(body, block, "text")
			if !ok || !gatewayIsJSONString(body, textSpan) {
				continue
			}
			d.add(reps, body, textSpan, block, anthropicPixel, live, d.opts.MinCompressChars, asIs)
		case "tool_result":
			content, ok := gatewayFindObjectField(body, block, "content")
			if !ok {
				continue
			}
			switch {
			case gatewayIsJSONString(body, content):
				d.add(reps, body, content, content, anthropicPixel, live, d.opts.MinToolResultChars, bracketed)
			case content.start < content.end && body[content.start] == '[':
				collectAnthropicPixelBlocks(body, content, d, live, reps)
			}
		}
	}
}

func transformOpenAIPixelLiveZone(body []byte, d pixelDecider) ([]byte, pixel.TransformInfo, error) {
	root, ok := gatewayRootObjectSpan(body)
	if !ok {
		return nil, pixel.TransformInfo{Reason: "parse_error"}, nil
	}
	messagesSpan, ok := gatewayFindObjectField(body, root, "messages")
	if ok {
		return transformOpenAIChatPixelLiveZone(body, messagesSpan, d)
	}
	inputSpan, ok := gatewayFindObjectField(body, root, "input")
	if ok {
		return transformOpenAIResponsesPixelLiveZone(body, inputSpan, d)
	}
	return nil, pixel.TransformInfo{Reason: "no_messages_or_input"}, nil
}

func transformOpenAIChatPixelLiveZone(body []byte, messagesSpan gatewayJSONSpan, d pixelDecider) ([]byte, pixel.TransformInfo, error) {
	if messagesSpan.start >= len(body) || body[messagesSpan.start] != '[' {
		return nil, pixel.TransformInfo{Reason: "messages_not_array"}, nil
	}
	messageSpans, ok := gatewayArrayElements(body, messagesSpan)
	if !ok {
		return nil, pixel.TransformInfo{Reason: "parse_error"}, nil
	}
	target := -1
	for i := len(messageSpans) - 1; i >= 0; i-- {
		if gatewayObjectStringField(body, messageSpans[i], "role") == "user" {
			target = i
			break
		}
	}
	if target < 0 {
		return nil, pixel.TransformInfo{Reason: "no_live_user"}, nil
	}
	var reps []pixelReplacement
	for i, msg := range messageSpans {
		if gatewayObjectStringField(body, msg, "role") != "user" {
			continue
		}
		content, ok := gatewayFindObjectField(body, msg, "content")
		if !ok {
			continue
		}
		live := i == target
		switch {
		case gatewayIsJSONString(body, content):
			d.add(&reps, body, content, content, openAIChatPixel, live, d.opts.MinCompressChars, bracketed)
		case content.start < content.end && body[content.start] == '[':
			parts, ok := gatewayArrayElements(body, content)
			if !ok {
				continue
			}
			for _, part := range parts {
				if part.start >= part.end || body[part.start] != '{' {
					continue
				}
				if typ := gatewayObjectStringField(body, part, "type"); typ != "text" && typ != "input_text" {
					continue
				}
				textSpan, ok := gatewayFindObjectField(body, part, "text")
				if !ok || !gatewayIsJSONString(body, textSpan) {
					continue
				}
				d.add(&reps, body, textSpan, part, openAIChatPixel, live, d.opts.MinCompressChars, asIs)
			}
		}
	}
	return applyPixelReplacements(body, reps)
}

func transformOpenAIResponsesPixelLiveZone(body []byte, inputSpan gatewayJSONSpan, d pixelDecider) ([]byte, pixel.TransformInfo, error) {
	if gatewayIsJSONString(body, inputSpan) {
		// A bare string input is one user turn; the next turn sends it back as a
		// message's content string, which shares this decision.
		var reps []pixelReplacement
		d.add(&reps, body, inputSpan, inputSpan, openAIResponsesPixel, true, d.opts.MinCompressChars, func(parts []byte) []byte {
			return append(append([]byte(`[{"type":"message","role":"user","content":[`), parts...), `]}]`...)
		})
		return applyPixelReplacements(body, reps)
	}
	if inputSpan.start >= inputSpan.end || body[inputSpan.start] != '[' {
		return nil, pixel.TransformInfo{Reason: "input_not_string_or_array"}, nil
	}
	items, ok := gatewayArrayElements(body, inputSpan)
	if !ok {
		return nil, pixel.TransformInfo{Reason: "parse_error"}, nil
	}
	latestUser, latestTool := -1, -1
	recoveredCalls := map[string]bool{}
	for i, item := range items {
		if item.start >= item.end || body[item.start] != '{' {
			continue
		}
		typ := gatewayObjectStringField(body, item, "type")
		switch {
		case typ == "function_call_output":
			latestTool = i
		case typ == "function_call":
			if providers.IsRecoveryToolName(gatewayObjectStringField(body, item, "name")) {
				if callID := gatewayObjectStringField(body, item, "call_id"); callID != "" {
					recoveredCalls[callID] = true
				}
			}
		case gatewayObjectStringField(body, item, "role") == "user":
			latestUser = i
		}
	}
	var reps []pixelReplacement
	for i, item := range items {
		if item.start >= item.end || body[item.start] != '{' {
			continue
		}
		if gatewayObjectStringField(body, item, "type") == "function_call_output" {
			if recoveredCalls[gatewayObjectStringField(body, item, "call_id")] {
				continue
			}
			if output, found := gatewayFindObjectField(body, item, "output"); found && gatewayIsJSONString(body, output) {
				d.add(&reps, body, output, output, openAIResponsesPixel, i == latestTool, d.opts.MinToolResultChars, bracketed)
			}
			continue
		}
		if gatewayObjectStringField(body, item, "role") == "user" {
			collectOpenAIResponsesUserPixelCandidates(body, item, d, i == latestUser, &reps)
		}
	}
	return applyPixelReplacements(body, reps)
}

func collectOpenAIResponsesUserPixelCandidates(body []byte, item gatewayJSONSpan, d pixelDecider, live bool, reps *[]pixelReplacement) {
	content, ok := gatewayFindObjectField(body, item, "content")
	if !ok {
		return
	}
	if gatewayIsJSONString(body, content) {
		d.add(reps, body, content, content, openAIResponsesPixel, live, d.opts.MinCompressChars, bracketed)
		return
	}
	if content.start >= content.end || body[content.start] != '[' {
		return
	}
	parts, ok := gatewayArrayElements(body, content)
	if !ok {
		return
	}
	for _, part := range parts {
		typ := gatewayObjectStringField(body, part, "type")
		if typ != "input_text" && typ != "text" {
			continue
		}
		textSpan, found := gatewayFindObjectField(body, part, "text")
		if !found || !gatewayIsJSONString(body, textSpan) {
			continue
		}
		d.add(reps, body, textSpan, part, openAIResponsesPixel, live, d.opts.MinCompressChars, asIs)
	}
}

func anthropicImageBlocks(text string, opts pixel.TransformOptions) ([]byte, int, int, int, bool) {
	images, before, after, imageBytes, ok := renderLiveZonePNGs(text, opts)
	if !ok {
		return nil, 0, 0, 0, false
	}
	blocks := make([]json.RawMessage, 0, len(images))
	for _, img := range images {
		b, _ := json.Marshal(map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": "image/png",
				"data":       base64.StdEncoding.EncodeToString(img.PNG),
			},
		})
		blocks = append(blocks, b)
	}
	return bytes.Join(rawMessagesToBytes(blocks), []byte(",")), before, after, imageBytes, true
}

func openAIImageParts(text string, opts pixel.TransformOptions) ([]byte, int, int, int, bool) {
	images, before, after, imageBytes, ok := renderLiveZonePNGs(text, opts)
	if !ok {
		return nil, 0, 0, 0, false
	}
	parts := make([]json.RawMessage, 0, len(images))
	for _, img := range images {
		b, _ := json.Marshal(map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(img.PNG),
			},
		})
		parts = append(parts, b)
	}
	return bytes.Join(rawMessagesToBytes(parts), []byte(",")), before, after, imageBytes, true
}

func openAIResponsesImageParts(text string, opts pixel.TransformOptions) ([]byte, int, int, int, int, bool) {
	images, before, after, imageBytes, ok := renderLiveZonePNGs(text, opts)
	if !ok {
		return nil, 0, 0, 0, 0, false
	}
	parts := make([]json.RawMessage, 0, len(images))
	for _, img := range images {
		b, _ := json.Marshal(map[string]any{
			"type":      "input_image",
			"image_url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(img.PNG),
			"detail":    "high",
		})
		parts = append(parts, b)
	}
	return bytes.Join(rawMessagesToBytes(parts), []byte(",")), before, after, imageBytes, len(images), true
}

func renderLiveZonePNGs(text string, opts pixel.TransformOptions) ([]pixel.RenderedImage, int, int, int, bool) {
	renderText := pixel.MinifyForRender(text)
	if opts.Reflow {
		if reflowed, ok := pixel.Reflow(renderText); ok {
			renderText = reflowed
		}
	}
	// Resolve density from the request's reader model + CAVE_PIXEL_DENSITY, exactly
	// like the transforms do (default balanced; a falsey env or an unrecognised model
	// fail closed to conservative standard-tier geometry). Without this the density
	// env would be dead ink on the proxy's own live-zone render.
	draw := pixel.ResolveDensityDraw(opts.Model, pixel.DensityFromEnv())
	// A non-mono ink scheme needs its reader note in-image so the model reads the
	// colour convention (zebra) or the two-layer overlay order correctly.
	if note := pixel.DensityInkNote(draw.Zebra, draw.Layers); note != "" {
		renderText = strings.TrimSpace(note) + "\n" + renderText
	}
	cols := pixel.MeasureContentCols(renderText, draw.Cols, 1)
	var images []pixel.RenderedImage
	var err error
	if draw.Layers == 2 {
		images, err = pixel.RenderTextToTwoLayerPNGs(renderText, cols, draw.CharBudget, draw.Style, draw.CanvasH)
	} else {
		images, err = pixel.RenderTextToPNGsWithCharLimit(renderText, cols, draw.CharBudget, draw.Style, draw.CanvasH, "")
	}
	if err != nil || len(images) == 0 {
		return nil, 0, 0, 0, false
	}
	before := max(1, int(float64(len(text))/opts.CharsPerToken))
	// Price each emitted image honestly by its actual pixels under the resolved tier
	// (hi-res canvases cost far more than the old flat 100/image would credit) — the
	// inferred savings must never over-count. Both sides stay `inferred`.
	after := 0
	var imageBytes int
	for _, img := range images {
		after += int(math.Ceil(float64(pixel.AnthropicImageTokens(img.Width, img.Height, draw.Tier)) * pixel.ImageCostSafetyMargin))
		imageBytes += len(img.PNG)
	}
	if after >= before {
		return nil, 0, 0, 0, false
	}
	return images, before, after, imageBytes, true
}

func rawMessagesToBytes(raw []json.RawMessage) [][]byte {
	out := make([][]byte, len(raw))
	for i := range raw {
		out[i] = raw[i]
	}
	return out
}

func applyPixelReplacements(body []byte, reps []pixelReplacement) ([]byte, pixel.TransformInfo, error) {
	sort.Slice(reps, func(i, j int) bool { return reps[i].span.start < reps[j].span.start })
	if len(reps) == 0 {
		return nil, pixel.TransformInfo{Reason: "no_profitable_live_blocks"}, nil
	}
	var out []byte
	last := 0
	info := pixel.TransformInfo{Compressed: true}
	for _, rep := range reps {
		if rep.span.start < last || rep.span.end > len(body) {
			return nil, info, fmt.Errorf("pixel live-zone splice overlap")
		}
		out = append(out, body[last:rep.span.start]...)
		out = append(out, rep.raw...)
		last = rep.span.end
		info.TextTokensEstimate += rep.before
		info.ImageTokensEstimate += rep.after
		info.ImageCount += rep.imageCount
		info.ImageBytes += rep.imageBytes
	}
	out = append(out, body[last:]...)
	if !json.Valid(out) {
		return nil, info, fmt.Errorf("pixel live-zone output invalid JSON")
	}
	return out, info, nil
}

func gatewayObjectStringField(body []byte, obj gatewayJSONSpan, field string) string {
	span, ok := gatewayFindObjectField(body, obj, field)
	if !ok || !gatewayIsJSONString(body, span) {
		return ""
	}
	value, ok := gatewayDecodeJSONString(body[span.start:span.end])
	if !ok {
		return ""
	}
	return value
}

func gatewayIsJSONString(body []byte, span gatewayJSONSpan) bool {
	return span.start < span.end && span.start >= 0 && span.end <= len(body) && body[span.start] == '"'
}

func gatewayDecodeJSONString(raw []byte) (string, bool) {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

func transformPixelBody(provider string, body []byte, opts pixel.TransformOptions) ([]byte, pixel.TransformInfo, error) {
	switch provider {
	case "anthropic", "bedrock":
		return pixel.TransformAnthropic(body, opts)
	case "openai", "azure_openai", "openai_compatible":
		return pixel.TransformOpenAI(body, opts)
	case "gemini":
		return pixel.TransformGemini(body, opts)
	case "vertex":
		shape := sniffVertexPixelShape(body)
		switch shape {
		case "gemini":
			return pixel.TransformGemini(body, opts)
		case "anthropic":
			return pixel.TransformAnthropic(body, opts)
		default:
			return nil, pixel.TransformInfo{}, nil
		}
	default:
		return nil, pixel.TransformInfo{}, nil
	}
}

func sniffVertexPixelShape(body []byte) string {
	var root map[string]json.RawMessage
	if json.Unmarshal(body, &root) != nil {
		return ""
	}
	if _, ok := root["contents"]; ok {
		return "gemini"
	}
	if _, ok := root["messages"]; ok {
		return "anthropic"
	}
	return ""
}
