package gateway

import (
	"errors"
	"strings"

	"github.com/klauspost/compress/zstd"
)

type chatGPTRequestEncoding string

const (
	chatGPTRequestIdentity chatGPTRequestEncoding = ""
	chatGPTRequestZstd     chatGPTRequestEncoding = "zstd"
)

var (
	// errChatGPTBodyOverLimit: the body decodes past chatGPTCaptureLimit, the
	// same logical cap an identity body on this route gets. ponytail: one cap
	// for both; raise them together if large Codex contexts need compressing.
	errChatGPTBodyOverLimit = errors.New("decoded request body exceeds the capture limit")
	errChatGPTUndecodable   = errors.New("request content encoding cannot be decoded")
)

func decodeChatGPTRequestBody(wire []byte, contentEncoding string) ([]byte, chatGPTRequestEncoding, error) {
	switch encoding := chatGPTRequestEncoding(strings.ToLower(strings.TrimSpace(contentEncoding))); encoding {
	case chatGPTRequestIdentity, "identity":
		return wire, chatGPTRequestIdentity, nil
	case chatGPTRequestZstd:
		decoder, err := zstd.NewReader(
			nil,
			zstd.WithDecoderConcurrency(1),
			zstd.WithDecoderMaxWindow(uint64(chatGPTCaptureLimit*2)),
			zstd.WithDecoderMaxMemory(uint64(chatGPTCaptureLimit*8)),
		)
		if err != nil {
			return nil, encoding, errChatGPTUndecodable
		}
		defer decoder.Close()
		decoded, err := decoder.DecodeAll(wire, nil)
		switch {
		case errors.Is(err, zstd.ErrDecoderSizeExceeded) || (err == nil && len(decoded) > chatGPTCaptureLimit):
			return nil, encoding, errChatGPTBodyOverLimit
		case err != nil:
			return nil, encoding, errChatGPTUndecodable
		}
		return decoded, encoding, nil
	default:
		return nil, encoding, errChatGPTUndecodable
	}
}

func encodeChatGPTRequestBody(logical []byte, encoding chatGPTRequestEncoding) ([]byte, bool) {
	switch encoding {
	case chatGPTRequestIdentity:
		return logical, true
	case chatGPTRequestZstd:
		encoder, err := zstd.NewWriter(
			nil,
			zstd.WithEncoderConcurrency(1),
			zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(3)),
		)
		if err != nil {
			return nil, false
		}
		defer encoder.Close()
		return encoder.EncodeAll(logical, nil), true
	default:
		return nil, false
	}
}
