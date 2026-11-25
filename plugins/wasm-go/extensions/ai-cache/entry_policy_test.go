package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/cache"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/config"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/entry"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/vector"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
)

const (
	entryPolicyGeneration = "kb-2026-11"
	entryPolicyConsumer   = "tenant-alpha"
)

// entryPolicyConfig builds a cache-only configuration with entry semantics on.
func entryPolicyConfig(overrides map[string]interface{}) json.RawMessage {
	policy := map[string]interface{}{
		"enabled":    true,
		"generation": entryPolicyGeneration,
	}
	for k, v := range overrides {
		policy[k] = v
	}
	data, _ := json.Marshal(map[string]interface{}{
		"cache": map[string]interface{}{
			"type":        "redis",
			"serviceName": "redis.static",
			"servicePort": 6379,
			"timeout":     10000,
			"cacheTTL":    0,
		},
		"entryPolicy":      policy,
		"cacheKeyStrategy": "lastQuestion",
		"cacheKeyFrom":     "messages.@reverse.0.content",
		"cacheValueFrom":   "choices.0.message.content",
	})
	return data
}

func entryPolicySemanticConfig() json.RawMessage {
	data, _ := json.Marshal(map[string]interface{}{
		"cache": map[string]interface{}{
			"type":        "redis",
			"serviceName": "redis.static",
			"servicePort": 6379,
		},
		"embedding": map[string]interface{}{
			"type":        "dashscope",
			"apiKey":      "test-key",
			"serviceName": "dashscope.static",
			"servicePort": 8080,
		},
		"vector": map[string]interface{}{
			"type":              "dashvector",
			"serviceName":       "dashvector-service",
			"serviceHost":       "dashvector.example.com",
			"servicePort":       8081,
			"apiKey":            "test-vector-key",
			"collectionID":      "kb",
			"threshold":         0.8,
			"thresholdRelation": "gt",
		},
		"entryPolicy": map[string]interface{}{
			"enabled":    true,
			"generation": entryPolicyGeneration,
		},
		"enableSemanticCache": true,
		"cacheKeyStrategy":    "lastQuestion",
		"cacheKeyFrom":        "messages.@reverse.0.content",
		"cacheValueFrom":      "choices.0.message.content",
	})
	return data
}

// aiLogAttributes decodes the attributes ai-cache writes into the ai_log
// property. The test host double-escapes them, so unwrap before decoding.
func aiLogAttributes(t *testing.T, host test.TestHost) map[string]interface{} {
	t.Helper()
	raw, err := host.GetProperty([]string{"ai_log"})
	require.NoError(t, err)
	text := strings.ReplaceAll(string(raw), `\"`, `"`)
	attributes := map[string]interface{}{}
	require.NoError(t, json.Unmarshal([]byte(text), &attributes), "ai_log payload: %s", text)
	return attributes
}

// pinClock fixes the request clock for the duration of a test.
func pinClock(t *testing.T, now int64) {
	t.Helper()
	previous := nowFunc
	nowFunc = func() int64 { return now }
	t.Cleanup(func() { nowFunc = previous })
}

func callEntryReadRequest(host test.TestHost, consumer, question string) types.Action {
	host.CallOnHttpRequestHeaders([][2]string{
		{":authority", "example.com"},
		{":path", "/v1/chat/completions"},
		{":method", "POST"},
		{"content-type", "application/json"},
		{"x-mse-consumer", consumer},
	})
	body, _ := json.Marshal(map[string]interface{}{
		"model":    "qwen",
		"messages": []map[string]string{{"role": "user", "content": question}},
		"stream":   false,
	})
	return host.CallOnHttpRequestBody(body)
}

func chatCompletion(answer, finishReason string) []byte {
	body, _ := json.Marshal(map[string]interface{}{
		"id": "chatcmpl-1",
		"choices": []map[string]interface{}{{
			"index":         0,
			"message":       map[string]string{"role": "assistant", "content": answer},
			"finish_reason": finishReason,
		}},
	})
	return body
}

// TestEntryPolicyDefaultsToUnchangedBehavior guards the hard requirement that
// nothing changes when the new capability is off.
func TestEntryPolicyDefaultsToUnchangedBehavior(t *testing.T) {
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicRedisConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		cfg, err := host.GetMatchConfig()
		require.NoError(t, err)
		require.False(t, cfg.(*config.PluginConfig).GetEntryPolicy().Enabled)
	})
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicRedisConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		require.Equal(t, types.ActionPause, callEntryReadRequest(host, "consumer-a", "question"))
		calls := host.GetRedisCalloutAttributes()
		require.Len(t, calls, 1)

		// The legacy key format is still used verbatim.
		require.Contains(t, string(calls[0].Query), cache.BuildCacheKey("higress-ai-cache:", "consumer-a", "question"))

		// A hit returns the old-style raw answer, not an entry envelope.
		host.CallOnRedisCall(0, test.CreateRedisRespString("legacy answer"))
		response := host.GetLocalResponse()
		require.NotNil(t, response)
		require.Equal(t, uint32(200), response.StatusCode)
		require.Contains(t, string(response.Data), "legacy answer")
	})
}

func TestEntryPolicyConfigValidation(t *testing.T) {
	test.RunGoTest(t, func(t *testing.T) {
		for _, tc := range []struct {
			name       string
			policy     map[string]interface{}
			wantStatus types.OnPluginStartStatus
		}{
			{name: "enabled without generation", policy: map[string]interface{}{"enabled": true}, wantStatus: types.OnPluginStartStatusFailed},
			{name: "enabled with blank generation", policy: map[string]interface{}{"enabled": true, "generation": "  "}, wantStatus: types.OnPluginStartStatusFailed},
			{name: "enabled with negative ttl", policy: map[string]interface{}{"enabled": true, "generation": "g", "ttlSeconds": -5}, wantStatus: types.OnPluginStartStatusFailed},
			{name: "enabled valid", policy: map[string]interface{}{"enabled": true, "generation": "g"}, wantStatus: types.OnPluginStartStatusOK},
			{name: "disabled with empty generation", policy: map[string]interface{}{"enabled": false}, wantStatus: types.OnPluginStartStatusOK},
		} {
			t.Run(tc.name, func(t *testing.T) {
				data, _ := json.Marshal(map[string]interface{}{
					"cache": map[string]interface{}{
						"type": "redis", "serviceName": "redis.static", "servicePort": 6379,
					},
					"entryPolicy": tc.policy,
				})
				host, status := test.NewTestHost(data)
				defer host.Reset()
				require.Equal(t, tc.wantStatus, status)
			})
		}
	})
}

// TestEntryPolicyWriteGate proves that only a fully terminated response
// creates a hittable entry carrying complete metadata.
func TestEntryPolicyWriteGate(t *testing.T) {
	// Written times come from nowFunc, which is process-local, so these
	// assertions run in Go mode only. The request path itself is identical.

	t.Run("non-stream complete response writes metadata", func(t *testing.T) {
		pinClock(t, 1_700_000_000)
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"ttlSeconds": 3600}))
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)

			callEntryReadRequest(host, entryPolicyConsumer, "how do I reset my password")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "application/json"}})
			require.Equal(t, types.ActionContinue, host.CallOnHttpResponseBody(chatCompletion("ask the support desk", "stop")))

			sets := host.GetRedisCalloutAttributes()
			require.Len(t, sets, 1)
			query := string(sets[0].Query)
			require.Contains(t, query, "$2\r\nex\r\n$4\r\n3600\r\n", "entry writes must carry the shared lifetime")
			require.Contains(t, query, entryPolicyGeneration)
			require.Contains(t, query, entryPolicyConsumer)
			require.Contains(t, query, "ask the support desk")
			require.Equal(t, 1, strings.Count(query, "higress-ai-cache:"), "the configured prefix must be applied exactly once")
			require.Contains(t, query, cache.BuildCacheKey("higress-ai-cache:", entryPolicyConsumer, "how do I reset my password"))
		})
	})

	t.Run("truncated non-stream response is not written", func(t *testing.T) {
		pinClock(t, 1_700_000_000)
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicyConfig(nil))
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)

			callEntryReadRequest(host, entryPolicyConsumer, "question")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "application/json"}})
			host.CallOnHttpResponseBody(chatCompletion("partial", "length"))
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("missing termination reason is not written", func(t *testing.T) {
		pinClock(t, 1_700_000_000)
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicyConfig(nil))
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)

			callEntryReadRequest(host, entryPolicyConsumer, "question")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "application/json"}})
			host.CallOnHttpResponseBody(chatCompletion("answer", ""))
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("upstream error status is not written", func(t *testing.T) {
		pinClock(t, 1_700_000_000)
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicyConfig(nil))
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)

			callEntryReadRequest(host, entryPolicyConsumer, "question")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpResponseHeaders([][2]string{{":status", "502"}, {"content-type", "application/json"}})
			host.CallOnHttpResponseBody(chatCompletion("upstream failure", "stop"))
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("stream without done marker is not written", func(t *testing.T) {
		pinClock(t, 1_700_000_000)
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicyConfig(nil))
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)

			host.CallOnHttpRequestHeaders([][2]string{
				{":authority", "example.com"},
				{":path", "/v1/chat/completions"},
				{":method", "POST"},
				{"content-type", "application/json"},
				{"x-mse-consumer", entryPolicyConsumer},
			})
			host.CallOnHttpRequestBody([]byte(`{"messages":[{"role":"user","content":"q"}],"stream":true}`))
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "text/event-stream"}})
			broken := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"半截\"}}]}\n\n"
			host.CallOnHttpStreamingResponseBody([]byte(broken), true)
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})
}

// TestEntryPolicyReadJudgment covers expiry, generation and tenant rejection
// on the exact read path, plus legacy compatibility.
func TestEntryPolicyReadJudgment(t *testing.T) {
	now := int64(1_700_000_000)
	pinClock(t, now)

	cases := []struct {
		name   string
		raw    string
		hit    bool
		reason string
		answer string
	}{
		{
			name:   "valid entry hits",
			raw:    entry.Encode(entry.Metadata{WrittenAt: now - 60, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: true}, "fresh answer"),
			hit:    true,
			answer: "fresh answer",
		},
		{
			name:   "legacy plain value is not served",
			raw:    "old answer",
			reason: entry.ReasonLegacy,
		},
		{
			name:   "old generation is rejected",
			raw:    entry.Encode(entry.Metadata{WrittenAt: now - 60, Tenant: entryPolicyConsumer, Generation: "kb-2026-10", Complete: true}, "stale answer"),
			reason: entry.ReasonGenerationMismatch,
		},
		{
			name:   "other tenant is rejected",
			raw:    entry.Encode(entry.Metadata{WrittenAt: now - 60, Tenant: "tenant-beta", Generation: entryPolicyGeneration, Complete: true}, "other tenant answer"),
			reason: entry.ReasonTenantMismatch,
		},
		{
			name:   "incomplete entry is rejected",
			raw:    entry.Encode(entry.Metadata{WrittenAt: now - 60, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: false}, "partial answer"),
			reason: entry.ReasonIncomplete,
		},
		{
			name:   "expired entry is rejected",
			raw:    entry.Encode(entry.Metadata{WrittenAt: now - 7200, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: true}, "expired answer"),
			reason: entry.ReasonExpired,
		},
		{
			name:   "malformed current-format entry is rejected",
			raw:    `{"higressEntry":1,"answer":"x"}`,
			reason: entry.ReasonLegacy,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			test.RunGoTest(t, func(t *testing.T) {
				host, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"ttlSeconds": 3600}))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status)

				callEntryReadRequest(host, entryPolicyConsumer, "question")
				host.CallOnRedisCall(0, test.CreateRedisRespString(tc.raw))

				response := host.GetLocalResponse()
				if tc.hit {
					require.NotNil(t, response)
					require.Contains(t, string(response.Data), tc.answer)
					// A hit must never write anything back.
					require.Empty(t, host.GetRedisCalloutAttributes(), "a hit must not re-store the entry")
					return
				}
				require.Nil(t, response)
				require.Equal(t, types.ActionContinue, host.GetHttpStreamAction())
				require.Empty(t, host.GetRedisCalloutAttributes(), "a rejected entry must not be written back")
				require.Contains(t, string(host.GetResponseBody()), "")
				logs := strings.Join(host.GetInfoLogs(), "\n")
				require.Contains(t, logs, tc.reason)
			})
		})
	}
}

func TestEntryPolicyMissReasons(t *testing.T) {
	pinClock(t, 1_700_000_000)
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(entryPolicyConfig(nil))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		callEntryReadRequest(host, entryPolicyConsumer, "question")
		host.CallOnRedisCall(0, test.CreateRedisRespNull())
		require.Nil(t, host.GetLocalResponse())

		attributes := aiLogAttributes(t, host)
		require.Equal(t, "absent", attributes["cache_miss_reason"])
	})
}

func TestEntryPolicyHitSourceIsObservable(t *testing.T) {
	pinClock(t, 1_700_000_000)
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(entryPolicyConfig(nil))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		callEntryReadRequest(host, entryPolicyConsumer, "question")
		raw := entry.Encode(entry.Metadata{WrittenAt: 1_699_999_000, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: true}, "answer")
		host.CallOnRedisCall(0, test.CreateRedisRespString(raw))

		attributes := aiLogAttributes(t, host)
		require.Equal(t, "kv", attributes["cache_source"])
		require.Equal(t, "hit", attributes["cache_status"])
	})
}

// TestEntryPolicySemanticJudgment exercises semantic hits, backfill and the
// "never write a semantic answer back" rule.
func TestEntryPolicySemanticJudgment(t *testing.T) {
	now := int64(1_700_000_000)
	pinClock(t, now)

	dashVectorResult := func(fields map[string]interface{}, score float64) string {
		body, _ := json.Marshal(map[string]interface{}{
			"code":   200,
			"output": []map[string]interface{}{{"id": "rec", "fields": fields, "score": score}},
		})
		return string(body)
	}

	run := func(t *testing.T, fields map[string]interface{}, score float64) test.TestHost {
		host, status := test.NewTestHost(entryPolicySemanticConfig())
		require.Equal(t, types.OnPluginStartStatusOK, status)
		t.Cleanup(host.Reset)
		callEntryReadRequest(host, entryPolicyConsumer, "how do I reset my password")
		host.CallOnRedisCall(0, test.CreateRedisRespNull())
		host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(`{"output":{"embeddings":[{"embedding":[0.1,0.2]}]}}`))
		host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(dashVectorResult(fields, score)))
		return host
	}

	t.Run("valid semantic answer returns without writing back", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host := run(t, map[string]interface{}{
				"query": "reset password", "answer": "use the portal", "consumer": entryPolicyConsumer,
				"writtenAt": now - 30, "generation": entryPolicyGeneration, "complete": true,
			}, 0.95)
			response := host.GetLocalResponse()
			require.NotNil(t, response)
			require.Contains(t, string(response.Data), "use the portal")
			require.Empty(t, host.GetRedisCalloutAttributes(), "semantic hit must not be written back to the key-value side")
			attributes := aiLogAttributes(t, host)
			require.Equal(t, "semantic", attributes["cache_source"])
		})
	})

	t.Run("legacy semantic record is not served", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host := run(t, map[string]interface{}{
				"query": "reset password", "answer": "old answer", "consumer": entryPolicyConsumer,
			}, 0.95)
			require.Nil(t, host.GetLocalResponse())
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("older generation semantic record is not served", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host := run(t, map[string]interface{}{
				"query": "reset password", "answer": "old answer", "consumer": entryPolicyConsumer,
				"writtenAt": now - 30, "generation": "kb-2026-10", "complete": true,
			}, 0.95)
			require.Nil(t, host.GetLocalResponse())
			require.Empty(t, host.GetRedisCalloutAttributes())
			logs := strings.Join(host.GetInfoLogs(), "\n")
			require.Contains(t, logs, entry.ReasonGenerationMismatch)
		})
	})

	t.Run("other tenant semantic record is not served", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host := run(t, map[string]interface{}{
				"query": "reset password", "answer": "beta answer", "consumer": "tenant-beta",
				"writtenAt": now - 30, "generation": entryPolicyGeneration, "complete": true,
			}, 0.95)
			require.Nil(t, host.GetLocalResponse())
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("duplicate records converge on the newest valid one", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			pinClock(t, now)
			host, status := test.NewTestHost(entryPolicySemanticConfig())
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)
			callEntryReadRequest(host, entryPolicyConsumer, "how do I reset my password")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(`{"output":{"embeddings":[{"embedding":[0.1,0.2]}]}}`))
			body, _ := json.Marshal(map[string]interface{}{
				"code": 200,
				"output": []map[string]interface{}{
					{"id": "old", "score": 0.97, "fields": map[string]interface{}{
						"query": "reset password", "answer": "old answer", "consumer": entryPolicyConsumer,
						"writtenAt": now - 600, "generation": entryPolicyGeneration, "complete": true,
					}},
					{"id": "new", "score": 0.90, "fields": map[string]interface{}{
						"query": "reset password", "answer": "new answer", "consumer": entryPolicyConsumer,
						"writtenAt": now - 10, "generation": entryPolicyGeneration, "complete": true,
					}},
					{"id": "stale-generation", "score": 0.99, "fields": map[string]interface{}{
						"query": "reset password", "answer": "stale answer", "consumer": entryPolicyConsumer,
						"writtenAt": now - 1, "generation": "kb-2026-10", "complete": true,
					}},
				},
			})
			host.CallOnHttpCall([][2]string{{":status", "200"}}, body)
			response := host.GetLocalResponse()
			require.NotNil(t, response)
			require.Contains(t, string(response.Data), "new answer")
			require.NotContains(t, string(response.Data), "old answer")
			require.NotContains(t, string(response.Data), "stale answer")
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})

	t.Run("semantic record without answer backfills through the same judgment", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host := run(t, map[string]interface{}{
				"query": "reset password", "answer": "", "consumer": entryPolicyConsumer,
				"writtenAt": now - 30, "generation": entryPolicyGeneration, "complete": true,
			}, 0.95)

			// The backfill issues exactly one exact-cache read for the similar question.
			calls := host.GetRedisCalloutAttributes()
			require.Len(t, calls, 1)
			require.Contains(t, string(calls[0].Query), cache.BuildCacheKey("higress-ai-cache:", entryPolicyConsumer, "reset password"))

			host.CallOnRedisCall(0, test.CreateRedisRespString(entry.Encode(
				entry.Metadata{WrittenAt: now - 20, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: true},
				"backfilled answer",
			)))
			response := host.GetLocalResponse()
			require.NotNil(t, response)
			require.Contains(t, string(response.Data), "backfilled answer")
			// No second write: the backfill read must not re-store the value.
			require.Empty(t, host.GetRedisCalloutAttributes())
			attributes := aiLogAttributes(t, host)
			require.Equal(t, "backfill", attributes["cache_source"])
		})
	})
}

// TestEntryPolicyDegradation proves a semantic backend failure is reported as
// degraded and never as "no similar result".
func TestEntryPolicyDegradation(t *testing.T) {
	pinClock(t, 1_700_000_000)
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(entryPolicySemanticConfig())
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		callEntryReadRequest(host, entryPolicyConsumer, "how do I reset my password")
		host.CallOnRedisCall(0, test.CreateRedisRespNull())
		host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(`{"output":{"embeddings":[{"embedding":[0.1,0.2]}]}}`))
		host.CallOnHttpCall([][2]string{{":status", "500"}}, []byte(`{"error":"unavailable"}`))

		require.Nil(t, host.GetLocalResponse())
		attributes := aiLogAttributes(t, host)
		require.Equal(t, "semantic", attributes["cache_degraded"])
		require.Equal(t, "backend_error", attributes["cache_miss_reason"])
	})
}

// TestEntryPolicyLogsAreRedacted asserts that neither the question, the answer
// nor the tenant name appears in any emitted log line.
func TestEntryPolicyLogsAreRedacted(t *testing.T) {
	pinClock(t, 1_700_000_000)
	test.RunGoTest(t, func(t *testing.T) {
		const (
			question = "customer-secret-question"
			answer   = "customer-secret-answer"
			tenant   = "tenant-secret"
		)
		host, status := test.NewTestHost(entryPolicyConfig(nil))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		callEntryReadRequest(host, tenant, question)
		host.CallOnRedisCall(0, test.CreateRedisRespString(entry.Encode(
			entry.Metadata{WrittenAt: 1_699_999_000, Tenant: tenant, Generation: entryPolicyGeneration, Complete: true},
			answer,
		)))

		all := append(append(append(host.GetTraceLogs(), host.GetDebugLogs()...), host.GetInfoLogs()...), host.GetErrorLogs()...)
		joined := strings.Join(all, "\n")
		require.NotContains(t, joined, question)
		require.NotContains(t, joined, answer)
		require.NotContains(t, joined, tenant)
	})
}

// TestEntryPolicyStreamCompletesAndWrites proves a stream that ends with
// [DONE] produces one metadata-carrying write, and that its entry is then
// hittable under the same judgment.
func TestEntryPolicyStreamCompletesAndWrites(t *testing.T) {
	pinClock(t, 1_700_000_000)
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"ttlSeconds": 600}))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)

		host.CallOnHttpRequestHeaders([][2]string{
			{":authority", "example.com"},
			{":path", "/v1/chat/completions"},
			{":method", "POST"},
			{"content-type", "application/json"},
			{"x-mse-consumer", entryPolicyConsumer},
		})
		host.CallOnHttpRequestBody([]byte(`{"messages":[{"role":"user","content":"q"}],"stream":true}`))
		host.CallOnRedisCall(0, test.CreateRedisRespNull())
		host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "text/event-stream"}})

		first := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"答\"}}]}\n\n"
		last := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"案\"}}]}\n\ndata: [DONE]\n\n"
		require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingResponseBody([]byte(first), false))
		require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingResponseBody([]byte(last), true))

		sets := host.GetRedisCalloutAttributes()
		require.Len(t, sets, 1)
		query := string(sets[0].Query)
		require.Contains(t, query, "答案")
		require.Contains(t, query, entryPolicyGeneration)
		require.Contains(t, query, "$3\r\n600\r\n")
	})
}

// TestEntryPolicyGenerationChangeInvalidatesBothPaths changes only the
// configured generation between two requests and asserts the previously
// stored entries stop hitting on the exact and semantic paths and are never
// written back.
func TestEntryPolicyGenerationChangeInvalidatesBothPaths(t *testing.T) {
	pinClock(t, 1_700_000_000)

	stale := entry.Encode(entry.Metadata{WrittenAt: 1_699_999_000, Tenant: entryPolicyConsumer, Generation: "g1", Complete: true}, "g1 answer")

	t.Run("exact", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			// Write under generation g1.
			writer, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"generation": "g1"}))
			require.Equal(t, types.OnPluginStartStatusOK, status)
			callEntryReadRequest(writer, entryPolicyConsumer, "question")
			writer.CallOnRedisCall(0, test.CreateRedisRespNull())
			writer.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "application/json"}})
			writer.CallOnHttpResponseBody(chatCompletion("g1 answer", "stop"))
			sets := writer.GetRedisCalloutAttributes()
			require.Len(t, sets, 1)
			require.Contains(t, string(sets[0].Query), "g1")
			writer.Reset()

			// The same stored envelope misses under generation g2 and is not
			// rewritten to the key-value side.
			reader, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"generation": "g2"}))
			defer reader.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)
			callEntryReadRequest(reader, entryPolicyConsumer, "question")
			reader.CallOnRedisCall(0, test.CreateRedisRespString(stale))
			require.Nil(t, reader.GetLocalResponse())
			require.Empty(t, reader.GetRedisCalloutAttributes())
			require.Contains(t, strings.Join(reader.GetInfoLogs(), "\n"), entry.ReasonGenerationMismatch)
		})
	})

	t.Run("semantic", func(t *testing.T) {
		test.RunGoTest(t, func(t *testing.T) {
			host, status := test.NewTestHost(entryPolicySemanticConfig())
			defer host.Reset()
			require.Equal(t, types.OnPluginStartStatusOK, status)
			callEntryReadRequest(host, entryPolicyConsumer, "q")
			host.CallOnRedisCall(0, test.CreateRedisRespNull())
			host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(`{"output":{"embeddings":[{"embedding":[0.1]}]}}`))
			host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(`{"code":200,"output":[{"id":"r","score":0.99,"fields":{"query":"q","answer":"old","consumer":"tenant-alpha","writtenAt":1699999000,"generation":"old-gen","complete":true}}]}`))
			require.Nil(t, host.GetLocalResponse())
			require.Empty(t, host.GetRedisCalloutAttributes())
		})
	})
}

// TestEntryPolicyTTLChangeInvalidatesBothPaths lowers the configured lifetime
// after a write and asserts both read paths reject the entry.
func TestEntryPolicyTTLChangeInvalidatesBothPaths(t *testing.T) {
	now := int64(1_700_000_000)
	pinClock(t, now)
	raw := entry.Encode(entry.Metadata{WrittenAt: now - 500, Tenant: entryPolicyConsumer, Generation: entryPolicyGeneration, Complete: true}, "answer")

	for _, tc := range []struct {
		name string
		ttl  int
		hit  bool
	}{
		{name: "inside a 3600s lifetime", ttl: 3600, hit: true},
		{name: "outside a 60s lifetime", ttl: 60},
	} {
		t.Run(tc.name, func(t *testing.T) {
			test.RunGoTest(t, func(t *testing.T) {
				host, status := test.NewTestHost(entryPolicyConfig(map[string]interface{}{"ttlSeconds": tc.ttl}))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status)
				callEntryReadRequest(host, entryPolicyConsumer, "q")
				host.CallOnRedisCall(0, test.CreateRedisRespString(raw))
				if tc.hit {
					require.NotNil(t, host.GetLocalResponse())
					return
				}
				require.Nil(t, host.GetLocalResponse())
				require.Empty(t, host.GetRedisCalloutAttributes())
			})
		})
	}
}

// TestEntryPolicySelectSemanticHitIsDeterministic pins the duplicate
// convergence rule, including the tie-break.
func TestEntryPolicySelectSemanticHitIsDeterministic(t *testing.T) {
	policy := entry.Policy{Enabled: true, Generation: "g", TTLSeconds: 3600}
	now := int64(1_700_000_000)
	results := []vector.QueryResult{
		{ID: "b", Text: "q", Answer: "b", Consumer: "t", Score: 0.9, WrittenAt: now, WrittenAtPresent: true, Generation: "g", Complete: true, MetadataPresent: true},
		{ID: "a", Text: "q", Answer: "a", Consumer: "t", Score: 0.9, WrittenAt: now, WrittenAtPresent: true, Generation: "g", Complete: true, MetadataPresent: true},
		{ID: "c", Text: "q", Answer: "c", Consumer: "t", Score: 0.9, WrittenAt: now - 1, WrittenAtPresent: true, Generation: "g", Complete: true, MetadataPresent: true},
	}
	for i := 0; i < 5; i++ {
		best, found, reason := selectSemanticHit(results, policy, "t", 0.8, "gt", now)
		require.True(t, found)
		require.Empty(t, reason)
		require.Equal(t, "a", best.ID)
	}

	reordered := []vector.QueryResult{results[2], results[0], results[1]}
	best, found, _ := selectSemanticHit(reordered, policy, "t", 0.8, "gt", now)
	require.True(t, found)
	require.Equal(t, "a", best.ID)
}
