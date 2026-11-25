package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/cache"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/config"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/entry"
	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/vector"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	logs "github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/resp"
)

// nowFunc is the single clock used for entry write times and validity checks.
// It is a variable so tests can pin time without touching the request path.
var nowFunc = func() int64 { return time.Now().Unix() }

// CheckCacheForKey checks if the key is in the cache, or triggers similarity search if not found.
// source records which read path produced a hit so observability can tell exact,
// semantic and backfill hits apart.
func CheckCacheForKey(key string, ctx wrapper.HttpContext, c config.PluginConfig, log logs.Log, stream bool, useSimilaritySearch bool, source string) error {
	consumer := vector.ConsumerFromContext(ctx)
	if consumer == "" {
		return logAndReturnError(log, "[CheckCacheForKey] authenticated consumer is required")
	}
	policy := c.GetEntryPolicy()
	if policy.Enabled {
		vector.SetEntryMetadataEnabled(ctx)
	}
	activeCacheProvider := c.GetCacheProvider()
	if activeCacheProvider == nil {
		log.Debugf("[%s] [CheckCacheForKey] no cache provider configured, performing similarity search", PLUGIN_NAME)
		return performSimilaritySearch(key, ctx, c, log, key, stream, source)
	}

	queryKey := cache.BuildCacheKey(activeCacheProvider.GetCacheKeyPrefix(), consumer, key)
	log.Debugf("[%s] [CheckCacheForKey] querying cache for question %s", PLUGIN_NAME, questionForLog(c, key))

	err := activeCacheProvider.Get(queryKey, func(response resp.Value) {
		handleCacheResponse(key, response, ctx, log, stream, c, useSimilaritySearch, source)
	})
	if err != nil {
		log.Errorf("[%s] [CheckCacheForKey] failed to retrieve entry for question %s, error: %v", PLUGIN_NAME, questionForLog(c, key), err)
		return err
	}

	return nil
}

// handleCacheResponse processes cache response and handles cache hits and misses.
func handleCacheResponse(key string, response resp.Value, ctx wrapper.HttpContext, log logs.Log, stream bool, c config.PluginConfig, useSimilaritySearch bool, source string) {
	policy := c.GetEntryPolicy()
	consumer := vector.ConsumerFromContext(ctx)
	missReason := ""

	switch {
	case response.Error() != nil:
		log.Errorf("[%s] [handleCacheResponse] cache backend error for question %s: %v", PLUGIN_NAME, questionForLog(c, key), response.Error())
		missReason = entry.ReasonBackendError
		observeDegraded(ctx, c, "kv")
	case response.IsNull():
		missReason = entry.ReasonAbsent
	case policy.Enabled:
		metadata, answer, status := entry.Decode(response.String())
		switch status {
		case entry.DecodeLegacy, entry.DecodeMalformed:
			missReason = entry.ReasonLegacy
			log.Debugf("[%s] [handleCacheResponse] entry for question %s is not in a recognizable format", PLUGIN_NAME, questionForLog(c, key))
		default:
			if strings.TrimSpace(answer) == "" {
				missReason = entry.ReasonIncomplete
				break
			}
			if reason := policy.Check(metadata, consumer, nowFunc()); reason != "" {
				missReason = reason
				log.Debugf("[%s] [handleCacheResponse] entry for question %s rejected: %s", PLUGIN_NAME, questionForLog(c, key), reason)
				break
			}
			processCacheHit(ctx, c, key, answer, stream, source, log)
			return
		}
	default:
		// Behavior is unchanged while the entry policy is off.
		processCacheHit(ctx, c, key, response.String(), stream, source, log)
		return
	}

	log.Infof("[%s] [handleCacheResponse] cache miss for question %s: %s", PLUGIN_NAME, questionForLog(c, key), missReason)
	if useSimilaritySearch && c.EnableSemanticCache {
		if err := performSimilaritySearch(key, ctx, c, log, key, stream, "semantic"); err != nil {
			log.Errorf("[%s] [handleCacheResponse] failed to perform similarity search for question %s, error: %v", PLUGIN_NAME, questionForLog(c, key), err)
			observeDegraded(ctx, c, "semantic")
			resumeAsMiss(ctx, c, log, entry.ReasonBackendError)
		}
		return
	}
	resumeAsMiss(ctx, c, log, missReason)
}

// processCacheHit handles a successful cache hit. It never writes anything
// back: no hit path may lengthen, revive or re-store an entry.
func processCacheHit(ctx wrapper.HttpContext, c config.PluginConfig, key string, response string, stream bool, source string, log logs.Log) {
	if strings.TrimSpace(response) == "" {
		log.Warnf("[%s] [processCacheHit] cached response for question %s is empty", PLUGIN_NAME, questionForLog(c, key))
		resumeAsMiss(ctx, c, log, entry.ReasonIncomplete)
		return
	}

	log.Debugf("[%s] [processCacheHit] cached answer for question %s: %s", PLUGIN_NAME, questionForLog(c, key), answerForLog(c, response))

	// Escape the response to ensure consistent formatting
	escapedResponse := strings.Trim(strconv.Quote(response), "\"")

	ctx.SetContext(CACHE_KEY_CONTEXT_KEY, nil)

	ctx.SetUserAttribute("cache_status", "hit")
	if c.GetEntryPolicy().Enabled {
		ctx.SetUserAttribute("cache_source", source)
	}
	ctx.WriteUserAttributeToLogWithKey(wrapper.AILogKey)

	if stream {
		proxywasm.SendHttpResponseWithDetail(200, "ai-cache.hit", [][2]string{{"content-type", "text/event-stream; charset=utf-8"}}, []byte(fmt.Sprintf(c.StreamResponseTemplate, escapedResponse)), -1)
	} else {
		proxywasm.SendHttpResponseWithDetail(200, "ai-cache.hit", [][2]string{{"content-type", "application/json; charset=utf-8"}}, []byte(fmt.Sprintf(c.ResponseTemplate, escapedResponse)), -1)
	}
}

// performSimilaritySearch determines the appropriate similarity search method to use.
func performSimilaritySearch(key string, ctx wrapper.HttpContext, c config.PluginConfig, log logs.Log, queryString string, stream bool, source string) error {
	if vector.ConsumerFromContext(ctx) == "" {
		return logAndReturnError(log, "[performSimilaritySearch] authenticated consumer is required")
	}
	activeVectorProvider := c.GetVectorProvider()
	if activeVectorProvider == nil {
		return logAndReturnError(log, "[performSimilaritySearch] no vector provider configured for similarity search")
	}
	if c.GetEntryPolicy().Enabled {
		vector.SetEntryMetadataEnabled(ctx)
	}

	// Check if the active vector provider implements the StringQuerier interface.
	if _, ok := activeVectorProvider.(vector.StringQuerier); ok {
		log.Debugf("[%s] [performSimilaritySearch] active vector provider implements StringQuerier interface, performing string query", PLUGIN_NAME)
		return performStringQuery(key, queryString, ctx, c, log, stream, source)
	}

	// Check if the active vector provider implements the EmbeddingQuerier interface.
	if _, ok := activeVectorProvider.(vector.EmbeddingQuerier); ok {
		log.Debugf("[%s] [performSimilaritySearch] active vector provider implements EmbeddingQuerier interface, performing embedding query", PLUGIN_NAME)
		return performEmbeddingQuery(key, ctx, c, log, stream, source)
	}

	return logAndReturnError(log, "[performSimilaritySearch] no suitable querier or embedding provider available for similarity search")
}

// performStringQuery executes the string-based similarity search.
func performStringQuery(key string, queryString string, ctx wrapper.HttpContext, c config.PluginConfig, log logs.Log, stream bool, _ string) error {
	stringQuerier, ok := c.GetVectorProvider().(vector.StringQuerier)
	if !ok {
		return logAndReturnError(log, "[performStringQuery] active vector provider does not implement StringQuerier interface")
	}

	return stringQuerier.QueryString(queryString, ctx, log, func(results []vector.QueryResult, ctx wrapper.HttpContext, log logs.Log, err error) {
		handleQueryResults(key, results, ctx, log, stream, c, err)
	})
}

// performEmbeddingQuery executes the embedding-based similarity search.
func performEmbeddingQuery(key string, ctx wrapper.HttpContext, c config.PluginConfig, log logs.Log, stream bool, _ string) error {
	embeddingQuerier, ok := c.GetVectorProvider().(vector.EmbeddingQuerier)
	if !ok {
		return logAndReturnError(log, fmt.Sprintf("[performEmbeddingQuery] active vector provider does not implement EmbeddingQuerier interface"))
	}

	activeEmbeddingProvider := c.GetEmbeddingProvider()
	if activeEmbeddingProvider == nil {
		return logAndReturnError(log, fmt.Sprintf("[performEmbeddingQuery] no embedding provider configured for similarity search"))
	}

	return activeEmbeddingProvider.GetEmbedding(key, ctx, func(textEmbedding []float64, err error) {
		log.Debugf("[%s] [performEmbeddingQuery] GetEmbedding completed, length of embedding: %d, hasError: %v", PLUGIN_NAME, len(textEmbedding), err != nil)
		if err != nil {
			handleInternalError(ctx, c, err, fmt.Sprintf("[%s] [performEmbeddingQuery] error getting embedding for question %s", PLUGIN_NAME, questionForLog(c, key)), log)
			return
		}
		ctx.SetContext(CACHE_KEY_EMBEDDING_KEY, textEmbedding)

		err = embeddingQuerier.QueryEmbedding(textEmbedding, ctx, log, func(results []vector.QueryResult, ctx wrapper.HttpContext, log logs.Log, err error) {
			handleQueryResults(key, results, ctx, log, stream, c, err)
		})
		if err != nil {
			handleInternalError(ctx, c, err, fmt.Sprintf("[%s] [performEmbeddingQuery] error querying vector database for question %s", PLUGIN_NAME, questionForLog(c, key)), log)
		}
	})
}

func firstResultForConsumer(results []vector.QueryResult, consumer string) (vector.QueryResult, bool) {
	if consumer == "" {
		return vector.QueryResult{}, false
	}
	for _, result := range results {
		if result.Consumer == consumer {
			return result, true
		}
	}
	return vector.QueryResult{}, false
}

// handleQueryResults processes the results of similarity search and determines next actions.
func handleQueryResults(key string, results []vector.QueryResult, ctx wrapper.HttpContext, log logs.Log, stream bool, c config.PluginConfig, err error) {
	policy := c.GetEntryPolicy()
	if err != nil {
		// A backend failure is a degraded read, never "no similar result".
		observeDegraded(ctx, c, "semantic")
		handleInternalError(ctx, c, err, fmt.Sprintf("[%s] [handleQueryResults] error querying vector database for question %s", PLUGIN_NAME, questionForLog(c, key)), log)
		return
	}

	if len(results) == 0 {
		log.Warnf("[%s] [handleQueryResults] no similar keys found for question %s", PLUGIN_NAME, questionForLog(c, key))
		resumeAsMiss(ctx, c, log, entry.ReasonAbsent)
		return
	}

	consumer := vector.ConsumerFromContext(ctx)
	if policy.Enabled {
		handlePolicyQueryResults(key, results, ctx, log, stream, c, consumer)
		return
	}

	// Existing behavior when the entry policy is off.
	mostSimilarData, ok := firstResultForConsumer(results, consumer)
	if !ok {
		log.Infof("[%s] [handleQueryResults] no result has matching consumer metadata", PLUGIN_NAME)
		resumeAsMiss(ctx, c, log, entry.ReasonTenantMismatch)
		return
	}

	log.Debugf("[%s] [handleQueryResults] for question %s, the most similar key found: %s with score: %f", PLUGIN_NAME, questionForLog(c, key), questionForLog(c, mostSimilarData.Text), mostSimilarData.Score)
	simThreshold := c.GetVectorProviderConfig().Threshold
	simThresholdRelation := c.GetVectorProviderConfig().ThresholdRelation
	if compare(simThresholdRelation, mostSimilarData.Score, simThreshold) {
		log.Infof("[%s] key accepted: %s with score: %f", PLUGIN_NAME, questionForLog(c, mostSimilarData.Text), mostSimilarData.Score)
		if mostSimilarData.Answer != "" {
			// direct return the answer if available
			cacheResponse(ctx, c, key, mostSimilarData.Answer, log)
			processCacheHit(ctx, c, key, mostSimilarData.Answer, stream, entry.SourceSemantic, log)
		} else {
			if c.GetCacheProvider() != nil {
				_ = CheckCacheForKey(mostSimilarData.Text, ctx, c, log, stream, false, entry.SourceBackfill)
			} else {
				// Otherwise, do not check the cache, directly return
				log.Infof("[%s] cache hit for question %s, but no corresponding answer found in the vector database", PLUGIN_NAME, questionForLog(c, mostSimilarData.Text))
				resumeAsMiss(ctx, c, log, entry.ReasonAbsent)
			}
		}
	} else {
		log.Infof("[%s] score not meet the threshold %f: %s with score %f", PLUGIN_NAME, simThreshold, questionForLog(c, mostSimilarData.Text), mostSimilarData.Score)
		resumeAsMiss(ctx, c, log, entry.ReasonAbsent)
	}
}

// handlePolicyQueryResults applies the shared entry judgment to every semantic
// candidate, keeps the newest valid one and never writes a candidate back.
func handlePolicyQueryResults(key string, results []vector.QueryResult, ctx wrapper.HttpContext, log logs.Log, stream bool, c config.PluginConfig, consumer string) {
	policy := c.GetEntryPolicy()
	simThreshold := c.GetVectorProviderConfig().Threshold
	simThresholdRelation := c.GetVectorProviderConfig().ThresholdRelation

	best, found, reason := selectSemanticHit(results, policy, consumer, simThreshold, simThresholdRelation, nowFunc())
	if !found {
		if reason == "" {
			reason = entry.ReasonAbsent
		}
		log.Infof("[%s] [handleQueryResults] no usable semantic result for question %s: %s", PLUGIN_NAME, questionForLog(c, key), reason)
		resumeAsMiss(ctx, c, log, reason)
		return
	}

	log.Debugf("[%s] [handleQueryResults] accepted semantic entry for question %s with score: %f", PLUGIN_NAME, questionForLog(c, key), best.Score)

	if strings.TrimSpace(best.Answer) == "" {
		// The record carries no answer, so the only way to serve it is the
		// key-value backfill. That read passes through the same judgment.
		if c.GetCacheProvider() != nil {
			_ = CheckCacheForKey(best.Text, ctx, c, log, stream, false, entry.SourceBackfill)
			return
		}
		log.Infof("[%s] [handleQueryResults] semantic entry for question %s has no answer", PLUGIN_NAME, questionForLog(c, key))
		resumeAsMiss(ctx, c, log, entry.ReasonAbsent)
		return
	}

	processCacheHit(ctx, c, key, best.Answer, stream, entry.SourceSemantic, log)
}

// selectSemanticHit returns the newest valid candidate. Ties are broken
// deterministically so repeated calls with the same input agree.
func selectSemanticHit(results []vector.QueryResult, policy entry.Policy, consumer string, threshold float64, relation string, now int64) (vector.QueryResult, bool, string) {
	var (
		best   vector.QueryResult
		found  bool
		reason string
	)
	for _, result := range results {
		// Threshold eligibility is evaluated first so a semantically irrelevant
		// candidate can never contribute a tenant/generation reason to the miss.
		if !compare(relation, result.Score, threshold) {
			continue
		}
		if result.Consumer != consumer {
			reason = preferMissReason(reason, entry.ReasonTenantMismatch)
			continue
		}
		metadata, ok := result.EntryMetadata()
		if !ok {
			reason = preferMissReason(reason, entry.ReasonLegacy)
			continue
		}
		if rejected := policy.Check(metadata, consumer, now); rejected != "" {
			reason = preferMissReason(reason, rejected)
			continue
		}
		if !found || newerSemanticResult(result, best) {
			best = result
			found = true
		}
	}
	return best, found, reason
}

// newerSemanticResult defines the deterministic "newest wins" ordering.
func newerSemanticResult(candidate, current vector.QueryResult) bool {
	if candidate.WrittenAt != current.WrittenAt {
		return candidate.WrittenAt > current.WrittenAt
	}
	candidateKey := candidate.ID + "\x00" + candidate.Text + "\x00" + candidate.Answer
	currentKey := current.ID + "\x00" + current.Text + "\x00" + current.Answer
	return candidateKey < currentKey
}

// missReasonPriority lets one deterministic reason describe a miss even when
// several candidates were rejected for different reasons.
func preferMissReason(current, candidate string) string {
	priority := func(reason string) int {
		switch reason {
		case entry.ReasonTenantMismatch:
			return 5
		case entry.ReasonGenerationMismatch:
			return 4
		case entry.ReasonIncomplete:
			return 3
		case entry.ReasonExpired:
			return 2
		case entry.ReasonLegacy:
			return 1
		default:
			return 0
		}
	}
	if priority(candidate) > priority(current) {
		return candidate
	}
	return current
}

// logAndReturnError logs an error and returns it.
func logAndReturnError(log logs.Log, message string) error {
	message = fmt.Sprintf("[%s] %s", PLUGIN_NAME, message)
	log.Errorf(message)
	return errors.New(message)
}

// handleInternalError logs an error and resumes the HTTP request.
func handleInternalError(ctx wrapper.HttpContext, c config.PluginConfig, err error, message string, log logs.Log) {
	if err != nil {
		log.Errorf("[%s] [handleInternalError] %s: %v", PLUGIN_NAME, message, err)
	} else {
		log.Errorf("[%s] [handleInternalError] %s", PLUGIN_NAME, message)
	}
	// The caller always continues to the upstream; backend errors are never
	// surfaced as a plugin response.
	resumeAsMiss(ctx, c, log, entry.ReasonBackendError)
}

// resumeAsMiss records why the read missed and lets the request continue.
func resumeAsMiss(ctx wrapper.HttpContext, c config.PluginConfig, log logs.Log, reason string) {
	if c.GetEntryPolicy().Enabled {
		ctx.SetUserAttribute("cache_status", "miss")
		ctx.SetUserAttribute("cache_miss_reason", reason)
		ctx.WriteUserAttributeToLogWithKey(wrapper.AILogKey)
	}
	proxywasm.ResumeHttpRequest()
}

// observeDegraded marks a partial backend failure so it is never mistaken for
// "no similar result".
func observeDegraded(ctx wrapper.HttpContext, c config.PluginConfig, side string) {
	if !c.GetEntryPolicy().Enabled {
		return
	}
	ctx.SetUserAttribute("cache_degraded", side)
}

// Caches the response value
func cacheResponse(ctx wrapper.HttpContext, c config.PluginConfig, key string, value string, log logs.Log) {
	consumer := vector.ConsumerFromContext(ctx)
	if consumer == "" {
		return
	}
	if strings.TrimSpace(value) == "" {
		log.Warnf("[%s] [cacheResponse] cached value for question %s is empty", PLUGIN_NAME, questionForLog(c, key))
		return
	}

	activeCacheProvider := c.GetCacheProvider()
	if activeCacheProvider == nil {
		return
	}
	queryKey := cache.BuildCacheKey(activeCacheProvider.GetCacheKeyPrefix(), consumer, key)

	if !c.GetEntryPolicy().Enabled {
		_ = activeCacheProvider.Set(queryKey, value, nil)
		if cache.PayloadLoggingOff() {
			log.Debugf("[%s] [cacheResponse] cache set success, length of value: %d", PLUGIN_NAME, len(value))
		} else {
			log.Debugf("[%s] [cacheResponse] cache set success, value: %s", PLUGIN_NAME, value)
		}
		return
	}

	metadata, ok := ctx.GetContext(ENTRY_METADATA_CONTEXT_KEY).(entry.Metadata)
	if !ok {
		log.Warnf("[%s] [cacheResponse] refusing to write without entry metadata", PLUGIN_NAME)
		return
	}
	if !c.GetEntryPolicy().WritableHolds(metadata, consumer, nowFunc()) {
		log.Infof("[%s] [cacheResponse] write rejected: entry is not a complete, current-generation result", PLUGIN_NAME)
		return
	}
	ttl := c.GetEntryPolicy().Remaining(metadata, nowFunc())
	if err := activeCacheProvider.SetWithTTL(queryKey, entry.Encode(metadata, value), ttl, nil); err != nil {
		log.Warnf("[%s] [cacheResponse] cache set failed, length of value: %d, error: %v", PLUGIN_NAME, len(value), err)
		return
	}
	log.Debugf("[%s] [cacheResponse] cache set success, length of value: %d", PLUGIN_NAME, len(value))
}

// Handles embedding upload if available
func uploadEmbeddingAndAnswer(ctx wrapper.HttpContext, c config.PluginConfig, key string, value string, log logs.Log) {
	if vector.ConsumerFromContext(ctx) == "" {
		return
	}
	embedding := ctx.GetContext(CACHE_KEY_EMBEDDING_KEY)
	if embedding == nil {
		return
	}

	emb, ok := embedding.([]float64)
	if !ok {
		log.Errorf("[%s] [uploadEmbeddingAndAnswer] embedding is not of expected type []float64", PLUGIN_NAME)
		return
	}

	activeVectorProvider := c.GetVectorProvider()
	if activeVectorProvider == nil {
		log.Debugf("[%s] [uploadEmbeddingAndAnswer] no vector provider configured for uploading embedding", PLUGIN_NAME)
		return
	}

	if c.GetEntryPolicy().Enabled {
		metadata, ok := ctx.GetContext(ENTRY_METADATA_CONTEXT_KEY).(entry.Metadata)
		if !ok {
			log.Warnf("[%s] [uploadEmbeddingAndAnswer] refusing to upload without entry metadata", PLUGIN_NAME)
			return
		}
		if !c.GetEntryPolicy().WritableHolds(metadata, vector.ConsumerFromContext(ctx), nowFunc()) {
			log.Infof("[%s] [uploadEmbeddingAndAnswer] write rejected: entry is not a complete, current-generation result", PLUGIN_NAME)
			return
		}
		vector.SetEntryMetadataEnabled(ctx)
		vector.SetEntryMetadata(ctx, metadata)
	}

	onUploadComplete := func(_ wrapper.HttpContext, log logs.Log, err error) {
		if err != nil {
			log.Warnf("[%s] [uploadEmbeddingAndAnswer] asynchronous embedding upload failed: %v", PLUGIN_NAME, err)
		}
	}

	// Attempt to upload answer embedding first
	if ansEmbUploader, ok := activeVectorProvider.(vector.AnswerAndEmbeddingUploader); ok {
		log.Infof("[%s] uploading answer embedding for question %s", PLUGIN_NAME, questionForLog(c, key))
		err := ansEmbUploader.UploadAnswerAndEmbedding(key, emb, value, ctx, log, onUploadComplete)
		if err != nil {
			log.Warnf("[%s] [uploadEmbeddingAndAnswer] failed to upload answer embedding for question %s, error: %v", PLUGIN_NAME, questionForLog(c, key), err)
		} else {
			return // If successful, return early
		}
	}

	// If answer embedding upload fails, attempt normal embedding upload
	if embUploader, ok := activeVectorProvider.(vector.EmbeddingUploader); ok {
		log.Infof("[%s] uploading embedding for question %s", PLUGIN_NAME, questionForLog(c, key))
		err := embUploader.UploadEmbedding(key, emb, ctx, log, onUploadComplete)
		if err != nil {
			log.Warnf("[%s] [uploadEmbeddingAndAnswer] failed to upload embedding for question %s, error: %v", PLUGIN_NAME, questionForLog(c, key), err)
		}
	}
}

// 主要用于相似度/距离/点积判断
// 余弦相似度度量的是两个向量在方向上的相似程度。相似度越高，两个向量越接近。
// 距离度量的是两个向量在空间上的远近程度。距离越小，两个向量越接近。
// compare 函数根据操作符进行判断并返回结果
func compare(operator string, value1 float64, value2 float64) bool {
	switch operator {
	case "gt":
		return value1 > value2
	case "gte":
		return value1 >= value2
	case "lt":
		return value1 < value2
	case "lte":
		return value1 <= value2
	default:
		return false
	}
}

// questionForLog returns a loggable identifier. When the entry policy is on,
// the question never appears in plaintext.
func questionForLog(c config.PluginConfig, question string) string {
	if c.GetEntryPolicy().Enabled {
		return entry.Redact(question)
	}
	return question
}

// answerForLog returns a loggable identifier for an answer.
func answerForLog(c config.PluginConfig, answer string) string {
	if c.GetEntryPolicy().Enabled {
		return entry.Redact(answer)
	}
	return answer
}
