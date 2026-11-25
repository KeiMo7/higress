package vector

import (
	"errors"
	"strconv"
	"strings"

	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/entry"
)

const (
	PROVIDER_TYPE_DASH_VECTOR = "dashvector"
	PROVIDER_TYPE_CHROMA      = "chroma"
	PROVIDER_TYPE_ES          = "elasticsearch"
	PROVIDER_TYPE_WEAVIATE    = "weaviate"
	PROVIDER_TYPE_PINECONE    = "pinecone"
	PROVIDER_TYPE_QDRANT      = "qdrant"
	PROVIDER_TYPE_MILVUS      = "milvus"

	ConsumerContextKey = "aiCacheAuthenticatedConsumer"
	TenantField        = "consumer"

	// Entry metadata fields stored alongside a vector record. They mirror the
	// key-value entry metadata so both sides obey the same judgment.
	WrittenAtField  = "writtenAt"
	GenerationField = "generation"
	CompleteField   = "complete"

	entryMetadataContextKey        = "aiCacheEntryMetadata"
	entryMetadataEnabledContextKey = "aiCacheEntryMetadataEnabled"
)

// SetEntryMetadataEnabled records that the reader/writer should carry entry
// metadata. When unset, providers keep their previous payload and query shape
// so existing schemas and behavior are untouched.
func SetEntryMetadataEnabled(ctx wrapper.HttpContext) {
	if ctx != nil {
		ctx.SetContext(entryMetadataEnabledContextKey, true)
	}
}

// EntryMetadataEnabled reports whether entry metadata must be carried.
func EntryMetadataEnabled(ctx wrapper.HttpContext) bool {
	if ctx == nil {
		return false
	}
	enabled, _ := ctx.GetContext(entryMetadataEnabledContextKey).(bool)
	return enabled
}

// SetEntryMetadata records the metadata to attach to the next upload.
func SetEntryMetadata(ctx wrapper.HttpContext, m entry.Metadata) {
	if ctx != nil {
		ctx.SetContext(entryMetadataContextKey, m)
	}
}

// EntryMetadataFromContext returns the metadata recorded for the next upload.
func EntryMetadataFromContext(ctx wrapper.HttpContext) (entry.Metadata, bool) {
	if ctx == nil {
		return entry.Metadata{}, false
	}
	m, ok := ctx.GetContext(entryMetadataContextKey).(entry.Metadata)
	return m, ok
}

// ConsumerFromContext returns the consumer recorded after consumer
// authentication. Ai-cache must run after that authentication stage.
func ConsumerFromContext(ctx wrapper.HttpContext) string {
	if ctx == nil {
		return ""
	}
	consumer, _ := ctx.GetContext(ConsumerContextKey).(string)
	return consumer
}

type providerInitializer interface {
	ValidateConfig(ProviderConfig) error
	CreateProvider(ProviderConfig) (Provider, error)
}

var (
	ErrConsumerRequired = errors.New("authenticated consumer is required")

	providerInitializers = map[string]providerInitializer{
		PROVIDER_TYPE_DASH_VECTOR: &dashVectorProviderInitializer{},
		PROVIDER_TYPE_CHROMA:      &chromaProviderInitializer{},
		PROVIDER_TYPE_ES:          &esProviderInitializer{},
		PROVIDER_TYPE_WEAVIATE:    &weaviateProviderInitializer{},
		PROVIDER_TYPE_PINECONE:    &pineconeProviderInitializer{},
		PROVIDER_TYPE_QDRANT:      &qdrantProviderInitializer{},
		PROVIDER_TYPE_MILVUS:      &milvusProviderInitializer{},
	}
)

// QueryResult 定义通用的查询结果的结构体
type QueryResult struct {
	Text      string    // 相似的文本
	Embedding []float64 // 相似文本的向量
	Score     float64   // 文本的向量相似度或距离等度量
	Answer    string    // 相似文本对应的LLM生成的回答
	Consumer  string    // 上传时记录的认证 consumer
	WrittenAt int64     // 上传时记录的写入时刻（秒）
	// WrittenAtPresent distinguishes an absent write time from a stored zero.
	WrittenAtPresent bool
	Generation       string
	Complete         bool
	// MetadataPresent records that the record carried the current-format entry
	// metadata. A record without it is a legacy record and must not be served.
	MetadataPresent bool
	// ID is the backend record identifier, used only for logging.
	ID string
}

// EntryMetadata derives entry metadata from a query result.
func (r QueryResult) EntryMetadata() (entry.Metadata, bool) {
	if !r.MetadataPresent || !r.WrittenAtPresent {
		return entry.Metadata{}, false
	}
	return entry.Metadata{
		WrittenAt:  r.WrittenAt,
		Tenant:     r.Consumer,
		Generation: r.Generation,
		Complete:   r.Complete,
	}, true
}

// parseInt64Value accepts a JSON number or a numeric string, because vector
// backends differ on how they store numeric payload fields.
func parseInt64Value(value gjson.Result) (int64, bool) {
	if !value.Exists() {
		return 0, false
	}
	switch value.Type {
	case gjson.Number:
		return value.Int(), true
	case gjson.String:
		parsed, err := strconv.ParseInt(strings.TrimSpace(value.String()), 10, 64)
		if err != nil {
			return 0, false
		}
		return parsed, true
	default:
		return 0, false
	}
}

// parseBoolValue accepts a JSON boolean or its string form.
func parseBoolValue(value gjson.Result) (bool, bool) {
	if !value.Exists() {
		return false, false
	}
	switch value.Type {
	case gjson.True:
		return true, true
	case gjson.False:
		return false, true
	case gjson.String:
		switch strings.TrimSpace(value.String()) {
		case "true":
			return true, true
		case "false":
			return false, true
		}
	}
	return false, false
}

// applyMetadata fills the entry metadata of a result. MetadataPresent stays
// false unless every metadata field is readable, so a partial record is
// recognized as legacy instead of being guessed into a valid entry. The record
// ID is only retained for current-format records so legacy results keep their
// previous shape.
func applyMetadata(result *QueryResult, id string, writtenAt, generation, complete gjson.Result) {
	written, writtenOK := parseInt64Value(writtenAt)
	done, doneOK := parseBoolValue(complete)
	if !writtenOK || !doneOK || !generation.Exists() || generation.Type != gjson.String {
		return
	}
	result.WrittenAt = written
	result.WrittenAtPresent = true
	result.Generation = generation.String()
	result.Complete = done
	result.MetadataPresent = true
	result.ID = id
}

// applyMapMetadata is the map-shaped variant used by providers whose response
// decodes into generic JSON objects. It applies exactly the same all-or-nothing
// rule as applyMetadata.
func applyMapMetadata(result *QueryResult, id string, fields map[string]interface{}) {
	written, writtenOK := parseInt64Any(fields[WrittenAtField])
	generation, generationOK := fields[GenerationField].(string)
	done, doneOK := parseBoolAny(fields[CompleteField])
	if !writtenOK || !generationOK || !doneOK {
		return
	}
	result.WrittenAt = written
	result.WrittenAtPresent = true
	result.Generation = generation
	result.Complete = done
	result.MetadataPresent = true
	result.ID = id
}

func parseInt64Any(value interface{}) (int64, bool) {
	switch v := value.(type) {
	case float64:
		return int64(v), true
	case int64:
		return v, true
	case int:
		return int64(v), true
	case string:
		parsed, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return 0, false
		}
		return parsed, true
	default:
		return 0, false
	}
}

func parseBoolAny(value interface{}) (bool, bool) {
	switch v := value.(type) {
	case bool:
		return v, true
	case string:
		switch strings.TrimSpace(v) {
		case "true":
			return true, true
		case "false":
			return false, true
		}
	}
	return false, false
}

// metadataPayload returns the fields to attach to an upload. It is empty when
// the new entry semantics are off.
func metadataPayload(ctx wrapper.HttpContext) (entry.Metadata, bool) {
	if !EntryMetadataEnabled(ctx) {
		return entry.Metadata{}, false
	}
	return EntryMetadataFromContext(ctx)
}

// payloadLoggingOff is set when entry semantics are enabled. Backend payloads
// then contain questions and answers, so they must never reach the log. The
// value is fixed at config load and read-only afterwards.
var payloadLoggingOff bool

// SetPayloadLoggingOff disables raw payload logging. Config load calls it once
// when entryPolicy.enabled is true.
func SetPayloadLoggingOff(off bool) {
	payloadLoggingOff = off
}

// LogPayload logs a backend payload with a non-reversible size marker when
// payload logging is disabled.
func LogPayload(log log.Log, ctx wrapper.HttpContext, prefix string, status int, body []byte) {
	if log == nil {
		return
	}
	if payloadLoggingOff || EntryMetadataEnabled(ctx) {
		log.Debugf("%s: status=%d bodyBytes=%d", prefix, status, len(body))
		return
	}
	log.Debugf("%s: %d, %s", prefix, status, body)
}

// LogPayloadError logs a malformed-payload diagnosis without echoing the body.
func LogPayloadError(log log.Log, ctx wrapper.HttpContext, prefix string, body []byte) {
	if log == nil {
		return
	}
	if payloadLoggingOff || EntryMetadataEnabled(ctx) {
		log.Errorf("%s: bodyBytes=%d", prefix, len(body))
		return
	}
	log.Errorf("%s: %s", prefix, body)
}

type Provider interface {
	GetProviderType() string
}

type EmbeddingQuerier interface {
	QueryEmbedding(
		emb []float64,
		ctx wrapper.HttpContext,
		log log.Log,
		callback func(results []QueryResult, ctx wrapper.HttpContext, log log.Log, err error)) error
}

type EmbeddingUploader interface {
	UploadEmbedding(
		queryString string,
		queryEmb []float64,
		ctx wrapper.HttpContext,
		log log.Log,
		callback func(ctx wrapper.HttpContext, log log.Log, err error)) error
}

type AnswerAndEmbeddingUploader interface {
	UploadAnswerAndEmbedding(
		queryString string,
		queryEmb []float64,
		answer string,
		ctx wrapper.HttpContext,
		log log.Log,
		callback func(ctx wrapper.HttpContext, log log.Log, err error)) error
}

type StringQuerier interface {
	QueryString(
		queryString string,
		ctx wrapper.HttpContext,
		log log.Log,
		callback func(results []QueryResult, ctx wrapper.HttpContext, log log.Log, err error)) error
}

type ProviderConfig struct {
	// @Title zh-CN 向量存储服务提供者类型
	// @Description zh-CN 向量存储服务提供者类型，例如 dashvector、chroma
	typ string
	// @Title zh-CN 向量存储服务名称
	// @Description zh-CN 向量存储服务名称
	serviceName string
	// @Title zh-CN 向量存储服务域名
	// @Description zh-CN 向量存储服务域名
	serviceHost string
	// @Title zh-CN 向量存储服务端口
	// @Description zh-CN 向量存储服务端口
	servicePort int64
	// @Title zh-CN 向量存储服务 API Key
	// @Description zh-CN 向量存储服务 API Key
	apiKey string
	// @Title zh-CN 返回TopK结果
	// @Description zh-CN 返回TopK结果，默认为 1
	topK int
	// @Title zh-CN 请求超时
	// @Description zh-CN 请求向量存储服务的超时时间，单位为毫秒。默认值是10000，即10秒
	timeout uint32
	// @Title zh-CN 向量存储服务 Collection ID
	// @Description zh-CN 向量存储服务的 Collection ID
	collectionID string
	// @Title zh-CN 相似度度量阈值
	// @Description zh-CN 启用语义缓存时必须显式配置，无数值默认值。
	Threshold    float64
	thresholdSet bool
	// @Title zh-CN 相似度度量比较方式
	// @Description zh-CN 启用语义缓存时必须显式配置。
	// 相似度度量方式有 Cosine, DotProduct, Euclidean 等，前两者值越大相似度越高，后者值越小相似度越高。
	// 所以需要允许自定义比较方式，对于 Cosine 和 DotProduct 选择 gt，对于 Euclidean 则选择 lt。
	// 所有条件包括 lt (less than，小于)、lte (less than or equal to，小等于)、gt (greater than，大于)、gte (greater than or equal to，大等于)
	ThresholdRelation    string
	thresholdRelationSet bool

	// ES 配置
	// @Title zh-CN ES 用户名
	// @Description zh-CN ES 用户名
	esUsername string
	// @Title zh-CN ES 密码
	// @Description zh-CN ES 密码
	esPassword string
}

func (c *ProviderConfig) GetProviderType() string {
	return c.typ
}

func (c *ProviderConfig) HasThreshold() bool {
	return c.thresholdSet
}

func (c *ProviderConfig) HasThresholdRelation() bool {
	return c.thresholdRelationSet
}

func (c *ProviderConfig) FromJson(json gjson.Result) {
	c.typ = json.Get("type").String()
	c.serviceName = json.Get("serviceName").String()
	c.serviceHost = json.Get("serviceHost").String()
	c.servicePort = int64(json.Get("servicePort").Int())
	if c.servicePort == 0 {
		c.servicePort = 443
	}
	c.apiKey = json.Get("apiKey").String()
	c.collectionID = json.Get("collectionID").String()
	c.topK = int(json.Get("topK").Int())
	if c.topK == 0 {
		c.topK = 1
	}
	c.timeout = uint32(json.Get("timeout").Int())
	if c.timeout == 0 {
		c.timeout = 10000
	}
	threshold := json.Get("threshold")
	c.thresholdSet = threshold.Exists() && threshold.Type != gjson.Null
	c.Threshold = threshold.Float()
	relation := json.Get("thresholdRelation")
	c.thresholdRelationSet = relation.Exists() && relation.String() != ""
	c.ThresholdRelation = relation.String()

	// ES
	c.esUsername = json.Get("esUsername").String()
	c.esPassword = json.Get("esPassword").String()
}

func (c *ProviderConfig) Validate() error {
	if c.typ == "" {
		return errors.New("vector database service is required")
	}
	initializer, has := providerInitializers[c.typ]
	if !has {
		return errors.New("unknown vector database service provider type: " + c.typ)
	}
	if c.ThresholdRelation != "" && !isRelationValid(c.ThresholdRelation) {
		return errors.New("invalid thresholdRelation: " + c.ThresholdRelation)
	}
	if err := initializer.ValidateConfig(*c); err != nil {
		return err
	}
	return nil
}

func CreateProvider(pc ProviderConfig) (Provider, error) {
	initializer, has := providerInitializers[pc.typ]
	if !has {
		return nil, errors.New("unknown provider type: " + pc.typ)
	}
	return initializer.CreateProvider(pc)
}

func isRelationValid(relation string) bool {
	for _, r := range []string{"lt", "lte", "gt", "gte"} {
		if r == relation {
			return true
		}
	}
	return false
}
