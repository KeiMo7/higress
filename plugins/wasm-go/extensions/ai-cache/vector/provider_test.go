package vector

import (
	"net/http"
	"testing"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-cache/entry"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type consumerTestContext struct {
	wrapper.HttpContext
	consumer string
}

func (c consumerTestContext) GetContext(key string) interface{} {
	if key == ConsumerContextKey {
		return c.consumer
	}
	return nil
}

type recordingClient struct {
	wrapper.HttpClient
	method string
	body   []byte
}

func (c *recordingClient) Post(_ string, _ [][2]string, body []byte, _ wrapper.ResponseCallback, _ ...uint32) error {
	c.method = http.MethodPost
	c.body = append([]byte(nil), body...)
	return nil
}

func (c *recordingClient) Put(_ string, _ [][2]string, body []byte, _ wrapper.ResponseCallback, _ ...uint32) error {
	c.method = http.MethodPut
	c.body = append([]byte(nil), body...)
	return nil
}

type discardLog struct {
	log.Log
}

func (discardLog) Debugf(string, ...interface{}) {}
func (discardLog) Infof(string, ...interface{})  {}
func (discardLog) Errorf(string, ...interface{}) {}

type tenantProvider interface {
	Provider
	EmbeddingQuerier
	AnswerAndEmbeddingUploader
}

func TestProvidersUploadAndFilterByConsumer(t *testing.T) {
	providerConfig := ProviderConfig{topK: 1, collectionID: "Higress"}
	for _, tc := range []struct {
		name        string
		newProvider func(wrapper.HttpClient) tenantProvider
		queryPath   string
		uploadPath  string
		queryJSON   bool
	}{
		{name: "dashvector", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &DvProvider{config: providerConfig, client: client}
		}, queryPath: "filter", uploadPath: "docs.0.fields.consumer"},
		{name: "chroma", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &ChromaProvider{config: providerConfig, client: client}
		}, queryPath: "where.consumer", uploadPath: "metadatas.0.consumer"},
		{name: "elasticsearch", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &ESProvider{config: providerConfig, client: client}
		}, queryPath: "knn.filter.term.consumer", uploadPath: "consumer"},
		{name: "weaviate", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &WeaviateProvider{config: providerConfig, client: client}
		}, queryPath: "query", uploadPath: "properties.consumer", queryJSON: true},
		{name: "pinecone", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &pineconeProvider{config: providerConfig, client: client}
		}, queryPath: "filter.consumer.$eq", uploadPath: "vectors.0.metadata.consumer"},
		{name: "qdrant", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &qdrantProvider{config: providerConfig, client: client}
		}, queryPath: "filter.must.0.match.value", uploadPath: "points.0.payload.consumer"},
		{name: "milvus", newProvider: func(client wrapper.HttpClient) tenantProvider {
			return &milvusProvider{config: providerConfig, client: client}
		}, queryPath: "filter", uploadPath: "data.0.consumer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &recordingClient{}
			provider := tc.newProvider(client)
			authenticatedCtx := consumerTestContext{consumer: "consumer-a"}
			require.NoError(t, provider.QueryEmbedding([]float64{0.1}, authenticatedCtx, discardLog{}, nil))
			queryValue := gjson.GetBytes(client.body, tc.queryPath).String()
			if tc.queryJSON || tc.name == "dashvector" || tc.name == "milvus" {
				require.Contains(t, queryValue, "consumer-a")
			} else {
				require.Equal(t, "consumer-a", queryValue)
			}
			if tc.name == "weaviate" {
				require.Contains(t, queryValue, `path: ["consumer"]`)
			}

			client.body = nil
			require.NoError(t, provider.UploadAnswerAndEmbedding("question", []float64{0.1}, "answer", authenticatedCtx, discardLog{}, nil))
			require.Equal(t, "consumer-a", gjson.GetBytes(client.body, tc.uploadPath).String())

			client.body = nil
			anonymousCtx := consumerTestContext{}
			require.ErrorIs(t, provider.QueryEmbedding([]float64{0.1}, anonymousCtx, discardLog{}, nil), ErrConsumerRequired)
			require.ErrorIs(t, provider.UploadAnswerAndEmbedding("question", []float64{0.1}, "answer", anonymousCtx, discardLog{}, nil), ErrConsumerRequired)
			require.Nil(t, client.body)
		})
	}
}

func TestQueryResponsesRetainConsumerMetadata(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parse func([]byte) ([]QueryResult, error)
		body  string
	}{
		{name: "dashvector", parse: func(body []byte) ([]QueryResult, error) {
			return (&DvProvider{}).ParseQueryResponse(body, nil, discardLog{})
		}, body: `{"output":[{"fields":{"query":"q","answer":"a","consumer":"consumer-a"},"score":0.9}]}`},
		{name: "chroma", parse: func(body []byte) ([]QueryResult, error) {
			return (&ChromaProvider{}).parseQueryResponse(body, discardLog{})
		}, body: `{"ids":[["q"]],"distances":[[0.1]],"documents":[["a"]],"metadatas":[[{"consumer":"consumer-a"}]]}`},
		{name: "elasticsearch", parse: func(body []byte) ([]QueryResult, error) { return (&ESProvider{}).parseQueryResponse(body, nil) }, body: `{"hits":{"total":{"value":1},"hits":[{"_score":0.9,"_source":{"question":"q","answer":"a","consumer":"consumer-a"}}]}}`},
		{name: "weaviate", parse: func(body []byte) ([]QueryResult, error) {
			return (&WeaviateProvider{config: ProviderConfig{collectionID: "Higress"}}).parseQueryResponse(body, discardLog{})
		}, body: `{"data":{"Get":{"Higress":[{"question":"q","answer":"a","consumer":"consumer-a","_additional":{"distance":0.1}}]}}}`},
		{name: "pinecone", parse: func(body []byte) ([]QueryResult, error) {
			return (&pineconeProvider{}).parseQueryResponse(body, discardLog{})
		}, body: `{"matches":[{"score":0.9,"metadata":{"question":"q","answer":"a","consumer":"consumer-a"}}]}`},
		{name: "qdrant", parse: func(body []byte) ([]QueryResult, error) {
			return (&qdrantProvider{}).parseQueryResponse(body, discardLog{})
		}, body: `{"result":[{"score":0.9,"payload":{"question":"q","answer":"a","consumer":"consumer-a"}}]}`},
		{name: "milvus", parse: func(body []byte) ([]QueryResult, error) {
			return (&milvusProvider{}).parseQueryResponse(body, discardLog{})
		}, body: `{"data":[{"distance":0.1,"question":"q","answer":"a","consumer":"consumer-a"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			results, err := tc.parse([]byte(tc.body))
			require.NoError(t, err)
			require.Len(t, results, 1)
			require.Equal(t, "consumer-a", results[0].Consumer)
		})
	}
}

type metadataCtx struct {
	wrapper.HttpContext
	values map[string]interface{}
}

func (c metadataCtx) GetContext(key string) interface{} { return c.values[key] }

// TestProvidersCarryEntryMetadataOnlyWhenEnabled pins the opt-in boundary: with
// entry semantics off the payload shape is byte-for-byte the legacy one.
func TestProvidersCarryEntryMetadataOnlyWhenEnabled(t *testing.T) {
	providerConfig := ProviderConfig{topK: 1, collectionID: "Higress"}
	metadata := entry.Metadata{WrittenAt: 1_700_000_000, Tenant: "consumer-a", Generation: "g1", Complete: true}

	for _, tc := range []struct {
		name        string
		newProvider func(wrapper.HttpClient) tenantProvider
		uploadPath  string
	}{
		{name: "dashvector", newProvider: func(c wrapper.HttpClient) tenantProvider { return &DvProvider{config: providerConfig, client: c} }, uploadPath: "docs.0.fields"},
		{name: "chroma", newProvider: func(c wrapper.HttpClient) tenantProvider { return &ChromaProvider{config: providerConfig, client: c} }, uploadPath: "metadatas.0"},
		{name: "elasticsearch", newProvider: func(c wrapper.HttpClient) tenantProvider { return &ESProvider{config: providerConfig, client: c} }, uploadPath: ""},
		{name: "weaviate", newProvider: func(c wrapper.HttpClient) tenantProvider { return &WeaviateProvider{config: providerConfig, client: c} }, uploadPath: "properties"},
		{name: "pinecone", newProvider: func(c wrapper.HttpClient) tenantProvider { return &pineconeProvider{config: providerConfig, client: c} }, uploadPath: "vectors.0.metadata"},
		{name: "qdrant", newProvider: func(c wrapper.HttpClient) tenantProvider { return &qdrantProvider{config: providerConfig, client: c} }, uploadPath: "points.0.payload"},
		{name: "milvus", newProvider: func(c wrapper.HttpClient) tenantProvider { return &milvusProvider{config: providerConfig, client: c} }, uploadPath: "data.0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Legacy: no metadata fields.
			client := &recordingClient{}
			provider := tc.newProvider(client)
			legacyCtx := metadataCtx{values: map[string]interface{}{ConsumerContextKey: "consumer-a"}}
			require.NoError(t, provider.UploadAnswerAndEmbedding("q", []float64{0.1}, "a", legacyCtx, discardLog{}, nil))
			legacyBody := string(client.body)
			require.NotContains(t, legacyBody, WrittenAtField)
			require.NotContains(t, legacyBody, GenerationField)
			require.NotContains(t, legacyBody, CompleteField)

			// Enabled: the same upload carries all three fields.
			client = &recordingClient{}
			provider = tc.newProvider(client)
			enabledCtx := metadataCtx{values: map[string]interface{}{
				ConsumerContextKey:             "consumer-a",
				entryMetadataEnabledContextKey: true,
				entryMetadataContextKey:        metadata,
			}}
			require.NoError(t, provider.UploadAnswerAndEmbedding("q", []float64{0.1}, "a", enabledCtx, discardLog{}, nil))
			enabledBody := string(client.body)
			require.Contains(t, enabledBody, WrittenAtField)
			require.Contains(t, enabledBody, GenerationField)
			require.Contains(t, enabledBody, CompleteField)
			if tc.uploadPath != "" {
				require.Equal(t, "g1", gjson.GetBytes(client.body, tc.uploadPath+"."+GenerationField).String())
			} else {
				require.Equal(t, "g1", gjson.GetBytes(client.body, GenerationField).String())
			}
		})
	}
}

// TestQueryResultsExposeEntryMetadata checks that each parser recognizes the
// current metadata and leaves legacy records metadata-free.
func TestQueryResultsExposeEntryMetadata(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parse func([]byte) ([]QueryResult, error)
		body  string
	}{
		{name: "chroma", parse: func(b []byte) ([]QueryResult, error) {
			return (&ChromaProvider{}).parseQueryResponse(b, discardLog{})
		}, body: `{"ids":[["q"]],"distances":[[0.1]],"documents":[["a"]],"metadatas":[[{"consumer":"consumer-a","writtenAt":"1700000000","generation":"g1","complete":"true"}]]}`},
		{name: "elasticsearch", parse: func(b []byte) ([]QueryResult, error) { return (&ESProvider{}).parseQueryResponse(b, nil) }, body: `{"hits":{"total":{"value":1},"hits":[{"_id":"i","_score":0.9,"_source":{"question":"q","answer":"a","consumer":"consumer-a","writtenAt":1700000000,"generation":"g1","complete":true}}]}}`},
		{name: "pinecone", parse: func(b []byte) ([]QueryResult, error) {
			return (&pineconeProvider{}).parseQueryResponse(b, discardLog{})
		}, body: `{"matches":[{"id":"i","score":0.9,"metadata":{"question":"q","answer":"a","consumer":"consumer-a","writtenAt":1700000000,"generation":"g1","complete":true}}]}`},
		{name: "qdrant", parse: func(b []byte) ([]QueryResult, error) {
			return (&qdrantProvider{}).parseQueryResponse(b, discardLog{})
		}, body: `{"result":[{"id":"i","score":0.9,"payload":{"question":"q","answer":"a","consumer":"consumer-a","writtenAt":1700000000,"generation":"g1","complete":true}}]}`},
		{name: "dashvector", parse: func(b []byte) ([]QueryResult, error) {
			return (&DvProvider{}).ParseQueryResponse(b, nil, discardLog{})
		}, body: `{"output":[{"id":"i","fields":{"query":"q","answer":"a","consumer":"consumer-a","writtenAt":"1700000000","generation":"g1","complete":"true"},"score":0.9}]}`},
		{name: "milvus", parse: func(b []byte) ([]QueryResult, error) {
			return (&milvusProvider{}).parseQueryResponse(b, discardLog{})
		}, body: `{"data":[{"distance":0.1,"question":"q","answer":"a","consumer":"consumer-a","writtenAt":1700000000,"generation":"g1","complete":true}]}`},
		{name: "weaviate", parse: func(b []byte) ([]QueryResult, error) {
			return (&WeaviateProvider{config: ProviderConfig{collectionID: "Higress"}}).parseQueryResponse(b, discardLog{})
		}, body: `{"data":{"Get":{"Higress":[{"question":"q","answer":"a","consumer":"consumer-a","writtenAt":1700000000,"generation":"g1","complete":true,"_additional":{"distance":0.1}}]}}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			results, err := tc.parse([]byte(tc.body))
			require.NoError(t, err)
			require.Len(t, results, 1)
			meta, ok := results[0].EntryMetadata()
			require.True(t, ok)
			require.Equal(t, int64(1_700_000_000), meta.WrittenAt)
			require.Equal(t, "g1", meta.Generation)
			require.True(t, meta.Complete)
			require.Equal(t, "consumer-a", meta.Tenant)
		})
	}
}

// TestPartialMetadataIsTreatedAsLegacy ensures a record missing any metadata
// field is never guessed into a valid entry.
func TestPartialMetadataIsTreatedAsLegacy(t *testing.T) {
	provider := &ChromaProvider{}
	results, err := provider.parseQueryResponse([]byte(`{"ids":[["q"]],"distances":[[0.1]],"documents":[["a"]],"metadatas":[[{"consumer":"consumer-a","writtenAt":"1700000000","generation":"g1"}]]}`), discardLog{})
	require.NoError(t, err)
	require.Len(t, results, 1)
	_, ok := results[0].EntryMetadata()
	require.False(t, ok)
	require.False(t, results[0].MetadataPresent)
}

// TestMilvusOutputFieldsFollowPolicy keeps the legacy field set intact.
func TestMilvusOutputFieldsFollowPolicy(t *testing.T) {
	require.Equal(t, []string{"question", "answer", TenantField}, outputFields(metadataCtx{values: map[string]interface{}{}}))
	require.Equal(t,
		[]string{"question", "answer", TenantField, WrittenAtField, GenerationField, CompleteField},
		outputFields(metadataCtx{values: map[string]interface{}{entryMetadataEnabledContextKey: true}}))
}
