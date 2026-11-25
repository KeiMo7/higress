package vector

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"
)

type milvusProviderInitializer struct{}

func (c *milvusProviderInitializer) ValidateConfig(config ProviderConfig) error {
	if len(config.serviceName) == 0 {
		return errors.New("[Milvus] serviceName is required")
	}
	if len(config.collectionID) == 0 {
		return errors.New("[Milvus] collectionID is required")
	}
	return nil
}

func (c *milvusProviderInitializer) CreateProvider(config ProviderConfig) (Provider, error) {
	return &milvusProvider{
		config: config,
		client: wrapper.NewClusterClient(wrapper.FQDNCluster{
			FQDN: config.serviceName,
			Host: config.serviceHost,
			Port: int64(config.servicePort),
		}),
	}, nil
}

type milvusProvider struct {
	config ProviderConfig
	client wrapper.HttpClient
}

func (c *milvusProvider) GetProviderType() string {
	return PROVIDER_TYPE_MILVUS
}

type milvusData struct {
	Vector     []float64 `json:"vector"`
	Question   string    `json:"question,omitempty"`
	Answer     string    `json:"answer,omitempty"`
	Consumer   string    `json:"consumer"`
	WrittenAt  int64     `json:"writtenAt,omitempty"`
	Generation string    `json:"generation,omitempty"`
	Complete   bool      `json:"complete,omitempty"`
}

type milvusInsertRequest struct {
	CollectionName string       `json:"collectionName"`
	Data           []milvusData `json:"data"`
}

func (d *milvusProvider) UploadAnswerAndEmbedding(
	queryString string,
	queryEmb []float64,
	queryAnswer string,
	ctx wrapper.HttpContext,
	log log.Log,
	callback func(ctx wrapper.HttpContext, log log.Log, err error)) error {
	consumer := ConsumerFromContext(ctx)
	if consumer == "" {
		return ErrConsumerRequired
	}
	// 最少需要填写的参数为 collectionName, data 和 Authorization. question, answer 可选
	// 需要填写 id，否则 v2.4.13-hotfix 提示 invalid syntax: invalid parameter[expected=Int64][actual=]
	// 如果不填写 id，要在创建 collection 的时候设置 autoId 为 true
	// 下面是一个例子
	// {
	// 	"collectionName": "higress",
	// 	"data": [
	// 	  {
	// 	    "question": "这里是问题",
	// 	  	"answer": "这里是答案"
	// 	    "vector": [
	// 	      0.9,
	// 	      0.1,
	// 	      0.1
	// 	    ]
	// 	  }
	//   ]
	// }
	data := milvusData{
		Question: queryString,
		Answer:   queryAnswer,
		Vector:   queryEmb,
		Consumer: consumer,
	}
	if m, ok := metadataPayload(ctx); ok {
		data.WrittenAt = m.WrittenAt
		data.Generation = m.Generation
		data.Complete = m.Complete
	}
	requestBody, err := json.Marshal(milvusInsertRequest{
		CollectionName: d.config.collectionID,
		Data:           []milvusData{data},
	})

	if err != nil {
		log.Errorf("[Milvus] Failed to marshal upload embedding request body: %v", err)
		return err
	}

	return d.client.Post(
		"/v2/vectordb/entities/insert",
		[][2]string{
			{"Content-Type", "application/json"},
			{"Authorization", fmt.Sprintf("Bearer %s", d.config.apiKey)},
		},
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			LogPayload(log, ctx, "[Milvus] upload response", statusCode, responseBody)
			callback(ctx, log, err)
		},
		d.config.timeout,
	)
}

type milvusQueryRequest struct {
	CollectionName string      `json:"collectionName"`
	Data           [][]float64 `json:"data"`
	AnnsField      string      `json:"annsField"`
	Limit          int         `json:"limit"`
	OutputFields   []string    `json:"outputFields"`
	Filter         string      `json:"filter"`
}

func (d *milvusProvider) QueryEmbedding(
	emb []float64,
	ctx wrapper.HttpContext,
	log log.Log,
	callback func(results []QueryResult, ctx wrapper.HttpContext, log log.Log, err error)) error {
	consumer := ConsumerFromContext(ctx)
	if consumer == "" {
		return ErrConsumerRequired
	}
	// 最少需要填写的参数为 collectionName, data, annsField. outputFields 为可选参数
	// 下面是一个例子
	// {
	// 	"collectionName": "quick_setup",
	// 	"data": [
	// 		[
	// 			0.3580376395471989,
	// 			"Unknown type",
	// 			0.18414012509913835,
	// 			"Unknown type",
	// 			0.9029438446296592
	// 		]
	// 	],
	// 	"annsField": "vector",
	// 	"limit": 3,
	// 	"outputFields": [
	// 		"color"
	// 	]
	// }
	consumerString, err := json.Marshal(consumer)
	if err != nil {
		return err
	}
	requestBody, err := json.Marshal(milvusQueryRequest{
		CollectionName: d.config.collectionID,
		Data:           [][]float64{emb},
		AnnsField:      "vector",
		Limit:          d.config.topK,
		OutputFields:   outputFields(ctx),
		Filter:         fmt.Sprintf("consumer == %s", consumerString),
	})
	if err != nil {
		log.Errorf("[Milvus] Failed to marshal query embedding: %v", err)
		return err
	}

	return d.client.Post(
		"/v2/vectordb/entities/search",
		[][2]string{
			{"Content-Type", "application/json"},
			{"Authorization", fmt.Sprintf("Bearer %s", d.config.apiKey)},
		},
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			LogPayload(log, ctx, "[Milvus] Query embedding response", statusCode, responseBody)
			results, err := d.parseQueryResponse(responseBody, log)
			if err != nil {
				err = fmt.Errorf("[Milvus] Failed to parse query response: %v", err)
			}
			callback(results, ctx, log, err)
		},
		d.config.timeout,
	)
}

func (d *milvusProvider) parseQueryResponse(responseBody []byte, log log.Log) ([]QueryResult, error) {
	if !gjson.GetBytes(responseBody, "data.0.distance").Exists() {
		LogPayloadError(log, nil, "[Milvus] No distance found in response", responseBody)
		return nil, errors.New("[Milvus] No distance found in response body")
	}

	if !gjson.GetBytes(responseBody, "data.0.question").Exists() {
		LogPayloadError(log, nil, "[Milvus] No question found in response", responseBody)
		return nil, errors.New("[Milvus] No question found in response body")
	}

	if !gjson.GetBytes(responseBody, "data.0.answer").Exists() {
		LogPayloadError(log, nil, "[Milvus] No answer found in response", responseBody)
		return nil, errors.New("[Milvus] No answer found in response body")
	}

	resultNum := gjson.GetBytes(responseBody, "data.#").Int()
	results := make([]QueryResult, 0, resultNum)
	for i := 0; i < int(resultNum); i++ {
		prefix := fmt.Sprintf("data.%d", i)
		result := QueryResult{
			Text:     gjson.GetBytes(responseBody, prefix+".question").String(),
			Score:    gjson.GetBytes(responseBody, prefix+".distance").Float(),
			Answer:   gjson.GetBytes(responseBody, prefix+".answer").String(),
			Consumer: gjson.GetBytes(responseBody, prefix+".consumer").String(),
		}
		applyMetadata(&result, "",
			gjson.GetBytes(responseBody, prefix+"."+WrittenAtField),
			gjson.GetBytes(responseBody, prefix+"."+GenerationField),
			gjson.GetBytes(responseBody, prefix+"."+CompleteField))
		results = append(results, result)
	}

	return results, nil
}

// outputFields keeps the legacy field set unless entry metadata is enabled.
func outputFields(ctx wrapper.HttpContext) []string {
	fields := []string{"question", "answer", TenantField}
	if EntryMetadataEnabled(ctx) {
		fields = append(fields, WrittenAtField, GenerationField, CompleteField)
	}
	return fields
}
