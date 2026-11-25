package vector

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"
)

type qdrantProviderInitializer struct{}

func (c *qdrantProviderInitializer) ValidateConfig(config ProviderConfig) error {
	if len(config.serviceName) == 0 {
		return errors.New("[Qdrant] serviceName is required")
	}
	if len(config.collectionID) == 0 {
		return errors.New("[Qdrant] collectionID is required")
	}
	return nil
}

func (c *qdrantProviderInitializer) CreateProvider(config ProviderConfig) (Provider, error) {
	return &qdrantProvider{
		config: config,
		client: wrapper.NewClusterClient(wrapper.FQDNCluster{
			FQDN: config.serviceName,
			Host: config.serviceHost,
			Port: int64(config.servicePort),
		}),
	}, nil
}

type qdrantProvider struct {
	config ProviderConfig
	client wrapper.HttpClient
}

func (c *qdrantProvider) GetProviderType() string {
	return PROVIDER_TYPE_QDRANT
}

type qdrantPayload struct {
	Question   string `json:"question"`
	Answer     string `json:"answer"`
	Consumer   string `json:"consumer"`
	WrittenAt  int64  `json:"writtenAt,omitempty"`
	Generation string `json:"generation,omitempty"`
	Complete   bool   `json:"complete,omitempty"`
}

type qdrantPoint struct {
	ID      string        `json:"id"`
	Vector  []float64     `json:"vector"`
	Payload qdrantPayload `json:"payload"`
}

type qdrantInsertRequest struct {
	Points []qdrantPoint `json:"points"`
}

func (d *qdrantProvider) UploadAnswerAndEmbedding(
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
	// 最少需要填写的参数为 id 和 vector. payload 可选
	// 下面是一个例子
	// {
	// 	"points": [
	// 	  {
	// 	    "id": "76874cce-1fb9-4e16-9b0b-f085ac06ed6f",
	// 	    "payload": {
	// 	      "question": "这里是问题",
	// 	  	  "answer": "这里是答案"
	// 	    },
	// 	    "vector": [
	// 	      0.9,
	// 	      0.1,
	// 	      0.1
	// 	    ]
	// 	  }
	//   ]
	// }
	payload := qdrantPayload{Question: queryString, Answer: queryAnswer, Consumer: consumer}
	if m, ok := metadataPayload(ctx); ok {
		payload.WrittenAt = m.WrittenAt
		payload.Generation = m.Generation
		payload.Complete = m.Complete
	}
	requestBody, err := json.Marshal(qdrantInsertRequest{
		Points: []qdrantPoint{
			{
				ID:      uuid.New().String(),
				Vector:  queryEmb,
				Payload: payload,
			},
		},
	})

	if err != nil {
		log.Errorf("[Qdrant] Failed to marshal upload embedding request body: %v", err)
		return err
	}

	return d.client.Put(
		fmt.Sprintf("/collections/%s/points", d.config.collectionID),
		[][2]string{
			{"Content-Type", "application/json"},
			{"api-key", d.config.apiKey},
		},
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			LogPayload(log, ctx, "[Qdrant] upload response", statusCode, responseBody)
			callback(ctx, log, err)
		},
		d.config.timeout,
	)
}

type qdrantQueryRequest struct {
	Vector      []float64    `json:"vector"`
	Limit       int          `json:"limit"`
	WithPayload bool         `json:"with_payload"`
	Filter      qdrantFilter `json:"filter"`
}

type qdrantFilter struct {
	Must []qdrantCondition `json:"must"`
}

type qdrantCondition struct {
	Key   string      `json:"key"`
	Match qdrantMatch `json:"match"`
}

type qdrantMatch struct {
	Value string `json:"value"`
}

func (d *qdrantProvider) QueryEmbedding(
	emb []float64,
	ctx wrapper.HttpContext,
	log log.Log,
	callback func(results []QueryResult, ctx wrapper.HttpContext, log log.Log, err error)) error {
	consumer := ConsumerFromContext(ctx)
	if consumer == "" {
		return ErrConsumerRequired
	}
	// 最少需要填写的参数为 vector 和 limit. with_payload 可选，为了直接得到问题答案，所以这里需要
	// 下面是一个例子
	// {
	// 	"vector": [
	// 	  0.2,
	// 	  0.1,
	// 	  0.9,
	// 	  0.7
	// 	],
	// 	"limit": 1
	// }
	requestBody, err := json.Marshal(qdrantQueryRequest{
		Vector:      emb,
		Limit:       d.config.topK,
		WithPayload: true,
		Filter: qdrantFilter{Must: []qdrantCondition{{
			Key: TenantField, Match: qdrantMatch{Value: consumer},
		}}},
	})
	if err != nil {
		log.Errorf("[Qdrant] Failed to marshal query embedding: %v", err)
		return err
	}

	return d.client.Post(
		fmt.Sprintf("/collections/%s/points/search", d.config.collectionID),
		[][2]string{
			{"Content-Type", "application/json"},
			{"api-key", d.config.apiKey},
		},
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			LogPayload(log, ctx, "[Qdrant] Query embedding response", statusCode, responseBody)
			results, err := d.parseQueryResponse(responseBody, log)
			if err != nil {
				err = fmt.Errorf("[Qdrant] Failed to parse query response: %v", err)
			}
			callback(results, ctx, log, err)
		},
		d.config.timeout,
	)
}

func (d *qdrantProvider) parseQueryResponse(responseBody []byte, log log.Log) ([]QueryResult, error) {
	// 返回的内容例子如下
	// {
	// 	"time": 0.002,
	// 	"status": "ok",
	// 	"result": [
	// 	  {
	// 		"id": 42,
	// 		"version": 3,
	// 		"score": 0.75,
	// 		"payload": {
	// 		  "question": "London",
	// 		  "answer": "green"
	// 		},
	// 		"shard_key": "region_1",
	// 		"order_value": 42
	// 	  }
	// 	]
	// }
	if !gjson.GetBytes(responseBody, "result.0.score").Exists() {
		LogPayloadError(log, nil, "[Qdrant] No distance found in response", responseBody)
		return nil, errors.New("[Qdrant] No distance found in response body")
	}

	if !gjson.GetBytes(responseBody, "result.0.payload.answer").Exists() {
		LogPayloadError(log, nil, "[Qdrant] No answer found in response", responseBody)
		return nil, errors.New("[Qdrant] No answer found in response body")
	}

	resultNum := gjson.GetBytes(responseBody, "result.#").Int()
	results := make([]QueryResult, 0, resultNum)
	for i := 0; i < int(resultNum); i++ {
		prefix := fmt.Sprintf("result.%d.payload", i)
		result := QueryResult{
			Text:     gjson.GetBytes(responseBody, prefix+".question").String(),
			Score:    gjson.GetBytes(responseBody, fmt.Sprintf("result.%d.score", i)).Float(),
			Answer:   gjson.GetBytes(responseBody, prefix+".answer").String(),
			Consumer: gjson.GetBytes(responseBody, prefix+".consumer").String(),
		}
		applyMetadata(&result,
			gjson.GetBytes(responseBody, fmt.Sprintf("result.%d.id", i)).String(),
			gjson.GetBytes(responseBody, prefix+"."+WrittenAtField),
			gjson.GetBytes(responseBody, prefix+"."+GenerationField),
			gjson.GetBytes(responseBody, prefix+"."+CompleteField))
		results = append(results, result)
	}

	return results, nil
}
