// Package entry defines what a cache entry is, when it is trusted, when it
// expires and who may see it.
//
// The same validity judgment (Policy.Check) is used by the exact key-value
// read, the semantic query and the key-value backfill read, and by both write
// paths. Keeping the four dimensions (tenant, generation, completeness,
// lifetime) in one place is the point of this package: the key-value side and
// the vector side must never disagree about whether an entry is usable.
package entry

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"

	"github.com/tidwall/gjson"
)

// Miss reasons. The first five are the required user-visible reasons; the
// remaining two describe compatibility and backend conditions without ever
// presenting them as "no similar result".
const (
	ReasonAbsent             = "absent"
	ReasonExpired            = "expired"
	ReasonGenerationMismatch = "generation_mismatch"
	ReasonIncomplete         = "incomplete"
	ReasonTenantMismatch     = "tenant_mismatch"

	// ReasonLegacy marks an entry that predates entry metadata, or whose
	// metadata cannot be recognized. Such an entry is never served.
	ReasonLegacy = "legacy_format"
	// ReasonBackendError marks a degraded read (backend unavailable, timeout or
	// malformed payload). It must stay distinguishable from a genuine miss.
	ReasonBackendError = "backend_error"
)

// Hit sources for observability.
const (
	SourceKV       = "kv"
	SourceSemantic = "semantic"
	SourceBackfill = "backfill"
)

const (
	markerField    = "higressEntry"
	currentVersion = 1
	// maxClockSkewSeconds tolerates small clock differences between gateway
	// nodes. A write time further in the future is treated as an invalid time,
	// never as a valid entry.
	maxClockSkewSeconds int64 = 300
)

// Policy is the configured identity and lifetime contract for cache entries.
type Policy struct {
	// Enabled turns the new entry semantics on. When false every existing
	// key format, value format, template and skip behavior is unchanged.
	Enabled bool
	// Generation identifies a knowledge/answer revision. A generation change
	// invalidates old entries lazily on read, without scanning or downtime.
	Generation string
	// TTLSeconds is the effective lifetime shared by the key-value and vector
	// sides. Zero means no expiry. It is resolved at load time: an explicit
	// entryPolicy.ttlSeconds wins, otherwise cache.cacheTTL is inherited.
	TTLSeconds int
	// TTLConfigured records whether ttlSeconds was explicitly configured.
	TTLConfigured bool
}

// Metadata is the complete set of facts a stored entry must carry.
type Metadata struct {
	WrittenAt  int64  `json:"writtenAt"`
	Tenant     string `json:"tenant"`
	Generation string `json:"generation"`
	Complete   bool   `json:"complete"`
}

// FromJson parses the entryPolicy config block. A configured ttlSeconds wins;
// otherwise the lifetime is inherited from cache.cacheTTL through Resolve, so
// existing configurations keep their previous lifetime.
func (p *Policy) FromJson(json gjson.Result) {
	p.Enabled = json.Get("enabled").Bool()
	p.Generation = json.Get("generation").String()
	ttl := json.Get("ttlSeconds")
	if ttl.Exists() && ttl.Type != gjson.Null {
		p.TTLSeconds = int(ttl.Int())
		p.TTLConfigured = true
	} else {
		p.TTLSeconds = 0
		p.TTLConfigured = false
	}
}

// Resolve fills the inherited lifetime when ttlSeconds was not configured.
func (p *Policy) Resolve(inheritedTTL int) {
	if !p.TTLConfigured {
		p.TTLSeconds = inheritedTTL
	}
}

// Validate rejects illegal values at load time instead of guessing defaults at
// request time.
func (p *Policy) Validate() error {
	if !p.Enabled {
		return nil
	}
	if strings.TrimSpace(p.Generation) == "" {
		return errors.New("entryPolicy.generation is required when entryPolicy.enabled is true")
	}
	if p.TTLSeconds < 0 {
		return errors.New("entryPolicy.ttlSeconds must be greater than or equal to 0")
	}
	return nil
}

// Check is the single validity judgment. It returns an empty string when the
// entry may be used, otherwise one of the Reason* constants.
//
// Order is tenant, generation, completeness, lifetime. Every read path and
// both write paths call exactly this function, so a value can never be
// accepted by one path and rejected by another.
func (p Policy) Check(m Metadata, tenant string, now int64) string {
	if !p.Enabled {
		return ""
	}
	if tenant == "" || m.Tenant != tenant {
		return ReasonTenantMismatch
	}
	if p.Generation == "" || m.Generation != p.Generation {
		return ReasonGenerationMismatch
	}
	if !m.Complete {
		return ReasonIncomplete
	}
	if m.WrittenAt <= 0 || m.WrittenAt > now+maxClockSkewSeconds {
		// An unreadable or impossible write time must never be guessed into a
		// valid entry.
		return ReasonExpired
	}
	if p.TTLSeconds > 0 && now-m.WrittenAt >= int64(p.TTLSeconds) {
		return ReasonExpired
	}
	return ""
}

// Writable is the write-side name of the same judgment. Callers build the
// metadata they are about to store and ask whether it may become hittable.
func (p Policy) Writable(m Metadata, tenant string, now int64) string {
	return p.Check(m, tenant, now)
}

// Remaining returns the lifetime left for an entry, in seconds. A non-positive
// result means the entry has expired. When the policy has no expiry it returns
// zero, meaning "store without a time limit".
func (p Policy) Remaining(m Metadata, now int64) int {
	if p.TTLSeconds <= 0 {
		return 0
	}
	remaining := int64(p.TTLSeconds) - (now - m.WrittenAt)
	if remaining <= 0 {
		return -1
	}
	return int(remaining)
}

// WritableHolds reports whether the metadata can be stored as a hittable entry.
// On top of the shared judgment it requires a non-zero tenant and a known
// generation, so a half-open write cannot pass the read check by accident.
func (p Policy) WritableHolds(m Metadata, tenant string, now int64) bool {
	if !p.Enabled || strings.TrimSpace(tenant) == "" || strings.TrimSpace(m.Generation) == "" {
		return false
	}
	return p.Check(m, tenant, now) == ""
}

// Encode serializes an entry into the key-value value. The answer stays the
// only payload; metadata is prepended so a read can judge the entry before
// using the answer.
func Encode(m Metadata, answer string) string {
	body, _ := json.Marshal(struct {
		Version    int    `json:"higressEntry"`
		WrittenAt  int64  `json:"writtenAt"`
		Tenant     string `json:"tenant"`
		Generation string `json:"generation"`
		Complete   bool   `json:"complete"`
		Answer     string `json:"answer"`
	}{
		Version:    currentVersion,
		WrittenAt:  m.WrittenAt,
		Tenant:     m.Tenant,
		Generation: m.Generation,
		Complete:   m.Complete,
		Answer:     answer,
	})
	return string(body)
}

// DecodeStatus describes how a stored value was recognized.
type DecodeStatus int

const (
	// DecodeOK is a well-formed current-format entry.
	DecodeOK DecodeStatus = iota
	// DecodeLegacy is an old-format value without entry metadata.
	DecodeLegacy
	// DecodeMalformed claims the current format but has invalid fields.
	DecodeMalformed
)

// Decode reads a key-value value. It never returns an error: an unreadable
// value is data to classify, not a reason to fail the request.
func Decode(raw string) (Metadata, string, DecodeStatus) {
	value := strings.TrimSpace(raw)
	if value == "" || !strings.HasPrefix(value, "{") {
		return Metadata{}, "", DecodeLegacy
	}
	if !gjson.Valid(value) {
		return Metadata{}, "", DecodeMalformed
	}
	marker := gjson.Get(value, markerField)
	if !marker.Exists() || marker.Type != gjson.Number {
		return Metadata{}, "", DecodeLegacy
	}
	if marker.Int() != currentVersion {
		return Metadata{}, "", DecodeLegacy
	}

	writtenAt := gjson.Get(value, "writtenAt")
	complete := gjson.Get(value, "complete")
	answer := gjson.Get(value, "answer")
	if !writtenAt.Exists() || writtenAt.Type != gjson.Number {
		return Metadata{}, "", DecodeMalformed
	}
	if !complete.Exists() || (complete.Type != gjson.True && complete.Type != gjson.False) {
		return Metadata{}, "", DecodeMalformed
	}
	if !answer.Exists() || answer.Type != gjson.String {
		return Metadata{}, "", DecodeMalformed
	}

	return Metadata{
		WrittenAt:  writtenAt.Int(),
		Tenant:     gjson.Get(value, "tenant").String(),
		Generation: gjson.Get(value, "generation").String(),
		Complete:   complete.Bool(),
	}, answer.String(), DecodeOK
}

// Redact returns a stable, non-reversible identifier for a tenant, question or
// answer so logs can correlate events without exposing plaintext.
func Redact(value string) string {
	if value == "" {
		return ""
	}
	sum := sha256.Sum256([]byte("higress-ai-cache\x00" + value))
	return "sha256:" + hex.EncodeToString(sum[:6])
}
