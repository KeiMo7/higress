package entry

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestPolicyCheckIsTheSingleJudgment(t *testing.T) {
	policy := Policy{Enabled: true, Generation: "g2", TTLSeconds: 3600}
	now := int64(1_700_000_000)

	cases := []struct {
		name   string
		meta   Metadata
		tenant string
		now    int64
		want   string
	}{
		{name: "valid", meta: Metadata{WrittenAt: now - 10, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now},
		{name: "tenant mismatch", meta: Metadata{WrittenAt: now - 10, Tenant: "t2", Generation: "g2", Complete: true}, tenant: "t1", now: now, want: ReasonTenantMismatch},
		{name: "empty tenant never matches", meta: Metadata{WrittenAt: now - 10, Tenant: "", Generation: "g2", Complete: true}, tenant: "", now: now, want: ReasonTenantMismatch},
		{name: "generation mismatch", meta: Metadata{WrittenAt: now - 10, Tenant: "t1", Generation: "g1", Complete: true}, tenant: "t1", now: now, want: ReasonGenerationMismatch},
		{name: "incomplete", meta: Metadata{WrittenAt: now - 10, Tenant: "t1", Generation: "g2", Complete: false}, tenant: "t1", now: now, want: ReasonIncomplete},
		{name: "expired", meta: Metadata{WrittenAt: now - 3600, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now, want: ReasonExpired},
		{name: "just inside ttl", meta: Metadata{WrittenAt: now - 3599, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now},
		{name: "missing write time", meta: Metadata{WrittenAt: 0, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now, want: ReasonExpired},
		{name: "future write time", meta: Metadata{WrittenAt: now + 10_000, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now, want: ReasonExpired},
		{name: "small clock skew tolerated", meta: Metadata{WrittenAt: now + 60, Tenant: "t1", Generation: "g2", Complete: true}, tenant: "t1", now: now},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, policy.Check(tc.meta, tc.tenant, tc.now))
		})
	}

	// Writable and Check are the same judgment for a complete entry.
	valid := Metadata{WrittenAt: now, Tenant: "t1", Generation: "g2", Complete: true}
	require.Equal(t, policy.Check(valid, "t1", now), policy.Writable(valid, "t1", now))
}

func TestPolicyDisabledAcceptsEverything(t *testing.T) {
	policy := Policy{}
	require.Equal(t, "", policy.Check(Metadata{}, "", 0))
	require.False(t, policy.WritableHolds(Metadata{}, "", 0))
}

func TestWritableHoldsRequiresTenantAndGeneration(t *testing.T) {
	policy := Policy{Enabled: true, Generation: "g2", TTLSeconds: 60}
	now := int64(1_700_000_000)
	require.True(t, policy.WritableHolds(Metadata{WrittenAt: now, Tenant: "t1", Generation: "g2", Complete: true}, "t1", now))
	require.False(t, policy.WritableHolds(Metadata{WrittenAt: now, Tenant: "", Generation: "g2", Complete: true}, "", now))
	require.False(t, policy.WritableHolds(Metadata{WrittenAt: now, Tenant: "t1", Generation: "g2", Complete: true}, "", now))
	require.False(t, policy.WritableHolds(Metadata{WrittenAt: now, Tenant: "t1", Generation: "", Complete: true}, "t1", now))
	require.False(t, policy.WritableHolds(Metadata{WrittenAt: now, Tenant: "t1", Generation: "g2", Complete: false}, "t1", now))
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	metadata := Metadata{WrittenAt: 1_700_000_000, Tenant: "tenant-a", Generation: "g3", Complete: true}
	raw := Encode(metadata, "the answer")

	decoded, answer, status := Decode(raw)
	require.Equal(t, DecodeOK, status)
	require.Equal(t, metadata, decoded)
	require.Equal(t, "the answer", answer)
}

func TestDecodeClassifiesLegacyAndMalformed(t *testing.T) {
	cases := []struct {
		name   string
		raw    string
		status DecodeStatus
	}{
		{name: "plain old answer", raw: "the answer", status: DecodeLegacy},
		{name: "old json answer", raw: `{"foo":"bar"}`, status: DecodeLegacy},
		{name: "empty", raw: "", status: DecodeLegacy},
		{name: "wrong version", raw: `{"higressEntry":9}`, status: DecodeLegacy},
		{name: "truncated json", raw: `{"higressEntry":1,"answer":"a"`, status: DecodeMalformed},
		{name: "missing writtenAt", raw: `{"higressEntry":1,"tenant":"t","generation":"g","complete":true,"answer":"a"}`, status: DecodeMalformed},
		{name: "non-numeric writtenAt", raw: `{"higressEntry":1,"writtenAt":"x","tenant":"t","generation":"g","complete":true,"answer":"a"}`, status: DecodeMalformed},
		{name: "non-bool complete", raw: `{"higressEntry":1,"writtenAt":1,"tenant":"t","generation":"g","complete":"yes","answer":"a"}`, status: DecodeMalformed},
		{name: "missing answer", raw: `{"higressEntry":1,"writtenAt":1,"tenant":"t","generation":"g","complete":true}`, status: DecodeMalformed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, status := Decode(tc.raw)
			require.Equal(t, tc.status, status)
		})
	}
}

func TestRemaining(t *testing.T) {
	policy := Policy{Enabled: true, TTLSeconds: 100}
	now := int64(1_000)
	require.Equal(t, 60, policy.Remaining(Metadata{WrittenAt: now - 40}, now))
	require.Equal(t, -1, policy.Remaining(Metadata{WrittenAt: now - 100}, now))
	require.Equal(t, 0, Policy{}.Remaining(Metadata{WrittenAt: now - 100}, now))
}

func TestPolicyFromJsonValidation(t *testing.T) {
	// Disabled ignores illegal-looking values so existing configs keep loading.
	var disabled Policy
	disabled.FromJson(gjson.Parse(`{"enabled":false,"generation":""}`))
	disabled.Resolve(0)
	require.NoError(t, disabled.Validate())

	var missingGeneration Policy
	missingGeneration.FromJson(gjson.Parse(`{"enabled":true}`))
	require.ErrorContains(t, missingGeneration.Validate(), "generation is required")

	var negativeTTL Policy
	negativeTTL.FromJson(gjson.Parse(`{"enabled":true,"generation":"g1","ttlSeconds":-1}`))
	require.ErrorContains(t, negativeTTL.Validate(), "ttlSeconds")

	var ok Policy
	ok.FromJson(gjson.Parse(`{"enabled":true,"generation":"g1","ttlSeconds":30}`))
	require.NoError(t, ok.Validate())
	require.Equal(t, 30, ok.TTLSeconds)

	// Inheriting the legacy lifetime keeps existing behavior.
	var inherited Policy
	inherited.FromJson(gjson.Parse(`{"enabled":true,"generation":"g1"}`))
	inherited.Resolve(7200)
	require.Equal(t, 7200, inherited.TTLSeconds)
	require.NoError(t, inherited.Validate())
}

func TestRedactIsStableAndOpaque(t *testing.T) {
	require.Equal(t, Redact("tenant-a"), Redact("tenant-a"))
	require.NotEqual(t, Redact("tenant-a"), Redact("tenant-b"))
	require.NotContains(t, Redact("tenant-a"), "tenant-a")
	require.Empty(t, Redact(""))
	require.Contains(t, Redact("x"), "sha256:")
}
