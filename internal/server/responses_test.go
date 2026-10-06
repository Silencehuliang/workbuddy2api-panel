package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// ---------- 请求翻译 ----------

// TestResponsesTranslateBasics Responses 请求 → chat 请求体的基本映射：
// instructions→system、developer→system、input_text→content、工具扁平→嵌套、
// reasoning.effort→reasoning_effort、max_output_tokens→max_completion_tokens，
// 以及 store/include/prompt_cache_key 等网关侧不入站字段的丢弃。
func TestResponsesTranslateBasics(t *testing.T) {
	raw := []byte(`{
	  "model": "gpt-5.3-codex",
	  "instructions": "You are a coding agent.",
	  "input": [
	    {"type":"message","role":"developer","content":[{"type":"input_text","text":"项目规范"}]},
	    {"type":"message","role":"user","content":[{"type":"input_text","text":"列个目录"}]}
	  ],
	  "tools": [
	    {"type":"function","name":"exec_command","description":"跑命令","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"strict":false},
	    {"type":"web_search"}
	  ],
	  "tool_choice": "auto",
	  "reasoning": {"effort":"medium","summary":"auto"},
	  "max_output_tokens": 4096,
	  "stream": true,
	  "store": false,
	  "include": ["reasoning.encrypted_content"],
	  "prompt_cache_key": "sess-1",
	  "client_metadata": {"x": 1}
	}`)

	out, err := responsesFE{}.translate(raw)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(out, &body); err != nil {
		t.Fatalf("translated body is not JSON: %v", err)
	}
	if body["model"] != "gpt-5.3-codex" {
		t.Errorf("model=%v", body["model"])
	}
	// 服务端状态类字段不得进 chat 出站体。
	for _, k := range []string{"store", "include", "client_metadata", "input", "instructions"} {
		if _, ok := body[k]; ok {
			t.Errorf("field %q must not leak into chat body", k)
		}
	}
	// prompt_cache_key 例外：原值保留（注入器优先级「客户端自带则不覆盖」）。
	if body["prompt_cache_key"] != "sess-1" {
		t.Errorf("client prompt_cache_key must be preserved, got %v", body["prompt_cache_key"])
	}
	if body["reasoning_effort"] != "medium" {
		t.Errorf("reasoning.effort must map to reasoning_effort, got %v", body["reasoning_effort"])
	}
	if body["max_completion_tokens"].(float64) != 4096 {
		t.Errorf("max_output_tokens must map to max_completion_tokens, got %v", body["max_completion_tokens"])
	}
	if body["stream"] != true {
		t.Errorf("stream must be preserved")
	}

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%d want 3 (system + 2 input messages), got %#v", len(msgs), msgs)
	}
	m0 := msgs[0].(map[string]any)
	if m0["role"] != "system" || m0["content"] != "You are a coding agent." {
		t.Errorf("instructions must become first system message, got %#v", m0)
	}
	// developer 角色映射为 system（网关 prompt 层按 system 处理）。
	m1 := msgs[1].(map[string]any)
	if m1["role"] != "system" || m1["content"] != "项目规范" {
		t.Errorf("developer role must map to system, got %#v", m1)
	}
	m2 := msgs[2].(map[string]any)
	if m2["role"] != "user" || m2["content"] != "列个目录" {
		t.Errorf("user message mismatch: %#v", m2)
	}

	// 工具：嵌套 shape；非 function 类型丢弃。
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools=%d want 1 (web_search must be dropped), got %#v", len(tools), tools)
	}
	tl := tools[0].(map[string]any)
	if tl["type"] != "function" {
		t.Errorf("tool type=%v", tl["type"])
	}
	fn, ok := tl["function"].(map[string]any)
	if !ok {
		t.Fatalf("tool must be nested under function, got %#v", tl)
	}
	if fn["name"] != "exec_command" || fn["description"] != "跑命令" {
		t.Errorf("tool function mismatch: %#v", fn)
	}
	if _, ok := fn["parameters"].(map[string]any); !ok {
		t.Errorf("tool parameters must be preserved: %#v", fn)
	}
}

// TestResponsesTranslateFunctionCallRoundTrip function_call / function_call_output
// 必须转成 chat 的 tool_calls / role:"tool" 且 call_id 对齐（否则上游报 tool 配对断裂）；
// reasoning 项丢弃。
func TestResponsesTranslateFunctionCallRoundTrip(t *testing.T) {
	raw := []byte(`{
	  "model": "m",
	  "input": [
	    {"type":"message","role":"user","content":"看看目录"},
	    {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"xxx"},
	    {"type":"function_call","id":"fc_1","call_id":"call_abc","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"},
	    {"type":"function_call_output","call_id":"call_abc","output":"file1\nfile2"}
	  ]
	}`)
	out, err := responsesFE{}.translate(raw)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(out, &body)
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%d want 3 (reasoning dropped), got %#v", len(msgs), msgs)
	}
	asst := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("function_call must become assistant message, got %#v", asst)
	}
	tcs, _ := asst["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls=%#v", asst["tool_calls"])
	}
	tc := tcs[0].(map[string]any)
	if tc["id"] != "call_abc" {
		t.Errorf("tool_call id must be call_id, got %v", tc["id"])
	}
	tfn := tc["function"].(map[string]any)
	if tfn["name"] != "exec_command" || tfn["arguments"] != `{"cmd":"ls"}` {
		t.Errorf("tool_call function mismatch: %#v", tfn)
	}
	tool := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_abc" || tool["content"] != "file1\nfile2" {
		t.Errorf("function_call_output must become role:tool with paired call id, got %#v", tool)
	}
}

// TestResponsesTranslateImagePart 含图请求：input_image 必须转成 chat 的 image_url
// 形态（url 为字符串 → {url:...} 对象），且形态必须是 part 数组（不能压成字符串）。
func TestResponsesTranslateImagePart(t *testing.T) {
	raw := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[
	  {"type":"input_text","text":"这张图里是什么"},
	  {"type":"input_image","image_url":"data:image/png;base64,AAAA"}
	]}]}`)
	out, err := responsesFE{}.translate(raw)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(out, &body)
	msgs, _ := body["messages"].([]any)
	parts, ok := msgs[0].(map[string]any)["content"].([]any)
	if !ok {
		t.Fatalf("content with image must stay a part array, got %#v", msgs[0])
	}
	if len(parts) != 2 {
		t.Fatalf("parts=%#v", parts)
	}
	img := parts[1].(map[string]any)
	if img["type"] != "image_url" {
		t.Fatalf("image part type=%v want image_url", img["type"])
	}
	iu, ok := img["image_url"].(map[string]any)
	if !ok || iu["url"] != "data:image/png;base64,AAAA" {
		t.Errorf("image_url must be {url:...} object, got %#v", img["image_url"])
	}
}

// TestResponsesTranslateInputString input 允许是纯字符串（Responses 规范）。
func TestResponsesTranslateInputString(t *testing.T) {
	out, err := responsesFE{}.translate([]byte(`{"model":"m","input":"你好"}`))
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(out, &body)
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 1 || msgs[0].(map[string]any)["content"] != "你好" {
		t.Errorf("string input must become a single user message, got %#v", msgs)
	}
}

func TestResponsesTranslateInvalidJSON(t *testing.T) {
	if _, err := (responsesFE{}).translate([]byte(`{not json`)); err == nil {
		t.Fatal("invalid JSON must return an error (handler maps it to 400)")
	}
}

// ---------- 流式事件序列 ----------

// rEvent 解析出的 Responses SSE 事件（Name == "done" 表示 data: [DONE]）。
type rEvent struct {
	Name string
	Data map[string]any
}

func parseResponsesSSE(t *testing.T, raw string) []rEvent {
	t.Helper()
	var out []rEvent
	cur := ""
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "event: "):
			cur = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				out = append(out, rEvent{Name: "done"})
				continue
			}
			var m map[string]any
			if err := json.Unmarshal([]byte(payload), &m); err != nil {
				t.Fatalf("data line is not JSON: %v (%s)", err, payload)
			}
			out = append(out, rEvent{Name: cur, Data: m})
		}
	}
	return out
}

func eventNames(evs []rEvent) []string {
	names := make([]string, 0, len(evs))
	for _, e := range evs {
		names = append(names, e.Name)
	}
	return names
}

func runResponsesSink(t *testing.T, upstreamSSE string) ([]rEvent, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	sink := newResponsesSink(rec, frontendMeta{
		Model:     "gpt-5.3-codex",
		BareModel: "gpt-5.3-codex",
		CreatedAt: time.Unix(1753600000, 0),
	})
	err := upstream.StreamTo(sink, strings.NewReader(upstreamSSE))
	if err != nil && !upstream.IsEmptyStreamError(err) {
		t.Fatalf("StreamTo: %v", err)
	}
	return parseResponsesSSE(t, rec.Body.String()), rec.Body.String()
}

// TestResponsesSinkTextAndReasoning 推理 + 正文流：事件序列必须严格是
// created → 推理项 → 消息项 → completed → [DONE]，且 completed 回放完整 output
// 与 usage（Codex 靠它构造下一轮 input，store:false 下没有服务端状态）。
func TestResponsesSinkTextAndReasoning(t *testing.T) {
	sse := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"先想\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"一下\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"世界\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n" +
		"data: [DONE]\n\n"

	evs, raw := runResponsesSink(t, sse)
	want := []string{
		"response.created",
		"response.output_item.added",             // reasoning
		"response.reasoning_summary_part.added",  // reasoning
		"response.reasoning_summary_text.delta",  // 先想
		"response.reasoning_summary_text.delta",  // 一下
		"response.reasoning_summary_text.done",   // 正文开始 → 收推理项
		"response.reasoning_summary_part.done",   //
		"response.output_item.done",              // reasoning 项闭合
		"response.output_item.added",             // message
		"response.content_part.added",            // message
		"response.output_text.delta",             // 你好
		"response.output_text.delta",             // 世界
		"response.output_text.done",              // Finish 收尾
		"response.content_part.done",             //
		"response.output_item.done",              // message 项闭合
		"response.completed",
		"done",
	}
	if got := eventNames(evs); !equalStrings(got, want) {
		t.Fatalf("event sequence mismatch:\n got %v\nwant %v\nraw:\n%s", got, want, raw)
	}

	// 幂等：sequence_number 必须严格递增（Codex 按序处理事件）。
	for i, e := range evs {
		if e.Name == "done" {
			continue
		}
		if int(e.Data["sequence_number"].(float64)) != i {
			t.Fatalf("event %d (%s) sequence_number=%v want %d", i, e.Name, e.Data["sequence_number"], i)
		}
	}

	created := evs[0].Data["response"].(map[string]any)
	if created["status"] != "in_progress" || created["model"] != "gpt-5.3-codex" || created["object"] != "response" {
		t.Errorf("response.created payload mismatch: %#v", created)
	}

	// reasoning 项：装配后的摘要文本完整。
	reasonAdded := evs[1].Data["item"].(map[string]any)
	if reasonAdded["type"] != "reasoning" {
		t.Errorf("first output item must be reasoning, got %#v", reasonAdded)
	}

	// 正文增量文本。
	var text strings.Builder
	for _, e := range evs {
		if e.Name == "response.output_text.delta" {
			text.WriteString(e.Data["delta"].(string))
		}
	}
	if text.String() != "你好世界" {
		t.Errorf("text deltas=%q want 你好世界", text.String())
	}

	completed := evs[len(evs)-2]
	if completed.Name != "response.completed" {
		t.Fatalf("terminal event must be response.completed, got %s", completed.Name)
	}
	resp := completed.Data["response"].(map[string]any)
	if resp["status"] != "completed" {
		t.Errorf("completed status=%v", resp["status"])
	}
	output, _ := resp["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("completed output=%d items want 2 (reasoning + message), got %#v", len(output), output)
	}
	rItem := output[0].(map[string]any)
	if rItem["type"] != "reasoning" {
		t.Fatalf("output[0] type=%v", rItem["type"])
	}
	summary := rItem["summary"].([]any)[0].(map[string]any)
	if summary["text"] != "先想一下" {
		t.Errorf("reasoning summary=%q want 先想一下", summary["text"])
	}
	mItem := output[1].(map[string]any)
	if mItem["type"] != "message" || mItem["role"] != "assistant" || mItem["status"] != "completed" {
		t.Fatalf("output[1]=%#v", mItem)
	}
	content := mItem["content"].([]any)[0].(map[string]any)
	if content["text"] != "你好世界" {
		t.Errorf("message content=%q want 你好世界", content["text"])
	}

	// usage：chat 口径 → Responses 口径（含缓存明细）。
	u := resp["usage"].(map[string]any)
	if numOf(t, u["input_tokens"]) != 10 || numOf(t, u["output_tokens"]) != 5 || numOf(t, u["total_tokens"]) != 15 {
		t.Errorf("usage token mapping: %#v", u)
	}
	if got := numOf(t, u["input_tokens_details"].(map[string]any)["cached_tokens"]); got != 4 {
		t.Errorf("cached_tokens=%v want 4", got)
	}
}

// TestResponsesSinkToolCall 工具调用流：function_call 项必须带 call_id 与完整
// arguments（分片累加），completed 回放的项要能被 Codex 直接用作下一轮 function_call。
func TestResponsesSinkToolCall(t *testing.T) {
	sse := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"exec_command\",\"arguments\":\"\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls\\\"}\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	evs, raw := runResponsesSink(t, sse)
	var added map[string]any
	var argDeltas []string
	for _, e := range evs {
		switch e.Name {
		case "response.output_item.added":
			added = e.Data["item"].(map[string]any)
		case "response.function_call_arguments.delta":
			argDeltas = append(argDeltas, e.Data["delta"].(string))
		}
	}
	if added == nil {
		t.Fatalf("no function_call output_item.added:\n%s", raw)
	}
	if added["type"] != "function_call" || added["call_id"] != "call_1" || added["name"] != "exec_command" {
		t.Fatalf("function_call item=%#v", added)
	}
	if added["status"] != "in_progress" {
		t.Errorf("added item status=%v want in_progress", added["status"])
	}
	if strings.Join(argDeltas, "") != `{"cmd":"ls"}` {
		t.Errorf("argument deltas=%v", argDeltas)
	}

	completed := evs[len(evs)-2]
	if completed.Name != "response.completed" {
		t.Fatalf("terminal=%s", completed.Name)
	}
	resp := completed.Data["response"].(map[string]any)
	output := resp["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%#v", output)
	}
	fc := output[0].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_1" ||
		fc["name"] != "exec_command" || fc["arguments"] != `{"cmd":"ls"}` || fc["status"] != "completed" {
		t.Fatalf("completed function_call item=%#v", fc)
	}
}

// TestResponsesSinkErrorFrame 上游 error 帧（6004 限流等）→ response.failed：
// 上游 error 原文（code/message/requestId）必须保留，且**不得**再发 response.completed
// （假成功会让 Codex 以为这一轮正常结束）。
func TestResponsesSinkErrorFrame(t *testing.T) {
	sse := "data: {\"error\":{\"code\":6004,\"message\":\"model rate limited\",\"requestId\":\"req-9\"}}\n\n" +
		"data: [DONE]\n\n"
	rec := httptest.NewRecorder()
	sink := newResponsesSink(rec, frontendMeta{
		Model: "m", CreatedAt: time.Unix(1753600000, 0),
		HintFn: func(string) string { return "hint-text" },
	})
	err := upstream.StreamTo(sink, strings.NewReader(sse))
	if err != nil {
		t.Fatalf("error frame is a valid frame, StreamTo must not fail: %v", err)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	names := eventNames(evs)
	if !equalStrings(names, []string{"response.created", "response.failed", "done"}) {
		t.Fatalf("events=%v want [response.created response.failed done]", names)
	}
	failed := evs[1].Data["response"].(map[string]any)
	if failed["status"] != "failed" {
		t.Errorf("failed status=%v", failed["status"])
	}
	errObj := failed["error"].(map[string]any)
	if errObj["message"] != "model rate limited" || errObj["requestId"] != "req-9" {
		t.Errorf("upstream error must be preserved verbatim: %#v", errObj)
	}
	if errObj["code"].(float64) != 6004 {
		t.Errorf("upstream error code lost: %#v", errObj)
	}
	if errObj["gateway_hint"] != "hint-text" {
		t.Errorf("gateway_hint missing: %#v", errObj)
	}
}

// TestResponsesSinkEmptyStream 上游 200 空流 → response.failed（+ errEmptyStream 供
// 调用方把观测收敛到失败口径），不能出现 response.completed。
func TestResponsesSinkEmptyStream(t *testing.T) {
	rec := httptest.NewRecorder()
	sink := newResponsesSink(rec, frontendMeta{Model: "m", CreatedAt: time.Unix(1753600000, 0)})
	err := upstream.StreamTo(sink, strings.NewReader("data: [DONE]\n\n"))
	if !upstream.IsEmptyStreamError(err) {
		t.Fatalf("empty stream must return errEmptyStream, got %v", err)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	names := eventNames(evs)
	if !equalStrings(names, []string{"response.created", "response.failed", "done"}) {
		t.Fatalf("events=%v want [response.created response.failed done]", names)
	}
	resp := evs[1].Data["response"].(map[string]any)
	if resp["status"] != "failed" {
		t.Errorf("status=%v want failed", resp["status"])
	}
}

// TestResponsesSinkNoChunksAtAll 上游连一个 data 帧都没有（纯空）：仍必须以
// response.created 开流——Codex 在缺该事件时报 "stream closed before response.completed"。
func TestResponsesSinkNoChunksAtAll(t *testing.T) {
	rec := httptest.NewRecorder()
	sink := newResponsesSink(rec, frontendMeta{Model: "m", CreatedAt: time.Unix(1753600000, 0)})
	_ = upstream.StreamTo(sink, strings.NewReader(""))
	evs := parseResponsesSSE(t, rec.Body.String())
	if len(evs) == 0 || evs[0].Name != "response.created" {
		t.Fatalf("stream must start with response.created, got %v", eventNames(evs))
	}
}

// ---------- 非流式 ----------

// TestResponsesFromAggregate 非流式聚合响应 → Responses 响应对象。
func TestResponsesFromAggregate(t *testing.T) {
	agg := map[string]any{
		"id":    "chatcmpl-1",
		"model": "m",
		"choices": []any{map[string]any{
			"index": 0.0,
			"message": map[string]any{
				"role":              "assistant",
				"content":           "答案",
				"reasoning_content": "推理",
				"tool_calls": []any{map[string]any{
					"id": "call_1", "type": "function",
					"function": map[string]any{"name": "exec_command", "arguments": `{"cmd":"ls"}`},
				}},
			},
			"finish_reason": "tool_calls",
		}},
		"usage": map[string]any{"prompt_tokens": 3.0, "completion_tokens": 2.0, "total_tokens": 5.0},
	}
	resp := responsesFromAggregate(agg, frontendMeta{Model: "gpt-5.3-codex", CreatedAt: time.Unix(1753600000, 0)})
	if resp["object"] != "response" || resp["status"] != "completed" || resp["model"] != "gpt-5.3-codex" {
		t.Fatalf("envelope=%#v", resp)
	}
	output, _ := resp["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("output=%d want 3 (reasoning + message + function_call): %#v", len(output), output)
	}
	if output[0].(map[string]any)["type"] != "reasoning" {
		t.Errorf("output[0]=%#v", output[0])
	}
	if output[1].(map[string]any)["type"] != "message" {
		t.Errorf("output[1]=%#v", output[1])
	}
	fc := output[2].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_1" || fc["arguments"] != `{"cmd":"ls"}` {
		t.Errorf("output[2]=%#v", fc)
	}
	u := resp["usage"].(map[string]any)
	if numOf(t, u["input_tokens"]) != 3 || numOf(t, u["output_tokens"]) != 2 || numOf(t, u["total_tokens"]) != 5 {
		t.Errorf("usage=%#v", u)
	}
}

// numOf 取数值字段（Go map 里可能是 int64，经 JSON 往返后是 float64——两条路径都要能断言）。
func numOf(t *testing.T, v any) float64 {
	t.Helper()
	switch n := v.(type) {
	case float64:
		return n
	case int64:
		return float64(n)
	case int:
		return float64(n)
	}
	t.Fatalf("not a number: %#v", v)
	return 0
}

// TestResponsesUsageRequiredDetailKeys Codex 的 ResponseUsage 把两个明细字段定为
// **必填**（非 Option）：缺 `cached_tokens` / `reasoning_tokens` 会让整个
// response.completed 帧解析失败——实测 Codex 0.160.0 报
// "failed to parse ResponseCompleted: missing field `cached_tokens`" 后判定流中断并
// 反复重连（3 次请求才算一轮）。故无论上游是否给缓存/推理明细，键必须恒在。
func TestResponsesUsageRequiredDetailKeys(t *testing.T) {
	cases := []map[string]any{
		// 无任何明细
		{"prompt_tokens": 10.0, "completion_tokens": 5.0, "total_tokens": 15.0},
		// 仅缓存命中明细
		{"prompt_tokens": 10.0, "completion_tokens": 5.0, "total_tokens": 15.0,
			"prompt_tokens_details": map[string]any{"cached_tokens": 4.0}},
		// 仅推理明细
		{"prompt_tokens": 10.0, "completion_tokens": 5.0, "total_tokens": 15.0,
			"completion_tokens_details": map[string]any{"reasoning_tokens": 3.0}},
	}
	for i, u := range cases {
		got := responsesUsage(u)
		in, ok := got["input_tokens_details"].(map[string]any)
		if !ok {
			t.Fatalf("case %d: input_tokens_details missing: %#v", i, got)
		}
		if _, ok := in["cached_tokens"]; !ok {
			t.Errorf("case %d: cached_tokens must always be present (Codex required field): %#v", i, in)
		}
		outD, ok := got["output_tokens_details"].(map[string]any)
		if !ok {
			t.Fatalf("case %d: output_tokens_details missing: %#v", i, got)
		}
		if _, ok := outD["reasoning_tokens"]; !ok {
			t.Errorf("case %d: reasoning_tokens must always be present (Codex required field): %#v", i, outD)
		}
	}
	// 上游零用量观测时同样必须带键（客户端读 completed.usage，缺键=整帧解析失败）。
	e := emptyResponsesUsage()
	if _, ok := e["input_tokens_details"].(map[string]any)["cached_tokens"]; !ok {
		t.Errorf("emptyResponsesUsage must carry cached_tokens: %#v", e)
	}
	if _, ok := e["output_tokens_details"].(map[string]any)["reasoning_tokens"]; !ok {
		t.Errorf("emptyResponsesUsage must carry reasoning_tokens: %#v", e)
	}
}

// TestResponsesFromAggregateWithoutUsage usage 缺失时给零值对象（而非省略）：
// 客户端读 response.usage 统计 token，缺失比零值更易触发告警。
func TestResponsesFromAggregateWithoutUsage(t *testing.T) {
	agg := map[string]any{"choices": []any{}}
	resp := responsesFromAggregate(agg, frontendMeta{Model: "m", CreatedAt: time.Unix(0, 0)})
	if _, ok := resp["usage"].(map[string]any); !ok {
		t.Fatalf("usage must be present even when absent upstream: %#v", resp)
	}
	if _, ok := resp["output"].([]any); !ok {
		t.Fatalf("output must be an array: %#v", resp)
	}
}

// ---------- 端到端 ----------

// TestResponsesEndpointEndToEnd 走完整 handler：POST /v1/responses 必须
// 1) 以 SSE 应答且事件序列完整；2) 出站上游体是**翻译后的 chat 体**（instructions→system、
// 工具嵌套、reasoning_effort）；3) 与 chat 端点共用同一套调度（成功率/粘性记账）。
func TestResponsesEndpointEndToEnd(t *testing.T) {
	var mu sync.Mutex
	var upstreamBody string
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			upstreamBody = string(b)
			mu.Unlock()
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(
					"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-5.3-codex\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"}}]}\n\n" +
						"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1,\"total_tokens\":8}}\n\n" +
						"data: [DONE]\n\n")),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	reqBody := `{"model":"gpt-5.3-codex","instructions":"You are a coding agent.","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"exec_command","description":"d","parameters":{"type":"object"}}],"reasoning":{"effort":"high"},"prompt_cache_key":"sess-42","store":false,"stream":true}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(reqBody)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type=%q want text/event-stream", ct)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	names := eventNames(evs)
	if names[0] != "response.created" || names[len(names)-2] != "response.completed" || names[len(names)-1] != "done" {
		t.Fatalf("event sequence incomplete: %v", names)
	}

	mu.Lock()
	got := upstreamBody
	mu.Unlock()
	var ub map[string]any
	if err := json.Unmarshal([]byte(got), &ub); err != nil {
		t.Fatalf("upstream body not JSON: %v (%s)", err, got)
	}
	msgs, _ := ub["messages"].([]any)
	if len(msgs) == 0 || msgs[0].(map[string]any)["role"] != "system" {
		t.Fatalf("instructions must reach upstream as system message: %#v", ub["messages"])
	}
	if msgs[0].(map[string]any)["content"] != "You are a coding agent." {
		t.Errorf("system content=%v", msgs[0].(map[string]any)["content"])
	}
	if ub["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort must reach upstream, got %v", ub["reasoning_effort"])
	}
	tools, _ := ub["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["function"] == nil {
		t.Fatalf("tools must reach upstream in nested shape: %#v", ub["tools"])
	}
	// 服务端状态类字段不得出站；prompt_cache_key 保留客户端原值并直达上游
	// （上游前缀缓存按会话复用；注入器见客户端已带则不覆盖）。
	for _, k := range []string{"store", "include", "instructions", "input"} {
		if _, ok := ub[k]; ok {
			t.Errorf("gateway-side field %q leaked upstream: %s", k, got)
		}
	}
	if ub["prompt_cache_key"] != "sess-42" {
		t.Errorf("client prompt_cache_key must reach upstream verbatim, got %v", ub["prompt_cache_key"])
	}
	// 与 chat 端点同源的成功记账：账号被记成功（非熔断/非冷却）。
	st, _ := p.Status("u1")
	if st.Disabled || st.Cooling {
		t.Errorf("successful responses request must not penalize account: %+v", st)
	}
}

// TestResponsesEndpointBadBodyIsRequestError 非法 JSON → 400 且不打上游
// （请求形态问题，与账号无关；换号必然同样失败）。
func TestResponsesEndpointBadBodyIsRequestError(t *testing.T) {
	var calls int
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		calls++
		return 200, sseOK, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{bad json`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400", rec.Code)
	}
	if calls != 0 {
		t.Errorf("upstream calls=%d want 0", calls)
	}
}

// TestResponsesEndpointRequiresAuth 与 chat 端点同口径的 Bearer 鉴权。
func TestResponsesEndpointRequiresAuth(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "secret"})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"m","input":"hi"}`)))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("code=%d want 401 without api key", rec.Code)
	}
}

// TestRequestTracePathPerEndpoint 归档路径按端点区分：两个协议共用同一套观测字段，
// 只有 path 能说明客户端用的是 chat 还是 responses（此前硬编码 chat，responses 流量
// 在面板/日志里被标成 chat.completions）。空 path 回退 chat，兼容直接构造 trace 的调用点。
func TestRequestTracePathPerEndpoint(t *testing.T) {
	if got := (&requestTrace{path: "/v1/responses"}).event(200).Path; got != "/v1/responses" {
		t.Errorf("responses request path=%q want /v1/responses", got)
	}
	if got := (&requestTrace{path: "/v1/chat/completions"}).event(200).Path; got != "/v1/chat/completions" {
		t.Errorf("chat request path=%q", got)
	}
	if got := (&requestTrace{}).event(200).Path; got != "/v1/chat/completions" {
		t.Errorf("empty path must fall back to chat, got %q", got)
	}
}

// ---------- chat 回归 ----------

// TestChatFrontendIsIdentity 重构后 chat 前端必须是恒等翻译：任何 body 逐字节不变，
// 且出帧走既有 StreamHint 路径（本仓库全部 chat 测试即为该契约的回归面）。
func TestChatFrontendIsIdentity(t *testing.T) {
	raw := []byte(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	out, err := chatFE{}.translate(raw)
	if err != nil {
		t.Fatalf("chat translate must not fail: %v", err)
	}
	if string(out) != string(raw) {
		t.Errorf("chat translate must be identity, got %s", out)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
