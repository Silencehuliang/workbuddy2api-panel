// responses.go OpenAI Responses API 前端（POST /v1/responses）。
//
// 为什么需要它：Codex 0.160.0 起 `wire_api = "chat"` 被上游客户端移除
// （配置直接报 "`wire_api = \"chat\"` is no longer supported"），只认
// wire_api = "responses"，即 POST /v1/responses。网关此前只提供 chat 协议，
// Codex 请求会落到 Go 默认 mux 的 404。本文件把 Responses 协议翻译到既有 chat
// 管线，使 Codex 能用上网关的全部账号调度能力。
//
// 职责边界（见 frontend.go）：
//   - translate：入站 Responses 请求 → 出站 chat 请求体（含 instructions/input/tools
//     的结构差异与 developer→system、扁平 function 工具→嵌套 shape 的改写）；
//   - stream：上游归一化 chat 增量帧 → Responses 事件序列（FrameSink 实现）；
//   - writeAggregate：聚合出的 chat.completion → Responses 响应对象（非流式）。
//
// 事件序列（经 Codex 0.160.0 实测跑通）：
//
//	response.created → [推理项 events] → [消息项 events] → [工具调用项 events]
//	→ response.completed → data: [DONE]
//
// 每类输出项（message / reasoning / function_call）以 output_item.added 开、
// output_item.done 收，中间是该类型的增量事件；response.completed 必须回放完整
// output 数组与 usage——Codex 用它构造下一轮的 input（store:false 下无服务端状态）。
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/linguo2625469/workbuddy2api-panel/internal/session"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// responsesFE Responses 前端。
type responsesFE struct{}

// translate 把 Responses 请求翻译成 chat 请求体。
//
// 映射关系：
//   - instructions                → 首条 system 消息
//   - input[] message             → messages（developer→system；input_text/output_text→正文；
//     input_image→image_url part）
//   - input[] function_call       → assistant 消息 + tool_calls（call_id 作 tool_call id，
//     与 function_call_output 的 call_id 对齐）
//   - input[] function_call_output→ role:"tool" + tool_call_id
//   - input[] reasoning           → 丢弃（思维链不参与上下文回放；网关侧由
//     reasoning_content 出帧，客户端不会把明文推理塞回来）
//   - tools[]（扁平）              → chat 的嵌套 {type:"function",function:{...}}
//   - reasoning.effort            → reasoning_effort（上游 payload/thinking 链路认这个字段）
//   - max_output_tokens           → max_completion_tokens（payload.go 会翻译成上游认的 max_tokens）
//   - prompt_cache_key            → 原值保留（客户端显式带 key 时注入器不覆盖）
//
// 不支持项（本地工具 / 服务端状态）直接丢弃：store、include、client_metadata、
// 非 function 类型的工具（web_search 等网关无对应能力）。
func (responsesFE) translate(raw []byte) ([]byte, error) {
	var req map[string]any
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("invalid JSON body: %v", err)
	}
	out := map[string]any{}
	if v, ok := req["model"].(string); ok && v != "" {
		out["model"] = v
	}

	var msgs []any
	if s, ok := req["instructions"].(string); ok && strings.TrimSpace(s) != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": s})
	}
	switch in := req["input"].(type) {
	case string:
		if in != "" {
			msgs = append(msgs, map[string]any{"role": "user", "content": in})
		}
	case []any:
		for _, it := range in {
			item, ok := it.(map[string]any)
			if !ok {
				continue
			}
			msgs = append(msgs, responsesItemToMessages(item)...)
		}
	}
	out["messages"] = msgs

	// 工具：Responses 的 function 工具是扁平结构（name/description/parameters 在顶层），
	// chat 规范嵌在 function 之下。非 function 类型（web_search / file_search 等本地
	// 工具）网关无对应能力，丢弃——转发会让上游整请求 400，不如降级为「无该工具」。
	if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
		conv := make([]any, 0, len(tools))
		for _, t := range tools {
			tm, ok := t.(map[string]any)
			if !ok || tm["type"] != "function" {
				continue
			}
			name, _ := tm["name"].(string)
			if name == "" {
				continue
			}
			fn := map[string]any{"name": name}
			if d, ok := tm["description"].(string); ok && d != "" {
				fn["description"] = d
			}
			if p, ok := tm["parameters"]; ok && p != nil {
				fn["parameters"] = p
			}
			if s, ok := tm["strict"]; ok && s != nil {
				fn["strict"] = s
			}
			conv = append(conv, map[string]any{"type": "function", "function": fn})
		}
		if len(conv) > 0 {
			out["tools"] = conv
		}
	}

	// tool_choice 形态两协议一致（"auto"/"none"/"required"/对象），原样透传，
	// 由 payload.go 的 normalizeToolChoice 做上游兼容收敛。
	if v, ok := req["tool_choice"]; ok && v != nil {
		out["tool_choice"] = v
	}
	// 推理档位：Responses 是嵌套对象 reasoning.effort，chat/上游认扁平 reasoning_effort。
	if r, ok := req["reasoning"].(map[string]any); ok {
		if e, ok := r["effort"].(string); ok && e != "" {
			out["reasoning_effort"] = e
		}
	}
	if v, ok := req["max_output_tokens"]; ok && v != nil {
		out["max_completion_tokens"] = v
	}
	for _, k := range []string{"temperature", "top_p", "stream"} {
		if v, ok := req[k]; ok && v != nil {
			out[k] = v
		}
	}
	// prompt_cache_key：客户端自带则原值保留——网关出站注入器的既有优先级规则是
	// 「客户端显式带 key → 绝不覆盖」（cache_key.go）。Codex 的该字段就是会话级
	// 稳定值，保留它同时喂给上游前缀缓存，与粘性路由指向同一会话，命中率最高。
	// 缺失时由注入器按账号+会话生成，行为不变。
	if v, ok := req["prompt_cache_key"].(string); ok && v != "" {
		out["prompt_cache_key"] = v
	}
	return json.Marshal(out)
}

// responsesItemToMessages 把一个 Responses 输入项翻译成 0..n 条 chat 消息。
func responsesItemToMessages(item map[string]any) []any {
	switch item["type"] {
	case "message":
		role, _ := item["role"].(string)
		switch role {
		case "developer":
			// Responses 的 developer 角色即 chat 的 system（网关 prompt 层也按 system 处理）。
			role = "system"
		case "user", "assistant", "system":
		default:
			// 未知角色（tool 等）已由 function_call_output 分支覆盖，其余丢弃。
			return nil
		}
		content, hasImage := responsesContentToChat(item["content"])
		if content == nil {
			return nil
		}
		_ = hasImage
		return []any{map[string]any{"role": role, "content": content}}

	case "function_call":
		callID, _ := item["call_id"].(string)
		name, _ := item["name"].(string)
		if callID == "" || name == "" {
			return nil
		}
		args, _ := item["arguments"].(string)
		return []any{map[string]any{
			"role":    "assistant",
			"content": "",
			"tool_calls": []any{map[string]any{
				"id":       callID,
				"type":     "function",
				"function": map[string]any{"name": name, "arguments": args},
			}},
		}}

	case "function_call_output":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			return nil
		}
		return []any{map[string]any{
			"role":         "tool",
			"tool_call_id": callID,
			"content":      responsesOutputText(item["output"]),
		}}
	}
	// reasoning / item_reference / 未知类型：不参与 chat 上下文。
	return nil
}

// responsesContentToChat 把 Responses 的 content（字符串或 part 数组）翻译成 chat content。
// 纯文本且单段时返回字符串（兼容性最好：上游与既有 payload 归一化都以字符串为主路径）；
// 含图片时返回 part 数组（chat 的 image_url 形态）。
func responsesContentToChat(v any) (any, bool) {
	switch c := v.(type) {
	case string:
		if c == "" {
			return nil, false
		}
		return c, false
	case []any:
		var textParts []string
		var withImage []any
		hasImage := false
		for _, p := range c {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch pm["type"] {
			case "input_text", "output_text", "text":
				if s, ok := pm["text"].(string); ok && s != "" {
					textParts = append(textParts, s)
					withImage = append(withImage, map[string]any{"type": "text", "text": s})
				}
			case "input_image", "image_url":
				// 两种形态：Responses 的 image_url 是字符串，chat 的是 {url:...} 对象。
				url := ""
				switch iu := pm["image_url"].(type) {
				case string:
					url = iu
				case map[string]any:
					url, _ = iu["url"].(string)
				}
				if url == "" {
					continue
				}
				hasImage = true
				withImage = append(withImage, map[string]any{
					"type":      "image_url",
					"image_url": map[string]any{"url": url},
				})
			}
		}
		if hasImage {
			if len(withImage) == 0 {
				return nil, false
			}
			return withImage, true
		}
		if len(textParts) == 0 {
			return nil, false
		}
		return strings.Join(textParts, "\n"), false
	}
	return nil, false
}

// responsesOutputText 取 function_call_output.output 的文本（字符串或 part 数组）。
func responsesOutputText(v any) string {
	switch o := v.(type) {
	case string:
		return o
	case []any:
		var b strings.Builder
		for _, p := range o {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if s, ok := pm["text"].(string); ok {
				b.WriteString(s)
			}
		}
		return b.String()
	}
	return ""
}

// stream 流式出帧：把上游归一化 chat 增量帧翻译成 Responses 事件序列。
func (responsesFE) stream(w http.ResponseWriter, r io.Reader, meta frontendMeta, opts ...upstream.StreamOption) error {
	return upstream.StreamTo(newResponsesSink(w, meta), r, opts...)
}

// writeAggregate 非流式出帧：聚合出的 chat.completion → Responses 响应对象。
func (responsesFE) writeAggregate(w http.ResponseWriter, agg map[string]any, meta frontendMeta) {
	writeJSON(w, http.StatusOK, responsesFromAggregate(agg, meta))
}

// responsesFromAggregate 把聚合的 chat.completion 转成 Responses 响应对象。
func responsesFromAggregate(agg map[string]any, meta frontendMeta) map[string]any {
	resp := map[string]any{
		"id":         newResponsesID("resp"),
		"object":     "response",
		"created_at": meta.CreatedAt.Unix(),
		"status":     "completed",
		"model":      meta.Model,
		"output":     []any{},
	}
	var items []any
	if choices, ok := agg["choices"].([]any); ok {
		for _, ci := range choices {
			c, ok := ci.(map[string]any)
			if !ok {
				continue
			}
			msg, ok := c["message"].(map[string]any)
			if !ok {
				continue
			}
			items = append(items, responsesItemsFromMessage(msg)...)
		}
	}
	resp["output"] = items
	if u, ok := agg["usage"].(map[string]any); ok {
		resp["usage"] = responsesUsage(u)
	} else {
		resp["usage"] = emptyResponsesUsage()
	}
	return resp
}

// responsesItemsFromMessage 把 chat message 转成 Responses 输出项（推理 / 正文 / 工具调用）。
func responsesItemsFromMessage(msg map[string]any) []any {
	var items []any
	if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
		items = append(items, reasoningItem(rc))
	}
	if c, ok := msg["content"].(string); ok && c != "" {
		items = append(items, messageItem(c))
	}
	if tcs, ok := msg["tool_calls"].([]any); ok {
		for _, tci := range tcs {
			tc, ok := tci.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := tc["function"].(map[string]any)
			name, _ := fn["name"].(string)
			args, _ := fn["arguments"].(string)
			if name == "" {
				continue
			}
			id, _ := tc["id"].(string)
			items = append(items, functionCallItem(id, name, args))
		}
	}
	return items
}

// responsesSink 把归一化 chat 增量帧翻译成 Responses 事件序列（upstream.FrameSink 实现）。
//
// 状态机：输出项按「先推理、再正文、最后工具调用」的实际到达顺序惰性开启；
// 开启即发 output_item.added（+ 该类型的 part.added），关闭即发 done 系列。
// 一旦正文开始，迟到的推理增量被丢弃——推理项若在正文项之后重开，两类项的事件会
// 交错，客户端对交错事件序列的容忍度未验证，而推理摘要本身不参与正确性。
type responsesSink struct {
	w    http.ResponseWriter
	fl   http.Flusher
	meta frontendMeta

	seq       int
	respID    string
	createdAt int64
	started   bool

	usage     map[string]any
	failed    bool
	textBegun bool

	output    []*responsesItem
	curMsg    *responsesItem
	curReason *responsesItem
	tools     map[int]*responsesItem
	toolOrder []int
}

func newResponsesSink(w http.ResponseWriter, meta frontendMeta) *responsesSink {
	fl, _ := w.(http.Flusher)
	return &responsesSink{
		w:         w,
		fl:        fl,
		meta:      meta,
		respID:    newResponsesID("resp"),
		createdAt: meta.CreatedAt.Unix(),
		tools:     map[int]*responsesItem{},
	}
}

func (s *responsesSink) SetHeaders() {
	h := s.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}

// emit 写一个 Responses 事件（event: <type> + data: <json>）并 flush。
func (s *responsesSink) emit(evType string, payload map[string]any) error {
	payload["type"] = evType
	payload["sequence_number"] = s.seq
	s.seq++
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(s.w, "event: "+evType+"\ndata: "+string(raw)+"\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// ensureStarted 惰性发 response.created：Codex 要求流以该事件开始，否则报
// "stream closed before response.completed"。所有出帧路径（含失败/空流）先经此处。
func (s *responsesSink) ensureStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	return s.emit("response.created", map[string]any{"response": s.responseObject("in_progress", false)})
}

// responseObject 构造 Responses 响应对象。withUsage=false 时省略 usage
// （response.created 阶段尚无用量）。
func (s *responsesSink) responseObject(status string, withUsage bool) map[string]any {
	resp := map[string]any{
		"id":         s.respID,
		"object":     "response",
		"created_at": s.createdAt,
		"status":     status,
		"model":      s.meta.Model,
		"output":     s.outputItems(),
	}
	if withUsage {
		if s.usage != nil {
			resp["usage"] = responsesUsage(s.usage)
		} else {
			resp["usage"] = emptyResponsesUsage()
		}
	}
	return resp
}

// outputItems 按顺序返回全部输出项的完成态快照。
func (s *responsesSink) outputItems() []any {
	items := make([]any, 0, len(s.output))
	for _, it := range s.output {
		items = append(items, it.wire("completed"))
	}
	return items
}

// openItem 开启一个新输出项并发出 added 事件（含类型专属的 part.added）。
// configure 在**发事件之前**填身份字段（function_call 的 call_id/name 必须出现在
// output_item.added 里——客户端据此知道正在调用哪个函数）。
func (s *responsesSink) openItem(kind string, configure func(*responsesItem)) (*responsesItem, error) {
	it := &responsesItem{
		index: len(s.output),
		kind:  kind,
		id:    newResponsesID(responsesIDPrefix(kind)),
		open:  true,
	}
	if configure != nil {
		configure(it)
	}
	s.output = append(s.output, it)
	if err := s.ensureStarted(); err != nil {
		return nil, err
	}
	if err := s.emit("response.output_item.added", map[string]any{
		"output_index": it.index,
		"item":         it.wire("in_progress"),
	}); err != nil {
		return nil, err
	}
	switch kind {
	case "message":
		return it, s.emit("response.content_part.added", map[string]any{
			"item_id":       it.id,
			"output_index":  it.index,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		})
	case "reasoning":
		return it, s.emit("response.reasoning_summary_part.added", map[string]any{
			"item_id":       it.id,
			"output_index":  it.index,
			"summary_index": 0,
			"part":          map[string]any{"type": "summary_text", "text": ""},
		})
	}
	return it, nil
}

// closeItem 关闭输出项：先发类型专属的 done 系列，再发 output_item.done。
func (s *responsesSink) closeItem(it *responsesItem) error {
	if it == nil || !it.open || it.closed {
		return nil
	}
	it.closed = true
	it.open = false
	switch it.kind {
	case "message":
		text := it.text.String()
		if err := s.emit("response.output_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0, "text": text,
		}); err != nil {
			return err
		}
		if err := s.emit("response.content_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}},
		}); err != nil {
			return err
		}
	case "reasoning":
		text := it.text.String()
		if err := s.emit("response.reasoning_summary_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0, "text": text,
		}); err != nil {
			return err
		}
		if err := s.emit("response.reasoning_summary_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": text},
		}); err != nil {
			return err
		}
	case "function_call":
		if err := s.emit("response.function_call_arguments.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "arguments": it.arguments.String(),
		}); err != nil {
			return err
		}
	}
	return s.emit("response.output_item.done", map[string]any{
		"output_index": it.index,
		"item":         it.wire("completed"),
	})
}

// Chunk 处理一个归一化 chat 增量帧。
func (s *responsesSink) Chunk(frame map[string]any) error {
	if u, ok := frame["usage"].(map[string]any); ok && u != nil {
		s.usage = mergeUsage(s.usage, u)
	}
	choices, _ := frame["choices"].([]any)
	for _, ci := range choices {
		c, ok := ci.(map[string]any)
		if !ok {
			continue
		}
		delta, _ := c["delta"].(map[string]any)
		if delta == nil {
			continue
		}
		if d, ok := delta["reasoning_content"].(string); ok && d != "" {
			if err := s.writeReasoning(d); err != nil {
				return err
			}
		}
		if d, ok := delta["content"].(string); ok && d != "" {
			if err := s.writeText(d); err != nil {
				return err
			}
		}
		if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
			if err := s.writeToolCalls(tcs); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *responsesSink) writeReasoning(delta string) error {
	if s.textBegun && (s.curReason == nil || !s.curReason.open) {
		// 正文已开始且推理项已关闭：丢弃迟到推理（见类型注释），不重开交错项。
		return nil
	}
	if s.curReason == nil || s.curReason.closed {
		it, err := s.openItem("reasoning", nil)
		if err != nil {
			return err
		}
		s.curReason = it
	}
	s.curReason.text.WriteString(delta)
	return s.emit("response.reasoning_summary_text.delta", map[string]any{
		"item_id": s.curReason.id, "output_index": s.curReason.index,
		"summary_index": 0, "delta": delta,
	})
}

func (s *responsesSink) writeText(delta string) error {
	if s.curReason != nil && s.curReason.open {
		if err := s.closeItem(s.curReason); err != nil {
			return err
		}
	}
	if s.curMsg == nil || s.curMsg.closed {
		it, err := s.openItem("message", nil)
		if err != nil {
			return err
		}
		s.curMsg = it
	}
	s.textBegun = true
	s.curMsg.text.WriteString(delta)
	return s.emit("response.output_text.delta", map[string]any{
		"item_id": s.curMsg.id, "output_index": s.curMsg.index,
		"content_index": 0, "delta": delta,
	})
}

// writeToolCalls 处理 tool_calls 增量：按 index 归位到输出项，首片建项（带 id/name），
// 后续分片只追加 arguments（name 已由上游归一化收敛为每 index 一次）。
func (s *responsesSink) writeToolCalls(tcs []any) error {
	for _, tci := range tcs {
		tc, ok := tci.(map[string]any)
		if !ok {
			continue
		}
		idx := 0
		if v, ok := tc["index"].(float64); ok {
			idx = int(v)
		}
		fn, _ := tc["function"].(map[string]any)
		args, _ := fn["arguments"].(string)
		name, _ := fn["name"].(string)
		id, _ := tc["id"].(string)

		it, seen := s.tools[idx]
		if !seen {
			// 正文/推理项在工具调用开始前收尾：输出项按产生顺序排列，工具调用是
			// 模型"说完再做"的分界点。
			if s.curReason != nil && s.curReason.open {
				if err := s.closeItem(s.curReason); err != nil {
					return err
				}
			}
			if s.curMsg != nil && s.curMsg.open {
				if err := s.closeItem(s.curMsg); err != nil {
					return err
				}
			}
			var err error
			it, err = s.openItem("function_call", func(it *responsesItem) {
				it.callID = id
				it.name = name
			})
			if err != nil {
				return err
			}
			s.tools[idx] = it
			s.toolOrder = append(s.toolOrder, idx)
		}
		if it.callID == "" && id != "" {
			it.callID = id
		}
		if it.name == "" && name != "" {
			it.name = name
		}
		if args != "" {
			it.arguments.WriteString(args)
			if err := s.emit("response.function_call_arguments.delta", map[string]any{
				"item_id": it.id, "output_index": it.index, "delta": args,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

// RawData 上游 data 行 JSON 解析失败：Responses 无「原样透传」形态，忽略
// （不计数，仍可能触发空流兜底，与 chat 的计数口径一致）。
func (s *responsesSink) RawData(string) error { return nil }

// Comment 上游注释/keep-alive 行：Responses 无对应形态，忽略。
func (s *responsesSink) Comment(string) error { return nil }

// ErrorFrame 上游 error 帧（6004 限流 / 内容拦截 / 审核）→ response.failed 事件。
// 上游 error 对象（code/message/requestId）原样搬进 response.error，另附 gateway_hint。
func (s *responsesSink) ErrorFrame(raw string) error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	s.failed = true
	var errObj map[string]any
	var obj map[string]any
	if json.Unmarshal([]byte(raw), &obj) == nil {
		if e, ok := obj["error"].(map[string]any); ok {
			errObj = e
		}
	}
	if errObj == nil {
		errObj = map[string]any{"message": raw, "type": "upstream_error"}
	}
	if s.meta.HintFn != nil {
		if hint := s.meta.HintFn(raw); hint != "" {
			errObj["gateway_hint"] = hint
		}
	}
	resp := s.responseObject("failed", false)
	resp["error"] = errObj
	return s.emit("response.failed", map[string]any{"response": resp})
}

// Finish 收尾：关闭未闭合项，然后按结果发 response.completed / response.failed，
// 最后补 data: [DONE]（与 chat 前端一致的流结束标记）。
func (s *responsesSink) Finish(empty bool) error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	if s.failed {
		// ErrorFrame 已发 response.failed：不再补发项事件（失败响应不回放半成品项），
		// 只保证流正常收尾。
		return s.writeDone()
	}
	for _, it := range s.output {
		if err := s.closeItem(it); err != nil {
			return err
		}
	}
	if empty {
		// 上游 200 但零有效帧：把已开的事件流收成失败态（HTTP 头已发出只能 200）。
		resp := s.responseObject("failed", false)
		resp["error"] = map[string]any{
			"message": "empty upstream stream",
			"type":    "upstream_error",
			"code":    "upstream_parse",
		}
		if err := s.emit("response.failed", map[string]any{"response": resp}); err != nil {
			return err
		}
		return s.writeDone()
	}
	if err := s.emit("response.completed", map[string]any{"response": s.responseObject("completed", true)}); err != nil {
		return err
	}
	// 与 chat 前端一致：事件流之后仍需补 data: [DONE]（客户端据此判定流正常结束）。
	return s.writeDone()
}

func (s *responsesSink) writeDone() error {
	if _, err := io.WriteString(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// responsesItem Responses 输出项的状态（wire 形态见 wire）。
type responsesItem struct {
	index  int
	kind   string // message | reasoning | function_call
	id     string
	open   bool
	closed bool

	text      strings.Builder // message 正文 / reasoning 摘要
	callID    string          // function_call
	name      string          // function_call
	arguments strings.Builder // function_call
}

// wire 构造该输出项在指定状态下的 wire 形态。
func (it *responsesItem) wire(status string) map[string]any {
	switch it.kind {
	case "message":
		content := []any{}
		if it.text.Len() > 0 {
			content = append(content, map[string]any{
				"type": "output_text", "text": it.text.String(), "annotations": []any{},
			})
		}
		item := map[string]any{"type": "message", "id": it.id, "role": "assistant", "content": content}
		if status != "" {
			item["status"] = status
		}
		return item
	case "reasoning":
		summary := []any{}
		if it.text.Len() > 0 {
			summary = append(summary, map[string]any{"type": "summary_text", "text": it.text.String()})
		}
		return map[string]any{"type": "reasoning", "id": it.id, "summary": summary}
	case "function_call":
		item := map[string]any{
			"type": "function_call", "id": it.id, "call_id": it.callID,
			"name": it.name, "arguments": it.arguments.String(),
		}
		if status != "" {
			item["status"] = status
		}
		return item
	}
	return map[string]any{"id": it.id}
}

func messageItem(text string) map[string]any {
	return map[string]any{
		"type": "message", "id": newResponsesID("msg"), "role": "assistant", "status": "completed",
		"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
	}
}

func reasoningItem(text string) map[string]any {
	return map[string]any{
		"type": "reasoning", "id": newResponsesID("rs"),
		"summary": []any{map[string]any{"type": "summary_text", "text": text}},
	}
}

func functionCallItem(callID, name, args string) map[string]any {
	return map[string]any{
		"type": "function_call", "id": newResponsesID("fc"), "call_id": callID,
		"name": name, "arguments": args, "status": "completed",
	}
}

func responsesIDPrefix(kind string) string {
	switch kind {
	case "message":
		return "msg"
	case "reasoning":
		return "rs"
	case "function_call":
		return "fc"
	}
	return "item"
}

// newResponsesID 生成 Responses 风格的对象 id（<prefix>_<32hex>）。客户端按 id 归并
// 事件与输出项，只要求同流内唯一、跨流不冲突，格式无额外约束。
func newResponsesID(prefix string) string {
	return prefix + "_" + session.NewMessageID()
}

// responsesUsage chat 口径 usage → Responses 口径（input/output_tokens + 明细）。
// 上游 normalizer 已把缓存别名归一（cached_tokens / prompt_tokens_details.cached_tokens），
// 这里只做键名平移，缺失的字段不强造。
func responsesUsage(u map[string]any) map[string]any {
	out := map[string]any{}
	in, hasIn := usageNumber(u, "prompt_tokens", "input_tokens")
	outTok, hasOut := usageNumber(u, "completion_tokens", "output_tokens")
	total, hasTotal := usageNumber(u, "total_tokens")
	if !hasTotal && hasIn && hasOut {
		total, hasTotal = in+outTok, true
	}
	out["input_tokens"] = clampNonNeg(in, hasIn)
	out["output_tokens"] = clampNonNeg(outTok, hasOut)
	out["total_tokens"] = clampNonNeg(total, hasTotal)

	// 两个明细对象恒给键：Codex 的 ResponseUsage 把 cached_tokens / reasoning_tokens
	// 定为**必填**（非 Option），缺键即 "failed to parse ResponseCompleted: missing
	// field `cached_tokens`"，整个响应被判失败并触发重连（实测 0.160.0）。未知填 0
	// 而非省略——0 与「未观测到缓存」语义一致，客户端也不会因缺键拒绝整帧。
	inDetails := map[string]any{"cached_tokens": int64(0)}
	if hit, ok := upstream.UsageCacheHitTokens(u); ok && hit > 0 {
		inDetails["cached_tokens"] = int64(hit)
	}
	out["input_tokens_details"] = inDetails

	outDetails := map[string]any{"reasoning_tokens": int64(0)}
	if rt, ok := usageNumber(u, "reasoning_tokens"); ok && rt > 0 {
		outDetails["reasoning_tokens"] = int64(rt)
	} else if cd, ok := u["completion_tokens_details"].(map[string]any); ok {
		if rt, ok := usageNumber(cd, "reasoning_tokens"); ok && rt > 0 {
			outDetails["reasoning_tokens"] = int64(rt)
		}
	}
	out["output_tokens_details"] = outDetails
	return out
}

// emptyResponsesUsage 无用量观测时的零值对象：Codex 读取 response.completed.usage
// 统计 token，且其明细字段必填（见 responsesUsage），故给完整零值而非省略或空对象。
func emptyResponsesUsage() map[string]any {
	return map[string]any{
		"input_tokens":          0,
		"output_tokens":         0,
		"total_tokens":          0,
		"input_tokens_details":  map[string]any{"cached_tokens": int64(0)},
		"output_tokens_details": map[string]any{"reasoning_tokens": int64(0)},
	}
}

func clampNonNeg(v float64, ok bool) int64 {
	if !ok || v < 0 {
		return 0
	}
	return int64(v)
}

// usageNumber 依次尝试若干键，返回首个数值。
func usageNumber(m map[string]any, keys ...string) (float64, bool) {
	for _, k := range keys {
		switch v := m[k].(type) {
		case float64:
			return v, true
		case int64:
			return float64(v), true
		case int:
			return float64(v), true
		case json.Number:
			if f, err := v.Float64(); err == nil {
				return f, true
			}
		}
	}
	return 0, false
}

// mergeUsage 合并多帧 usage（上游末帧可能分次给：先 token 数、后 credit）。
// 后到的非零字段覆盖先到的，避免零值帧把已观测到的用量冲掉。
func mergeUsage(dst, src map[string]any) map[string]any {
	if dst == nil {
		dst = map[string]any{}
	}
	for k, v := range src {
		if v == nil {
			continue
		}
		if f, ok := v.(float64); ok && f == 0 {
			if _, exists := dst[k]; exists {
				continue
			}
		}
		dst[k] = v
	}
	return dst
}
