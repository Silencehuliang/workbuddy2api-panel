// framesink.go 上游 SSE → 客户端协议的写出层。
//
// 分层动机：上游读取、帧归一化（白名单重建 / tool_call name 收敛 / 消息级 id 续传）、
// 空流兜底这些逻辑与客户端协议**无关**；协议差异只在「帧怎么写给客户端」：
//   - chatSink（本文件）：OpenAI chat.completion.chunk 以 data: 帧逐帧透传；
//   - responsesSink（internal/server/responses.go）：把同一批归一化增量帧翻译成
//     OpenAI Responses API 事件序列，供只认 /v1/responses 的客户端（Codex 等）。
//
// StreamTo 承载协议无关部分，两个前端各实现一个 FrameSink。
// 该拆分让 responses 前端与 chat 前端共用同一条经过实战的归一化链路（上游噪声剥离、
// tool_call 分片合并、error 帧透传、空流兜底），无需各自维护一份。
package upstream

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// FrameSink 客户端帧接收端。实现方自行负责 flush——每帧写完必须立即刷出，
// 否则流式退化成本地缓冲（客户端看到"一次性吐完"）。
type FrameSink interface {
	// SetHeaders 在开流前设置响应头（SSE 相关）。
	SetHeaders()
	// Chunk 写一个归一化后的 chat 增量帧（choices/delta/usage 形态，已白名单重建）。
	Chunk(frame map[string]any) error
	// RawData 写一个无法归一化的 data 帧（上游帧 JSON 解析失败，无法走白名单）：
	// chat 原样转发原文。
	RawData(payload string) error
	// ErrorFrame 写上游 error 帧：raw 为 data: 之后的原文，**未经白名单重建**
	// （必须保留 error.code/message/requestId 等上游字段）。
	ErrorFrame(raw string) error
	// Comment 写上游非 data 行（注释 / keep-alive 等）。chat 原样透传；
	// responses 无对应形态，忽略。
	Comment(line string) error
	// Finish 收尾，恰好调用一次。empty=true 表示上游零有效帧（空流）：
	// chat 先补一帧 error 兜底再写 [DONE]；responses 把已开事件流收成失败态。
	Finish(empty bool) error
}

// Stream 透传上游 SSE 到 w（逐帧规范化后 flush），保证至少写一个 [DONE]。
// 调用方必须先设置过 status 200；本函数自设 SSE headers。
// 流式策略：逐帧透传（规范化已剥空 content 噪声），恢复与上游一致的平滑流式。
//
// StreamHint 变体（gateway_hint 任务）：上游 error 帧透传时附加
// error.gateway_hint 字段——message 原文不动，hint 并列补充；hintFn 返回空串
// 或 nil 时与 Stream 行为逐字节一致。
func Stream(w http.ResponseWriter, r io.Reader) error {
	return StreamHint(w, r, nil)
}

// StreamHint 同 Stream，但上游 error 帧透出前把 hintFn(payload) 的返回值写入
// error.gateway_hint。hintFn 为 nil 或返回空串 → 原样透传（零改写）。
// 空流兜底 error 帧（"empty upstream stream"）不带 hint（网关本地故障形态
// 未覆盖，不编造）。
// opts：旁路观测开关（见 WithErrorFrameObserver），不改变任何透传字节。
func StreamHint(w http.ResponseWriter, r io.Reader, hintFn func(string) string, opts ...StreamOption) error {
	return StreamTo(newChatSink(w, hintFn), r, opts...)
}

// StreamTo 读取上游 SSE，把每一帧交给 sink 编码后写出（协议无关的公共链路）。
// 调用方必须先设置过 status 200；SSE 响应头由 sink.SetHeaders 设置。
//
// 归一化（两个前端共用）：
//   - 带 error 键的帧**不走白名单**（白名单会剥掉 error 字段，客户端就看不到上游
//     code/msg/requestId）→ 交 sink.ErrorFrame 原文转发，且计入有效帧
//     （避免误判空流、补写 "empty upstream stream"）；
//   - 其余帧先按 index 收敛 tool_calls.name（每 index 仅首片保留，见 stripToolCallNames）、
//     续传消息级 id，再按 OpenAI 流式规范白名单重建 → 交 sink.Chunk；
//   - JSON 解析失败的 data 行无法归一化 → 交 sink.RawData 原样降级转发（不计有效帧）。
//
// 收尾：恰好一次 sink.Finish（空流时 empty=true），空流另返回 errEmptyStream
// 供调用方把观测收敛到失败口径（HTTP 头已发出只能 200，但这不是成功）。
func StreamTo(sink FrameSink, r io.Reader, opts ...StreamOption) error {
	var o streamOptions
	for _, f := range opts {
		if f != nil {
			f(&o)
		}
	}
	sink.SetHeaders()

	// toolCallSeen 跨帧记录 delta.tool_calls 里已发过首片的 index，
	// 供逐 chunk 转发时收敛 name 为「每 index 一次」（对齐 OpenAI 官方流）。
	toolCallSeen := map[int]bool{}

	// firstID 透传流的消息级 id 基准：缓存首个非空上游 id，后续帧缺失/空串时复用
	// （issue #35：同一条 SSE 消息所有帧共用一个真实 id，后台按 id 归并；此前中间帧
	// 一律补 chatcmpl-wb2api 哨兵，造成同流 id 分裂）。全流无真实 id → 才出现哨兵。
	firstID := ""

	br := bufio.NewReaderSize(r, 64*1024)
	validFrames := 0
readLoop:
	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(trimmed, "data: [DONE]"):
			// 上游显式结束：停止读取，DONE 之后的任何数据（含垃圾帧）一律不再透传。
			// [DONE] 统一在循环结束后由 sink.Finish 写出，保证恰好一个。
			break readLoop
		case strings.HasPrefix(trimmed, "data: "):
			payload := strings.TrimPrefix(trimmed, "data: ")
			var obj map[string]any
			if json.Unmarshal([]byte(payload), &obj) != nil {
				// 解析失败：无法归一化，原样降级转发（不计数——空流判定只看有效帧）。
				if werr := sink.RawData(payload); werr != nil {
					return werr
				}
				break
			}
			if _, hasErr := obj["error"]; hasErr {
				// 上游错误帧（error-passthrough）：原文透出，白名单会剥掉 error 字段。
				// 旁路通知（账号处置用）：原始 payload 交给观察者，透传字节不变。
				if o.onErrorFrame != nil {
					o.onErrorFrame(payload)
				}
				if werr := sink.ErrorFrame(payload); werr != nil {
					return werr
				}
				validFrames++
				break
			}
			// 先按 index 收敛 tool_calls name（每 index 仅首片保留，后续分片删 name 键），再规范化。
			stripToolCallNames(obj, toolCallSeen)
			// id 续传：首帧非空真实 id 缓存；后续帧缺 id / 空 id 一律用缓存值，
			// 有自己 id 的帧保持原样（不同流分裂的帧允许各自 id）。
			if firstID == "" {
				if v, ok := obj["id"].(string); ok && v != "" {
					firstID = v
				}
			} else if v, ok := obj["id"].(string); !ok || v == "" {
				obj["id"] = firstID
			}
			if werr := sink.Chunk(normalizeFrame(obj)); werr != nil {
				return werr
			}
			validFrames++
		case trimmed != "":
			// 注释/其他行：原样透传
			if werr := sink.Comment(line); werr != nil {
				return werr
			}
		}
		// 空行（帧分隔）吞掉：sink 自产帧分隔
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}
	empty := validFrames == 0
	if err := sink.Finish(empty); err != nil {
		return err
	}
	if empty {
		return errEmptyStream
	}
	return nil
}

// chatSink OpenAI chat 前端：归一化帧以 data: 帧透传。
// 行为与拆分层前的 StreamHint 逐字节一致（含 error 帧的 gateway_hint 附加、
// 空流兜底帧、恰好一个 [DONE]）。
type chatSink struct {
	w      http.ResponseWriter
	fl     http.Flusher
	hintFn func(string) string
}

func newChatSink(w http.ResponseWriter, hintFn func(string) string) *chatSink {
	fl, _ := w.(http.Flusher)
	return &chatSink{w: w, fl: fl, hintFn: hintFn}
}

func (s *chatSink) SetHeaders() {
	h := s.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}

// write 写出一帧 data（payload 原样）+ flush。
func (s *chatSink) write(payload string) error {
	if _, err := io.WriteString(s.w, "data: "+payload+"\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

func (s *chatSink) Chunk(frame map[string]any) error {
	raw, err := json.Marshal(frame)
	if err != nil {
		// 理论不可达：帧由 json.Unmarshal 产出的 map 组成，必然可序列化。
		return err
	}
	return s.write(string(raw))
}

func (s *chatSink) RawData(payload string) error { return s.write(payload) }

// ErrorFrame 上游 error 帧透出：按 hintFn 附加 error.gateway_hint
// （error 对象上加一个键，message/code/requestId 等原文不动；hintFn 为
// nil / 空串 / 非 JSON 帧 → 原样写出，零改写）。
func (s *chatSink) ErrorFrame(raw string) error {
	if hint := frameGatewayHint(s.hintFn, raw); hint != "" {
		raw = attachHintToErrorFrame(raw, hint)
	}
	return s.write(raw)
}

func (s *chatSink) Comment(line string) error {
	if _, err := io.WriteString(s.w, line); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// Finish 空流先补一帧 error（绕过白名单原样保留 error 字段），再保证恰好一个 [DONE]。
// 兜底帧不走 hintFn：该形态未覆盖（不编造 hint），且 hintFn 闭包可能携带上一帧的
// 上下文造成误配。
func (s *chatSink) Finish(empty bool) error {
	if empty {
		if err := s.write(`{"error":{"message":"empty upstream stream","type":"upstream_error","code":"upstream_parse"}}`); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}
