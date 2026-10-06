// frontend.go 客户端协议前端：把「补全核心」与「客户端看到什么协议」解耦。
//
// 背景：Codex 0.160.0 起只认 Responses API（POST /v1/responses），而网关此前的
// 全部账号调度能力（挑号轮转、粘性、冷却/熔断、错误分类与账号处置、积分与用量
// 观测、面板日志）都长在 chatCompletions 这一个 handler 里。与其为 responses 再抄
// 一份调度逻辑（两份必然漂移），这里把协议差异收敛成三个钩子，核心链路
// （翻译 → 选号轮转 → 上游调用 → 处置 → 出帧）由 completions 独占：
//
//	translate     入站请求体 → 出站 chat 请求体（chat 前端恒等）
//	stream        流式出帧（chat: SSE 透传；responses: 事件序列翻译）
//	writeAggregate 非流式出帧（chat: 原样；responses: 响应对象转换）
//
// 错误出口两个前端共用 writeOpenAIError/writeOpenAIErrorHint：Responses API 的
// HTTP 错误体同样是 {"error":{...}}，无需分叉。
package server

import (
	"io"
	"net/http"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// frontend 客户端协议前端（见文件头注释）。
type frontend interface {
	// translate 把入站请求体翻译成出站 chat 请求体。返回错误时核心以 400 回客户端
	// （请求形态非法，与账号无关，不轮转）。
	translate(raw []byte) ([]byte, error)
	// stream 流式出帧：r 为上游 SSE 流（通常已包一层统计 reader）。
	stream(w http.ResponseWriter, r io.Reader, meta frontendMeta, opts ...upstream.StreamOption) error
	// writeAggregate 非流式出帧：agg 为 upstream.Aggregate 聚合出的 chat.completion 响应。
	writeAggregate(w http.ResponseWriter, agg map[string]any, meta frontendMeta)
}

// frontendMeta 出帧所需的请求侧上下文（两个前端共用）。
type frontendMeta struct {
	// Model 客户端请求的模型名（含 realm 前缀）——responses 出帧需回显该名，
	// Codex 据此校验响应与请求同模型。
	Model string
	// BareModel 出站裸模型名（剥 realm 前缀后），用于 hint 判定。
	BareModel string
	// CreatedAt 请求开始时间，responses 的 created_at 与事件时间戳基准。
	CreatedAt time.Time
	// HintFn SSE error 帧的 gateway_hint 判定函数（惰性求值，正常流零开销）。
	HintFn func(string) string
}

// chatFE chat 前端：入站即 chat 格式，出站按既有 chat 语义透传帧。
// 零翻译、零改写——既有 /v1/chat/completions 行为逐字节不变。
type chatFE struct{}

func (chatFE) translate(raw []byte) ([]byte, error) { return raw, nil }

func (chatFE) stream(w http.ResponseWriter, r io.Reader, meta frontendMeta, opts ...upstream.StreamOption) error {
	return upstream.StreamHint(w, r, meta.HintFn, opts...)
}

func (chatFE) writeAggregate(w http.ResponseWriter, agg map[string]any, _ frontendMeta) {
	writeJSON(w, http.StatusOK, agg)
}
