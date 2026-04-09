package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

// 一些全局变量
const (
	MaxTokens = 8000
	ColorReset  = "\033[0m"
	ColorCyan   = "\033[36m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
)

type AppConfig struct {
	systemPrompt string                     //系统提示词
	baseUrl      string                     //API基础URL
	apiKey       string                     //API密钥
	model        string                     //模型ID
	maxTokens    int64                      //最大token数
	tools        []anthropic.ToolUnionParam //工具列表
}

// LoadConfig 加载配置
func LoadConfig() (AppConfig, error) {
	//指定路径，覆盖系统环境变量
	if err := godotenv.Overload("../.env"); err != nil {
		return AppConfig{}, fmt.Errorf("failed to load .env file: %w", err)
	}

	baseURL := os.Getenv("ANTHROPIC_BASE_URL")
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	model := os.Getenv("MODEL_ID")

	if baseURL != "" {
		// 对接兼容 Anthropic 的网关时，避免系统环境里的 Bearer token 干扰鉴权。
		os.Unsetenv("ANTHROPIC_AUTH_TOKEN")
	}

	if apiKey == "" {
		return AppConfig{}, fmt.Errorf("ANTHROPIC_API_KEY is required")
	}
	if model == "" {
		return AppConfig{}, fmt.Errorf("MODEL_ID is required")
	}

	//加载系统提示词
	cwd, err := os.Getwd()
	if err != nil {
		return AppConfig{}, fmt.Errorf("failed to get current working directory: %w", err)
	}

	return AppConfig{
		systemPrompt: fmt.Sprintf(
			"You are a coding agent at %s. Use bash to inspect and change the workspace. Act first, then report clearly.",
			cwd,
		),
		baseUrl:   baseURL,
		apiKey:    apiKey,
		model:     model,
		maxTokens: MaxTokens,
		tools:     LoadTools(),
	}, nil
}

// 循环状态
type LoopState struct {
	Messages         []anthropic.MessageParam //对话历史
	TurnCount        int                      //对话轮数
	TransitionReason string                   //继续循环的原因
}

// 加载工具
func LoadTools() []anthropic.ToolUnionParam {
	toolParams := []anthropic.ToolParam{
		{
			Name:        "bash",
			Description: anthropic.String("Run a shell command in the current workspace."),
			// 输入参数的JSON Schema，要求必须有一个字符串类型的"command"字段
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"command": map[string]any{
						"type": "string",
					},
				},
				Required: []string{"command"},
			},
		},
	}

	tools := make([]anthropic.ToolUnionParam, len(toolParams))
	for i := range toolParams {
		tools[i] = anthropic.ToolUnionParam{OfTool: &toolParams[i]}
	}
	return tools
}

// 取出text内容
func extractText(content []anthropic.ContentBlockUnion) string {
	var parts []string
	for _, block := range content {
		if block.Type == "text" && block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func isDangerousCommand(command string) bool {
	dangerous := []string{
		"rm -rf /",
		"sudo",
		"shutdown",
		"reboot",
		"> /dev/",
	}

	for _, item := range dangerous {
		if strings.Contains(command, item) {
			return true
		}
	}
	return false
}

// 执行bash工具
func ExecuteBashTool(toolUse anthropic.ToolUseBlock) anthropic.ContentBlockParamUnion {
	var input struct {
		Command string `json:"command"`
	}

	if err := json.Unmarshal(toolUse.Input, &input); err != nil {
		log.Println("Error decoding bash tool input:", err)
		return anthropic.NewToolResultBlock(toolUse.ID, err.Error(), true)
	}
	if strings.TrimSpace(input.Command) == "" {
		return anthropic.NewToolResultBlock(toolUse.ID, "command is required", true)
	}
	if isDangerousCommand(input.Command) {
		return anthropic.NewToolResultBlock(toolUse.ID, "Error: dangerous command blocked", true)
	}
	fmt.Printf("%s$ %s%s\n", ColorYellow, input.Command, ColorReset)

	cwd, err := os.Getwd()
	if err != nil {
		log.Println("Error getting current working directory:", err)
		return anthropic.NewToolResultBlock(toolUse.ID, err.Error(), true)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "/bin/zsh", "-lc", input.Command)
	cmd.Dir = cwd

	output, err := cmd.CombinedOutput()
	result := strings.TrimSpace(string(output))
	if result == "" {
		result = "(no output)"
	}
	if len(result) > 50000 {
		result = result[:50000]
	}
	fmt.Println(result[:min(200, len(result))])

	if err != nil {
		log.Println("Error executing bash tool:", err)
		if ctx.Err() == context.DeadlineExceeded {
			return anthropic.NewToolResultBlock(toolUse.ID, "Error: timeout (120s)", true)
		}
		if _, ok := err.(*exec.ExitError); ok && result != "" {
			return anthropic.NewToolResultBlock(toolUse.ID, result, true)
		}
		return anthropic.NewToolResultBlock(toolUse.ID, err.Error(), true)
	}

	return anthropic.NewToolResultBlock(toolUse.ID, result, false)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 按照工具名称分发工具调用
func ExecuteToolCall(toolUse anthropic.ToolUseBlock) anthropic.ContentBlockParamUnion {
	switch toolUse.Name {
	case "bash":
		return ExecuteBashTool(toolUse)
	default:
		return anthropic.NewToolResultBlock(toolUse.ID, "unknown tool: "+toolUse.Name, true)
	}
}

// 跑一轮对话
func RunOneTurn(state *LoopState, client *anthropic.Client, config AppConfig) bool {
	// 发起一次模型调用
	response, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		MaxTokens: config.maxTokens,
		Messages:  state.Messages,
		Model:     config.model,
		System: []anthropic.TextBlockParam{
			{Text: config.systemPrompt},
		},
		Tools: config.tools,
	})
	if err != nil {
		log.Println("Error calling model:", err)
		return false
	}
	// 将模型的回复添加到对话历史中
	state.Messages = append(state.Messages, response.ToParam())
	finalText := extractText(response.Content)
	if finalText != "" {
		fmt.Printf("%sassistant >> %s%s\n\n", ColorGreen, finalText, ColorReset)
		fmt.Println()
	}

	// 检查模型是否调用了工具，如果调用了工具则继续循环，否则结束循环
	if response.StopReason != "tool_use" {
		state.TransitionReason = ""
		return false
	}

	//工具调用部分
	toolResults := make([]anthropic.ContentBlockParamUnion, 0)//收集工具结果的切片

	for _, block := range response.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.ToolUseBlock:
			toolResults = append(toolResults, ExecuteToolCall(variant))
		}
	}

	// 没有工具结果，结束循环
	if len(toolResults) == 0 {
		state.TransitionReason = ""
		return false
	}

	state.Messages = append(state.Messages, anthropic.NewUserMessage(toolResults...))//以user消息返回给模型
	state.TurnCount++
	state.TransitionReason = "tool_result"
	return true

}

// Agent循环
func AgentLoop(state *LoopState, client *anthropic.Client, config AppConfig) {
	for RunOneTurn(state, client, config) {
	}
}

func main() {
	//加载配置
	config, err := LoadConfig()
	if err != nil {
		panic(err.Error())
	}

	//创建Anthropic客户端
	client := anthropic.NewClient(
		option.WithAPIKey(config.apiKey),
		option.WithBaseURL(config.baseUrl),
	)

	// 简单的命令行界面，用户输入问题，模型回复并调用工具，循环往复直到用户退出
	scanner := bufio.NewScanner(os.Stdin)
	history := make([]anthropic.MessageParam, 0)

	for {
		fmt.Printf("%sgo-agent >> %s", ColorCyan, ColorReset)
		if !scanner.Scan() {
			break
		}

		// 读取用户输入，如果输入为空或者是"q"或"exit"，则退出循环
		query := strings.TrimSpace(scanner.Text())
		if query == "" || query == "q" || query == "exit" {
			break
		}

		// 将用户输入添加到对话历史中，并进入Agent循环
		history = append(history, anthropic.NewUserMessage(anthropic.NewTextBlock(query)))
		state := LoopState{
			Messages:         history,
			TurnCount:        1,
			TransitionReason: "",
		}
		AgentLoop(&state, &client, config)
		history = state.Messages
	}

	if err := scanner.Err(); err != nil {
		log.Println("Error reading input:", err)
	}
}
