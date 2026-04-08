package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

func extractText(content []anthropic.ContentBlockUnion) string {
	var parts []string
	for _, block := range content {
		if block.Type == "text" && block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func main() {
	//指定路径，覆盖系统环境变量
	if err := godotenv.Overload("../.env"); err != nil {
		panic(err)
	}

	baseURL := os.Getenv("ANTHROPIC_BASE_URL")
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	model := os.Getenv("MODEL_ID")

	if baseURL != "" {
		// 对接兼容 Anthropic 的网关时，避免系统环境里的 Bearer token 干扰鉴权。
		os.Unsetenv("ANTHROPIC_AUTH_TOKEN")
	}

	if apiKey == "" {
		panic("ANTHROPIC_API_KEY is empty")
	}
	if model == "" {
		panic("MODEL_ID is empty")
	}

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)

	message, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		MaxTokens: 1024,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("你好")),
		},
		Model: model,
	})
	if err != nil {
		panic(err.Error())
	}

	fmt.Println(extractText(message.Content))
}
