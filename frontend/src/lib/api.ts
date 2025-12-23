import type { AgentChatV2Request, AgentChatV2Response } from '../types/agent'

function normalizeBaseUrl(raw: string | undefined): string {
  const v = (raw ?? '').trim()
  if (!v) return 'http://127.0.0.1:8000'
  return v.endsWith('/') ? v.slice(0, -1) : v
}

export function getApiBaseUrl(): string {
  return normalizeBaseUrl((import.meta as any).env?.VITE_API_BASE_URL as string | undefined)
}

export async function chatV2(req: AgentChatV2Request, opts?: { signal?: AbortSignal }): Promise<AgentChatV2Response> {
  const baseUrl = getApiBaseUrl()

  const resp = await fetch(`${baseUrl}/v1/agent/chat_v2`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
    signal: opts?.signal,
  })

  if (!resp.ok) {
    const t = await resp.text().catch(() => '')
    throw new Error(t || `HTTP ${resp.status}`)
  }

  const data = (await resp.json()) as unknown
  return data as AgentChatV2Response
}
